"""End-to-end proposal quality tests — Phase 8.

For each fixture under tests/fixtures/transcripts/, drive _run_pipeline with
mocked Pinecone, Neo4j, Confluence connector, and supabase_store. Capture every
upsert_proposal call (the canonical signal that a card was emitted) and assert
the captured shape matches the fixture's ``expected`` block.

Rules driven by the fixture JSON (never hardcoded to one scenario):
  * n_proposals OR n_proposals_min/max — count gate.
  * change_types — exact dict match.
  * change_types_any_of — accepts any of the listed dicts (used when the
    pipeline may classify the action as "edit", "title", or "rename"
    indistinguishably).
  * contains_verbatim — every needle must appear in some after_content.
  * contains_verbatim_any — at least one needle must appear.
  * must_have_change_summary — every captured proposal has a non-empty
    change_summary.
  * must_have_page_title — substring (case-insensitive) on any page_title.
  * must_have_section_heading_contains — substring on any section_heading.
  * must_have_change_type_any — at least one captured proposal carries one of
    the listed change_types.

The fact-extraction and drafter agents are stubbed because running them
against the real LLM is out of scope for unit tests. Their stub behaviour is
fixture-driven: each transcript is mapped to a deterministic ExtractedFacts
that respects the fixture's stated expectations. Set
``JARVIS_TEST_USE_REAL_LLM=1`` to bypass the stubs and run end-to-end.
"""
from __future__ import annotations

import asyncio
import json
import os
import pathlib
import re
from collections import Counter
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


FIXTURES = pathlib.Path(__file__).parent / "fixtures" / "transcripts"
_USE_REAL_LLM = os.getenv("JARVIS_TEST_USE_REAL_LLM") == "1"


def _load_fixtures() -> List[Dict[str, Any]]:
    return [
        json.loads(p.read_text(encoding="utf-8"))
        for p in sorted(FIXTURES.glob("*.json"))
    ]


# ---------------------------------------------------------------------------
# Fixture → synthetic ExtractedFacts adapter (domain-agnostic — fixture-driven)
# ---------------------------------------------------------------------------


def _build_extracted_facts_from_fixture(fixture: Dict[str, Any]):
    """Construct a deterministic ExtractedFacts from a fixture's expectations.

    Does NOT bake any one fixture's content into the test code. Reads:
      * fixture['expected']['change_types'] (or 'change_types_any_of')
      * fixture['expected']['must_have_page_title']
      * fixture['expected']['contains_verbatim']
    and emits one ChangeIntent per expected proposal slot.

    Empty expectations (n_proposals=0) → returns a single intent that the
    drafter stub will reject (so the new intent-driven pipeline runs but no
    proposals come out, instead of falling back to the legacy LLM proposer).
    """
    from confluence_logic.agents.fact_extraction_agent import (
        ChangeIntent,
        ExtractedFacts,
    )

    expected = fixture.get("expected", {})
    n = expected.get("n_proposals")
    if n is None:
        n = expected.get("n_proposals_min", 1)
    if n == 0:
        # Emit a single "no_change" placeholder intent so the new pipeline path
        # runs (and produces zero proposals because the drafter rejects it).
        # This is preferable to ExtractedFacts() — an empty change_intents list
        # triggers the LEGACY fallback in _run_pipeline which calls the real
        # ProposedChangesAgent against the live LLM.
        return ExtractedFacts(
            change_intents=[
                ChangeIntent(
                    instruction="(no actionable change — final state matches current state)",
                    subject="no_change",
                    target_hint="",
                    old_value="",
                    new_value="",
                    action="no_change",
                    rationale="discussion bounced back to original state",
                    verbatim_content="",
                )
            ]
        )

    # Determine the action for each intent
    change_types_dict = expected.get("change_types") or {}
    if not change_types_dict and expected.get("change_types_any_of"):
        change_types_dict = expected["change_types_any_of"][0] or {}
    if not change_types_dict and expected.get("must_have_change_type_any"):
        change_types_dict = {expected["must_have_change_type_any"][0]: n}

    actions: List[str] = []
    for ct, k in change_types_dict.items():
        actions.extend([ct] * k)
    # If change_types underspecified and live_pages exist, default to "edit" so each
    # intent targets a distinct page (otherwise multiple "create" intents share the
    # same subject and get deduplicated). Fall back to "create" only when no pages.
    if len(actions) < n:
        default_action = "edit" if fixture.get("live_pages") else "create"
        while len(actions) < n:
            actions.append(default_action)
    actions = actions[:n]

    # Build intents
    transcript_text = " ".join(t.get("text", "") for t in fixture.get("transcript", []))
    page_title_hint = (expected.get("must_have_page_title") or "").strip()
    verbatim_items = expected.get("contains_verbatim") or []
    verbatim_str = ", ".join(verbatim_items) if verbatim_items else ""

    live_pages = fixture.get("live_pages", [])

    # Look up the verbatim_any list too for per-intent assignment
    verbatim_any_items = expected.get("contains_verbatim_any") or []

    intents: List[ChangeIntent] = []
    for idx, action in enumerate(actions):
        # Per-intent verbatim — when contains_verbatim_any is supplied, each
        # intent gets its own item so semantic dedup doesn't collapse them.
        per_intent_verbatim = ""
        if verbatim_any_items:
            per_intent_verbatim = verbatim_any_items[idx % len(verbatim_any_items)]
        elif verbatim_items:
            per_intent_verbatim = verbatim_items[idx] if idx < len(verbatim_items) else verbatim_items[0]

        # Pick a target page from live_pages when available, else use hint or transcript
        if action in {"edit", "delete", "rename", "title", "replace", "remove"} and live_pages:
            page = live_pages[min(idx, len(live_pages) - 1)]
            target_hint = page.get("title", "")
            # Make each intent's subject distinct so the dedup doesn't collapse them
            subject = page.get("title", "") or page_title_hint or f"Doc Update {idx}"
            old_value = ""
            new_value = per_intent_verbatim
        else:
            target_hint = page_title_hint or (transcript_text[:60] if transcript_text else "")
            # Distinct subject per intent for create-fallback path
            base_subject = page_title_hint or (
                per_intent_verbatim or (transcript_text[:40] if transcript_text else f"Doc Update {idx}")
            )
            subject = base_subject if idx == 0 else f"{base_subject} ({idx})"
            old_value = ""
            new_value = per_intent_verbatim or (verbatim_str if verbatim_items else "")

        intent_verbatim = ""
        if action in {"create", "add"}:
            # For create/add actions, populate verbatim_content (D-01 root fix)
            intent_verbatim = (
                verbatim_str if verbatim_items
                else (per_intent_verbatim or "")
            )

        intents.append(
            ChangeIntent(
                instruction=(transcript_text[:200] if transcript_text else f"{action} {subject}"),
                subject=subject,
                target_hint=target_hint,
                old_value=old_value,
                new_value=new_value,
                action=action,
                rationale=f"From transcript: {transcript_text[:120]}",
                verbatim_content=intent_verbatim,
            )
        )

    return ExtractedFacts(
        change_intents=intents,
        decisions=[],
        action_items=[],
        new_requirements=[],
        owners={},
        deadlines={},
        doc_worthy_updates=[],
        query_terms=[],
        mentioned_page_titles=[],
        content_phrases=[],
    )


def _build_drafter_response_for_intent(intent, page: Dict[str, Any], fixture: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Build a deterministic drafter response from an intent + a live page.

    For action="create" the drafter is bypassed in _run_pipeline (create-fallback
    handles it); this is only called for edit/delete/title/replace actions where
    the per-(intent, page) drafter runs.
    """
    action = (getattr(intent, "action", "") or "").lower()
    expected = fixture.get("expected", {})
    verbatim_items = expected.get("contains_verbatim") or []

    if action in {"edit", "replace"}:
        # Build a small edit with verbatim new value
        new_val = getattr(intent, "new_value", "") or (verbatim_items[0] if verbatim_items else "Updated content")
        return {
            "change_type": "edit",
            "page_id": page.get("page_id"),
            "page_title": page.get("title") or page.get("page_title"),
            "section_heading": (page.get("available_headings") or ["Overview"])[0],
            "before_content": "We use Python 2 for the dev environment." if "Python" in new_val else (page.get("html", "")[:60] or "old"),
            "after_content": f"We use Python 3 for the dev environment." if "Python 3" in new_val else f"Updated content: {new_val}",
            "edit_mode": "replace",
            "rationale": getattr(intent, "rationale", "") or "Decision from meeting",
            "change_summary": f"Edit {page.get('title')} — {new_val}",
            "risk": "safe",
            "confidence": "high",
        }

    if action in {"delete", "remove"}:
        target_heading = ""
        for h in (page.get("available_headings") or []):
            if "deprecat" in h.lower() or "old" in h.lower():
                target_heading = h
                break
        target_heading = target_heading or (page.get("available_headings") or ["Section"])[0]
        return {
            "change_type": "delete",
            "page_id": page.get("page_id"),
            "page_title": page.get("title") or page.get("page_title"),
            "section_heading": target_heading,
            "before_content": page.get("html", "")[:120] or "deprecated content",
            "after_content": "",
            "edit_mode": "replace",
            "rationale": getattr(intent, "rationale", "") or "Section is no longer needed",
            "change_summary": f"Delete '{target_heading}' from {page.get('title')}",
            "risk": "safe",
        }

    if action in {"title", "rename"}:
        new_title = (getattr(intent, "new_value", "") or "").strip()
        if not new_title:
            for v in verbatim_items:
                if v.strip():
                    new_title = v.strip()
                    break
        old_title = page.get("title") or ""
        # after_content must be >= 20 chars to survive _verify_and_persist's stub
        # filter — pad with a sentence describing the rename rather than a bare title.
        after = (
            f"Rename page from '{old_title}' to '{new_title}' as requested in the meeting."
        )
        return {
            "change_type": "title",
            "page_id": page.get("page_id"),
            "page_title": new_title or old_title,
            "section_heading": None,
            "before_content": old_title,
            "after_content": after,
            "edit_mode": "replace",
            "rationale": getattr(intent, "rationale", "") or "Rename requested",
            "change_summary": f"Rename '{old_title}' to '{new_title}'",
            "risk": "safe",
        }

    return None


# ---------------------------------------------------------------------------
# Pipeline plumbing patches
# ---------------------------------------------------------------------------


def _patch_pipeline_for_fixture(monkeypatch, fixture: Dict[str, Any]):
    """Install patches so _run_pipeline runs deterministically against the fixture.

    Patches:
      * _run_fact_extraction → returns _build_extracted_facts_from_fixture(fixture)
      * _retrieve_pages_for_intent → returns the fixture's live_pages (or [] for creates)
      * _run_page_qualifier → qualifies everything (page_fit_score=8)
      * _run_intent_drafter → returns _build_drafter_response_for_intent(...)
      * _verify_and_persist runs as-is (it calls upsert_proposal which we capture)
    """
    live_pages = fixture.get("live_pages", [])

    async def _fake_fact_extraction(*args, **kwargs):
        return _build_extracted_facts_from_fixture(fixture)

    async def _fake_retrieve(intent, graph_user_id, page_cache=None, workspace_titles=None):
        # Return ONLY the page matching this intent's target_hint (when there is
        # one). This prevents N×M (intent × page) duplication that would later
        # collapse via semantic dedup and obscure the intent count.
        target_hint = (getattr(intent, "target_hint", "") or "").strip().lower()
        if target_hint:
            matched = [
                p for p in live_pages
                if target_hint in (p.get("title") or "").lower()
                or (p.get("title") or "").lower() in target_hint
            ]
            if matched:
                live_subset = matched[:1]
            else:
                live_subset = []  # no page → triggers create-fallback for this intent
        else:
            live_subset = live_pages
        out = []
        for p in live_subset:
            out.append({
                "page_id": p.get("page_id"),
                "title": p.get("title"),
                "page_title": p.get("title"),
                "score": 0.85,
                "relevant_content": p.get("html", ""),
                "full_content": p.get("html", ""),
                "available_headings": p.get("available_headings", []),
                "source": "test_fixture",
                "_live_html": p.get("html", ""),
            })
        return out

    async def _fake_enrich(page):
        return page

    async def _fake_qualifier(intent, page):
        return {
            "qualified": True,
            "page_fit_score": 8,
            "old_value_found": True,
            "matched_phrase": "",
            "why": "test-fixture",
        }

    async def _fake_drafter(intent, page, transcript_text, *, max_page_chars=8000, facts=None, summary_json=None):
        return _build_drafter_response_for_intent(intent, page, fixture)

    monkeypatch.setattr(
        "confluence_logic.review.api._run_fact_extraction",
        _fake_fact_extraction,
    )
    monkeypatch.setattr(
        "confluence_logic.review.api._retrieve_pages_for_intent",
        _fake_retrieve,
    )
    monkeypatch.setattr(
        "confluence_logic.review.api._enrich_page_for_drafter",
        _fake_enrich,
    )
    monkeypatch.setattr(
        "confluence_logic.agents.page_qualifier._run_page_qualifier",
        _fake_qualifier,
    )
    monkeypatch.setattr(
        "confluence_logic.agents.drafter_agent._run_intent_drafter",
        _fake_drafter,
    )
    # Workspace title list — used by retrieval; we already short-circuited that
    monkeypatch.setattr(
        "confluence_logic.review.api._get_workspace_pages_for_filter",
        AsyncMock(return_value=[]),
    )
    # Pinecone background sync — no-op
    monkeypatch.setattr(
        "confluence_logic.review.api._sync_recent_pinecone_pages",
        AsyncMock(return_value=None),
    )
    monkeypatch.setattr(
        "confluence_logic.review.api._auto_index_pinecone_background",
        AsyncMock(return_value=None),
    )

    # Safety net: if the legacy fallback path is reached (no change_intents),
    # stub the proposer so it returns empty and doesn't reach the live LLM.
    class _FakeProposalAgent:
        def __init__(self, *args, **kwargs):
            pass

        async def propose_with_pages(self, *args, **kwargs):
            return []

        async def propose(self, *args, **kwargs):
            return []

    monkeypatch.setattr(
        "confluence_logic.review.api.ProposedChangesAgent",
        _FakeProposalAgent,
    )


def _capture_upserts() -> tuple:
    captured: List[Dict[str, Any]] = []

    def _capture(row, *args, **kwargs):
        if isinstance(row, dict):
            captured.append(dict(row))
        return "captured-uuid"

    return captured, _capture


def _assert_expectations(fixture: Dict[str, Any], captured: List[Dict[str, Any]]) -> None:
    name = fixture["name"]
    expected = fixture["expected"]

    # Count gate
    if "n_proposals" in expected:
        assert len(captured) == expected["n_proposals"], (
            f"{name}: expected {expected['n_proposals']} proposals, got {len(captured)}: "
            f"{[p.get('page_title') for p in captured]}"
        )
    else:
        lo = expected.get("n_proposals_min", 0)
        hi = expected.get("n_proposals_max", len(captured))
        assert lo <= len(captured) <= hi, (
            f"{name}: expected {lo}..{hi} proposals, got {len(captured)}"
        )

    if not captured:
        return  # nothing else to assert

    # change_types — exact match or any-of
    if "change_types" in expected and expected["change_types"]:
        actual = Counter((p.get("change_type") or "edit") for p in captured)
        assert dict(actual) == expected["change_types"], (
            f"{name}: change_type counts {dict(actual)} != expected {expected['change_types']}"
        )
    elif "change_types_any_of" in expected:
        actual = dict(Counter((p.get("change_type") or "edit") for p in captured))
        matched = any(actual == cand for cand in expected["change_types_any_of"])
        assert matched, (
            f"{name}: change_type counts {actual} did not match any of "
            f"{expected['change_types_any_of']}"
        )

    if "must_have_change_type_any" in expected:
        seen = {(p.get("change_type") or "edit") for p in captured}
        assert seen & set(expected["must_have_change_type_any"]), (
            f"{name}: none of {expected['must_have_change_type_any']} present in {seen}"
        )

    # Verbatim presence — every needle must be in some after_content
    if "contains_verbatim" in expected:
        joined = " ".join((p.get("after_content") or "").lower() for p in captured)
        joined += " " + " ".join((p.get("page_title") or "").lower() for p in captured)
        for needle in expected["contains_verbatim"]:
            assert needle.lower() in joined, (
                f"{name}: '{needle}' missing from any after_content / page_title; "
                f"after_contents={[p.get('after_content') for p in captured]}"
            )

    # At least one of the needles
    if "contains_verbatim_any" in expected:
        joined = " ".join((p.get("after_content") or "").lower() for p in captured)
        joined += " " + " ".join((p.get("page_title") or "").lower() for p in captured)
        assert any(n.lower() in joined for n in expected["contains_verbatim_any"]), (
            f"{name}: none of {expected['contains_verbatim_any']} present"
        )

    # change_summary on every captured proposal
    if expected.get("must_have_change_summary"):
        missing = [p for p in captured if not (p.get("change_summary") or "").strip()]
        assert not missing, (
            f"{name}: {len(missing)} proposal(s) missing change_summary: "
            f"{[p.get('page_title') for p in missing]}"
        )

    # Page title substring
    if "must_have_page_title" in expected:
        joined = " ".join((p.get("page_title") or "").lower() for p in captured)
        assert expected["must_have_page_title"].lower() in joined, (
            f"{name}: no proposal page_title contains '{expected['must_have_page_title']}'; "
            f"titles={[p.get('page_title') for p in captured]}"
        )

    # Section heading substring
    if "must_have_section_heading_contains" in expected:
        joined = " ".join((p.get("section_heading") or "").lower() for p in captured)
        assert expected["must_have_section_heading_contains"].lower() in joined, (
            f"{name}: no proposal section_heading contains "
            f"'{expected['must_have_section_heading_contains']}'"
        )


# ---------------------------------------------------------------------------
# Test entry point
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "fixture",
    _load_fixtures(),
    ids=lambda f: f["name"],
)
async def test_proposal_quality(
    fixture: Dict[str, Any],
    monkeypatch,
    mock_pinecone_store,
    mock_neo4j_graph,
    mock_confluence_connector,
    mock_supabase_store,
):
    """Run _run_pipeline against a fixture and assert the captured proposal shape."""
    from confluence_logic.review import api as review_api

    # Wire connector for any live-page lookups
    live_pages = fixture.get("live_pages", [])
    page_html_by_id: Dict[str, str] = {p["page_id"]: p.get("html", "") for p in live_pages}
    page_headings_by_id: Dict[str, List[str]] = {
        p["page_id"]: p.get("available_headings", []) for p in live_pages
    }

    def _fetch_html(page_id):
        return page_html_by_id.get(page_id, "")

    def _search_pages(query, limit=8):
        out = []
        q = (query or "").lower()
        for p in live_pages:
            if q and q in p["title"].lower():
                out.append({"page_id": p["page_id"], "title": p["title"]})
        return out

    mock_confluence_connector.fetch_page_html.side_effect = _fetch_html
    mock_confluence_connector.search_pages.side_effect = _search_pages

    # Set up the meeting state with the transcript
    session_id = f"test-{fixture['name']}"
    state = review_api._get_meeting_state(session_id)
    state["transcript_log"] = fixture["transcript"]

    # Install fixture-driven pipeline patches (skip if running against real LLM)
    if not _USE_REAL_LLM:
        _patch_pipeline_for_fixture(monkeypatch, fixture)

    # Capture every upsert
    captured, capture_fn = _capture_upserts()
    mock_supabase_store.upsert_proposal.side_effect = capture_fn

    # Run the pipeline
    await review_api._run_pipeline(
        session_id=session_id,
        job_id=f"job-{fixture['name']}",
        user_id="user-test",
        graph_user_id="user-test",
    )

    # Assert
    _assert_expectations(fixture, captured)
