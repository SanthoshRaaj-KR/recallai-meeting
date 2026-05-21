"""End-to-end proposal quality scorecard — Phase 8.

Run:
    conda activate ml && python -m tests.e2e_proposal_quality_eval

Loads each fixture in tests/fixtures/transcripts/, runs _run_pipeline with the
same fixture-driven mocks as tests/test_proposal_quality.py, prints PASS/FAIL
per assertion, exits nonzero on any failure.

Designed to be safe to run in CI: no network calls, no LLM calls.
"""
from __future__ import annotations

import asyncio
import json
import pathlib
import sys
from collections import Counter
from typing import Any, Dict, List, Tuple
from unittest.mock import AsyncMock, MagicMock


FIXTURES = pathlib.Path(__file__).parent / "fixtures" / "transcripts"


# ---------------------------------------------------------------------------
# Reuse the proposal-quality test's mock-builders so the harness stays in sync
# with the pytest assertions. We import lazily because importing the test
# module at top-level would also collect its pytest-parametrize decorators.
# ---------------------------------------------------------------------------


def _import_helpers():
    from tests.test_proposal_quality import (
        _build_extracted_facts_from_fixture,
        _build_drafter_response_for_intent,
    )
    return _build_extracted_facts_from_fixture, _build_drafter_response_for_intent


# ---------------------------------------------------------------------------
# Per-fixture evaluation
# ---------------------------------------------------------------------------


async def _run_fixture(fixture: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Run _run_pipeline against a fixture and return every captured upsert payload."""
    from confluence_logic.review import api as review_api
    from confluence_logic.review import supabase_store
    from confluence_logic.confluence_page_graph import (
        query_user_confluence_graph,  # noqa: F401
    )

    build_facts, build_draft = _import_helpers()
    live_pages = fixture.get("live_pages", [])
    page_html_by_id = {p["page_id"]: p.get("html", "") for p in live_pages}

    captured: List[Dict[str, Any]] = []

    # ── connector stub ──
    connector = MagicMock()
    connector.fetch_page_html.side_effect = lambda pid: page_html_by_id.get(pid, "")
    connector.get_page_metadata.return_value = {"version": {"number": 1}}
    connector.push_update.return_value = True
    connector.create_page.return_value = {"id": "new-page-id"}
    connector.search_pages.side_effect = lambda q, limit=8: [
        {"page_id": p["page_id"], "title": p["title"]}
        for p in live_pages
        if (q or "").lower() in p["title"].lower()
    ]
    connector.get_workspace_titles = MagicMock(return_value=[])

    # ── async stubs ──
    async def _fake_fact_extraction(*args, **kwargs):
        return build_facts(fixture)

    async def _fake_retrieve(intent, graph_user_id, page_cache=None, workspace_titles=None):
        return [
            {
                "page_id": p["page_id"],
                "title": p["title"],
                "page_title": p["title"],
                "score": 0.85,
                "relevant_content": p.get("html", ""),
                "full_content": p.get("html", ""),
                "available_headings": p.get("available_headings", []),
                "source": "test_fixture",
                "_live_html": p.get("html", ""),
            }
            for p in live_pages
        ]

    async def _fake_enrich(page):
        return page

    async def _fake_qualifier(intent, page):
        return {
            "qualified": True,
            "page_fit_score": 8,
            "old_value_found": True,
            "matched_phrase": "",
            "why": "e2e-test",
        }

    async def _fake_drafter(intent, page, transcript_text, *, max_page_chars=8000, facts=None, summary_json=None):
        return build_draft(intent, page, fixture)

    def _upsert_capture(row, *args, **kwargs):
        if isinstance(row, dict):
            captured.append(dict(row))
        return "captured-uuid"

    # ── monkeypatch.setattr equivalent — direct attribute swap ──
    originals: List[Tuple[Any, str, Any]] = []

    def _swap(module, name, replacement):
        originals.append((module, name, getattr(module, name)))
        setattr(module, name, replacement)

    try:
        import confluence_logic.confluence_page_graph as _cpg
        import confluence_logic.agents.fact_extraction_agent as _fea
        import confluence_logic.agents.page_qualifier as _pq
        import confluence_logic.agents.drafter_agent as _da
        import confluence_logic.review.supabase_store as _ss

        _swap(review_api, "_get_connector", lambda: connector)
        _swap(review_api, "_resolve_page_id", AsyncMock(side_effect=lambda pid, title=None: pid))
        _swap(review_api, "_run_fact_extraction", _fake_fact_extraction)
        _swap(review_api, "_retrieve_pages_for_intent", _fake_retrieve)
        _swap(review_api, "_enrich_page_for_drafter", _fake_enrich)
        _swap(review_api, "_get_workspace_pages_for_filter", AsyncMock(return_value=[]))
        _swap(review_api, "_sync_recent_pinecone_pages", AsyncMock(return_value=None))
        _swap(review_api, "_auto_index_pinecone_background", AsyncMock(return_value=None))
        _swap(review_api, "_merged_rag_retrieval", AsyncMock(return_value=[]))

        _swap(_pq, "_run_page_qualifier", _fake_qualifier)
        _swap(_da, "_run_intent_drafter", _fake_drafter)
        _swap(_cpg, "ensure_user_confluence_graph", AsyncMock(return_value=None))
        _swap(_cpg, "query_user_confluence_graph", AsyncMock(return_value=[]))
        _swap(_fea, "_store", None)

        _swap(_ss, "is_configured", lambda: True)
        _swap(_ss, "upsert_proposal", _upsert_capture)
        _swap(_ss, "update_pipeline_job", lambda *a, **kw: None)
        _swap(_ss, "create_pipeline_job", lambda *a, **kw: "job-test")
        _swap(_ss, "get_history_item", lambda *a, **kw: {})

        # Set up meeting state with the transcript
        session_id = f"e2e-{fixture['name']}"
        state = review_api._get_meeting_state(session_id)
        state["transcript_log"] = fixture["transcript"]

        await review_api._run_pipeline(
            session_id=session_id,
            job_id=f"job-{fixture['name']}",
            user_id="user-test",
            graph_user_id="user-test",
        )
    finally:
        # Restore originals so subsequent fixtures see clean modules
        for module, name, original in reversed(originals):
            setattr(module, name, original)

    return captured


def _evaluate_against_expectations(
    fixture: Dict[str, Any], captured: List[Dict[str, Any]]
) -> List[Tuple[str, bool, str]]:
    """Return list of (assertion_name, passed, detail) tuples for the fixture."""
    results: List[Tuple[str, bool, str]] = []
    expected = fixture.get("expected", {})

    # n_proposals
    if "n_proposals" in expected:
        ok = len(captured) == expected["n_proposals"]
        results.append((
            "n_proposals",
            ok,
            f"expected={expected['n_proposals']}, got={len(captured)}",
        ))
    else:
        lo = expected.get("n_proposals_min", 0)
        hi = expected.get("n_proposals_max", 10)
        ok = lo <= len(captured) <= hi
        results.append((
            "n_proposals_range",
            ok,
            f"expected={lo}..{hi}, got={len(captured)}",
        ))

    if not captured:
        return results

    # change_types
    actual_ct = dict(Counter((p.get("change_type") or "edit") for p in captured))
    if expected.get("change_types"):
        ok = actual_ct == expected["change_types"]
        results.append((
            "change_types",
            ok,
            f"expected={expected['change_types']}, got={actual_ct}",
        ))
    elif expected.get("change_types_any_of"):
        ok = any(actual_ct == cand for cand in expected["change_types_any_of"])
        results.append((
            "change_types_any_of",
            ok,
            f"expected one of {expected['change_types_any_of']}, got={actual_ct}",
        ))

    if "must_have_change_type_any" in expected:
        seen = {(p.get("change_type") or "edit") for p in captured}
        ok = bool(seen & set(expected["must_have_change_type_any"]))
        results.append((
            "must_have_change_type_any",
            ok,
            f"expected any of {expected['must_have_change_type_any']}, got {seen}",
        ))

    # contains_verbatim
    if "contains_verbatim" in expected:
        joined = " ".join((p.get("after_content") or "").lower() for p in captured)
        joined += " " + " ".join((p.get("page_title") or "").lower() for p in captured)
        missing = [n for n in expected["contains_verbatim"] if n.lower() not in joined]
        ok = not missing
        results.append((
            "contains_verbatim",
            ok,
            "all present" if ok else f"missing: {missing}",
        ))

    if "contains_verbatim_any" in expected:
        joined = " ".join((p.get("after_content") or "").lower() for p in captured)
        joined += " " + " ".join((p.get("page_title") or "").lower() for p in captured)
        ok = any(n.lower() in joined for n in expected["contains_verbatim_any"])
        results.append((
            "contains_verbatim_any",
            ok,
            f"at least one of {expected['contains_verbatim_any']}"
            if ok else f"none of {expected['contains_verbatim_any']} present",
        ))

    # must_have_change_summary
    if expected.get("must_have_change_summary"):
        missing = [
            p.get("page_title")
            for p in captured
            if not (p.get("change_summary") or "").strip()
        ]
        ok = not missing
        results.append((
            "must_have_change_summary",
            ok,
            "all have summaries" if ok else f"missing on: {missing}",
        ))

    # must_have_page_title
    if "must_have_page_title" in expected:
        joined = " ".join((p.get("page_title") or "").lower() for p in captured)
        ok = expected["must_have_page_title"].lower() in joined
        results.append((
            "must_have_page_title",
            ok,
            f"need substring '{expected['must_have_page_title']}'"
            f" in titles {[p.get('page_title') for p in captured]}",
        ))

    # must_have_section_heading_contains
    if "must_have_section_heading_contains" in expected:
        joined = " ".join((p.get("section_heading") or "").lower() for p in captured)
        ok = expected["must_have_section_heading_contains"].lower() in joined
        results.append((
            "must_have_section_heading_contains",
            ok,
            f"need substring '{expected['must_have_section_heading_contains']}'"
            f" in headings {[p.get('section_heading') for p in captured]}",
        ))

    return results


async def evaluate_fixture(fixture: Dict[str, Any]) -> List[Tuple[str, bool, str]]:
    """Run the pipeline for one fixture and grade it. Returns the result list."""
    try:
        captured = await _run_fixture(fixture)
    except Exception as exc:  # noqa: BLE001
        return [("pipeline_executed", False, f"raised {type(exc).__name__}: {exc}")]
    return _evaluate_against_expectations(fixture, captured)


# ---------------------------------------------------------------------------
# Scorecard main
# ---------------------------------------------------------------------------


async def main() -> int:
    fixtures = [
        json.loads(p.read_text(encoding="utf-8"))
        for p in sorted(FIXTURES.glob("*.json"))
    ]
    total_assertions = 0
    total_passed = 0
    failed_fixtures: List[str] = []

    print("\n=== Proposal Quality Scorecard — Phase 8 ===\n")
    for fx in fixtures:
        results = await evaluate_fixture(fx)
        f_pass = sum(1 for r in results if r[1])
        f_total = len(results)
        total_assertions += f_total
        total_passed += f_pass
        status = "PASS" if f_pass == f_total else "FAIL"
        print(f"  [{status}]  {fx['name']:40s}  {f_pass}/{f_total}")
        if f_pass != f_total:
            failed_fixtures.append(fx["name"])
            for name, ok, detail in results:
                if not ok:
                    print(f"           - {name}: {detail}")

    print(
        f"\nTotal: {total_passed}/{total_assertions} assertions passed "
        f"across {len(fixtures)} fixtures"
    )
    if failed_fixtures:
        print(f"FAILED fixtures: {', '.join(failed_fixtures)}")
        return 1
    print("All fixtures GREEN.")
    return 0


if __name__ == "__main__":
    rc = asyncio.run(main())
    sys.exit(rc)
