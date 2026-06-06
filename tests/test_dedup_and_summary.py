"""Regression tests for the post-Phase-8 follow-ups.

Two production bugs the user reported AFTER Phase 8 shipped:

1. Same change appearing twice in the card list — the existing strict/semantic
   dedup keys included edit_mode and section_heading, so the same final
   after_content on the same page could leak through if the drafter picked a
   different heading or one called it "replace" while another called it
   "append". Fix: a third "outcome-only" dedup layer on (page_id, change_type,
   normalized after_content).

2. UI didn't communicate WHAT was changing — the synthesized change_summary
   was too generic ("Edit section 'X' in 'Y'"). Fix: when before/after content
   is available, the summary now reads "Replace 'snippet…' with 'snippet…'"
   for edits and "Add to 'section': 'snippet…'" for appends.

These tests pin both fixes so they cannot regress silently.
"""
from __future__ import annotations

from typing import Any, Dict, List

import pytest


# ---------------------------------------------------------------------------
# Dedup helper tests — _dedupe_proposals() is a pure function so we exercise
# it directly with crafted proposal dicts.
# ---------------------------------------------------------------------------


def _make_proposal(**overrides: Any) -> Dict[str, Any]:
    """Build a baseline proposal dict; tests override only what they need."""
    base: Dict[str, Any] = {
        "page_id": "page-x",
        "page_title": "Enterprise Customer Feedback",
        "change_type": "edit",
        "edit_mode": "replace",
        "section_heading": "Customer Feedback",
        "before_content": "Microsoft Teams integration is mandatory.",
        "after_content": "Teams integration will be supported.",
    }
    base.update(overrides)
    return base


def test_dedup_strict_layer_collapses_identical_keys():
    """Two proposals with identical strict keys (page_id, change_type,
    edit_mode, section_heading) must collapse to one regardless of after."""
    from confluence_logic.review.api import _dedupe_proposals

    ps = [
        _make_proposal(after_content="version 1 of the same text"),
        _make_proposal(after_content="version 2 — different but same strict key"),
    ]
    out = _dedupe_proposals(ps)
    assert len(out) == 1, f"strict-key dedup failed: got {len(out)} proposals"
    assert out[0] is ps[0], "first occurrence must be the one preserved"


def test_dedup_outcome_layer_catches_different_heading_same_content():
    """The user-reported regression: two cards for the same final after_content
    on the same page, but with different section_heading guesses, must collapse
    to one. This is the third OUTCOME-only dedup layer's whole job."""
    from confluence_logic.review.api import _dedupe_proposals

    ps = [
        _make_proposal(section_heading="Customer Feedback"),
        _make_proposal(section_heading="Customer Demand for Teams Integration"),
    ]
    out = _dedupe_proposals(ps)
    assert len(out) == 1, (
        f"outcome-key dedup failed for different headings: got {len(out)}; "
        f"second proposal's heading was: {ps[1]['section_heading']!r}"
    )


def test_dedup_outcome_layer_catches_replace_vs_append_same_content():
    """Same after_content on the same page, one as replace and the other as
    append — collapse to one. Without the OUTCOME layer, the strict and
    semantic keys both include edit_mode and would let this pass through."""
    from confluence_logic.review.api import _dedupe_proposals

    ps = [
        _make_proposal(edit_mode="replace", before_content="something old"),
        _make_proposal(edit_mode="append", before_content=""),
    ]
    out = _dedupe_proposals(ps)
    assert len(out) == 1, (
        f"outcome-key dedup failed across edit modes: got {len(out)}"
    )


def test_dedup_keeps_genuinely_different_changes():
    """Sanity: different after_content on the same page must NOT be deduped."""
    from confluence_logic.review.api import _dedupe_proposals

    ps = [
        _make_proposal(after_content="Teams integration will be supported."),
        _make_proposal(
            section_heading="Slack Improvements",
            before_content="Slack formatting is slow.",
            after_content="Slack rich-text formatting is prioritized for Q3.",
        ),
    ]
    out = _dedupe_proposals(ps)
    assert len(out) == 2, (
        f"dedup over-collapsed: got {len(out)}, both proposals should survive"
    )


def test_dedup_keeps_different_pages_same_content():
    """Same after_content but different pages — both must survive."""
    from confluence_logic.review.api import _dedupe_proposals

    ps = [
        _make_proposal(page_id="page-A"),
        _make_proposal(page_id="page-B"),
    ]
    out = _dedupe_proposals(ps)
    assert len(out) == 2, "dedup must not collapse across pages"


def test_dedup_tolerates_malformed_audit_field():
    """WR-05: upstream stages occasionally set _audit to non-dict types
    (debug strings, lists) during refactors. The dedup helper must NOT crash
    on these — it should treat malformed audit as missing audit data and
    continue."""
    from confluence_logic.review.api import _dedupe_proposals

    ps = [
        _make_proposal(_audit="oops a string instead of a dict"),
        _make_proposal(_audit=["list", "instead", "of", "dict"]),
        _make_proposal(_audit=None),
        _make_proposal(_audit=12345),
    ]
    # All four have identical strict/outcome keys → collapse to one.
    out = _dedupe_proposals(ps)
    assert len(out) == 1, (
        f"dedup must be robust to malformed _audit; got {len(out)} proposals"
    )


def test_dedup_tolerates_missing_page_id_and_title():
    """Sanity: proposals with NULL page_id AND NULL page_title still get
    deduped on the rest of their signature without crashing."""
    from confluence_logic.review.api import _dedupe_proposals

    ps = [
        _make_proposal(page_id=None, page_title=None),
        _make_proposal(page_id=None, page_title=None),
    ]
    out = _dedupe_proposals(ps)
    assert len(out) == 1


def test_dedup_preserves_order_of_first_occurrence():
    """If duplicates appear, only the FIRST occurrence is kept — downstream
    rationales/audit metadata on the first card stays meaningful."""
    from confluence_logic.review.api import _dedupe_proposals

    first = _make_proposal(rationale="First reasoning — high confidence")
    second = _make_proposal(
        section_heading="Customer Demand",
        rationale="Second reasoning — should be dropped",
    )
    out = _dedupe_proposals([first, second])
    assert len(out) == 1
    assert out[0]["rationale"] == "First reasoning — high confidence"


# ---------------------------------------------------------------------------
# change_summary synthesis tests — exercised through _verify_and_persist.
# Mock the connector + supabase + SSE emit so we can capture the persisted row
# and assert on its change_summary.
# ---------------------------------------------------------------------------


def _patch_verify_and_persist(monkeypatch, live_html: str = "") -> Dict[str, Any]:
    """Patch all external dependencies of _verify_and_persist and return a
    capture dict that will hold the persisted row after the call.

    ``live_html`` controls what the fake Confluence connector returns when the
    verifier fetches the page — supply something that contains the drafter's
    before_content if you want it preserved through the pre-validate step.
    """
    capture: Dict[str, Any] = {"row": None}

    class _FakeConnector:
        def fetch_page_html(self, page_id):
            return live_html
        def search_pages(self, title, limit=5):
            return []
        def get_page_metadata(self, page_id):
            return {"version": {"number": 1}}
    monkeypatch.setattr("confluence_logic.review.api._get_connector", lambda: _FakeConnector())

    def _capture_upsert(row):
        capture["row"] = row
        return "row-id-1"
    monkeypatch.setattr("confluence_logic.review.supabase_store.upsert_proposal", _capture_upsert)
    monkeypatch.setattr("confluence_logic.review.api._emit", lambda *a, **kw: None)
    return capture


@pytest.mark.asyncio
async def test_change_summary_for_replace_includes_before_and_after_snippets(monkeypatch):
    """For change_type=edit with both before_content and after_content, the
    synthesized change_summary must include short snippets of BOTH so the user
    can read what's being changed without expanding the card."""
    from confluence_logic.review.api import _verify_and_persist

    # Fake page must contain the drafter's full before_content as a clean
    # substring (whitespace-collapsed match) so the verifier's pre-validate
    # step keeps it instead of clearing.
    before_text = "Microsoft Teams integration is mandatory and prioritized"
    live_html = (
        f"<h2>Customer Feedback</h2><p>{before_text} ahead of Slack improvements.</p>"
    )
    capture = _patch_verify_and_persist(monkeypatch, live_html=live_html)
    draft = {
        "change_type": "edit",
        "page_id": "pg-1",
        "page_title": "Enterprise Customer Feedback",
        "section_heading": "Customer Feedback",
        "before_content": before_text,
        "after_content": "Teams integration will be supported in Q2.",
        "edit_mode": "replace",
        "rationale": "Confirm prioritization.",
    }
    await _verify_and_persist(draft, "transcript-x", "job-1", "sess-1", "user-1")

    row = capture["row"]
    assert row is not None, "upsert was not called"
    summary = row.get("change_summary") or ""
    assert summary, "change_summary was not synthesized"
    assert "replace" in summary.lower(), (
        f"summary should name the action 'replace'; got: {summary!r}"
    )
    assert "Microsoft Teams" in summary or "Teams integration is mandatory" in summary, (
        f"summary missing before-snippet: {summary!r}"
    )
    assert "Teams integration will be supported" in summary, (
        f"summary missing after-snippet: {summary!r}"
    )


@pytest.mark.asyncio
async def test_change_summary_for_append_includes_after_snippet(monkeypatch):
    """For edits with NO before_content (append-only), the change_summary must
    describe what is being added to which section."""
    from confluence_logic.review.api import _verify_and_persist

    # The verifier may auto-populate before_content from the live page when
    # the drafter left it empty. The synthesis must still respect edit_mode=
    # "append" and use Add-to-section phrasing, NOT Replace-with phrasing.
    capture = _patch_verify_and_persist(
        monkeypatch,
        live_html="<h2>Setup</h2><p>Some existing setup instructions.</p>",
    )
    draft = {
        "change_type": "edit",
        "page_id": "pg-1",
        "page_title": "API Runbook",
        "section_heading": "Setup",
        "before_content": "",
        "after_content": "Add a step: install pre-commit hooks before first push.",
        "edit_mode": "append",
        "rationale": "Tooling decision",
    }
    await _verify_and_persist(draft, "transcript-x", "job-1", "sess-1", "user-1")

    row = capture["row"]
    assert row is not None
    summary = (row.get("change_summary") or "").lower()
    assert "add" in summary, f"append summary missing 'add' verb: {summary!r}"
    assert "setup" in summary, f"append summary missing section name: {summary!r}"
    assert "install pre-commit" in summary, (
        f"append summary missing the actual content snippet: {summary!r}"
    )


@pytest.mark.asyncio
async def test_change_summary_for_create_includes_seed_content(monkeypatch):
    """For change_type=create with after_content seeding the new page, the
    synthesized change_summary must surface a snippet of what the new page
    will contain."""
    from confluence_logic.review.api import _verify_and_persist

    capture = _patch_verify_and_persist(monkeypatch)
    draft = {
        "change_type": "create",
        "page_id": None,
        "page_title": "Architecture Overview",
        "section_heading": None,
        "before_content": "",
        "after_content": "Service boundaries, datastores, and external APIs.",
        "edit_mode": "",
        "rationale": "New documentation",
        "confluence_space_key": "TEST",
    }
    await _verify_and_persist(draft, "transcript-x", "job-1", "sess-1", "user-1")

    row = capture["row"]
    assert row is not None
    summary = row.get("change_summary") or ""
    assert "Create" in summary, f"create summary missing verb: {summary!r}"
    assert "Architecture Overview" in summary, (
        f"create summary missing page title: {summary!r}"
    )
    assert "Service boundaries" in summary or "datastores" in summary, (
        f"create summary should surface a snippet of the seed content: {summary!r}"
    )


@pytest.mark.asyncio
async def test_change_summary_preserved_when_drafter_already_supplied_it(monkeypatch):
    """When the drafter has already produced a change_summary, the synthesizer
    must NOT overwrite it — the drafter's wording usually has the meeting
    context the verifier can't reconstruct."""
    from confluence_logic.review.api import _verify_and_persist

    # Page contains the before_content so the pre-validate doesn't clear it.
    capture = _patch_verify_and_persist(
        monkeypatch,
        live_html="<h2>Setup</h2><p>Python 2 is the default.</p>",
    )
    draft = {
        "change_type": "edit",
        "page_id": "pg-1",
        "page_title": "Dev Runbook",
        "section_heading": "Setup",
        "before_content": "Python 2 is the default.",
        "after_content": "Python 3.12 is the default.",
        "edit_mode": "replace",
        "change_summary": "Bump default Python from 2 to 3.12 per Q1 platform standardization.",
    }
    await _verify_and_persist(draft, "transcript-x", "job-1", "sess-1", "user-1")

    row = capture["row"]
    assert row is not None
    # Drafter's summary preserved verbatim
    assert row.get("change_summary") == (
        "Bump default Python from 2 to 3.12 per Q1 platform standardization."
    )


# ---------------------------------------------------------------------------
# IN-01 — additional coverage requested by code review.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_change_summary_for_title_rename_uses_rename_phrasing(monkeypatch):
    """change_type="title" must produce a "Rename 'X' to 'Y'" summary even
    when after_content is short (which the stub-length guard previously
    silently dropped — WR-03)."""
    from confluence_logic.review.api import _verify_and_persist

    capture = _patch_verify_and_persist(monkeypatch, live_html="")
    draft = {
        "change_type": "title",
        "page_id": "pg-1",
        "page_title": "Old Team Roster",
        "section_heading": None,
        "before_content": "",
        # WR-03: short new title — was being dropped by the stub-length guard
        "after_content": "Q3 Plan",
        "edit_mode": "",
        "rationale": "Renaming for clarity",
    }
    await _verify_and_persist(draft, "transcript-x", "job-1", "sess-1", "user-1")

    row = capture["row"]
    assert row is not None, "title rename was DROPPED by the stub guard (WR-03 regression)"
    summary = row.get("change_summary") or ""
    assert "Rename" in summary, f"title summary should use 'Rename' verb: {summary!r}"
    assert "Old Team Roster" in summary, f"missing old title: {summary!r}"
    assert "Q3 Plan" in summary, f"missing new title: {summary!r}"


@pytest.mark.asyncio
async def test_change_summary_for_delete_section_names_section(monkeypatch):
    """change_type="delete" with a section_heading must name the section."""
    from confluence_logic.review.api import _verify_and_persist

    capture = _patch_verify_and_persist(monkeypatch, live_html="")
    draft = {
        "change_type": "delete",
        "page_id": "pg-1",
        "page_title": "API Runbook",
        "section_heading": "Deprecated Endpoints",
        "before_content": "",
        "after_content": "",
        "edit_mode": "",
        "rationale": "Cleanup",
    }
    await _verify_and_persist(draft, "transcript-x", "job-1", "sess-1", "user-1")

    row = capture["row"]
    assert row is not None
    summary = row.get("change_summary") or ""
    assert "Delete section" in summary, f"delete summary missing 'Delete section': {summary!r}"
    assert "Deprecated Endpoints" in summary, f"missing section name: {summary!r}"
    assert "API Runbook" in summary, f"missing page name: {summary!r}"


@pytest.mark.asyncio
async def test_change_summary_for_delete_whole_page_names_page(monkeypatch):
    """change_type="delete" without a section means delete the whole page."""
    from confluence_logic.review.api import _verify_and_persist

    capture = _patch_verify_and_persist(monkeypatch, live_html="")
    draft = {
        "change_type": "delete",
        "page_id": "pg-1",
        "page_title": "Obsolete Service",
        "section_heading": None,
        "before_content": "",
        "after_content": "",
        "edit_mode": "",
        "rationale": "Service decommissioned",
    }
    await _verify_and_persist(draft, "transcript-x", "job-1", "sess-1", "user-1")

    row = capture["row"]
    assert row is not None
    summary = row.get("change_summary") or ""
    assert "Delete page" in summary, f"whole-page delete summary missing 'Delete page': {summary!r}"
    assert "Obsolete Service" in summary, f"missing page name: {summary!r}"


@pytest.mark.asyncio
async def test_change_summary_strips_html_tags_from_snippets(monkeypatch):
    """WR-04: when the drafter occasionally produces Confluence storage-format
    HTML instead of markdown, the snippet helper must strip the tags so they
    don't leak into the headline as visible text."""
    from confluence_logic.review.api import _verify_and_persist

    # Live page contains the BEFORE text so pre-validate keeps it.
    capture = _patch_verify_and_persist(
        monkeypatch,
        live_html="<p>The default is mandatory training.</p>",
    )
    draft = {
        "change_type": "edit",
        "page_id": "pg-1",
        "page_title": "Onboarding",
        "section_heading": "Training",
        "before_content": "The default is mandatory training.",
        # Drafter produced storage-format HTML — tags must NOT leak into summary
        "after_content": "<p>Training is <strong>optional</strong> for senior hires.</p>",
        "edit_mode": "replace",
        "rationale": "Policy update",
    }
    await _verify_and_persist(draft, "transcript-x", "job-1", "sess-1", "user-1")

    row = capture["row"]
    assert row is not None
    summary = row.get("change_summary") or ""
    assert "<p>" not in summary, f"HTML tag leaked into summary: {summary!r}"
    assert "<strong>" not in summary, f"HTML tag leaked into summary: {summary!r}"
    assert "optional" in summary, f"summary lost the actual content: {summary!r}"


# ---------------------------------------------------------------------------
# _normalize_proposal_text direct unit tests.
# ---------------------------------------------------------------------------


def test_normalize_proposal_text_lowercases_and_collapses():
    from confluence_logic.review.api import _normalize_proposal_text

    out = _normalize_proposal_text("Hello   WORLD\n\nfoo", 100)
    assert out == "hello world foo"


def test_normalize_proposal_text_strips_light_markdown():
    from confluence_logic.review.api import _normalize_proposal_text

    out = _normalize_proposal_text("**Bold** _italic_ `code` #header", 100)
    # Light markdown chars become spaces; hyphens preserved
    assert "*" not in out
    assert "_" not in out
    assert "`" not in out
    assert "#" not in out
    assert "bold" in out and "italic" in out and "code" in out and "header" in out


def test_normalize_proposal_text_truncates_to_max_chars():
    from confluence_logic.review.api import _normalize_proposal_text

    out = _normalize_proposal_text("a" * 1000, 50)
    assert len(out) == 50


def test_normalize_proposal_text_handles_none_and_empty():
    from confluence_logic.review.api import _normalize_proposal_text

    assert _normalize_proposal_text(None, 50) == ""  # type: ignore[arg-type]
    assert _normalize_proposal_text("", 50) == ""
    assert _normalize_proposal_text("   \t\n  ", 50) == ""


# ---------------------------------------------------------------------------
# Editor-answer failure detection direct unit tests (WR-06).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("answer,expected", [
    ("ERROR: The update failed.", True),
    ("Error: anything", True),
    ("The requested change did not complete because …", True),
    ("I encountered an issue while running the agent workflow.", True),
    ("I could not find the section.", True),
    ("I couldn't locate the page.", True),
    ("I was unable to commit.", True),
    ("I'm unable to apply.", True),
    ("I am unable to apply.", True),
    ("I can't find that.", True),
    ("I cannot proceed.", True),
    ("Unable to apply the change.", True),
    ("Could not find target page.", True),
    ("Couldn't find anything.", True),
    ("No matching page.", True),
    ("The update failed. Reason X.", True),
    ("Page creation failed: reason.", True),
    ("The deletion failed.", True),
    ("Sorry, that's not possible.", True),
    ("", True),  # empty answer also signals failure
    (None, True),  # None too
    ("Done.", False),
    ("Applied the change to Confluence.", False),
    ("Edit committed successfully.", False),
])
def test_editor_answer_indicates_failure(answer, expected):
    from confluence_logic.review.api import _editor_answer_indicates_failure

    assert _editor_answer_indicates_failure(answer) is expected, (
        f"answer={answer!r} → expected {expected}"
    )
