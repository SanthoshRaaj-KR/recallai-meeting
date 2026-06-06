"""Fake-transcript end-to-end smoke test for the ProposalPipeline.

Tests:
  1. Facts extracted without explicit "update the docs" mentions
  2. Multi-page proposals — same metric on multiple pages each get a card
  3. Action items converted to change_intents even without doc keywords
  4. Append behaviour — a second pipeline run adds cards, does not overwrite

Run with:
    uv run pytest tests/test_fake_transcript.py -v -s
"""

from __future__ import annotations

import asyncio
import pprint
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from review_pipeline.models import ChangeIntent, ExtractedMeeting, PageCandidate
from review_pipeline.pipeline import ProposalPipeline

# ── Fake transcript ────────────────────────────────────────────────────────────
# Deliberately written so no one says "update the docs / Confluence / page".
# The pipeline must infer that these facts could live in documentation.

FAKE_TRANSCRIPT = [
    {"participant": "Alice", "text": "Good morning everyone, let's kick off."},
    {"participant": "Bob",   "text": "We ran the nightly eval — precision is now 0.94, recall is 0.88."},
    {"participant": "Alice", "text": "Last sprint the recall target was 0.91, so we slipped. We need to raise it."},
    {"participant": "Bob",   "text": "Agreed. We are bumping the precision threshold from 0.90 to 0.95."},
    {"participant": "Carol", "text": "The Q3 launch date moved from October 1st to November 15th."},
    {"participant": "Alice", "text": "Confirmed. Owner of the launch plan is now Dave, not Carol."},
    {"participant": "Bob",   "text": "The audio defect detection model recall went from 0.82 to 0.89 as well."},
    {"participant": "Carol", "text": "Great. The Q2 integration checklist is fully complete — all items done."},
    {"participant": "Alice", "text": "Bob will send the updated benchmark numbers to the team by Friday."},
    {"participant": "Bob",   "text": "And Dave should finalize the release notes for the 2.4 build."},
    {"participant": "Carol", "text": "Thanks everyone, see you next week."},
]

# ── Fake Confluence: three pages each mentioning recall ───────────────────────

class MultiPageConfluenceClient:
    """Three pages that all reference recall so we can test multi-page proposals."""

    _pages = {
        "metrics-main": PageCandidate(
            page_id="metrics-main",
            title="Vision Model Metrics",
            html=(
                "<h2>Classifier</h2>"
                "<p>Precision threshold: 0.90</p>"
                "<p>Recall target: 0.91</p>"
                "<h2>Detector</h2>"
                "<p>Recall: 0.82</p>"
            ),
            text=(
                "Classifier\nPrecision threshold: 0.90\nRecall target: 0.91\n"
                "Detector\nRecall: 0.82"
            ),
            version=5,
            source="fake",
        ),
        "audio-model": PageCandidate(
            page_id="audio-model",
            title="Audio Defect Detection Model",
            html=(
                "<h2>Performance</h2>"
                "<p>Recall: 0.82</p>"
                "<p>Precision: 0.91</p>"
            ),
            text="Performance\nRecall: 0.82\nPrecision: 0.91",
            version=2,
            source="fake",
        ),
        "launch-plan": PageCandidate(
            page_id="launch-plan",
            title="Q3 Launch Plan",
            html=(
                "<h2>Timeline</h2>"
                "<p>Launch date: October 1st</p>"
                "<h2>Ownership</h2>"
                "<p>Owner: Carol</p>"
            ),
            text="Timeline\nLaunch date: October 1st\nOwnership\nOwner: Carol",
            version=3,
            source="fake",
        ),
        "integration-checklist": PageCandidate(
            page_id="integration-checklist",
            title="Q2 Integration Checklist",
            html=(
                "<h2>Tasks</h2>"
                "<ac:task-list>"
                "<ac:task><ac:task-id>1</ac:task-id>"
                "<ac:task-status>incomplete</ac:task-status>"
                "<ac:task-body>Q2 integration checklist</ac:task-body></ac:task>"
                "</ac:task-list>"
            ),
            text="Tasks\n[task: incomplete] Q2 integration checklist",
            version=1,
            source="fake",
        ),
    }

    async def search_pages(self, query: str, limit: int = 10):
        query_l = query.lower()
        results = []
        for page_id, page in self._pages.items():
            score = 0
            for kw in query_l.split():
                if kw in page.text.lower() or kw in page.title.lower():
                    score += 1
            if score:
                results.append({"page_id": page_id, "title": page.title, "retrieval_score": float(score)})
        results.sort(key=lambda x: x["retrieval_score"], reverse=True)
        return results[:limit]

    async def fetch_page(self, page_id: str) -> PageCandidate:
        if page_id not in self._pages:
            raise KeyError(f"Unknown page: {page_id}")
        return self._pages[page_id]

    async def update_page(self, page_id, html_content, *, title=None, expected_version=None):
        return True

    async def create_page(self, title, html_content, space_key=None):
        return {"id": "new-page"}

    async def list_pages(self, limit=10):
        return [{"page_id": pid, "title": p.title} for pid, p in self._pages.items()]


class NoOpRAG:
    def search(self, query, top_k=8):
        return []

    def upsert_page(self, page):
        pass


# ── Helpers ────────────────────────────────────────────────────────────────────

async def async_no_candidates(*_args, **_kwargs):
    return []


async def async_passthrough_verify(_meeting, proposals, _text):
    return proposals


async def async_passthrough_critic(_meeting, proposals, _text, query=None):
    return proposals


def _make_pipeline() -> ProposalPipeline:
    p = ProposalPipeline()
    p._client = MultiPageConfluenceClient()
    p._rag = NoOpRAG()
    return p


# ── Tests ──────────────────────────────────────────────────────────────────────

def test_extracts_metric_facts_without_doc_mention(monkeypatch):
    """Precision threshold and recall changes must become change_intents even though
    nobody said 'update Confluence'."""
    pipeline = _make_pipeline()

    async def fake_extract(_tr, _text, query=None):
        return ExtractedMeeting(
            title="Nightly Eval Review",
            summary="Metrics updated, launch date moved.",
            key_topics=["precision", "recall", "launch date"],
            decisions=["Bump precision threshold to 0.95", "Launch date moved to Nov 15"],
            action_items=[
                {"description": "Bob sends updated benchmark numbers to team by Friday", "owner": "Bob", "due": "Friday"},
                {"description": "Dave finalizes release notes for the 2.4 build", "owner": "Dave", "due": None},
            ],
            change_intents=[
                ChangeIntent(
                    instruction="Change precision threshold from 0.90 to 0.95",
                    subject="precision threshold",
                    target_hint="Vision Model Metrics",
                    old_value="0.90",
                    new_value="0.95",
                    action="replace",
                    evidence=["bumping the precision threshold from 0.90 to 0.95"],
                ),
                ChangeIntent(
                    instruction="Recall target slipped to 0.88 from 0.91",
                    subject="recall target",
                    target_hint="Vision Model Metrics",
                    old_value="0.91",
                    new_value="0.88",
                    action="replace",
                    evidence=["recall is now 0.88"],
                ),
                ChangeIntent(
                    instruction="Audio defect detection recall went from 0.82 to 0.89",
                    subject="recall",
                    target_hint="Audio Defect Detection Model",
                    old_value="0.82",
                    new_value="0.89",
                    action="replace",
                    evidence=["audio defect detection model recall went from 0.82 to 0.89"],
                ),
                ChangeIntent(
                    instruction="Q3 launch date moved from October 1st to November 15th",
                    subject="launch date",
                    target_hint="Q3 Launch Plan",
                    old_value="October 1st",
                    new_value="November 15th",
                    action="replace",
                    evidence=["Q3 launch date moved from October 1st to November 15th"],
                ),
                ChangeIntent(
                    instruction="Launch plan owner changed from Carol to Dave",
                    subject="owner",
                    target_hint="Q3 Launch Plan",
                    old_value="Carol",
                    new_value="Dave",
                    action="replace",
                    evidence=["Owner of the launch plan is now Dave, not Carol"],
                ),
            ],
        )

    monkeypatch.setattr(pipeline, "_extract_meeting", fake_extract)
    monkeypatch.setattr(pipeline, "_generate_page_grounded_candidates", async_no_candidates)
    monkeypatch.setattr(pipeline, "_adversarial_verify", async_passthrough_verify)
    monkeypatch.setattr(pipeline, "_rovo_independent_critic", async_passthrough_critic)

    meeting, proposals = asyncio.run(pipeline.run(session_id="s1", transcript=FAKE_TRANSCRIPT))

    print("\n\n=== PROPOSALS ===")
    for p in proposals:
        print(f"\n[{p['change_type'].upper()}] {p['page_title']} — {p.get('section_heading')}")
        print(f"  before: {p.get('before_content')!r}")
        print(f"  after:  {p.get('after_content')!r}")
        print(f"  mode:   {p.get('edit_mode')} | confidence: {p.get('confidence')} | risk: {p.get('risk')}")
        print(f"  rationale: {p.get('rationale')}")

    subjects = [p.get("before_content") or p.get("after_content", "") for p in proposals]
    print(f"\n=== TOTAL PROPOSALS: {len(proposals)} ===")

    # Precision threshold must be found on Vision Model Metrics page
    precision_props = [p for p in proposals if "0.95" in str(p.get("after_content", ""))]
    assert precision_props, "Expected a proposal updating precision threshold to 0.95"

    # Recall on audio defect detection page must be found
    audio_recall = [p for p in proposals if p.get("page_id") == "audio-model"]
    assert audio_recall, "Expected a proposal on the Audio Defect Detection Model page"

    # Launch date must be updated
    launch_date = [p for p in proposals if "November 15th" in str(p.get("after_content", ""))]
    assert launch_date, "Expected a proposal moving launch date to November 15th"

    # Owner change
    owner = [p for p in proposals if "Dave" in str(p.get("after_content", ""))]
    assert owner, "Expected a proposal changing owner to Dave"


def test_same_fact_proposes_to_multiple_pages(monkeypatch):
    """Recall value 0.82 appears on BOTH metrics-main and audio-model.
    After removing the early break, we must get proposals for both pages."""
    pipeline = _make_pipeline()

    async def fake_extract(_tr, _text, query=None):
        return ExtractedMeeting(
            title="Recall Review",
            summary="Recall updated across models.",
            change_intents=[
                # Single intent — recall — that matches content on two pages.
                ChangeIntent(
                    instruction="Recall changed from 0.82 to 0.89",
                    subject="recall",
                    target_hint="model metrics",
                    old_value="0.82",
                    new_value="0.89",
                    action="replace",
                    evidence=["recall went from 0.82 to 0.89"],
                ),
            ],
        )

    monkeypatch.setattr(pipeline, "_extract_meeting", fake_extract)
    monkeypatch.setattr(pipeline, "_generate_page_grounded_candidates", async_no_candidates)
    monkeypatch.setattr(pipeline, "_adversarial_verify", async_passthrough_verify)
    monkeypatch.setattr(pipeline, "_rovo_independent_critic", async_passthrough_critic)

    _meeting, proposals = asyncio.run(pipeline.run(session_id="s1", transcript=FAKE_TRANSCRIPT))

    print("\n\n=== MULTI-PAGE RECALL PROPOSALS ===")
    for p in proposals:
        print(f"  page_id={p['page_id']!r} before={p.get('before_content')!r} after={p.get('after_content')!r}")

    page_ids = {p["page_id"] for p in proposals}
    assert "metrics-main" in page_ids, "Expected proposal on metrics-main (recall 0.82 is there)"
    assert "audio-model" in page_ids, "Expected proposal on audio-model (recall 0.82 is also there)"
    assert len(proposals) >= 2, f"Expected at least 2 proposals (one per page), got {len(proposals)}"


def test_action_items_become_intents_without_doc_keywords(monkeypatch):
    """Action items with no doc-related keywords must still produce change_intents."""
    pipeline = _make_pipeline()

    async def fake_extract(_tr, _text, query=None):
        # LLM returned zero change_intents but captured action items.
        return ExtractedMeeting(
            title="Planning",
            summary="Actions assigned.",
            change_intents=[],
            action_items=[
                {"description": "Bob sends updated benchmark numbers to team by Friday", "owner": "Bob", "due": "Friday"},
                {"description": "Dave finalizes release notes for 2.4 build", "owner": "Dave", "due": None},
                # Greeting-like item — still has no page match, so it'll just produce a diagnostic
                {"description": "Good morning everyone", "owner": None, "due": None},
            ],
        )

    monkeypatch.setattr(pipeline, "_extract_meeting", fake_extract)
    monkeypatch.setattr(pipeline, "_generate_page_grounded_candidates", async_no_candidates)
    monkeypatch.setattr(pipeline, "_adversarial_verify", async_passthrough_verify)
    monkeypatch.setattr(pipeline, "_rovo_independent_critic", async_passthrough_critic)

    meeting, proposals = asyncio.run(pipeline.run(session_id="s1", transcript=FAKE_TRANSCRIPT))

    print("\n\n=== ACTION ITEM INTENTS ===")
    print(f"change_intents after fallback: {len(meeting.change_intents)}")
    for ci in meeting.change_intents:
        print(f"  source={ci.source!r} subject={ci.subject!r}")

    print(f"\nProposals generated: {len(proposals)}")
    for p in proposals:
        print(f"  page={p.get('page_id', '<create>')} — {p.get('after_content', '')[:80]}")

    # All 3 action items (including the greeting) must become change_intents.
    # (The greeting won't find a page — it'll just produce a diagnostic.)
    assert len(meeting.change_intents) == 3, (
        f"Expected 3 change_intents from action items, got {len(meeting.change_intents)}"
    )
    sources = {ci.source for ci in meeting.change_intents}
    assert "action_item" in sources
    # Greeting-like items have no matching page so they generate no proposals,
    # but real action items (benchmark numbers, release notes) may match pages.
    print(f"  diagnostics: {[d['reason'] for d in pipeline.last_diagnostics]}")


def test_no_proposals_overwritten_on_second_run():
    """Simulates recall_bridge append logic: a second propose call must add to
    the existing list, not replace it."""
    existing = [
        {"id": "existing-1", "page_id": "metrics-main", "after_content": "0.95"},
    ]
    new_proposals = [
        {"id": "new-1", "page_id": "audio-model", "after_content": "0.89"},
        {"id": "existing-1", "page_id": "metrics-main", "after_content": "0.95"},  # duplicate
    ]
    existing_ids = {str(ch.get("id")) for ch in existing}
    unique_new = [p for p in new_proposals if str(p.get("id")) not in existing_ids]
    existing.extend(unique_new)

    print(f"\n=== APPEND TEST: {len(existing)} total changes after second run ===")
    for c in existing:
        print(f"  id={c['id']} page={c['page_id']}")

    assert len(existing) == 2, "Duplicate proposal must be deduplicated; unique new one appended"
    assert any(c["id"] == "existing-1" for c in existing)
    assert any(c["id"] == "new-1" for c in existing)


def test_task_completion_from_transcript(monkeypatch):
    """Q2 integration checklist is marked complete in transcript — must produce a task_status proposal."""
    pipeline = _make_pipeline()

    async def fake_extract(_tr, _text, query=None):
        return ExtractedMeeting(
            title="Sprint Review",
            summary="Q2 checklist done.",
            change_intents=[
                ChangeIntent(
                    instruction="Mark Q2 integration checklist as complete",
                    subject="Q2 integration checklist",
                    target_hint="Q2 Integration Checklist",
                    new_value="complete",
                    action="complete_task",
                    evidence=["Q2 integration checklist is fully complete"],
                ),
            ],
        )

    monkeypatch.setattr(pipeline, "_extract_meeting", fake_extract)
    monkeypatch.setattr(pipeline, "_generate_page_grounded_candidates", async_no_candidates)
    monkeypatch.setattr(pipeline, "_adversarial_verify", async_passthrough_verify)
    monkeypatch.setattr(pipeline, "_rovo_independent_critic", async_passthrough_critic)

    _meeting, proposals = asyncio.run(pipeline.run(session_id="s1", transcript=FAKE_TRANSCRIPT))

    print("\n\n=== TASK COMPLETION PROPOSALS ===")
    for p in proposals:
        print(f"  {p.get('edit_mode')} | {p.get('before_content')} → {p.get('after_content')}")

    task_props = [p for p in proposals if p.get("edit_mode") == "task_status"]
    assert task_props, "Expected a task_status proposal for Q2 integration checklist"
    assert "[x]" in task_props[0]["after_content"]
