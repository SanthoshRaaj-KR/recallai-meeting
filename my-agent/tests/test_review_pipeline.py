import asyncio
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from review_pipeline.models import ChangeIntent, ExtractedMeeting, PageCandidate, Proposal
from review_pipeline.pipeline import ProposalPipeline
from review_pipeline.rag import ConfluenceVectorIndex, VectorSearchHit, chunk_page


class FakeConfluenceClient:
    def __init__(self):
        self.updated_html = None
        self.updated_title = None

    async def search_pages(self, query, limit=10):
        if "2nd September" in query or "Q3 meeting plan" in query:
            return [{"page_id": "page-1", "title": "Q3 Meeting Plan"}]
        return []

    async def fetch_page(self, page_id):
        return PageCandidate(
            page_id=page_id,
            title="Q3 Meeting Plan",
            html=(
                "<h2>Timeline</h2>"
                "<p>The Q3 meeting plan is currently scheduled for 2nd September.</p>"
                "<h2>Owners</h2><p>Asha owns the plan.</p>"
            ),
            text=(
                "Timeline\nThe Q3 meeting plan is currently scheduled for 2nd September.\n"
                "Owners\nAsha owns the plan."
            ),
            version=3,
            source="fake",
        )

    async def update_page(self, page_id, html_content, *, title=None, expected_version=None):
        self.updated_html = html_content
        self.updated_title = title
        assert expected_version == 3
        return True

    async def create_page(self, title, html_content, space_key=None):
        return {"id": "new-page"}


class VectorOnlyConfluenceClient:
    async def search_pages(self, query, limit=10):
        return []

    async def fetch_page(self, page_id):
        return PageCandidate(
            page_id=page_id,
            title="Vision Model Metrics",
            html="<h2>Classifier</h2><p>Precision: 0.96</p><p>Recall: 0.91</p>",
            text="Classifier\nPrecision: 0.96\nRecall: 0.91",
            version=4,
            source="fake",
        )


class FakeVectorIndex:
    def __init__(self):
        self.upserted = []

    def search(self, query, top_k=8):
        if "recall" in query.lower():
            return [
                VectorSearchHit(
                    page_id="metrics-page",
                    title="Vision Model Metrics",
                    heading="Classifier",
                    score=0.88,
                    text="Recall: 0.91",
                    version=4,
                )
            ]
        return []

    def search_with_rerank(self, queries, *, top_k_per_query=25, top_n=8, rerank_model="bge-reranker-v2-m3"):
        seen: dict[tuple, VectorSearchHit] = {}
        for q in queries:
            for hit in self.search(q, top_k=top_k_per_query):
                key = (hit.page_id, hit.heading)
                if key not in seen or hit.score > seen[key].score:
                    seen[key] = hit
        return sorted(seen.values(), key=lambda h: h.score, reverse=True)[:top_n]

    def upsert_page(self, page):
        self.upserted.append(page.page_id)


class FakePineconeIndex:
    def __init__(self):
        self.vectors = {}
        self.upsert_calls = 0

    def fetch(self, ids, namespace=None):
        return {"vectors": {item_id: self.vectors[item_id] for item_id in ids if item_id in self.vectors}}

    def upsert_records(self, namespace=None, records=None):
        self.upsert_calls += 1
        for record in (records or []):
            record_id = record["_id"]
            self.vectors[record_id] = {"fields": record}

    def delete(self, ids, namespace=None):
        for item_id in ids:
            self.vectors.pop(item_id, None)


class LocalVectorIndex(ConfluenceVectorIndex):
    def __init__(self):
        super().__init__()
        self.fake_index = FakePineconeIndex()

    @property
    def enabled(self):
        return True

    def _index(self):
        return self.fake_index

    def _embed(self, texts):
        return [[0.1, 0.2, 0.3] for _ in texts]


def _make_proposal_dict(
    page_id, page_title, heading, before, after, edit_mode,
    confidence="high", session_id="s1",
):
    return Proposal(
        id=str(uuid.uuid4()),
        change_type="edit",
        page_id=page_id,
        page_title=page_title,
        section_heading=heading,
        before_content=before,
        after_content=after,
        timestamp="2026-01-01T00:00:00",
        session_id=session_id,
        confidence=confidence,
        risk="safe" if confidence == "high" else "review",
        confidence_score=0.9,
        edit_mode=edit_mode,
    ).to_dict()


def test_pipeline_drafts_exact_date_replacement(monkeypatch):
    pipeline = ProposalPipeline()
    pipeline._client = FakeConfluenceClient()

    async def fake_extract(_transcript, _text, query=None):
        return ExtractedMeeting(
            title="Planning Meeting",
            summary="Q3 plan moved.",
            change_intents=[
                ChangeIntent(
                    instruction="Move Q3 meeting plan from 2nd September to 3rd December",
                    subject="Q3 meeting plan date",
                    target_hint="Q3 meeting plan",
                    old_value="2nd September",
                    new_value="3rd December",
                    action="replace",
                    evidence=["we push Q3 meeting plan to 3rd of December"],
                )
            ],
        )

    async def fake_retrieve_all_sections(intents):
        return [(intents[0], [VectorSearchHit(
            page_id="page-1", title="Q3 Meeting Plan",
            heading="Timeline", score=0.91,
            text="scheduled for 2nd September", version=3,
        )])]

    def fake_draft_sync(*, session_id, meeting, intent_sections, section_cache, page_cache, query):
        return [_make_proposal_dict(
            "page-1", "Q3 Meeting Plan", "Timeline",
            "2nd September", "3rd December", "replace",
        )]

    monkeypatch.setattr(pipeline, "_extract_meeting", fake_extract)
    monkeypatch.setattr(pipeline, "_retrieve_all_sections", fake_retrieve_all_sections)
    monkeypatch.setattr(pipeline, "_draft_grounded_proposals_sync", fake_draft_sync)
    monkeypatch.setattr(pipeline, "_adversarial_verify", async_passthrough_verify)

    transcript = [{"participant": "Asha", "text": "We push Q3 meeting plan to 3rd of December."}]
    meeting, proposals = asyncio.run(pipeline.run(session_id="s1", transcript=transcript))

    assert meeting.title == "Planning Meeting"
    assert len(proposals) == 1
    proposal = proposals[0]
    assert proposal["page_id"] == "page-1"
    assert proposal["change_type"] == "edit"
    assert proposal["edit_mode"] == "replace"
    assert proposal["before_content"] == "2nd September"
    assert proposal["after_content"] == "3rd December"
    assert proposal["confidence"] == "high"
    assert proposal["section_heading"] == "Timeline"


def test_rag_chunks_one_page_without_mixing_pages():
    chunks = chunk_page(
        PageCandidate(
            page_id="page-1",
            title="Metrics",
            space_key="ENG",
            html="<h2>Classifier</h2><p>Recall: 0.91</p><h2>Detector</h2><p>Recall: 0.82</p>",
            version=3,
        ),
        max_words=20,
    )

    assert [chunk.page_id for chunk in chunks] == ["page-1", "page-1"]
    assert [chunk.heading for chunk in chunks] == ["Classifier", "Detector"]
    assert all(chunk.title == "Metrics" for chunk in chunks)


def test_rag_upsert_skips_unchanged_page_and_reindexes_changed_page():
    index = LocalVectorIndex()
    page = PageCandidate(
        page_id="page-1",
        title="Metrics",
        html="<h2>Classifier</h2><p>Recall: 0.91</p>",
        version=3,
    )

    index.upsert_page(page)
    index.upsert_page(page)
    changed_page = PageCandidate(
        page_id="page-1",
        title="Metrics",
        html="<h2>Classifier</h2><p>Recall: 0.98</p>",
        version=4,
    )
    index.upsert_page(changed_page)

    assert index.fake_index.upsert_calls == 2
    assert index.fake_index.vectors["page-1:0"]["fields"]["version"] == 4
    assert index.fake_index.vectors["page-1:0"]["fields"]["chunk_count"] == 1


def test_rag_reindex_only_rebuilds_changed_chunks(monkeypatch):
    """A section edit re-enriches only the changed chunk (plus the :0 freshness
    sentinel); unchanged chunks are reused, skipping their enrichment/re-embed."""
    monkeypatch.setattr("review_pipeline.rag._CONTEXTUAL_ENRICHMENT", True)
    index = LocalVectorIndex()
    enriched: list[str] = []
    monkeypatch.setattr(
        index,
        "_generate_chunk_context",
        lambda chunk: (enriched.append(chunk.id) or "ctx"),
    )

    page = PageCandidate(
        page_id="p1",
        title="Doc",
        html=(
            "<h2>Alpha</h2><p>first section original</p>"
            "<h2>Bravo</h2><p>second section original</p>"
            "<h2>Charlie</h2><p>third section original</p>"
        ),
        version=1,
    )
    index.upsert_page(page)
    assert sorted(enriched) == ["p1:0", "p1:1", "p1:2"]  # initial index enriches all

    enriched.clear()
    index.fake_index.upsert_calls = 0
    changed = PageCandidate(
        page_id="p1",
        title="Doc",
        html=(
            "<h2>Alpha</h2><p>first section original</p>"
            "<h2>Bravo</h2><p>second section UPDATED</p>"
            "<h2>Charlie</h2><p>third section original</p>"
        ),
        version=2,
    )
    index.upsert_page(changed)

    # Bravo (:1) changed -> re-enriched; :0 always rebuilt (sentinel); Charlie (:2)
    # unchanged -> reused, NOT re-enriched.
    assert "p1:1" in enriched
    assert "p1:2" not in enriched
    assert set(enriched) <= {"p1:0", "p1:1"}
    assert index.fake_index.upsert_calls == 1  # only rebuilt chunks upserted
    assert index.fake_index.vectors["p1:0"]["fields"]["version"] == 2  # sentinel advanced
    assert index.fake_index.vectors["p1:1"]["fields"]["version"] == 2  # changed chunk updated
    assert "UPDATED" in index.fake_index.vectors["p1:1"]["fields"]["text"]


def test_vector_rag_search_with_rerank_finds_metric_page(monkeypatch):
    pipeline = ProposalPipeline()
    pipeline._client = VectorOnlyConfluenceClient()
    pipeline._rag = FakeVectorIndex()

    async def fake_extract(_transcript, _text, query=None):
        return ExtractedMeeting(
            title="Metrics Meeting",
            summary="Recall changed.",
            change_intents=[
                ChangeIntent(
                    instruction="Change recall to 0.98 for the classifier",
                    subject="recall",
                    target_hint="classifier metrics",
                    new_value="0.98",
                    action="replace",
                    evidence=["recall should be 0.98"],
                )
            ],
        )

    def fake_draft_sync(*, session_id, meeting, intent_sections, section_cache, page_cache, query):
        # Verify retrieval found the right page via search_with_rerank
        assert len(intent_sections) == 1
        _intent, hits = intent_sections[0]
        assert any(h.page_id == "metrics-page" for h in hits)
        return [_make_proposal_dict(
            "metrics-page", "Vision Model Metrics", "Classifier",
            "Recall: 0.91", "Recall: 0.98", "replace",
        )]

    monkeypatch.setattr(pipeline, "_extract_meeting", fake_extract)
    monkeypatch.setattr(pipeline, "_draft_grounded_proposals_sync", fake_draft_sync)
    monkeypatch.setattr(pipeline, "_adversarial_verify", async_passthrough_verify)

    _meeting, proposals = asyncio.run(
        pipeline.run(
            session_id="s1",
            transcript=[{"participant": "Asha", "text": "Change recall to 0.98 for the classifier."}],
        )
    )

    assert len(proposals) == 1
    assert proposals[0]["page_id"] == "metrics-page"
    assert proposals[0]["edit_mode"] == "replace"
    assert proposals[0]["before_content"] == "Recall: 0.91"
    assert proposals[0]["after_content"] == "Recall: 0.98"
    # Confirm the page was indexed via upsert_page after fetch
    assert "metrics-page" in pipeline._rag.upserted


def test_custom_new_page_workflow_returns_create_proposal(monkeypatch):
    pipeline = ProposalPipeline()

    async def fake_extract(_transcript, _text, query=None):
        return ExtractedMeeting(
            title="Framework Discussion",
            summary="The team discussed the Nova framework.",
            key_topics=["Nova framework"],
            decisions=["Adopt Nova for the pilot."],
            action_items=[{"description": "Document Nova rollout plan", "owner": "Asha", "due": None}],
        )

    async def no_style_pages(*_args, **_kwargs):
        return []

    def fail_styled_draft(*_args, **_kwargs):
        raise RuntimeError("No LLM in test")

    monkeypatch.setattr(pipeline, "_extract_meeting", fake_extract)
    monkeypatch.setattr(pipeline, "_sample_style_pages", no_style_pages)
    monkeypatch.setattr(pipeline, "_draft_styled_new_page_sync", fail_styled_draft)

    _meeting, proposals = asyncio.run(
        pipeline.propose_custom_new_page(
            session_id="s1",
            transcript=[{"participant": "Asha", "text": "We should use Nova for the pilot."}],
            query="create page for the Nova framework discussed in meeting",
        )
    )

    assert len(proposals) == 1
    assert proposals[0]["change_type"] == "create"
    assert "Nova Framework" in proposals[0]["page_title"]
    assert "Adopt Nova for the pilot" in proposals[0]["after_content"]


def test_pipeline_returns_empty_without_transcript():
    pipeline = ProposalPipeline()
    meeting, proposals = asyncio.run(pipeline.run(session_id="s1", transcript=[]))

    assert "No transcript" in meeting.summary
    assert proposals == []


def test_action_items_fallback_creates_intents():
    pipeline = ProposalPipeline()
    meeting = ExtractedMeeting(
        change_intents=[],
        action_items=[{"description": "Update timeline in the Q3 plan", "owner": "Asha", "due": None}],
    )
    intents = pipeline._intents_from_action_items(meeting)

    assert len(intents) == 1
    assert intents[0].action == "add"
    assert "timeline" in intents[0].subject.lower()
    assert intents[0].source == "action_item"


def test_pipeline_records_no_sections_found_diagnostic(monkeypatch):
    pipeline = ProposalPipeline()

    async def fake_extract(_transcript, _text, query=None):
        return ExtractedMeeting(
            change_intents=[
                ChangeIntent(
                    instruction="Update launch owner from Asha to Ben",
                    subject="launch owner",
                    target_hint="launch plan",
                    old_value="Asha",
                    new_value="Ben",
                    action="replace",
                )
            ]
        )

    async def empty_retrieve_for_intent(_intent):
        return []

    monkeypatch.setattr(pipeline, "_extract_meeting", fake_extract)
    monkeypatch.setattr(pipeline, "_retrieve_sections_for_intent", empty_retrieve_for_intent)
    monkeypatch.setattr(pipeline, "_adversarial_verify", async_passthrough_verify)

    _meeting, proposals = asyncio.run(
        pipeline.run(session_id="s1", transcript=[{"participant": "Asha", "text": "Owner is now Ben."}])
    )

    assert proposals == []
    assert any(d["reason"] == "no_sections_found" for d in pipeline.last_diagnostics)


def test_looks_like_instruction_text_recognises_bracket_prompts():
    pipeline = ProposalPipeline()
    assert pipeline._looks_like_instruction_text(
        "This page is dedicated to audio-based defect detection. "
        "[Expand to a minimum of 2 paragraphs, including audio-based deepfake detection.]"
    )
    assert pipeline._looks_like_instruction_text(
        "Details here. [Add section summarizing the rollout plan.]"
    )


def test_looks_like_instruction_text_accepts_plain_content():
    pipeline = ProposalPipeline()
    assert not pipeline._looks_like_instruction_text("Recall: 0.98")
    assert not pipeline._looks_like_instruction_text("")
    assert not pipeline._looks_like_instruction_text("Launch review is on Friday.")


def test_adversarial_verdict_removes_wrong_or_unsupported_proposals():
    pipeline = ProposalPipeline()
    proposals = [
        {
            "id": "p1",
            "page_title": "Getting started in Confluence",
            "change_type": "edit",
            "before_content": "[ ] Select Add status",
            "after_content": "[x] Select Add status",
            "risk": "review",
            "confidence": "medium",
        },
        {
            "id": "p2",
            "page_title": "Audio-based Defect Detection Page",
            "change_type": "edit",
            "before_content": "old",
            "after_content": "new",
            "risk": "safe",
            "confidence": "high",
        },
    ]
    verdict = {
        "proposal_verdicts": [
            {
                "id": "p1",
                "supported": False,
                "page_fit": "wrong",
                "duplicate_of": None,
                "confidence": "low",
                "risk": "risky",
                "note": "Proposal targets the wrong page.",
            },
            {
                "id": "p2",
                "supported": True,
                "page_fit": "good",
                "duplicate_of": None,
                "confidence": "high",
                "risk": "safe",
                "note": "",
            },
        ],
        "missed_intents": [],
    }

    filtered = pipeline._apply_adversarial_verdict(verdict, proposals)

    assert [p["id"] for p in filtered] == ["p2"]
    assert pipeline.last_diagnostics[-1]["reason"] == "verifier_rejected_proposal"


def test_coverage_audit_records_diagnostics_without_polluting_cards():
    pipeline = ProposalPipeline()
    meeting = ExtractedMeeting(
        change_intents=[
            ChangeIntent(
                instruction="Create audio-based defect detection page",
                subject="audio-based defect detection",
                new_value="Audio defects should include deepfake detection.",
            )
        ]
    )
    proposals = [
        {
            "id": "p1",
            "page_title": "Some Page",
            "section_heading": "",
            "after_content": "Unrelated content",
            "verifier_note": "Clean note.",
            "risk": "safe",
        }
    ]

    updated = asyncio.run(pipeline._coverage_audit(meeting, proposals, "transcript"))

    assert updated[0]["verifier_note"] == "Clean note."
    assert updated[0]["risk"] == "safe"
    assert pipeline.last_diagnostics[-1]["reason"] == "coverage_audit_unanchored_intent"


def test_execute_append_inserts_into_selected_section():
    pipeline = ProposalPipeline()
    fake = FakeConfluenceClient()
    pipeline._client = fake
    proposal = {
        "change_type": "edit",
        "page_id": "page-1",
        "page_title": "Q3 Meeting Plan",
        "section_heading": "Timeline",
        "before_content": None,
        "after_content": "Review moved to 3rd December.",
        "edit_mode": "append",
    }

    result = asyncio.run(pipeline.execute(proposal))

    assert result["success"] is True
    assert "Review moved to 3rd December." in fake.updated_html
    assert fake.updated_html.index("Review moved to 3rd December.") < fake.updated_html.index("<h2>Owners</h2>")


def test_execute_replace_handles_text_wrapped_in_block():
    pipeline = ProposalPipeline()
    fake = FakeConfluenceClient()
    pipeline._client = fake
    proposal = {
        "change_type": "edit",
        "page_id": "page-1",
        "page_title": "Q3 Meeting Plan",
        "section_heading": "Timeline",
        "before_content": "The Q3 meeting plan is currently scheduled for 2nd September.",
        "after_content": "The Q3 meeting plan is currently scheduled for 3rd December.",
        "edit_mode": "replace",
    }

    result = asyncio.run(pipeline.execute(proposal))

    assert result["success"] is True
    assert "3rd December" in fake.updated_html
    assert "2nd September" not in fake.updated_html


def test_execute_schedules_rag_index_off_critical_path():
    """Accept returns as soon as the Confluence write lands; the RAG re-index runs
    after, off the request's critical path (freshness is best-effort)."""
    import threading

    indexed = threading.Event()

    class SlowRag:
        enabled = True

        def upsert_page(self, page):
            indexed.set()

    async def scenario():
        pipeline = ProposalPipeline()
        fake = FakeConfluenceClient()
        pipeline._client = fake
        pipeline._rag = SlowRag()
        proposal = {
            "change_type": "edit",
            "page_id": "page-1",
            "page_title": "Q3 Meeting Plan",
            "section_heading": "Timeline",
            "before_content": "The Q3 meeting plan is currently scheduled for 2nd September.",
            "after_content": "The Q3 meeting plan is currently scheduled for 3rd December.",
            "edit_mode": "replace",
        }

        result = await pipeline.execute(proposal)

        # The Confluence write is already applied and success is returned…
        assert result["success"] is True
        assert "3rd December" in fake.updated_html
        # …but the RAG re-index was scheduled, not awaited.
        assert pipeline._bg_tasks, "re-index must be scheduled as a background task"
        assert not indexed.is_set(), "execute must not block on the RAG re-index"

        # Draining the loop lets the background index complete.
        await asyncio.gather(*list(pipeline._bg_tasks))
        assert indexed.is_set()

    asyncio.run(scenario())


class TaskConfluenceClient:
    def __init__(self):
        self.updated_html = None
        self.updated_title = None

    async def search_pages(self, query, limit=10):
        if "Q2" in query or "plan" in query.lower():
            return [{"page_id": "page-task", "title": "Quarterly Planning"}]
        return []

    async def fetch_page(self, page_id):
        html = (
            "<h2>Quarterly plans</h2>"
            "<ac:task-list>"
            "<ac:task><ac:task-id>10</ac:task-id>"
            "<ac:task-status>incomplete</ac:task-status>"
            "<ac:task-body>Q2 plan</ac:task-body></ac:task>"
            "<ac:task><ac:task-id>11</ac:task-id>"
            "<ac:task-status>incomplete</ac:task-status>"
            "<ac:task-body>Q3 planning draft</ac:task-body></ac:task>"
            "</ac:task-list>"
        )
        return PageCandidate(
            page_id=page_id,
            title="Quarterly Planning",
            html=html,
            text="Quarterly plans\n[task: incomplete] Q2 plan\n[task: incomplete] Q3 planning draft",
            version=7,
            source="fake",
        )

    async def update_page(self, page_id, html_content, *, title=None, expected_version=None):
        self.updated_html = html_content
        self.updated_title = title
        assert page_id == "page-task"
        assert expected_version == 7
        return True

    async def create_page(self, title, html_content, space_key=None):
        return {"id": "new-page"}


def test_pipeline_drafts_indirect_task_completion(monkeypatch):
    pipeline = ProposalPipeline()
    pipeline._client = TaskConfluenceClient()

    async def fake_extract(_transcript, _text, query=None):
        return ExtractedMeeting(
            title="Planning Meeting",
            summary="Q2 plan completed.",
            change_intents=[
                ChangeIntent(
                    instruction="Mark the Q2 plan as complete",
                    subject="Q2 plan",
                    target_hint="Quarterly plans",
                    new_value="complete",
                    action="complete_task",
                    evidence=["we have completed the Q2 plan completely"],
                )
            ],
        )

    async def fake_retrieve_all_sections(intents):
        return [(intents[0], [VectorSearchHit(
            page_id="page-task", title="Quarterly Planning",
            heading="Quarterly plans", score=0.85,
            text="[task: incomplete] Q2 plan", version=7,
        )])]

    def fake_draft_sync(*, session_id, meeting, intent_sections, section_cache, page_cache, query):
        return [_make_proposal_dict(
            "page-task", "Quarterly Planning", "Quarterly plans",
            "[ ] Q2 plan", "[x] Q2 plan", "task_status",
        )]

    monkeypatch.setattr(pipeline, "_extract_meeting", fake_extract)
    monkeypatch.setattr(pipeline, "_retrieve_all_sections", fake_retrieve_all_sections)
    monkeypatch.setattr(pipeline, "_draft_grounded_proposals_sync", fake_draft_sync)
    monkeypatch.setattr(pipeline, "_adversarial_verify", async_passthrough_verify)

    _meeting, proposals = asyncio.run(
        pipeline.run(
            session_id="s1",
            transcript=[{"participant": "Asha", "text": "We have completed the Q2 plan completely."}],
        )
    )

    assert len(proposals) == 1
    proposal = proposals[0]
    assert proposal["page_id"] == "page-task"
    assert proposal["edit_mode"] == "task_status"
    assert proposal["before_content"] == "[ ] Q2 plan"
    assert proposal["after_content"] == "[x] Q2 plan"


def test_execute_task_status_updates_only_matching_checkbox():
    pipeline = ProposalPipeline()
    fake = TaskConfluenceClient()
    pipeline._client = fake
    proposal = {
        "change_type": "edit",
        "page_id": "page-task",
        "page_title": "Quarterly Planning",
        "section_heading": "Quarterly plans",
        "before_content": "[ ] Q2 plan",
        "after_content": "[x] Q2 plan",
        "edit_mode": "task_status",
    }

    result = asyncio.run(pipeline.execute(proposal))

    assert result["success"] is True
    assert "<ac:task-id>10</ac:task-id><ac:task-status>complete</ac:task-status>" in fake.updated_html
    assert "<ac:task-id>11</ac:task-id><ac:task-status>incomplete</ac:task-status>" in fake.updated_html


async def async_passthrough_verify(_meeting, proposals, _text):
    return proposals
