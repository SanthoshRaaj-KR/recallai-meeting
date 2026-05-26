import pytest

from confluence_logic.pipeline.context import PipelineContext
from confluence_logic.pipeline.contracts import (
    ChangeIntentV3,
    EvidenceSpan,
    ProposalCardV3,
    RetrievalResult,
    SectionCandidate,
    PlannedOperation,
)
from confluence_logic.pipeline.stages.gate import GateResult


pytestmark = pytest.mark.asyncio


async def test_run_expands_qualified_candidates_and_passes_page_content(monkeypatch):
    import confluence_logic.pipeline.run as run_mod

    intent = ChangeIntentV3(
        kind="fact_update",
        subject="SOC2 audit schedule",
        old_value="Q3",
        new_value="Q2",
        dedup_key="soc2-q3-q2",
        evidence=[EvidenceSpan(text="SOC2 moved from Q3 to Q2", start=0, end=25)],
    )
    candidates = [
        SectionCandidate(page_id="pg-1", page_title="SOC2", section_heading="Audit"),
        SectionCandidate(page_id="pg-2", page_title="Security", section_heading="Compliance"),
    ]
    page_html_seen = []
    gate_content_seen = []

    async def fake_load_transcript(session_id, ctx):
        ctx.transcript_text = "SOC2 moved from Q3 to Q2."
        return [{"text": ctx.transcript_text}]

    async def fake_extract(text, ctx=None):
        return [intent]

    async def fake_retrieve(intent_arg, corpus=None):
        return RetrievalResult(intent=intent_arg, candidates=candidates)

    async def fake_rerank(intent_arg, retrieval):
        return retrieval.candidates

    async def fake_plan(intent_arg, candidate, page_html=None):
        page_html_seen.append(page_html)
        return PlannedOperation(
            operation="edit_section",
            page_id=candidate.page_id,
            page_title=candidate.page_title,
            section_heading=candidate.section_heading,
            before_content="Q3",
            after_content="Q2",
            rationale="meeting update",
        )

    async def fake_gate(op, transcript_text="", current_page_content="", **kwargs):
        gate_content_seen.append(current_page_content)
        return GateResult(False, False, [], None, 0.9, "high")

    monkeypatch.setattr(run_mod, "load_transcript", fake_load_transcript)
    monkeypatch.setattr(run_mod, "extract_intents", fake_extract)
    monkeypatch.setattr(run_mod, "retrieve_candidates", fake_retrieve)
    monkeypatch.setattr(run_mod, "rerank_candidates", fake_rerank)
    monkeypatch.setattr(run_mod, "plan_operation", fake_plan)
    monkeypatch.setattr(run_mod, "apply_grounding_gate_v3", fake_gate)

    ctx = PipelineContext(
        session_id="sess-v3",
        user_id="user-v3",
        graph_user_id="graph-v3",
        section_corpus=[
            {"page_id": "pg-1", "title": "SOC2", "content_html": "<h2>Audit</h2><p>Q3</p>"},
            {"page_id": "pg-2", "title": "Security", "content_html": "<h2>Compliance</h2><p>Q3</p>"},
        ],
        contradiction_enabled=False,
    )

    cards = await run_mod.run(ctx)

    assert {card.page_id for card in cards} == {"pg-1", "pg-2"}
    assert all("Q3" in html for html in page_html_seen)
    assert all("Q3" in content for content in gate_content_seen)


async def test_run_pipeline_v3_emits_persisted_cards_and_proposal_count(monkeypatch):
    import confluence_logic.review.api as api
    import confluence_logic.pipeline.run as pipeline_run

    events = []
    upserts = []

    async def fake_run(ctx):
        return [
            ProposalCardV3(
                change_type="edit",
                page_id="pg-1",
                page_title="SOC2",
                section_heading="Audit",
                before_content="Q3",
                after_content="Q2",
                status="pending",
                change_summary="Move audit to Q2",
                operation_action="edit_section",
                confidence_score=0.9,
                confidence_bin="high",
                transcript_evidence=["SOC2 moved from Q3 to Q2"],
            )
        ]

    def fake_upsert(row):
        upserts.append(row)
        return "proposal-1"

    monkeypatch.setattr(pipeline_run, "run", fake_run)
    monkeypatch.setattr(api, "_emit", lambda job_id, event: events.append(event))
    monkeypatch.setattr(api.supabase_store, "update_pipeline_job", lambda *args, **kwargs: None)
    monkeypatch.setattr(api.supabase_store, "upsert_proposal", fake_upsert)

    await api._run_pipeline_v3("sess-1", "job-1", "user-1", "graph-1")

    assert upserts[0]["job_id"] == "job-1"
    dict_events = [event for event in events if isinstance(event, dict)]
    assert any(event.get("type") == "proposal_ready" and event.get("id") == "proposal-1" for event in dict_events)
    assert any(event.get("type") == "pipeline_complete" and event.get("proposal_count") == 1 for event in dict_events)
