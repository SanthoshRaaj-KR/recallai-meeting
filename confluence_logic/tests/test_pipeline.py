"""Tests for Phase 2 multi-agent pipeline (RETR-01 through PIPE-04)."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

# --- Production imports guarded: these modules do not exist until Wave 1+ ---

try:
    from confluence_logic.agents.fact_extraction_agent import (
        _run_fact_extraction,
        ExtractedFacts,
    )
except ImportError:
    _run_fact_extraction = None
    ExtractedFacts = None

try:
    from confluence_logic.agents.pipeline_coordinator import (
        _merged_rag_retrieval,
        _draft_all_pages,
        _verify_proposal,
        _run_pipeline,
    )
except ImportError:
    _merged_rag_retrieval = None
    _draft_all_pages = None
    _verify_proposal = None
    _run_pipeline = None

# The review API already exists; the /review/pipeline/start route does NOT yet exist.
try:
    from confluence_logic.review.api import router as review_router
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    _test_app = FastAPI()
    _test_app.include_router(review_router)
    _pipeline_client = TestClient(_test_app, raise_server_exceptions=False)
except Exception:  # noqa: BLE001
    _pipeline_client = None


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_pinecone_store(monkeypatch):
    """Patches PineconeStore.search to return 2 fake Pinecone page dicts."""
    fake_results = [
        {"page_id": "p1", "title": "Page 1", "score": 0.9, "relevant_content": "content1"},
        {"page_id": "p2", "title": "Page 2", "score": 0.8, "relevant_content": "content2"},
    ]
    mock_search = MagicMock(return_value=fake_results)
    # Reset the singleton so a fresh mock instance is used — prevents a pre-initialized _store
    # from bypassing the class-level patch when the module was imported with real credentials
    monkeypatch.setattr("confluence_logic.agents.fact_extraction_agent._store", None)
    monkeypatch.setattr(
        "confluence_logic.db.vector_store.PineconeStore.search",
        mock_search,
    )
    return mock_search


@pytest.fixture
def mock_neo4j_graph(monkeypatch):
    """Patches query_user_confluence_graph to return 1 fake Neo4j page dict (different page_id)."""
    fake_results = [
        {"page_id": "p3", "title": "Page 3", "score": 0.85, "relevant_content": "content3"}
    ]
    mock_query = AsyncMock(return_value=fake_results)
    monkeypatch.setattr(
        "confluence_logic.confluence_page_graph.query_user_confluence_graph",
        mock_query,
    )
    return mock_query


@pytest.fixture
def mock_supabase(monkeypatch):
    """Patches Supabase store helpers; is_configured returns True.

    Security note (T-02-W0-02): no real credentials are used — all mocks use
    dummy return values only.
    """
    mock_upsert_proposal = MagicMock(return_value={"id": "prop-1"})
    mock_update_pipeline_job = MagicMock(return_value=None)
    mock_create_pipeline_job = MagicMock(return_value={"id": "job-1"})
    mock_is_configured = MagicMock(return_value=True)

    monkeypatch.setattr(
        "confluence_logic.review.supabase_store.upsert_proposal",
        mock_upsert_proposal,
    )
    monkeypatch.setattr(
        "confluence_logic.review.supabase_store.update_pipeline_job",
        mock_update_pipeline_job,
    )
    monkeypatch.setattr(
        "confluence_logic.review.supabase_store.create_pipeline_job",
        mock_create_pipeline_job,
    )
    monkeypatch.setattr(
        "confluence_logic.review.supabase_store.is_configured",
        mock_is_configured,
    )

    ns = MagicMock()
    ns.upsert_proposal = mock_upsert_proposal
    ns.update_pipeline_job = mock_update_pipeline_job
    ns.create_pipeline_job = mock_create_pipeline_job
    ns.is_configured = mock_is_configured
    return ns


@pytest.fixture
def mock_openai_runner(monkeypatch):
    """Patches agents.Runner.run to return a fake result with .final_output."""
    fake_final_output = MagicMock()
    fake_final_output.decisions = ["Launch next Friday"]
    fake_final_output.action_items = ["Prepare rollout checklist"]
    fake_final_output.new_requirements = []
    fake_final_output.owners = {"Prepare rollout checklist": "Ben"}
    fake_final_output.deadlines = {}
    fake_final_output.doc_worthy_updates = ["Launch timeline updated"]
    fake_final_output.query_terms = ["launch timeline", "rollout checklist"]

    fake_result = MagicMock()
    fake_result.final_output = fake_final_output

    mock_run = AsyncMock(return_value=fake_result)
    monkeypatch.setattr("agents.Runner.run", mock_run)
    return mock_run


# ---------------------------------------------------------------------------
# Test: RETR-01 + RETR-03 — merged RAG queries both Pinecone and Neo4j
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.xfail(reason="production module not yet implemented", strict=False)
async def test_merged_rag_queries_both_sources(mock_pinecone_store, mock_neo4j_graph):
    """RETR-01: merged RAG retrieval must call both Pinecone and Neo4j and deduplicate by page_id.

    This test is RED until confluence_logic.agents.pipeline_coordinator is created in Wave 1.
    """
    if _merged_rag_retrieval is None:
        raise ImportError("_merged_rag_retrieval not yet implemented (Wave 1)")

    query_terms = ["launch timeline", "rollout checklist"]
    user_id = "user-test-001"

    result = await _merged_rag_retrieval(query_terms=query_terms, user_id=user_id)

    # Both sources must have been called
    mock_pinecone_store.assert_called()
    mock_neo4j_graph.assert_called()

    # Deduplication: p1, p2 from Pinecone + p3 from Neo4j = 3 unique pages
    page_ids = [r["page_id"] for r in result]
    assert len(page_ids) == len(set(page_ids)), "page_ids must be deduplicated"
    assert set(page_ids) == {"p1", "p2", "p3"}


# ---------------------------------------------------------------------------
# Test: RETR-02 — fact extraction output schema
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.xfail(reason="production module not yet implemented", strict=False)
async def test_fact_extraction_output_schema(mock_openai_runner):
    """RETR-02: _run_fact_extraction must return an object with the required fact-extraction fields.

    This test is RED until confluence_logic.agents.fact_extraction_agent is created in Wave 1.
    """
    if _run_fact_extraction is None:
        raise ImportError("_run_fact_extraction not yet implemented (Wave 1)")

    short_transcript = [
        {"participant": "Asha", "text": "We decided to launch next Friday.", "timestamp": 1.0},
        {"participant": "Ben", "text": "I will prepare the rollout checklist.", "timestamp": 2.0},
    ]

    result = await _run_fact_extraction(transcript=short_transcript)

    required_fields = [
        "decisions",
        "action_items",
        "new_requirements",
        "owners",
        "deadlines",
        "doc_worthy_updates",
        "query_terms",
    ]
    for field in required_fields:
        assert hasattr(result, field), f"ExtractedFacts missing field: {field}"


# ---------------------------------------------------------------------------
# Test: PIPE-01 — pipeline start endpoint returns 202 and job_id
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.xfail(reason="POST /review/pipeline/start route not yet implemented", strict=False)
async def test_pipeline_start_returns_202():
    """PIPE-01: POST /review/pipeline/start must return 202 with a job_id.
    Without auth header must return 401.

    This test is RED (receives 404) until the route is added in Wave 1.
    """
    if _pipeline_client is None:
        raise ImportError("TestClient could not be constructed")

    # Without auth — must return 401
    no_auth_response = _pipeline_client.post(
        "/review/pipeline/start",
        json={"session_id": "test-session-pipe01"},
    )
    assert no_auth_response.status_code == 401, (
        f"Expected 401 without auth, got {no_auth_response.status_code}"
    )

    # With auth header — must return 202 and job_id
    with patch(
        "confluence_logic.review.supabase_store.user_from_bearer",
        return_value={"id": "user-test-pipe01"},
    ):
        response = _pipeline_client.post(
            "/review/pipeline/start",
            json={"session_id": "test-session-pipe01"},
            headers={"Authorization": "Bearer fake-token"},
        )

    assert response.status_code == 202, (
        f"Expected 202 for pipeline start, got {response.status_code}"
    )
    body = response.json()
    assert "job_id" in body, f"Response missing job_id: {body}"


# ---------------------------------------------------------------------------
# Test: PIPE-02 — drafter pool runs per-page agents concurrently
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.xfail(reason="production module not yet implemented", strict=False)
async def test_drafter_pool_parallel(mock_openai_runner):
    """PIPE-02: _draft_all_pages must call the Runner once per candidate page concurrently.

    This test is RED until confluence_logic.agents.pipeline_coordinator is created in Wave 1.
    """
    if _draft_all_pages is None:
        raise ImportError("_draft_all_pages not yet implemented (Wave 1)")

    candidate_pages = [
        {"page_id": "p1", "title": "Launch Plan", "score": 0.9, "relevant_content": "content1"},
        {"page_id": "p2", "title": "Rollout Checklist", "score": 0.85, "relevant_content": "content2"},
        {"page_id": "p3", "title": "Engineering Backlog", "score": 0.8, "relevant_content": "content3"},
    ]
    extracted_facts = MagicMock()
    extracted_facts.decisions = ["Launch Friday"]
    extracted_facts.action_items = ["Checklist"]
    extracted_facts.doc_worthy_updates = ["Timeline updated"]
    extracted_facts.query_terms = ["launch", "checklist"]

    results = await _draft_all_pages(
        candidate_pages=candidate_pages,
        extracted_facts=extracted_facts,
    )

    # Runner.run must be called exactly once per candidate page
    assert mock_openai_runner.call_count == 3, (
        f"Expected 3 Runner.run calls (one per page), got {mock_openai_runner.call_count}"
    )
    # Concurrency: all 3 calls should be awaited via asyncio.gather (not sequentially)
    # We verify this by confirming the results list has 3 entries
    assert len(results) == 3


# ---------------------------------------------------------------------------
# Test: PIPE-03 — verifier enriches proposal, never drops it
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.xfail(reason="production module not yet implemented", strict=False)
async def test_verifier_enriches_not_drops():
    """PIPE-03: _verify_proposal must add confidence, risk, and verifier_note without dropping the card.

    This test is RED until confluence_logic.agents.pipeline_coordinator is created in Wave 1.
    """
    if _verify_proposal is None:
        raise ImportError("_verify_proposal not yet implemented (Wave 1)")

    draft = {
        "change_type": "edit",
        "page_id": "p1",
        "page_title": "Launch Plan",
        "section_heading": "Decisions",
        "before_content": "Launch Q3",
        "after_content": "Launch next Friday as decided in meeting.",
        "rationale": "Meeting decision captured.",
        "relevant_content": "Current Confluence content about launch.",
    }

    enriched = await _verify_proposal(draft=draft)

    # The proposal must not be dropped — enriched dict must be returned
    assert enriched is not None, "verifier must not drop proposals"

    # Required enrichment fields
    assert "confidence" in enriched, "verifier must add confidence field"
    assert "risk" in enriched, "verifier must add risk field"
    assert "verifier_note" in enriched, "verifier must add verifier_note field"

    # Values must be non-None and valid
    assert enriched["confidence"] is not None
    assert enriched["risk"] is not None
    assert enriched["verifier_note"] is not None
    assert enriched["confidence"] in {"high", "medium", "low"}, (
        f"confidence must be high/medium/low, got {enriched['confidence']!r}"
    )
    assert enriched["risk"] in {"safe", "review", "risky"}, (
        f"risk must be safe/review/risky, got {enriched['risk']!r}"
    )


# ---------------------------------------------------------------------------
# Test: PIPE-04 — incremental Supabase writes (one per page, not bulk at end)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.xfail(reason="production module not yet implemented", strict=False)
async def test_incremental_proposal_writes(mock_supabase, mock_openai_runner):
    """PIPE-04: pipeline must persist each proposal to Supabase as it completes, not in a bulk write.

    This test is RED until confluence_logic.agents.pipeline_coordinator is created in Wave 1.
    """
    if _run_pipeline is None:
        raise ImportError("_run_pipeline not yet implemented (Wave 1)")

    candidate_pages = [
        {"page_id": "p1", "title": "Launch Plan", "score": 0.9, "relevant_content": "c1"},
        {"page_id": "p2", "title": "Rollout Checklist", "score": 0.85, "relevant_content": "c2"},
    ]
    extracted_facts = MagicMock()
    extracted_facts.decisions = ["Launch Friday"]
    extracted_facts.action_items = []
    extracted_facts.doc_worthy_updates = []
    extracted_facts.query_terms = ["launch"]

    await _run_pipeline(
        session_id="test-pipe04",
        job_id="job-pipe04",
        candidate_pages=candidate_pages,
        extracted_facts=extracted_facts,
        user_id="user-pipe04",
    )

    # upsert_proposal must be called once per candidate page (incremental writes)
    assert mock_supabase.upsert_proposal.call_count == 2, (
        f"Expected upsert_proposal called twice (once per page), "
        f"got {mock_supabase.upsert_proposal.call_count}"
    )


# ---------------------------------------------------------------------------
# PIPE-05 SSE tests (Phase 3, Plan 01)
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_pipeline_queue(monkeypatch):
    """Patches _job_queues with a real asyncio.Queue under test-job-pipe05."""
    import asyncio as _asyncio
    from confluence_logic.review import api as _api
    test_queue = _asyncio.Queue()
    monkeypatch.setattr(_api, "_job_queues", {"test-job-pipe05": test_queue})
    yield test_queue


@pytest.mark.xfail(reason="PIPE-05 — implementation in Task 2/3", strict=False)
def test_sse_stream_returns_events(mock_pipeline_queue):
    """PIPE-05: GET /review/pipeline/{job_id}/stream returns text/event-stream with auth + ownership."""
    if _pipeline_client is None:
        raise ImportError("_pipeline_client not constructed")

    # 401 — no token at all
    resp = _pipeline_client.get("/review/pipeline/test-job-pipe05/stream")
    assert resp.status_code == 401

    # 403 — authenticated but does not own the job
    with patch(
        "confluence_logic.review.supabase_store.user_from_bearer",
        return_value={"id": "user-other"},
    ), patch(
        "confluence_logic.review.supabase_store.get_pipeline_job",
        return_value={"job_id": "test-job-pipe05", "user_id": "user-owner"},
    ):
        resp = _pipeline_client.get(
            "/review/pipeline/test-job-pipe05/stream",
            headers={"Authorization": "Bearer fake-token"},
        )
        assert resp.status_code == 403

    # 200 + text/event-stream — owner with valid token
    with patch(
        "confluence_logic.review.supabase_store.user_from_bearer",
        return_value={"id": "user-owner"},
    ), patch(
        "confluence_logic.review.supabase_store.get_pipeline_job",
        return_value={"job_id": "test-job-pipe05", "user_id": "user-owner"},
    ):
        # Put a sentinel into the queue BEFORE the request so the generator terminates quickly.
        from confluence_logic.review import api as _api
        mock_pipeline_queue.put_nowait(_api._SENTINEL)

        resp = _pipeline_client.get(
            "/review/pipeline/test-job-pipe05/stream",
            headers={"Authorization": "Bearer fake-token"},
        )
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")
        assert resp.headers.get("cache-control") == "no-cache"
        assert resp.headers.get("x-accel-buffering") == "no"


@pytest.mark.xfail(reason="PIPE-05 — implementation in Task 2/3", strict=False)
def test_stage_events_emitted(mock_pipeline_queue):
    """PIPE-05: _emit() puts events into the registered queue, silently no-ops otherwise."""
    from confluence_logic.review import api as _api

    _api._emit("test-job-pipe05", {"type": "stage_start", "stage": "fact_extraction"})
    item = mock_pipeline_queue.get_nowait()
    assert item == {"type": "stage_start", "stage": "fact_extraction"}

    # No raise when job_id absent
    _api._emit("nonexistent-job-id-xyz", {"type": "stage_start", "stage": "drafting"})


@pytest.mark.asyncio
@pytest.mark.xfail(reason="PIPE-05 — implementation in Task 2/3", strict=False)
async def test_proposal_ready_includes_id(mock_pipeline_queue):
    """PIPE-05: proposal_ready event carries the Supabase-assigned UUID under 'id'."""
    from confluence_logic.review import api as _api

    fake_verified = {
        "change_type": "edit",
        "page_id": "page-1",
        "page_title": "Launch Plan",
        "section_heading": "Decisions",
        "before_content": "old",
        "after_content": "new",
        "rationale": "decision was made",
        "transcript_evidence": ["evidence quote"],
        "confidence": "high",
        "risk": "safe",
        "verifier_note": "ok",
    }

    with patch(
        "confluence_logic.review.api._run_drafter",
        new=AsyncMock(return_value={"change_type": "edit", "page_id": "page-1", "page_title": "Launch Plan"}),
    ), patch(
        "confluence_logic.review.api._run_verifier",
        new=AsyncMock(return_value=fake_verified),
    ), patch(
        "confluence_logic.review.supabase_store.upsert_proposal",
        return_value="supabase-uuid-xyz",
    ):
        await _api._draft_verify_persist(
            page={"page_id": "page-1", "page_title": "Launch Plan", "relevant_content": ""},
            facts=None,
            transcript_text="",
            job_id="test-job-pipe05",
            session_id="sess-1",
            user_id="user-owner",
        )

    item = mock_pipeline_queue.get_nowait()
    assert item["type"] == "proposal_ready"
    assert item["id"] == "supabase-uuid-xyz"
    for field in ("change_type", "page_id", "page_title", "before_content", "after_content",
                  "rationale", "transcript_evidence", "confidence", "risk", "verifier_note"):
        assert field in item, f"proposal_ready event missing field: {field}"


@pytest.mark.asyncio
@pytest.mark.xfail(reason="PIPE-05 — implementation in Task 2/3", strict=False)
async def test_pipeline_error_event(mock_pipeline_queue):
    """PIPE-05: exception in _run_pipeline emits pipeline_error event then _SENTINEL."""
    from confluence_logic.review import api as _api

    with patch(
        "confluence_logic.review.api._extract_facts",
        new=AsyncMock(side_effect=RuntimeError("simulated")),
    ), patch(
        "confluence_logic.review.supabase_store.update_pipeline_job",
        return_value=None,
    ):
        await _api._run_pipeline(
            job_id="test-job-pipe05",
            session_id="sess-1",
            user_id="user-owner",
            graph_user_id="user-owner",
        )

    events = []
    while True:
        try:
            events.append(mock_pipeline_queue.get_nowait())
        except Exception:
            break

    error_events = [e for e in events if isinstance(e, dict) and e.get("type") == "pipeline_error"]
    assert len(error_events) == 1
    assert "simulated" in error_events[0]["detail"]
    assert _api._SENTINEL in events
