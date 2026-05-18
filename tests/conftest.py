"""Shared pytest fixtures for the Phase 8 test suite (tests/).

Mocks Pinecone, Neo4j, the Confluence connector, and supabase_store so the
post-meeting pipeline can be exercised end-to-end without network access.

Reuses the same shape as confluence_logic/tests/test_pipeline.py's fixtures so
both suites stay consistent.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import pytest
from unittest.mock import AsyncMock, MagicMock


# ---------------------------------------------------------------------------
# RAG mocks
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_pinecone_store(monkeypatch):
    """Patch PineconeStore.search to return an empty list by default.

    Per-test code can override via ``mock_pinecone_store.return_value = [...]``.
    Also resets the module-level singleton so the patch takes effect for the
    next call (matches confluence_logic/tests/test_pipeline.py:49 behaviour).
    """
    mock_search = MagicMock(return_value=[])
    monkeypatch.setattr("confluence_logic.agents.fact_extraction_agent._store", None)
    monkeypatch.setattr(
        "confluence_logic.db.vector_store.PineconeStore.search",
        mock_search,
    )
    # Also blank out the module-level normalisation helper's side-effects by
    # stubbing the merged retrieval used in the legacy fallback branch.
    monkeypatch.setattr(
        "confluence_logic.review.api._merged_rag_retrieval",
        AsyncMock(return_value=[]),
    )
    return mock_search


@pytest.fixture
def mock_neo4j_graph(monkeypatch):
    """Patch the Neo4j Confluence-graph query and the ensure-graph step.

    ``ensure_user_confluence_graph`` is a slow IO call inside the pipeline and
    has no value in a mocked run — short-circuit it to a no-op AsyncMock.
    """
    mock_query = AsyncMock(return_value=[])
    monkeypatch.setattr(
        "confluence_logic.confluence_page_graph.query_user_confluence_graph",
        mock_query,
    )
    monkeypatch.setattr(
        "confluence_logic.confluence_page_graph.ensure_user_confluence_graph",
        AsyncMock(return_value=None),
    )
    return mock_query


@pytest.fixture
def mock_confluence_connector(monkeypatch):
    """A drop-in MagicMock ConfluenceConnector wired into _get_connector().

    Defaults:
      - search_pages → []  (override per test)
      - fetch_page_html → ""
      - get_page_metadata → {"version": {"number": 1}}
      - push_update → True
      - create_page → {"id": "new-page-id"}
      - get_workspace_titles → []
    """
    connector = MagicMock()
    connector.search_pages.return_value = []
    connector.fetch_page_html.return_value = ""
    connector.get_page_metadata.return_value = {"version": {"number": 1}}
    connector.push_update.return_value = True
    connector.create_page.return_value = {"id": "new-page-id"}
    connector.get_workspace_titles = MagicMock(return_value=[])

    monkeypatch.setattr(
        "confluence_logic.review.api._get_connector",
        lambda: connector,
    )
    # The connector is also used by the regenerate flow + fallback; make sure
    # _resolve_page_id returns whatever was passed (so tests can supply page_id directly).
    monkeypatch.setattr(
        "confluence_logic.review.api._resolve_page_id",
        AsyncMock(side_effect=lambda pid, title=None: pid),
    )
    return connector


# ---------------------------------------------------------------------------
# Supabase mocks (used by tests that hit /sessions/.../review/regenerate)
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_supabase_store(monkeypatch):
    """A bag of patched supabase_store helpers exposed as attributes on the returned mock.

    Tests assert e.g. ``mock_supabase_store.update_proposal_full.called`` to
    confirm regenerate-in-place wrote back to Supabase.
    """
    ns = MagicMock()
    ns.upsert_proposal = MagicMock(return_value="prop-uuid-1")
    ns.update_pipeline_job = MagicMock(return_value=None)
    ns.create_pipeline_job = MagicMock(return_value="job-uuid-1")
    ns.is_configured = MagicMock(return_value=True)
    ns.get_proposal_with_intent = MagicMock(return_value=None)
    ns.update_proposal_full = MagicMock(return_value={"status": "pending"})
    ns.user_from_bearer = MagicMock(return_value={"id": "user-test"})
    ns.get_history_item = MagicMock(return_value={})
    ns.get_pipeline_job = MagicMock(return_value={"job_id": "job-test", "user_id": "user-test"})

    monkeypatch.setattr("confluence_logic.review.supabase_store.upsert_proposal", ns.upsert_proposal)
    monkeypatch.setattr("confluence_logic.review.supabase_store.update_pipeline_job", ns.update_pipeline_job)
    monkeypatch.setattr("confluence_logic.review.supabase_store.create_pipeline_job", ns.create_pipeline_job)
    monkeypatch.setattr("confluence_logic.review.supabase_store.is_configured", ns.is_configured)
    monkeypatch.setattr("confluence_logic.review.supabase_store.get_proposal_with_intent", ns.get_proposal_with_intent)
    monkeypatch.setattr("confluence_logic.review.supabase_store.update_proposal_full", ns.update_proposal_full)
    monkeypatch.setattr("confluence_logic.review.supabase_store.user_from_bearer", ns.user_from_bearer)
    monkeypatch.setattr("confluence_logic.review.supabase_store.get_history_item", ns.get_history_item)
    monkeypatch.setattr("confluence_logic.review.supabase_store.get_pipeline_job", ns.get_pipeline_job)
    return ns


# ---------------------------------------------------------------------------
# FastAPI app for TestClient-based tests
# ---------------------------------------------------------------------------


@pytest.fixture
def app():
    """A bare FastAPI app with only the review router mounted.

    Used by /sessions/{sid}/review/regenerate/{pid} tests so we don't pull in
    the full uvicorn app (which would also bring the WebSocket handlers).
    """
    from fastapi import FastAPI
    from confluence_logic.review.api import router as review_router

    test_app = FastAPI()
    test_app.include_router(review_router)
    return test_app


# ---------------------------------------------------------------------------
# OpenAI / agents.Runner mock for fact extraction
# ---------------------------------------------------------------------------


def _make_extracted_facts_with_create(subject: str, instruction: str, verbatim: str = "") -> Any:
    """Build a real ExtractedFacts with one create ChangeIntent.

    Centralised so test_fact_extraction_explicit_create.py can dispatch one of
    these per phrase without re-implementing the schema.
    """
    from confluence_logic.agents.fact_extraction_agent import ChangeIntent, ExtractedFacts

    return ExtractedFacts(
        change_intents=[
            ChangeIntent(
                instruction=instruction,
                subject=subject,
                target_hint=subject,
                old_value="",
                new_value="",
                action="create",
                rationale=instruction,
                verbatim_content=verbatim,
            )
        ]
    )


@pytest.fixture
def mock_openai_facts(monkeypatch):
    """Patches agents.Runner.run inside fact_extraction_agent to return a canned
    ExtractedFacts. Per-test code sets the desired payload via the returned
    helper before calling _run_fact_extraction.

    When env var JARVIS_TEST_USE_REAL_LLM=1, the patch is skipped so the real
    LLM is invoked instead.
    """
    import os

    helper = {"facts": None, "calls": 0}

    if os.getenv("JARVIS_TEST_USE_REAL_LLM") == "1":
        # Bypass — real LLM is in play
        helper["bypassed"] = True
        return helper

    async def _fake_run(_agent, _text, **_kwargs):
        helper["calls"] += 1
        result = MagicMock()
        result.final_output = helper.get("facts") or _make_extracted_facts_with_create(
            "Default Subject", "Default instruction"
        )
        return result

    monkeypatch.setattr("agents.Runner.run", _fake_run)
    helper["set"] = lambda subject, instruction, verbatim="": helper.update(
        {"facts": _make_extracted_facts_with_create(subject, instruction, verbatim)}
    )
    return helper
