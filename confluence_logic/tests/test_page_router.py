"""Tests for PageRouter (PROP-V2-03).

Phase 10 — auto-propose-pipeline-quality-redesign-v2.

PageRouter is the new routing stage between FactExtraction and
PageQualifier (D-05) that performs the three-signal merge: semantic
(Pinecone) + graph-aware (Neo4j heading match) + explicit-token-gate.

These tests pin the three-signal merge contract:
  1. explicit-token verbatim match in title or H1/H2 → force-promote to top
  2. graph-heading match outranks pure semantic match
  3. empty signals → empty list (drafter is skipped, never default-fallback)
  4. ``graph_user_id`` is an explicit parameter — NEVER read from ContextVar
     (Pitfall 4 — sequential-executor safety; ContextVars don't survive
     ``asyncio.to_thread`` hops nor parallel pipeline waves)
"""
from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

pytestmark = pytest.mark.asyncio

from confluence_logic.agents.page_router import route_intent  # noqa: F401, E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _intent(subject: str = "frameworks", target_hint: str = "", instruction: str = "") -> Any:
    """Build a real ChangeIntent so the router sees the canonical schema."""
    from confluence_logic.agents.fact_extraction_agent import ChangeIntent

    return ChangeIntent(
        instruction=instruction or f"Update {subject}",
        subject=subject,
        target_hint=target_hint,
        old_value="",
        new_value="",
        action="replace",
        rationale="",
        verbatim_content="",
    )


def _pinecone_match(page_id: str, score: float, title: str = "", heading: str = "") -> Dict[str, Any]:
    """Build a Pinecone-shaped match dict with metadata.page_id (Pitfall 2)."""
    return {
        "id": f"{page_id}_0",  # chunk id — NEVER the page id
        "score": score,
        "metadata": {
            "page_id": page_id,
            "title": title,
            "heading": heading,
            "space_key": "TEST",
            "markdown_content": f"content for {page_id}",
            "text_summary": f"summary for {page_id}",
        },
    }


def _graph_row(page_id: str, title: str, heading: str, score: float) -> Dict[str, Any]:
    """Build a row matching query_user_confluence_graph's return shape."""
    return {
        "page_id": page_id,
        "title": title,
        "space_key": "TEST",
        "version": 1,
        "heading": heading,
        "relevant_content": f"section text for {heading}",
        "score": score,
        "source": "neo4j_confluence_graph",
    }


def _list_page(page_id: str, title: str, headings: List[str]) -> Dict[str, Any]:
    """Build a workspace-page row that the explicit-token scan iterates."""
    return {
        "page_id": page_id,
        "title": title,
        "space_key": "TEST",
        "version": 1,
        "headings": headings,
    }


# ---------------------------------------------------------------------------
# Behaviour tests
# ---------------------------------------------------------------------------


async def test_explicit_token_force_promotes():
    """PROP-V2-03: a page whose title verbatim-matches the subject noun is force-promoted to the top.

    Scenario: intent.subject = "frameworks". Pinecone scores an unrelated
    page (p1 "Architecture Overview") highest. Graph also returns p1 first.
    But p2 has title="Frameworks" — the explicit-token gate MUST force-promote
    p2 to position 0 regardless of Pinecone/graph scores.
    """
    intent = _intent(subject="frameworks")

    pinecone_results = [
        _pinecone_match("p1", score=0.95, title="Architecture Overview"),
        _pinecone_match("p3", score=0.50, title="Random"),
    ]
    graph_results = [
        _graph_row("p1", "Architecture Overview", "Background", score=3),
    ]
    workspace_pages = [
        _list_page("p1", "Architecture Overview", ["Background"]),
        _list_page("p2", "Frameworks", ["Overview"]),  # title matches subject verbatim
        _list_page("p3", "Random", ["Stuff"]),
    ]

    mock_store = MagicMock()
    mock_store.search.return_value = pinecone_results

    with patch(
        "confluence_logic.agents.page_router._get_store",
        return_value=mock_store,
    ), patch(
        "confluence_logic.agents.page_router.query_user_confluence_graph",
        new=AsyncMock(return_value=graph_results),
    ), patch(
        "confluence_logic.agents.page_router.list_user_confluence_pages",
        new=AsyncMock(return_value=workspace_pages),
    ):
        ranked = await route_intent(intent, graph_user_id="user-1", top_n=5)

    assert ranked, "expected at least one ranked candidate"
    assert ranked[0]["page_id"] == "p2", (
        f"expected force-promoted page 'p2' (title 'Frameworks') at position 0, "
        f"got {[p.get('page_id') for p in ranked]}"
    )
    assert ranked[0].get("forced") is True, "force-promoted row must carry forced=True"


async def test_three_signal_merge_ranks_graph_above_semantic_when_heading_match():
    """PROP-V2-03: graph-heading match outranks pure semantic match.

    Scenario: intent.subject = "deployment". Pinecone returns p2 with high
    semantic score (no graph hit). Neo4j returns p1 with a section heading
    match ("Deployment Architecture"). The Cypher already weights
    section_score * 2 — the merged ranking MUST therefore put p1 above p2,
    even though p1 had no Pinecone match at all.
    """
    intent = _intent(subject="deployment")

    pinecone_results = [
        _pinecone_match("p2", score=0.92, title="Old Deployment Notes"),
    ]
    graph_results = [
        # section_score * 2 + page_score baked into score by existing Cypher
        _graph_row("p1", "System Overview", "Deployment Architecture", score=5),
    ]
    workspace_pages = [
        _list_page("p1", "System Overview", ["Deployment Architecture"]),
        _list_page("p2", "Old Deployment Notes", ["Notes"]),
    ]

    mock_store = MagicMock()
    mock_store.search.return_value = pinecone_results

    with patch(
        "confluence_logic.agents.page_router._get_store",
        return_value=mock_store,
    ), patch(
        "confluence_logic.agents.page_router.query_user_confluence_graph",
        new=AsyncMock(return_value=graph_results),
    ), patch(
        "confluence_logic.agents.page_router.list_user_confluence_pages",
        new=AsyncMock(return_value=workspace_pages),
    ):
        ranked = await route_intent(intent, graph_user_id="user-1", top_n=5)

    page_ids = [p["page_id"] for p in ranked]
    assert "p1" in page_ids and "p2" in page_ids, f"expected both pages, got {page_ids}"
    # p1 (heading section match in Deployment Architecture) is force-promoted by
    # the explicit-token gate because "deployment" appears verbatim in its H1/H2.
    # p2 only matches via title — title-only matches are also forced but at a
    # lower rank, so we just assert p1 is ahead of p2.
    assert page_ids.index("p1") < page_ids.index("p2"), (
        f"expected p1 (graph + heading) ranked above p2 (pinecone only), got order {page_ids}"
    )


async def test_returns_empty_when_no_signals_hit():
    """PROP-V2-03: when all three signals miss, route_intent returns []."""
    intent = _intent(subject="zorblax42")

    mock_store = MagicMock()
    mock_store.search.return_value = []

    workspace_pages = [
        _list_page("p1", "Architecture Overview", ["Background"]),
        _list_page("p2", "Other Page", ["Section"]),
    ]

    with patch(
        "confluence_logic.agents.page_router._get_store",
        return_value=mock_store,
    ), patch(
        "confluence_logic.agents.page_router.query_user_confluence_graph",
        new=AsyncMock(return_value=[]),
    ), patch(
        "confluence_logic.agents.page_router.list_user_confluence_pages",
        new=AsyncMock(return_value=workspace_pages),
    ):
        ranked = await route_intent(intent, graph_user_id="user-1", top_n=5)

    assert ranked == [], (
        f"expected empty list when no signal hits (NOT a default-fallback), got {ranked}"
    )


async def test_graph_user_id_passed_explicitly_not_via_contextvar():
    """PROP-V2-03: ``graph_user_id`` is an explicit parameter with NO default.

    Two structural guarantees (Pitfall 4):
      1. inspect.signature shows ``graph_user_id`` as a parameter without a
         default value (caller MUST pass it).
      2. The page_router source contains zero ContextVar reads — the router
         never reaches into ``_current_graph_user_id`` or
         ``get_current_graph_user_id``.
    """
    sig = inspect.signature(route_intent)
    assert "graph_user_id" in sig.parameters, (
        "route_intent must accept 'graph_user_id' as a named parameter"
    )
    param = sig.parameters["graph_user_id"]
    assert param.default is inspect.Parameter.empty, (
        f"graph_user_id must have NO default — caller must pass it explicitly "
        f"(Pitfall 4: ContextVars don't survive asyncio hops). Got default={param.default!r}"
    )

    # Source-level guarantee: no ContextVar reads in page_router.py.
    router_src = Path(
        "confluence_logic/agents/page_router.py"
    ).read_text(encoding="utf-8")

    banned_reads = (
        "get_current_graph_user_id",
        "_current_graph_user_id.get(",
        "confluence_graph_user_id.get(",
    )
    for needle in banned_reads:
        assert needle not in router_src, (
            f"page_router.py must not read ContextVar '{needle}' "
            f"(Pitfall 4 — graph_user_id is an explicit parameter)"
        )
