"""Wave 0 RED test scaffold for PageRouter (PROP-V2-03).

Phase 10 — auto-propose-pipeline-quality-redesign-v2.

PageRouter is the new routing stage between FactExtraction and
PageQualifier (D-05) that performs the three-signal merge: semantic
(Pinecone) + graph-aware (Neo4j heading match) + explicit-token-gate.
Wave 1 creates `confluence_logic/agents/page_router.py`; until then
this import fails RED.
"""
import pytest

pytestmark = pytest.mark.asyncio

from confluence_logic.agents.page_router import route_intent  # noqa: F401, E402


async def test_explicit_token_force_promotes():
    """PROP-V2-03: a page whose title or H1/H2 verbatim-matches the subject noun is force-promoted to the top."""
    pytest.fail("Wave 0 RED — PageRouter implementation pending (Wave 1)")


async def test_three_signal_merge_ranks_graph_above_semantic_when_heading_match():
    """PROP-V2-03: graph-heading match outranks pure semantic match when the intent subject aligns with a section heading."""
    pytest.fail("Wave 0 RED — PageRouter implementation pending (Wave 1)")


async def test_returns_empty_when_no_signals_hit():
    """PROP-V2-03: when all three signals miss, route_intent returns an empty candidate list (drafter is skipped)."""
    pytest.fail("Wave 0 RED — PageRouter implementation pending (Wave 1)")


async def test_graph_user_id_passed_explicitly_not_via_contextvar():
    """PROP-V2-03: route_intent receives ``graph_user_id`` as an explicit kwarg — never reads it from the ambient ContextVar (sequential-executor safety)."""
    pytest.fail("Wave 0 RED — PageRouter implementation pending (Wave 1)")
