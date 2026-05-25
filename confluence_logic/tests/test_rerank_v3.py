"""RED tests for reranking stage (RETR-V3-03 — Phase 11).

RETR-V3-03: Fused candidates from RRF pass through a reranker (LLM listwise
on gpt-5.4-nano) that reorders by true relevance; only the top-k proceed;
if the LLM reranker fails the stage falls back to RRF order.

These tests import from ``confluence_logic.pipeline.stages.rerank`` which
does not exist yet. Pytest collection fails with ImportError — expected RED
state for Wave 0 of Phase 11.
"""

import pytest
from unittest.mock import AsyncMock, patch

pytestmark = pytest.mark.asyncio

# ---------------------------------------------------------------------------
# Imports from not-yet-built pipeline targets (RED: ImportError at collection)
# ---------------------------------------------------------------------------
from confluence_logic.pipeline.stages.rerank import rerank_candidates
from confluence_logic.pipeline.contracts import (
    ChangeIntentV3,
    EvidenceSpan,
    SectionCandidate,
    RetrievalResult,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_intent(subject: str = "test") -> ChangeIntentV3:
    return ChangeIntentV3(
        kind="fact_update",
        subject=subject,
        old_value="old",
        new_value="new",
        dedup_key="test-intent",
        evidence=[EvidenceSpan(text="some evidence", start=0, end=13)],
    )


def _make_retrieval_result(*candidates: SectionCandidate) -> RetrievalResult:
    return RetrievalResult(candidates=list(candidates), fusion_log="rrf applied")


# ---------------------------------------------------------------------------
# RETR-V3-03-1: Reranker changes the order when LLM disagrees with RRF
# ---------------------------------------------------------------------------

async def test_reranker_reorders_candidates():
    """RETR-V3-03: if the LLM reranker puts candidate B above A, the output
    must reflect that ordering (not the original RRF order)."""
    candidate_a = SectionCandidate(
        page_id="pg-auth",
        section_heading="Provider",
        score=0.92,
        source="dense",
    )
    candidate_b = SectionCandidate(
        page_id="pg-soc2",
        section_heading="Audit Schedule",
        score=0.85,
        source="lexical",
    )
    retrieval = _make_retrieval_result(candidate_a, candidate_b)
    intent = _make_intent("authentication")

    # LLM reranker returns B, A (reverse of RRF order).
    with patch(
        "confluence_logic.pipeline.stages.rerank._llm_listwise_rerank",
        new=AsyncMock(return_value=["pg-soc2::Audit Schedule", "pg-auth::Provider"]),
    ):
        reranked = await rerank_candidates(intent, retrieval, top_k=10)

    assert reranked[0].page_id == "pg-soc2", (
        "Reranker must reorder: LLM placed pg-soc2 first"
    )
    assert reranked[1].page_id == "pg-auth"


# ---------------------------------------------------------------------------
# RETR-V3-03-2: Only top-k candidates proceed past the reranker
# ---------------------------------------------------------------------------

async def test_reranker_enforces_top_k():
    """RETR-V3-03: only top_k candidates must be returned; extras are dropped."""
    candidates = [
        SectionCandidate(page_id=f"pg-{i}", section_heading="Section", score=1.0 - i*0.05, source="dense")
        for i in range(10)
    ]
    retrieval = _make_retrieval_result(*candidates)
    intent = _make_intent()

    # LLM returns all 10 in some order.
    with patch(
        "confluence_logic.pipeline.stages.rerank._llm_listwise_rerank",
        new=AsyncMock(return_value=[f"pg-{i}::Section" for i in range(10)]),
    ):
        reranked = await rerank_candidates(intent, retrieval, top_k=3)

    assert len(reranked) == 3, (
        f"top_k=3 must return exactly 3 candidates, got {len(reranked)}"
    )


# ---------------------------------------------------------------------------
# RETR-V3-03-3: Fallback to RRF order when LLM reranker fails
# ---------------------------------------------------------------------------

async def test_reranker_falls_back_to_rrf_order_on_llm_error():
    """RETR-V3-03: if the LLM reranker raises an exception, candidates are
    returned in original RRF order (no crash, graceful degradation)."""
    candidate_a = SectionCandidate(
        page_id="pg-auth", section_heading="Provider", score=0.92, source="dense"
    )
    candidate_b = SectionCandidate(
        page_id="pg-soc2", section_heading="Audit Schedule", score=0.88, source="dense"
    )
    retrieval = _make_retrieval_result(candidate_a, candidate_b)
    intent = _make_intent()

    with patch(
        "confluence_logic.pipeline.stages.rerank._llm_listwise_rerank",
        new=AsyncMock(side_effect=Exception("LLM reranker unavailable")),
    ):
        reranked = await rerank_candidates(intent, retrieval, top_k=10)

    # Must not raise; must return original RRF order.
    assert len(reranked) == 2
    assert reranked[0].page_id == "pg-auth", (
        "Fallback must preserve RRF order: pg-auth was ranked first"
    )


# ---------------------------------------------------------------------------
# RETR-V3-03-4: Reranker must not introduce new page_ids
# ---------------------------------------------------------------------------

async def test_reranker_does_not_introduce_new_candidates():
    """RETR-V3-03: reranker may only reorder existing candidates, never add new ones."""
    candidates = [
        SectionCandidate(page_id="pg-auth", section_heading="Provider", score=0.9, source="dense"),
        SectionCandidate(page_id="pg-soc2", section_heading="Audit Schedule", score=0.8, source="lexical"),
    ]
    retrieval = _make_retrieval_result(*candidates)
    intent = _make_intent()

    # LLM attempts to introduce an unseen id.
    with patch(
        "confluence_logic.pipeline.stages.rerank._llm_listwise_rerank",
        new=AsyncMock(return_value=[
            "pg-auth::Provider",
            "pg-hallucinated::NewSection",  # not in input
            "pg-soc2::Audit Schedule",
        ]),
    ):
        reranked = await rerank_candidates(intent, retrieval, top_k=10)

    reranked_ids = {c.page_id for c in reranked}
    assert "pg-hallucinated" not in reranked_ids, (
        "Reranker must not introduce page_ids absent from the input candidates"
    )
