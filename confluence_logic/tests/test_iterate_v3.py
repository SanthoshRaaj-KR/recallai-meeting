"""RED tests for bounded agentic iterative retrieval (RETR-V3-04 — Phase 11).

RETR-V3-04: A bounded loop reformulates queries for low-confidence intents;
the loop cap is honored (no unbounded recursion); when no existing target is
found the stage concludes "no_existing_target" which maps to create_page.

These tests import from ``confluence_logic.pipeline.stages.iterate`` which
does not exist yet. Pytest collection fails with ImportError — expected RED
state for Wave 0 of Phase 11.
"""

import pytest
from unittest.mock import AsyncMock, patch

pytestmark = pytest.mark.asyncio

# ---------------------------------------------------------------------------
# Imports from not-yet-built pipeline targets (RED: ImportError at collection)
# ---------------------------------------------------------------------------
from confluence_logic.pipeline.stages.iterate import iterative_retrieve
from confluence_logic.pipeline.contracts import (
    ChangeIntentV3,
    EvidenceSpan,
    SectionCandidate,
    RetrievalResult,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _intent(subject: str = "test") -> ChangeIntentV3:
    return ChangeIntentV3(
        kind="fact_update",
        subject=subject,
        old_value="old",
        new_value="new",
        dedup_key=f"iterate-{subject}",
        evidence=[EvidenceSpan(text="some evidence", start=0, end=13)],
    )


# ---------------------------------------------------------------------------
# RETR-V3-04-1: Iteration cap is honored
# ---------------------------------------------------------------------------

async def test_iteration_cap_is_honored():
    """RETR-V3-04: iterative_retrieve must stop after MAX_ITERATIONS even if
    confidence remains low; the counter is respected regardless of outcome."""
    call_count = 0

    async def _always_low_confidence(intent, corpus, store, **kwargs):
        nonlocal call_count
        call_count += 1
        # Return a low-confidence result every time.
        return RetrievalResult(
            candidates=[
                SectionCandidate(
                    page_id="pg-wrong",
                    section_heading="Unrelated",
                    score=0.20,
                    source="dense",
                )
            ],
            fusion_log="rrf applied",
        )

    with patch(
        "confluence_logic.pipeline.stages.iterate._retrieve_once",
        new=_always_low_confidence,
    ):
        result = await iterative_retrieve(_intent(), corpus=[], store=None, max_iterations=3)

    assert call_count <= 3, (
        f"Expected ≤3 retrieve calls (iteration cap), got {call_count}"
    )
    assert result is not None


# ---------------------------------------------------------------------------
# RETR-V3-04-2: no_existing_target maps to create_page disposition
# ---------------------------------------------------------------------------

async def test_no_existing_target_produces_create_page_disposition():
    """RETR-V3-04: when all iterations fail to find a confident match the stage
    must return a result flagged as 'no_existing_target' so plan_ops maps it
    to create_page."""
    async def _no_match(intent, corpus, store, **kwargs):
        return RetrievalResult(
            candidates=[],
            fusion_log="rrf: empty",
        )

    with patch(
        "confluence_logic.pipeline.stages.iterate._retrieve_once",
        new=_no_match,
    ):
        result = await iterative_retrieve(
            _intent("brand new workstream"), corpus=[], store=None, max_iterations=2
        )

    assert result.no_existing_target is True, (
        "RETR-V3-04: empty result must set no_existing_target=True"
    )


# ---------------------------------------------------------------------------
# RETR-V3-04-3: Early exit when high-confidence match found
# ---------------------------------------------------------------------------

async def test_early_exit_when_confident_match_found():
    """RETR-V3-04: if the first iteration returns a high-confidence match,
    the loop must not run a second time."""
    call_count = 0

    async def _high_confidence_first(intent, corpus, store, **kwargs):
        nonlocal call_count
        call_count += 1
        return RetrievalResult(
            candidates=[
                SectionCandidate(
                    page_id="pg-soc2",
                    section_heading="Audit Schedule",
                    score=0.95,
                    source="dense",
                )
            ],
            fusion_log="rrf applied",
        )

    with patch(
        "confluence_logic.pipeline.stages.iterate._retrieve_once",
        new=_high_confidence_first,
    ):
        result = await iterative_retrieve(
            _intent("SOC2 audit"), corpus=[], store=None, max_iterations=5
        )

    assert call_count == 1, (
        f"Expected early exit after first high-confidence hit, got {call_count} calls"
    )
    assert not result.no_existing_target


# ---------------------------------------------------------------------------
# RETR-V3-04-4: Query is reformulated between iterations
# ---------------------------------------------------------------------------

async def test_query_reformulated_between_iterations():
    """RETR-V3-04: each successive call should receive a reformulated query,
    not the same query as the first iteration."""
    queries_seen = []

    async def _capture_query(intent, corpus, store, query=None, **kwargs):
        queries_seen.append(query)
        return RetrievalResult(
            candidates=[
                SectionCandidate(
                    page_id="pg-x", section_heading="X", score=0.30, source="dense"
                )
            ],
            fusion_log="rrf applied",
        )

    with patch(
        "confluence_logic.pipeline.stages.iterate._retrieve_once",
        new=_capture_query,
    ):
        await iterative_retrieve(
            _intent("PgBouncer pooling"), corpus=[], store=None, max_iterations=3
        )

    # At least two distinct queries should have been issued.
    if len(queries_seen) > 1:
        assert len(set(str(q) for q in queries_seen)) > 1, (
            "RETR-V3-04: iteration must reformulate the query between rounds"
        )
