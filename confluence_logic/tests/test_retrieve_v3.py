"""RED tests for hybrid retrieval stage (RETR-V3-01, RETR-V3-02 — Phase 11).

RETR-V3-01: Dense (Pinecone) + lexical (BM25) signals must both be present
in hybrid retrieval; fusion via RRF is deterministic and logged; neither
signal silently skipped.

RETR-V3-02: Candidates resolve to (page_id, section_heading) — not just
page_id — before drafting begins.

These tests import from ``confluence_logic.pipeline.stages.retrieve`` and
``confluence_logic.pipeline.retrieval.fusion`` which do not exist yet.
Pytest collection fails with ImportError — expected RED state for Wave 0.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

pytestmark = pytest.mark.asyncio

# ---------------------------------------------------------------------------
# Imports from not-yet-built pipeline targets (RED: ImportError at collection)
# ---------------------------------------------------------------------------
from confluence_logic.pipeline.stages.retrieve import retrieve_candidates
from confluence_logic.pipeline.retrieval.fusion import reciprocal_rank_fusion
from confluence_logic.pipeline.contracts import (
    ChangeIntentV3,
    EvidenceSpan,
    SectionCandidate,
    RetrievalResult,
)


# ---------------------------------------------------------------------------
# RETR-V3-01-1: Both dense and lexical signals present in ranked candidates
# ---------------------------------------------------------------------------

async def test_both_dense_and_lexical_signals_present(
    fake_section_corpus, fake_pinecone_store
):
    """RETR-V3-01: retrieve_candidates must use both Pinecone dense and BM25
    lexical signals and combine them via RRF — not skip either."""
    intent = ChangeIntentV3(
        kind="fact_update",
        subject="SOC2 audit schedule",
        old_value="Q3",
        new_value="Q2",
        dedup_key="soc2-audit-q3-q2",
        evidence=[EvidenceSpan(text="audit moved to Q2", start=0, end=17)],
    )

    with patch(
        "confluence_logic.pipeline.stages.retrieve._dense_search",
        new=AsyncMock(return_value=[
            SectionCandidate(
                page_id="pg-soc2",
                section_heading="Audit Schedule",
                score=0.92,
                source="dense",
            )
        ]),
    ), patch(
        "confluence_logic.pipeline.stages.retrieve._lexical_search",
        return_value=[
            SectionCandidate(
                page_id="pg-soc2",
                section_heading="Audit Schedule",
                score=0.85,
                source="lexical",
            )
        ],
    ) as mock_lex:
        result: RetrievalResult = await retrieve_candidates(
            intent, corpus=fake_section_corpus, pinecone_store=fake_pinecone_store
        )

    # Both signals must have been exercised.
    mock_lex.assert_called_once(), "lexical BM25 search must be called (RETR-V3-01)"
    assert result is not None
    assert len(result.candidates) > 0, "fusion result must have candidates"
    # Fusion must be logged/traced.
    assert result.fusion_log is not None, (
        "RETR-V3-01: fusion must emit a log/trace entry"
    )


async def test_neither_signal_silently_skipped_when_one_returns_empty(
    fake_section_corpus, fake_pinecone_store
):
    """RETR-V3-01: if dense returns empty, BM25 results must still be included."""
    intent = ChangeIntentV3(
        kind="fact_update",
        subject="connection pooling",
        old_value="default",
        new_value="PgBouncer",
        dedup_key="db-connection-pool",
        evidence=[EvidenceSpan(text="use PgBouncer", start=0, end=13)],
    )

    with patch(
        "confluence_logic.pipeline.stages.retrieve._dense_search",
        new=AsyncMock(return_value=[]),  # dense returns nothing
    ), patch(
        "confluence_logic.pipeline.stages.retrieve._lexical_search",
        return_value=[
            SectionCandidate(
                page_id="pg-db",
                section_heading="Connection Pooling",
                score=0.77,
                source="lexical",
            )
        ],
    ):
        result = await retrieve_candidates(
            intent, corpus=fake_section_corpus, pinecone_store=fake_pinecone_store
        )

    page_ids = [c.page_id for c in result.candidates]
    assert "pg-db" in page_ids, (
        "BM25 result must appear when dense is empty (RETR-V3-01 — no silent skip)"
    )


# ---------------------------------------------------------------------------
# RETR-V3-02: Candidates resolve to (page_id, section_heading)
# ---------------------------------------------------------------------------

async def test_candidates_resolve_to_section_heading(
    fake_section_corpus, fake_pinecone_store
):
    """RETR-V3-02: each candidate in RetrievalResult must have a non-None
    section_heading — page-level-only targeting is forbidden."""
    intent = ChangeIntentV3(
        kind="fact_update",
        subject="authentication provider",
        old_value="OAuth",
        new_value="SAML",
        dedup_key="auth-provider",
        evidence=[EvidenceSpan(text="switching to SAML", start=0, end=17)],
    )

    with patch(
        "confluence_logic.pipeline.stages.retrieve._dense_search",
        new=AsyncMock(return_value=[
            SectionCandidate(
                page_id="pg-auth",
                section_heading="Provider",
                score=0.91,
                source="dense",
            )
        ]),
    ), patch(
        "confluence_logic.pipeline.stages.retrieve._lexical_search",
        return_value=[
            SectionCandidate(
                page_id="pg-auth",
                section_heading="Provider",
                score=0.88,
                source="lexical",
            )
        ],
    ):
        result = await retrieve_candidates(
            intent, corpus=fake_section_corpus, pinecone_store=fake_pinecone_store
        )

    for candidate in result.candidates:
        assert candidate.section_heading is not None, (
            f"Candidate for page {candidate.page_id!r} has no section_heading — "
            "RETR-V3-02 requires section-level resolution"
        )
        assert candidate.page_id is not None


# ---------------------------------------------------------------------------
# RETR-V3-01-3: RRF fusion is deterministic (same input → same rank order)
# ---------------------------------------------------------------------------

def test_rrf_fusion_is_deterministic():
    """RETR-V3-01: reciprocal_rank_fusion must produce stable output for stable input."""
    rank_lists = [
        ["pg-auth::Provider", "pg-soc2::Audit Schedule", "pg-db::Connection Pooling"],
        ["pg-soc2::Audit Schedule", "pg-auth::Provider", "pg-infra::Regions"],
    ]
    result_a = reciprocal_rank_fusion(rank_lists, k=60)
    result_b = reciprocal_rank_fusion(rank_lists, k=60)
    assert result_a == result_b, "RRF must be deterministic"


def test_rrf_fusion_top_candidate_appears_in_both_lists():
    """RETR-V3-01: item ranked high in both lists should score highest in RRF."""
    rank_lists = [
        ["pg-auth::Provider", "pg-soc2::Audit Schedule"],
        ["pg-auth::Provider", "pg-db::Connection Pooling"],
    ]
    fused = reciprocal_rank_fusion(rank_lists, k=60)
    top_id, top_score = fused[0]
    assert top_id == "pg-auth::Provider", (
        f"Item ranked first in both lists should be RRF top-1, got {top_id}"
    )


def test_rrf_handles_empty_rank_list():
    """RETR-V3-01: RRF must not crash on an empty rank list."""
    result = reciprocal_rank_fusion([[], ["pg-auth::Provider"]], k=60)
    assert len(result) == 1
    assert result[0][0] == "pg-auth::Provider"


# ---------------------------------------------------------------------------
# RETR-V3-01-4: Pinecone match.id is not used as page_id (Pitfall 2)
# ---------------------------------------------------------------------------

async def test_pinecone_page_id_from_metadata_not_match_id(
    fake_section_corpus, fake_pinecone_store
):
    """RETR-V3-01: dense search must extract page_id from metadata.page_id,
    never from the vector match id (which is a chunk id)."""
    intent = ChangeIntentV3(
        kind="fact_update",
        subject="auth provider",
        old_value="OAuth",
        new_value="SAML",
        dedup_key="auth-pitfall2",
        evidence=[EvidenceSpan(text="switch to SAML", start=0, end=14)],
    )
    # fake_pinecone_store returns matches with id="chunk-001" but
    # metadata.page_id="pg-auth" — the retrieve stage must use metadata.page_id.
    result = await retrieve_candidates(
        intent, corpus=fake_section_corpus, pinecone_store=fake_pinecone_store
    )
    for candidate in result.candidates:
        assert not candidate.page_id.startswith("chunk-"), (
            f"page_id '{candidate.page_id}' looks like a chunk id — "
            "retrieve stage must read metadata.page_id (RETR-V3-01 Pitfall-2)"
        )
