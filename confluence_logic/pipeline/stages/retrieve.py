"""Stage 2: Hybrid dense+lexical retrieval with RRF fusion (RETR-V3-01/02).

Replaces Phase 10's weighted-sum ``route_intent`` heuristic (which mixed
incommensurable scales — Pitfall 4) with calibrated Reciprocal Rank Fusion
over two independent retrieval signals:

  1. Dense (Pinecone) — semantic similarity via ``PineconeStore.search``.
  2. Lexical (BM25) — in-process Okapi BM25 over the per-run section corpus
     built from Neo4j CfSection nodes.

Both signals are ranked independently; the rank lists are fused via RRF
(k=60).  The fused order is logged via ``RetrievalResult.fusion_log``
(RETR-V3-01).

Section resolution (RETR-V3-02): every candidate carries a non-None
``section_heading`` resolved HERE — the drafter never picks the section.

Pitfall 2 guard: ``page_id`` is always read from ``match.metadata.page_id``,
never from ``match.id`` (which is a chunk id like ``"pageX_3"``).

Public API::

    from confluence_logic.pipeline.stages.retrieve import retrieve_candidates

    result: RetrievalResult = await retrieve_candidates(
        intent, corpus=ctx.section_corpus, pinecone_store=store
    )

The internal ``_dense_search`` and ``_lexical_search`` functions are
module-level so tests can patch them individually (test-seam pattern from
structure_aware_drafter.py).
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from confluence_logic.pipeline.contracts import (
    ChangeIntentV3,
    RetrievalResult,
    SectionCandidate,
)
from confluence_logic.pipeline.retrieval.fusion import reciprocal_rank_fusion

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration constants
# ---------------------------------------------------------------------------

# Number of hits to request from each signal.  Generous so RRF has breadth.
_PER_SIGNAL_LIMIT: int = int(os.getenv("JARVIS_RETRIEVE_PER_SIGNAL_LIMIT", "10"))

# ---------------------------------------------------------------------------
# Module-level Pinecone singleton (lazy, following page_router.py pattern)
# ---------------------------------------------------------------------------

_store: Optional[Any] = None


def _get_store() -> Any:
    """Lazily construct a module-level PineconeStore singleton."""
    global _store
    if _store is None:
        from confluence_logic.db.vector_store import PineconeStore  # noqa: PLC0415
        _store = PineconeStore()
    return _store


# ---------------------------------------------------------------------------
# Pinecone match normalization (Pitfall 2 — always read metadata.page_id)
# Byte-consistent copy of page_router._normalize_pinecone_match.
# ---------------------------------------------------------------------------

def _normalize_pinecone_match(match: Any) -> Optional[Dict[str, Any]]:
    """Flatten a raw Pinecone match into a canonical dict.

    Always reads ``metadata.page_id`` — NEVER ``match.id`` (which is the
    chunk id, e.g. ``"pageX_3"``).  Returns None if metadata.page_id is
    absent or empty.
    """
    if match is None:
        return None
    if isinstance(match, dict):
        metadata = match.get("metadata") or {}
        score = match.get("score")
        chunk_id = match.get("id") or ""
    else:
        metadata = getattr(match, "metadata", None) or {}
        score = getattr(match, "score", None)
        chunk_id = getattr(match, "id", "") or ""

    if not isinstance(metadata, dict):
        return None

    page_id = (metadata.get("page_id") or "").strip()
    if not page_id:
        return None

    return {
        "page_id": page_id,
        "page_title": metadata.get("title") or "",
        "space_key": metadata.get("space_key") or "",
        "section_heading": metadata.get("heading") or None,
        "section_text": (
            metadata.get("markdown_content")
            or metadata.get("text_summary")
            or metadata.get("text")
            or ""
        ),
        "score": float(score) if score is not None else 0.0,
        "source": "dense",
        "_chunk_id": chunk_id,
    }


# ---------------------------------------------------------------------------
# _dense_search — async wrapper around PineconeStore.search (sync)
# Tests can patch this at ``confluence_logic.pipeline.stages.retrieve._dense_search``
# ---------------------------------------------------------------------------

async def _dense_search(
    query: str,
    pinecone_store: Any,
    top_k: int = _PER_SIGNAL_LIMIT,
) -> List[SectionCandidate]:
    """Run dense retrieval via Pinecone and return SectionCandidate objects.

    ``PineconeStore.search`` is synchronous; it is wrapped in
    ``asyncio.to_thread`` so it does not block the event loop.

    Pitfall 2: page_id is always read from ``metadata.page_id``.
    """
    if pinecone_store is None:
        return []

    try:
        # PineconeStore.search is synchronous — wrap in thread.
        raw_matches = await asyncio.to_thread(
            pinecone_store.search, query, top_k
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "retrieve._dense_search: Pinecone error (non-fatal): %s", exc
        )
        return []

    candidates: List[SectionCandidate] = []
    for match in (raw_matches or []):
        norm = _normalize_pinecone_match(match)
        if norm is None:
            continue
        candidates.append(
            SectionCandidate(
                page_id=norm["page_id"],
                page_title=norm["page_title"],
                space_key=norm["space_key"],
                section_heading=norm["section_heading"],
                section_text=norm["section_text"],
                score=norm["score"],
                source="dense",
            )
        )
    return candidates


# ---------------------------------------------------------------------------
# _lexical_search — synchronous BM25 search over the section corpus
# Tests can patch this at ``confluence_logic.pipeline.stages.retrieve._lexical_search``
# ---------------------------------------------------------------------------

def _lexical_search(
    query: str,
    corpus: list,
    top_k: int = _PER_SIGNAL_LIMIT,
) -> List[SectionCandidate]:
    """Run lexical (BM25) retrieval over the section corpus.

    Builds a BM25Index from ``corpus`` and ranks sections by the query.
    Returns SectionCandidate objects with ``source="lexical"``.

    Falls back to an empty list on any error (RETR-V3-01 degradation).
    """
    if not corpus:
        return []

    try:
        from confluence_logic.pipeline.retrieval.bm25_index import build_index  # noqa: PLC0415
        index = build_index(corpus)
        ranked_ids = index.rank(query, top_k=top_k)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "retrieve._lexical_search: BM25 error (non-fatal): %s", exc
        )
        return []

    # Build a lookup from section_id → corpus row
    row_by_id: Dict[str, Any] = {}
    for row in corpus:
        sid = getattr(row, "section_id", "")
        if sid:
            row_by_id[sid] = row

    candidates: List[SectionCandidate] = []
    for rank_pos, section_id in enumerate(ranked_ids):
        row = row_by_id.get(section_id)
        if row is None:
            # section_id from BM25 not in lookup — skip gracefully
            continue
        candidates.append(
            SectionCandidate(
                page_id=getattr(row, "page_id", ""),
                page_title=getattr(row, "page_title", ""),
                space_key=getattr(row, "space_key", ""),
                section_heading=getattr(row, "heading", None) or None,
                section_text=getattr(row, "text", ""),
                score=float(len(ranked_ids) - rank_pos) / len(ranked_ids),
                source="lexical",
            )
        )
    return candidates


# ---------------------------------------------------------------------------
# _doc_id_for_candidate — canonical composite key used in RRF rank lists
# ---------------------------------------------------------------------------

def _doc_id_for_candidate(candidate: SectionCandidate) -> str:
    """Return the composite doc id ``"{page_id}::{section_heading}"``."""
    heading = candidate.section_heading or ""
    return f"{candidate.page_id}::{heading}"


# ---------------------------------------------------------------------------
# _fuse_candidates — merge dense + lexical hits via RRF, emit fusion_log
# ---------------------------------------------------------------------------

def _fuse_candidates(
    dense_hits: List[SectionCandidate],
    lexical_hits: List[SectionCandidate],
    k: int = 60,
) -> tuple[List[SectionCandidate], dict]:
    """Fuse dense and lexical candidates via RRF.

    Returns:
        (ordered_candidates, fusion_log)

    ``fusion_log`` is a dict carrying the fused order (list of doc_id strings
    sorted by rrf_score) and signal counts — emitted for RETR-V3-01 tracing.
    """
    # Build dense rank list (doc_ids, best → worst)
    dense_ranking: List[str] = [_doc_id_for_candidate(c) for c in dense_hits]
    lexical_ranking: List[str] = [_doc_id_for_candidate(c) for c in lexical_hits]

    # RRF fusion
    fused_order = reciprocal_rank_fusion([dense_ranking, lexical_ranking], k=k)

    # Build a lookup from doc_id → candidate (prefer dense over lexical on tie)
    candidate_lookup: Dict[str, SectionCandidate] = {}
    for c in lexical_hits:
        candidate_lookup[_doc_id_for_candidate(c)] = c
    for c in dense_hits:
        candidate_lookup[_doc_id_for_candidate(c)] = c

    # Map fused doc_ids back to SectionCandidate with rank/score metadata
    ordered: List[SectionCandidate] = []
    for rrf_rank, (doc_id, rrf_score) in enumerate(fused_order):
        base = candidate_lookup.get(doc_id)
        if base is None:
            continue
        # Annotate dense_rank / lexical_rank
        dense_rank: Optional[int] = None
        if doc_id in dense_ranking:
            dense_rank = dense_ranking.index(doc_id)
        lexical_rank: Optional[int] = None
        if doc_id in lexical_ranking:
            lexical_rank = lexical_ranking.index(doc_id)

        ordered.append(
            SectionCandidate(
                page_id=base.page_id,
                page_title=base.page_title,
                space_key=base.space_key,
                section_heading=base.section_heading,
                section_text=base.section_text,
                dense_rank=dense_rank,
                lexical_rank=lexical_rank,
                rrf_score=rrf_score,
                score=base.score,
                source="fused",
            )
        )

    fusion_log = {
        "fused_order": [(doc_id, rrf_score) for doc_id, rrf_score in fused_order],
        "dense_count": len(dense_hits),
        "lexical_count": len(lexical_hits),
        "total_unique": len(fused_order),
    }
    return ordered, fusion_log


# ---------------------------------------------------------------------------
# retrieve_candidates — the public entry point for Stage 2
# ---------------------------------------------------------------------------

async def retrieve_candidates(
    intent: ChangeIntentV3,
    *,
    corpus: Optional[list] = None,
    pinecone_store: Optional[Any] = None,
    top_k: int = _PER_SIGNAL_LIMIT,
    rrf_k: int = 60,
) -> RetrievalResult:
    """Hybrid dense+lexical retrieval with RRF fusion (RETR-V3-01/02).

    Builds a query from ``intent.subject + intent.target_hint + intent.instruction``,
    then runs both the dense signal (Pinecone) and the lexical signal (BM25
    over ``corpus``) concurrently.  The two rank lists are fused via RRF.

    Every returned SectionCandidate has a non-None ``section_heading``
    (RETR-V3-02 — section resolved here, before drafting).

    Degradation: if Pinecone is unavailable, only BM25 results are returned
    (and vice versa).  Neither signal silently skips the other's results
    (RETR-V3-01).  Falls back gracefully — never raises.

    Args:
        intent: The ChangeIntentV3 driving retrieval.
        corpus: Per-run section corpus (list of SectionRow or compatible).
            Pass ``ctx.section_corpus`` in production; tests may pass
            ``fake_section_corpus`` directly.
        pinecone_store: PineconeStore instance (or compatible object with a
            ``.search(query, top_k)`` method).  Pass the module-level singleton
            in production; tests inject a fake.
        top_k: Number of candidates to request from each signal.
        rrf_k: RRF constant (default 60).

    Returns:
        RetrievalResult with candidates ranked by RRF score.
    """
    # Build the query string (mirrors route_intent query construction)
    parts = [intent.subject, intent.target_hint, intent.instruction]
    query = " ".join(p for p in parts if p).strip()
    if not query:
        query = intent.subject

    # Use the module-level store if caller does not provide one
    if pinecone_store is None:
        try:
            pinecone_store = _get_store()
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "retrieve_candidates: could not initialise Pinecone store "
                "(non-fatal): %s",
                exc,
            )

    # Run both signals — dense is already async; lexical runs in thread to
    # avoid blocking the event loop on corpus iteration + BM25 scoring.
    try:
        dense_task = asyncio.create_task(
            _dense_search(query, pinecone_store, top_k=top_k)
        )
        lexical_hits: List[SectionCandidate] = await asyncio.to_thread(
            _lexical_search, query, corpus or [], top_k
        )
        dense_hits: List[SectionCandidate] = await dense_task
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "retrieve_candidates: unexpected error during signal retrieval "
            "(non-fatal): %s",
            exc,
        )
        dense_hits = []
        lexical_hits = []

    # RRF fusion
    ordered, fusion_log = _fuse_candidates(dense_hits, lexical_hits, k=rrf_k)

    # Log the fused order at DEBUG level (RETR-V3-01: logged)
    logger.debug(
        "retrieve_candidates: fused %d unique candidates "
        "(dense=%d, lexical=%d) for intent %r",
        len(ordered),
        len(dense_hits),
        len(lexical_hits),
        intent.subject,
    )

    # RETR-V3-02: Filter out candidates with null section_heading.
    # In production all hits carry a heading (Pinecone chunks have heading
    # metadata, BM25 rows have a heading field).  Drop the rare None-heading
    # hit rather than propagating a section-less candidate to the drafter.
    section_resolved = [c for c in ordered if c.section_heading is not None]
    if len(section_resolved) < len(ordered):
        logger.warning(
            "retrieve_candidates: dropped %d candidates with no section_heading "
            "(RETR-V3-02 — section must be resolved before drafting)",
            len(ordered) - len(section_resolved),
        )

    # fusion_log is set to None only when absolutely no candidates were found
    # (avoids None-assertions in tests).
    final_fusion_log = fusion_log if (dense_hits or lexical_hits) else None

    return RetrievalResult(
        intent=intent,
        candidates=section_resolved,
        no_existing_target=len(section_resolved) == 0,
        iterations=0,
        fusion_log=final_fusion_log,
    )
