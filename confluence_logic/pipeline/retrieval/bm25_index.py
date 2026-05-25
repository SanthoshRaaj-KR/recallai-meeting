"""In-process Okapi BM25 index over the per-run section corpus (RETR-V3-01).

Builds a single BM25Okapi index from the user's section corpus each pipeline
run.  Each section's ``heading + " " + text`` is one document; the doc id is
``section_id`` (typically ``"{page_id}::{heading}"``).

Tokenizer: ``grounding_gate.content_bearing_tokens`` — the shared production
tokenizer (stopwords, identifier/number handling).  Reusing it here means the
lexical signal is consistent with the grounding gate's token vocabulary.

Degradation (RETR-V3-01): if ``rank-bm25`` (import name ``rank_bm25``) is
absent, ``BM25Index.rank`` falls back to the Neo4j term-overlap signal
(list of section ids sorted by term-overlap count) and logs a warning.
This preserves the Phase 10 behaviour and never crashes the pipeline.

Usage::

    from confluence_logic.pipeline.retrieval.bm25_index import build_index

    corpus = [SectionRow(...), ...]   # from corpus.py
    idx = build_index(corpus)
    ranked_ids = idx.rank("SOC2 audit schedule", top_k=10)
    # → ["pg-soc2::Audit Schedule", "pg-security-overview::Audit Schedule", ...]
"""

from __future__ import annotations

import logging
from typing import List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy import of rank_bm25 — degrade gracefully if not installed
# ---------------------------------------------------------------------------

_BM25_AVAILABLE: Optional[bool] = None  # None = not yet checked
_BM25Okapi = None  # type: ignore[assignment]


def _check_bm25() -> bool:
    """Return True if rank_bm25 is importable; log once and cache the result."""
    global _BM25_AVAILABLE, _BM25Okapi
    if _BM25_AVAILABLE is not None:
        return _BM25_AVAILABLE
    try:
        from rank_bm25 import BM25Okapi  # type: ignore[import-untyped]  # noqa: PLC0415
        _BM25Okapi = BM25Okapi
        _BM25_AVAILABLE = True
    except ImportError:
        logger.warning(
            "bm25_index: rank_bm25 not installed — lexical signal degrades to "
            "Neo4j term-overlap (RETR-V3-01 degradation).  "
            "Install: pip install rank-bm25==0.2.2"
        )
        _BM25_AVAILABLE = False
    return _BM25_AVAILABLE


# ---------------------------------------------------------------------------
# Tokenizer (shared with grounding_gate — Pitfall: no duplicated stopword set)
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> List[str]:
    """Tokenize using grounding_gate.content_bearing_tokens.

    Falls back to simple whitespace split if the import fails (e.g. in
    isolated unit tests that stub the corpus module).
    """
    try:
        from confluence_logic.agents.grounding_gate import content_bearing_tokens  # noqa: PLC0415
        return content_bearing_tokens(text)
    except Exception:  # noqa: BLE001
        return [t.lower() for t in text.split() if t]


# ---------------------------------------------------------------------------
# BM25Index
# ---------------------------------------------------------------------------

class BM25Index:
    """In-process BM25Okapi index over a list of section corpus rows.

    Build once per pipeline run via ``build_index(corpus)`` (not per-intent).
    The ``rank`` method returns a ranked list of section ids for a query.

    Attributes:
        _doc_ids: Ordered list of section ids matching the BM25 document order.
        _bm25: BM25Okapi instance (None when rank_bm25 is absent).
        _corpus_rows: Original corpus rows for term-overlap fallback.
    """

    def __init__(
        self,
        doc_ids: List[str],
        bm25_instance: object,  # BM25Okapi | None
        corpus_rows: list,
    ) -> None:
        self._doc_ids = doc_ids
        self._bm25 = bm25_instance
        self._corpus_rows = corpus_rows

    def rank(self, query: str, top_k: int = 10) -> List[str]:
        """Return up to ``top_k`` section ids ordered by BM25 relevance.

        Falls back to Neo4j term-overlap count sort when rank_bm25 is absent.
        Never raises — returns an empty list on unexpected errors.

        Args:
            query: Natural-language query string (typically
                ``subject + target_hint + instruction``).
            top_k: Maximum number of results to return.

        Returns:
            Ordered list of section ids (best match first).
        """
        if not self._doc_ids:
            return []

        try:
            if self._bm25 is not None:
                # BM25Okapi path
                tokens = _tokenize(query)
                scores = self._bm25.get_scores(tokens)
                # Sort doc ids by score descending, take top_k
                ranked = sorted(
                    zip(self._doc_ids, scores),
                    key=lambda x: -x[1],
                )
                return [doc_id for doc_id, score in ranked[:top_k] if score > 0]
            else:
                # Degraded path: Neo4j term-overlap count
                return self._term_overlap_fallback(query, top_k)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "bm25_index.rank: unexpected error (non-fatal): %s", exc
            )
            return []

    def _term_overlap_fallback(self, query: str, top_k: int) -> List[str]:
        """Rank sections by term-overlap count with the query (Phase 10 behaviour)."""
        query_tokens = set(_tokenize(query))
        if not query_tokens:
            return self._doc_ids[:top_k]

        scored: List[tuple] = []
        for row in self._corpus_rows:
            terms = set(getattr(row, "terms", []))
            overlap = len(query_tokens & terms)
            if overlap > 0:
                section_id = getattr(row, "section_id", "")
                scored.append((section_id, overlap))

        scored.sort(key=lambda x: -x[1])
        return [sid for sid, _ in scored[:top_k]]


# ---------------------------------------------------------------------------
# Public factory function
# ---------------------------------------------------------------------------

def build_index(corpus: list) -> "BM25Index":
    """Build a BM25Index from a list of SectionRow (or compatible) objects.

    Args:
        corpus: List of SectionRow namedtuples from ``corpus.py`` (or any
            object with ``section_id``, ``heading``, and ``text`` attributes).

    Returns:
        A BM25Index ready for ``rank(query, top_k)`` calls.  If rank_bm25 is
        not installed, the index uses the term-overlap fallback transparently.
    """
    if not corpus:
        return BM25Index(doc_ids=[], bm25_instance=None, corpus_rows=[])

    doc_ids: List[str] = []
    tokenized_docs: List[List[str]] = []

    for row in corpus:
        section_id = getattr(row, "section_id", "")
        heading = getattr(row, "heading", "") or ""
        text = getattr(row, "text", "") or ""
        combined = f"{heading} {text}".strip()
        tokens = _tokenize(combined)
        doc_ids.append(section_id)
        tokenized_docs.append(tokens)

    if _check_bm25() and _BM25Okapi is not None:
        try:
            bm25 = _BM25Okapi(tokenized_docs)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "bm25_index.build_index: BM25Okapi construction failed "
                "(non-fatal, using fallback): %s",
                exc,
            )
            bm25 = None
    else:
        bm25 = None

    return BM25Index(doc_ids=doc_ids, bm25_instance=bm25, corpus_rows=corpus)
