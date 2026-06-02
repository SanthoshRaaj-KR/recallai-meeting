"""Hybrid BM25 + FAISS retriever with RRF fusion and optional cross-encoder reranking.

Uses Reciprocal Rank Fusion (RRF, k=60) to combine BM25 and dense (FAISS)
rankings.  When ``sentence-transformers`` is installed and ``rerank=True``,
results are re-scored with ``cross-encoder/ms-marco-MiniLM-L-6-v2`` before
returning the final top-k.

Graceful degradation:
  - sentence-transformers absent → reranking silently skipped
  - FAISS index absent (use_embeddings=False) → BM25-only fusion
  - OpenAI client absent → zero-vector dense query (no real dense results)
"""

from __future__ import annotations

import logging
import re
from typing import Any, Optional

import numpy as np

from models.rag import ChunkRecord, RetrievalResult
from rag.indexer import (  # reuse shared tokenizer + sync-client helper
    DocumentIndex,
    _as_sync_embeddings_client,
    _tokenize,
)

logger = logging.getLogger(__name__)

_EMBED_DIM = 1536


def _get_sync_embeddings_client(attached):
    """Resolve a synchronous embeddings client from the attached client or env.

    The query path runs synchronously; an AsyncOpenAI client would return an
    un-awaited coroutine. Derive a sync client (from the attached client's
    api_key, or OPENAI_API_KEY) so dense retrieval actually works.
    """
    if attached is not None:
        return _as_sync_embeddings_client(attached)
    import os

    import openai

    key = os.getenv("OPENAI_API_KEY")
    return openai.OpenAI(api_key=key) if key else None


# ── Module-level RRF helper ───────────────────────────────────────────────────


def _rrf(
    bm25_ids: list[str],
    dense_ids: list[str],
    k: int = 60,
) -> list[tuple[str, float]]:
    """Reciprocal Rank Fusion of two ranked lists.

    Parameters
    ----------
    bm25_ids:
        Chunk IDs in BM25 rank order (best first).
    dense_ids:
        Chunk IDs in dense-retrieval rank order (best first).
    k:
        RRF smoothing constant (default 60, per RESEARCH.md).

    Returns
    -------
    list[tuple[str, float]]
        (chunk_id, rrf_score) pairs sorted by rrf_score descending.
    """
    scores: dict[str, float] = {}
    for rank, cid in enumerate(bm25_ids):
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
    for rank, cid in enumerate(dense_ids):
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


# ── Public class ──────────────────────────────────────────────────────────────


class HybridRetriever:
    """BM25 + FAISS hybrid retriever with optional cross-encoder reranking.

    Parameters
    ----------
    index:
        A :class:`rag.indexer.DocumentIndex` built by :func:`rag.indexer.build_index`.
    rerank:
        Whether to apply cross-encoder reranking on the RRF candidate set.
        Requires ``sentence-transformers`` to be installed; silently disabled
        when the package is absent.
    """

    def __init__(self, index: DocumentIndex, rerank: bool = True) -> None:
        self._index = index
        self.rerank = rerank
        self._reranker: Any = None  # lazy-loaded on first rerank call
        self._reranker_loaded = False

        # Try to import cross-encoder; disable reranking if unavailable
        try:
            import sentence_transformers  # noqa: F401
            self._sentence_transformers_available = True
        except ImportError:
            logger.warning(
                "sentence-transformers not installed; reranking disabled"
            )
            self._sentence_transformers_available = False
            self.rerank = False

    # ── Public query method ───────────────────────────────────────────────────

    def query(
        self,
        text: str,
        top_k: int = 3,
        bm25_only: bool = False,
        dense_only: bool = False,
    ) -> list[RetrievalResult]:
        """Retrieve the top *top_k* chunks for *text*.

        Parameters
        ----------
        text:
            Natural-language query string.
        top_k:
            Number of results to return.
        bm25_only:
            Use BM25 scores only (skip FAISS even if available).
        dense_only:
            Use FAISS scores only (skip BM25).

        Returns
        -------
        list[RetrievalResult]
            Ranked results ordered by rrf_score descending (or rerank_score
            when cross-encoder reranking is active).
        """
        chunks = self._index.chunks
        if not chunks:
            return []

        # When dense_only=True but FAISS index is unavailable, fall back to BM25.
        # This prevents silent empty results when no OpenAI client was provided
        # at index-build time (graceful degradation without credentials).
        has_dense = (self._index.faiss_index is not None) or (
            self._index.vector_db == "pinecone" and self._index.pinecone_namespace
        )
        effective_dense_only = dense_only and has_dense
        effective_bm25_only = bm25_only or (dense_only and not has_dense)

        # ── Step 1: BM25 top-20 ───────────────────────────────────────────────
        bm25_ids: list[str] = []
        bm25_rank_map: dict[str, int] = {}
        if not effective_dense_only:
            scores = self._index.bm25.get_scores(_tokenize(text))
            top_indices = np.argsort(scores)[::-1][:20]
            bm25_ids = [self._index.chunk_ids[i] for i in top_indices]
            bm25_rank_map = {cid: rank for rank, cid in enumerate(bm25_ids)}

        # ── Step 2: dense top-20 (FAISS in-process or Pinecone) ──────────────
        dense_ids: list[str] = []
        dense_rank_map: dict[str, int] = {}
        if not effective_bm25_only and has_dense:
            q_vec = self._embed_query(text)
            if self._index.vector_db == "pinecone":
                from rag import vector_store

                ids = vector_store.query(
                    self._index.pinecone_namespace, q_vec[0], min(20, len(chunks))
                )
                dense_ids = [cid for cid in ids if cid in self._index.id_to_chunk]
            else:
                D, I = self._index.faiss_index.search(q_vec, min(20, len(chunks)))
                dense_ids = [
                    self._index.chunk_ids[idx]
                    for idx in I[0]
                    if idx >= 0 and idx < len(self._index.chunk_ids)
                ]
            dense_rank_map = {cid: rank for rank, cid in enumerate(dense_ids)}

        # ── Step 3: RRF fusion ────────────────────────────────────────────────
        fused = _rrf(bm25_ids, dense_ids, k=60)
        candidates_10 = fused[:10]  # top-10 for reranking

        # ── Step 4: Optional cross-encoder reranking ──────────────────────────
        if self.rerank and self._sentence_transformers_available and len(candidates_10) >= 2:
            try:
                reranked = self._rerank(text, candidates_10)
                # Build RetrievalResult from reranked list
                results: list[RetrievalResult] = []
                for final_rank, (cid, rrf_score, rerank_score) in enumerate(
                    reranked[:top_k]
                ):
                    chunk = self._index.id_to_chunk.get(cid)
                    if chunk is None:
                        continue
                    results.append(
                        RetrievalResult(
                            chunk=chunk,
                            bm25_rank=bm25_rank_map.get(cid),
                            dense_rank=dense_rank_map.get(cid),
                            rrf_score=rrf_score,
                            rerank_score=rerank_score,
                            final_rank=final_rank + 1,
                        )
                    )
                return results
            except Exception as exc:
                logger.warning("Reranking failed: %s; falling back to RRF.", exc)

        # ── Step 5: Top-k from RRF (no reranking) ─────────────────────────────
        results = []
        for final_rank, (cid, rrf_score) in enumerate(candidates_10[:top_k]):
            chunk = self._index.id_to_chunk.get(cid)
            if chunk is None:
                continue
            results.append(
                RetrievalResult(
                    chunk=chunk,
                    bm25_rank=bm25_rank_map.get(cid),
                    dense_rank=dense_rank_map.get(cid),
                    rrf_score=rrf_score,
                    rerank_score=None,
                    final_rank=final_rank + 1,
                )
            )
        return results

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _embed_query(self, text: str) -> np.ndarray:
        """Embed a query string into a normalised FAISS-ready numpy array.

        Falls back to a zero vector when no OpenAI client is attached.
        """
        client = _get_sync_embeddings_client(
            getattr(self._index, "_openai_client", None)
        )
        if client is None:
            return np.zeros((1, _EMBED_DIM), dtype=np.float32)
        try:
            response = client.embeddings.create(
                input=[text],
                model="text-embedding-3-small",
            )
            vec = np.array([response.data[0].embedding], dtype=np.float32)
            import faiss as _faiss

            _faiss.normalize_L2(vec)
            return vec
        except Exception as exc:
            logger.warning("Query embedding failed: %s; using zero vector.", exc)
            return np.zeros((1, _EMBED_DIM), dtype=np.float32)

    def _rerank(
        self,
        query: str,
        candidates: list[tuple[str, float]],
    ) -> list[tuple[str, float, float]]:
        """Re-score candidates with a cross-encoder; sort by rerank_score desc."""
        # Lazy load the cross-encoder on first call
        if not self._reranker_loaded:
            from sentence_transformers import CrossEncoder  # type: ignore[import]

            self._reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
            self._reranker_loaded = True

        pairs = [
            (query, self._index.id_to_chunk[cid].content)
            for cid, _ in candidates
            if cid in self._index.id_to_chunk
        ]
        scores = self._reranker.predict(pairs)
        triples = [
            (cid, rrf_score, float(score))
            for (cid, rrf_score), score in zip(candidates, scores)
            if cid in self._index.id_to_chunk
        ]
        return sorted(triples, key=lambda x: x[2], reverse=True)
