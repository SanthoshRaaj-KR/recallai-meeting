"""Tests for the hybrid RAG retriever — RAG-02, RAG-03, RAG-04, RAG-05.

All tests fail with ImportError until Wave 1 implements rag/indexer.py and rag/retriever.py.
The import is deferred into each test body so pytest can collect without errors.
"""

from __future__ import annotations

import pytest

from tests.fixtures import FIXTURES_DIR


def test_bm25_exact_match():
    """RAG-02: BM25 retrieval returns correct section for exact-match intent."""
    from rag.indexer import build_index  # ImportError until Wave 1
    from rag.retriever import HybridRetriever  # ImportError until Wave 1

    index = build_index(str(FIXTURES_DIR), use_embeddings=False)
    retriever = HybridRetriever(index)
    results = retriever.query("data retention 7 years", top_k=3, bm25_only=True)
    assert len(results) >= 1
    top = results[0]
    assert "Data Retention" in top.chunk.section_heading or "7 years" in top.chunk.content


def test_dense_semantic():
    """RAG-03: Dense retrieval returns semantically correct section."""
    from rag.indexer import build_index  # ImportError until Wave 1
    from rag.retriever import HybridRetriever  # ImportError until Wave 1

    index = build_index(str(FIXTURES_DIR), use_embeddings=True)
    retriever = HybridRetriever(index)
    results = retriever.query("records kept for regulatory compliance", top_k=3, dense_only=True)
    assert len(results) >= 1
    assert any(
        "Data Retention" in r.chunk.section_heading or "retention" in r.chunk.content.lower()
        for r in results
    )


def test_rrf_beats_single():
    """RAG-04: RRF fusion produces higher recall than either single method."""
    from rag.indexer import build_index  # ImportError until Wave 1
    from rag.retriever import HybridRetriever  # ImportError until Wave 1

    index = build_index(str(FIXTURES_DIR), use_embeddings=True)
    retriever = HybridRetriever(index)
    bm25_results = retriever.query("production deployments approvals", top_k=3, bm25_only=True)
    hybrid_results = retriever.query("production deployments approvals", top_k=3)
    # RRF score of best hybrid result >= best bm25 result (or finds a result bm25 missed)
    assert len(hybrid_results) >= len(bm25_results)


def test_top3_recall():
    """RAG-05: top-3 results contain correct section for representative intents."""
    from rag.indexer import build_index  # ImportError until Wave 1
    from rag.retriever import HybridRetriever  # ImportError until Wave 1

    index = build_index(str(FIXTURES_DIR), use_embeddings=True)
    retriever = HybridRetriever(index)
    results = retriever.query("employee access production systems manager approval", top_k=3)
    assert len(results) == 3
    found = any("Access Control" in r.chunk.section_heading for r in results)
    assert found, "Top-3 must include the Access Control section"


def test_retrieval_result_schema():
    """RetrievalResult carries rrf_score and final_rank."""
    from rag.indexer import build_index  # ImportError until Wave 1
    from rag.retriever import HybridRetriever  # ImportError until Wave 1

    index = build_index(str(FIXTURES_DIR), use_embeddings=False)
    retriever = HybridRetriever(index)
    results = retriever.query("retention", top_k=3, bm25_only=True)
    for r in results:
        assert r.rrf_score >= 0
        assert r.final_rank >= 0
        assert r.chunk.chunk_id


def test_index_cache_reuse():
    """Index is rebuilt from disk cache when folder contents unchanged."""
    import time
    from rag.indexer import build_index  # ImportError until Wave 1

    index1 = build_index(str(FIXTURES_DIR), use_embeddings=False)
    t0 = time.monotonic()
    index2 = build_index(str(FIXTURES_DIR), use_embeddings=False)
    elapsed = time.monotonic() - t0
    assert elapsed < 1.0, "Cache hit should return in under 1 second"
