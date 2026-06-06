"""Regression tests for the FAISS/Pinecone vector-DB toggle (LDOC_VECTOR_DB).

These tests never touch a real Pinecone account — the backend is mocked. They
verify backend selection, graceful fallback, and that the indexer/retriever
route through Pinecone when (and only when) it is enabled.
"""
from __future__ import annotations

import numpy as np
import pytest

from rag import indexer, vector_store
from rag.indexer import DocumentIndex
from rag.retriever import HybridRetriever


def test_default_backend_is_faiss(monkeypatch):
    monkeypatch.delenv("LDOC_VECTOR_DB", raising=False)
    assert vector_store.vector_db_choice() == "faiss"
    assert vector_store.pinecone_enabled() is False


def test_pinecone_disabled_without_key(monkeypatch):
    monkeypatch.setenv("LDOC_VECTOR_DB", "pinecone")
    monkeypatch.delenv("PINECONE_API_KEY", raising=False)
    # No key -> must fall back to FAISS (returns False), never raise.
    assert vector_store.pinecone_enabled() is False


def test_build_index_routes_to_pinecone(monkeypatch, tmp_path):
    # A tiny doc so chunking yields >=1 chunk.
    doc = tmp_path / "policy.txt"
    doc.write_text(
        "SECTION 1. RETENTION\nLogs are kept for 90 days.\n\n"
        "SECTION 2. ACCESS\nAccess needs one manager approval.\n",
        encoding="utf-8",
    )

    captured = {}

    def fake_embed(chunks, client):
        return np.zeros((len(chunks), 1536), dtype=np.float32)

    def fake_upsert(ns, ids, vecs):
        captured["ns"] = ns
        captured["ids"] = list(ids)
        return True

    monkeypatch.setattr(indexer, "_embed_chunks", fake_embed)
    monkeypatch.setattr(vector_store, "pinecone_enabled", lambda: True)
    monkeypatch.setattr(vector_store, "namespace_count", lambda ns: 0)
    monkeypatch.setattr(vector_store, "upsert", fake_upsert)

    idx = indexer.build_index(
        str(tmp_path), use_embeddings=True, contextual_retrieval=False, openai_client=object()
    )

    assert idx.vector_db == "pinecone"
    assert idx.pinecone_namespace == idx.folder_hash
    assert idx.faiss_index is None
    assert captured["ns"] == idx.folder_hash
    assert captured["ids"] == idx.chunk_ids  # all chunks were upserted


def test_retriever_uses_pinecone_query(monkeypatch):
    from models.rag import ChunkRecord
    from rank_bm25 import BM25Okapi

    chunks = [
        ChunkRecord(chunk_id="a:0", source_path="/x/p.txt", source_format="txt",
                    section_heading="Retention", section_index=0,
                    content="Logs are kept for 90 days"),
        ChunkRecord(chunk_id="a:1", source_path="/x/p.txt", source_format="txt",
                    section_heading="Access", section_index=1,
                    content="Access needs one manager approval"),
    ]
    bm25 = BM25Okapi([indexer._tokenize(c.content) for c in chunks])
    index = DocumentIndex(
        chunks=chunks, bm25=bm25, faiss_index=None,
        id_to_chunk={c.chunk_id: c for c in chunks},
        folder_hash="h", chunk_ids=[c.chunk_id for c in chunks],
        vector_db="pinecone", pinecone_namespace="h",
    )

    calls = {}

    def fake_query(ns, vec, k):
        calls["ns"] = ns
        return ["a:1"]  # pretend Pinecone ranks the Access chunk first

    monkeypatch.setattr(vector_store, "query", fake_query)
    monkeypatch.setattr(HybridRetriever, "_embed_query",
                        lambda self, text: np.zeros((1, 1536), dtype=np.float32))

    r = HybridRetriever(index, rerank=False)
    results = r.query("who approves access", top_k=2)

    assert calls.get("ns") == "h"               # Pinecone path was taken
    ids = [res.chunk.chunk_id for res in results]
    assert "a:1" in ids                          # Pinecone-provided chunk surfaced
