"""BM25 + FAISS document indexer with disk cache.

FAISS FlatIP is O(N) search; adequate for <=5K chunks (~500 pages). For
larger corpora, replace faiss.IndexFlatIP with
faiss.IndexIVFFlat(quantizer, dim, nlist) where nlist=int(sqrt(N)). This
requires calling index.train(vecs) before index.add(vecs).

Index is persisted to disk under ``local_doc_change/index/<folder_hash>/``
and reloaded on subsequent calls when folder contents are unchanged (same
file-modification-time fingerprint).

Security — T-12-02: only supported extensions are scanned when building the
index.  Unknown file types are silently skipped.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import pickle
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import faiss  # type: ignore[import]
import numpy as np
from rank_bm25 import BM25Okapi

from models.rag import ChunkRecord
from rag import vector_store
from rag.chunker import chunk_document
from rag.contextualizer import add_context_prefixes

logger = logging.getLogger(__name__)

# ── Cache directory (relative to project root; created at module load time) ───
_PROJECT_ROOT = Path(__file__).parent.parent
CACHE_DIR = _PROJECT_ROOT / "index"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# ── Supported document extensions (must match chunker allowlist) ──────────────
_DOC_EXTENSIONS = {".docx", ".odt", ".pdf", ".txt", ".md", ".rtf"}

# ── FAISS embedding dimension (text-embedding-3-small) ───────────────────────
_EMBED_DIM = 1536


# ── Public dataclass ──────────────────────────────────────────────────────────


@dataclass
class DocumentIndex:
    """In-memory RAG index for a folder of documents."""

    chunks: list[ChunkRecord]
    bm25: BM25Okapi
    faiss_index: Optional[Any]  # faiss.IndexFlatIP or None (no embeddings / pinecone)
    id_to_chunk: dict[str, ChunkRecord]
    folder_hash: str
    # Internal: stored alongside index for retrieval
    chunk_ids: list[str] = field(default_factory=list)
    # Dense backend selector: "faiss" (default) or "pinecone"
    vector_db: str = "faiss"
    pinecone_namespace: Optional[str] = None


# ── Public function ───────────────────────────────────────────────────────────


def build_index(
    folder_path: str,
    use_embeddings: bool = True,
    contextual_retrieval: bool = True,
    openai_client=None,
) -> DocumentIndex:
    """Build (or load from cache) a BM25 + FAISS index for a folder.

    Parameters
    ----------
    folder_path:
        Directory containing the documents to index.
    use_embeddings:
        When True and *openai_client* is not None, generate dense embeddings
        via the OpenAI Embeddings API and add them to a FAISS FlatIP index.
        When False, ``faiss_index`` is ``None``.
    contextual_retrieval:
        When True and *openai_client* is not None, prepend an LLM-generated
        context description to each chunk before BM25/embedding (improves
        retrieval quality for short sections).
    openai_client:
        An initialised ``openai.OpenAI`` or ``openai.AsyncOpenAI`` client.
        Required for contextual retrieval and dense embeddings.

    Returns
    -------
    DocumentIndex
        Populated index ready for :class:`rag.retriever.HybridRetriever`.
    """
    folder_hash = _compute_folder_hash(folder_path)
    cache_path = CACHE_DIR / folder_hash

    # ── Fast path: load from disk cache ──────────────────────────────────────
    chunks_file = cache_path / "chunks.json"
    bm25_file = cache_path / "bm25.pkl"
    faiss_file = cache_path / "faiss.bin"

    # Pinecone cache hit: chunks+BM25 on disk AND vectors already in the namespace.
    pinecone_cache_ok = (
        chunks_file.exists()
        and bm25_file.exists()
        and use_embeddings
        and vector_store.pinecone_enabled()
        and vector_store.namespace_count(folder_hash) > 0
    )
    if pinecone_cache_ok:
        try:
            chunks = _load_chunks(chunks_file)
            with open(bm25_file, "rb") as f:
                bm25 = pickle.load(f)
            id_to_chunk = {c.chunk_id: c for c in chunks}
            chunk_ids = [c.chunk_id for c in chunks]
            logger.info("Loaded index from cache %s via Pinecone (%d chunks)", folder_hash, len(chunks))
            return DocumentIndex(
                chunks=chunks, bm25=bm25, faiss_index=None, id_to_chunk=id_to_chunk,
                folder_hash=folder_hash, chunk_ids=chunk_ids,
                vector_db="pinecone", pinecone_namespace=folder_hash,
            )
        except Exception as exc:
            logger.warning("Pinecone cache load failed (%s); rebuilding index.", exc)

    # FAISS cache hit (default backend).
    if chunks_file.exists() and bm25_file.exists() and not vector_store.pinecone_enabled():
        try:
            chunks = _load_chunks(chunks_file)
            with open(bm25_file, "rb") as f:
                bm25 = pickle.load(f)
            faiss_idx: Optional[Any] = None
            if faiss_file.exists():
                faiss_idx = faiss.read_index(str(faiss_file))
            id_to_chunk = {c.chunk_id: c for c in chunks}
            chunk_ids = [c.chunk_id for c in chunks]
            logger.info(
                "Loaded index from cache %s (%d chunks)", folder_hash, len(chunks)
            )
            return DocumentIndex(
                chunks=chunks,
                bm25=bm25,
                faiss_index=faiss_idx,
                id_to_chunk=id_to_chunk,
                folder_hash=folder_hash,
                chunk_ids=chunk_ids,
            )
        except Exception as exc:
            logger.warning("Cache load failed (%s); rebuilding index.", exc)

    # ── Slow path: build from scratch ────────────────────────────────────────
    chunks = _gather_chunks(folder_path)
    if not chunks:
        logger.warning("No supported documents found in %s", folder_path)

    # Optional contextual retrieval (LLM prefix enrichment)
    if contextual_retrieval and openai_client is not None and chunks:
        try:
            chunks = asyncio.run(
                add_context_prefixes(chunks, openai_client)
            )
        except RuntimeError:
            # Already inside an event loop (e.g. pytest-asyncio)
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(
                    asyncio.run, add_context_prefixes(chunks, openai_client)
                )
                chunks = future.result()
        except Exception as exc:
            logger.warning("Contextual retrieval enrichment failed: %s", exc)

    # BM25 index
    tokenized = [_tokenize(c.context_prefix + " " + c.content) for c in chunks]
    bm25 = BM25Okapi(tokenized)

    id_to_chunk = {c.chunk_id: c for c in chunks}
    chunk_ids = [c.chunk_id for c in chunks]

    # Dense index — FAISS (default) or Pinecone (LDOC_VECTOR_DB=pinecone).
    faiss_idx = None
    vector_db = "faiss"
    pinecone_ns: Optional[str] = None
    want_dense = use_embeddings and openai_client is not None and chunks
    if want_dense and vector_store.pinecone_enabled():
        vecs = _embed_chunks(chunks, openai_client)
        if vecs is not None and vector_store.upsert(folder_hash, chunk_ids, vecs):
            vector_db = "pinecone"
            pinecone_ns = folder_hash
        else:
            logger.warning("Pinecone dense build failed; falling back to FAISS for this folder.")
            faiss_idx = _build_faiss_index(chunks, openai_client)
    elif want_dense:
        faiss_idx = _build_faiss_index(chunks, openai_client)

    # Persist to cache (chunks + BM25 always; FAISS only for the FAISS backend —
    # Pinecone vectors live server-side under the folder-hash namespace).
    try:
        cache_path.mkdir(parents=True, exist_ok=True)
        _save_chunks(chunks, chunks_file)
        with open(bm25_file, "wb") as f:
            pickle.dump(bm25, f)
        if faiss_idx is not None:
            faiss.write_index(faiss_idx, str(faiss_file))
    except Exception as exc:
        logger.warning("Failed to persist index cache: %s", exc)

    return DocumentIndex(
        chunks=chunks,
        bm25=bm25,
        faiss_index=faiss_idx,
        id_to_chunk=id_to_chunk,
        folder_hash=folder_hash,
        chunk_ids=chunk_ids,
        vector_db=vector_db,
        pinecone_namespace=pinecone_ns,
    )


# ── Internal helpers ──────────────────────────────────────────────────────────


def _is_backup_file(path: Path) -> bool:
    """True for SafeApply backup files (e.g. policy.backup.20260601T....docx).

    SafeApply writes backups into the same folder as the source document; they
    must NOT be re-indexed as if they were real documents, or accepted changes
    would pollute the corpus and the pipeline could propose edits to backups.
    """
    return ".backup." in path.name


def _iter_doc_files(folder: Path):
    """Yield supported document files in a folder, excluding backups."""
    for ext in sorted(_DOC_EXTENSIONS):
        for fpath in sorted(folder.rglob(f"*{ext}")):
            if _is_backup_file(fpath):
                continue
            yield fpath


def _compute_folder_hash(folder_path: str) -> str:
    """Fingerprint a folder by path + mtime of all supported files."""
    folder = Path(folder_path)
    entries: list[str] = []
    for fpath in _iter_doc_files(folder):
        try:
            mtime = os.path.getmtime(fpath)
            entries.append(f"{fpath!s}:{mtime}")
        except OSError:
            pass
    fingerprint = "\n".join(entries).encode()
    return hashlib.md5(fingerprint).hexdigest()[:16]


def _gather_chunks(folder_path: str) -> list[ChunkRecord]:
    """Chunk all supported documents in *folder_path* (excluding backups)."""
    folder = Path(folder_path)
    chunks: list[ChunkRecord] = []
    for fpath in _iter_doc_files(folder):
        try:
            file_chunks = chunk_document(str(fpath))
            chunks.extend(file_chunks)
        except Exception as exc:
            logger.warning("Failed to chunk %s: %s", fpath, exc)
    return chunks


def _as_sync_embeddings_client(openai_client):
    """Return a synchronous OpenAI client suitable for embeddings.

    Accepts either a sync OpenAI client (returned as-is) or an AsyncOpenAI
    client (a new sync client is built from its api_key). Returns None if no
    API key can be resolved.
    """
    import openai

    if isinstance(openai_client, openai.OpenAI):
        return openai_client
    api_key = getattr(openai_client, "api_key", None) or os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    return openai.OpenAI(api_key=api_key)


def _embed_batch_with_retry(sync_client, batch, offset, max_retries=6):
    """Embed one batch with backoff on rate limits.

    On large corpora the embeddings endpoint hits the per-minute token limit
    (HTTP 429). Zero-filling on the first failure silently destroys dense
    retrieval for those chunks, so retry with exponential backoff first and only
    zero-fill as a last resort.
    """
    import time as _time

    delay = 2.0
    for attempt in range(max_retries):
        try:
            resp = sync_client.embeddings.create(
                input=batch, model="text-embedding-3-small"
            )
            return [item.embedding for item in resp.data]
        except Exception as exc:
            is_rate = "429" in str(exc) or "rate_limit" in str(exc).lower()
            if attempt < max_retries - 1 and is_rate:
                logger.warning(
                    "Embedding batch %d rate-limited (attempt %d/%d); retrying in %.1fs",
                    offset, attempt + 1, max_retries, delay,
                )
                _time.sleep(delay)
                delay = min(delay * 2, 30.0)
                continue
            logger.warning("Embedding batch %d failed: %s; using zeros.", offset, exc)
            return [[0.0] * _EMBED_DIM] * len(batch)
    return [[0.0] * _EMBED_DIM] * len(batch)


def _embed_chunks(chunks: list[ChunkRecord], openai_client) -> Optional[np.ndarray]:
    """Embed all chunks; returns a (N, 1536) float32 array or None.

    build_index() runs synchronously (often from inside a running event loop in
    the async pipeline), so we must use a *synchronous* embeddings client here.
    The pipeline passes an AsyncOpenAI client (needed by the contextualizer);
    calling .embeddings.create() on it returns an un-awaited coroutine and the
    embeddings silently become zero vectors. Derive a sync client instead.
    """
    sync_client = _as_sync_embeddings_client(openai_client)
    if sync_client is None:
        logger.warning("No usable embeddings client; skipping dense index.")
        return None

    texts = [c.context_prefix + " " + c.content for c in chunks]
    batch_size = 100
    all_embeddings: list[list[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        all_embeddings.extend(_embed_batch_with_retry(sync_client, batch, i))

    vecs = np.array(all_embeddings, dtype=np.float32)
    if vecs.ndim != 2 or vecs.shape[1] != _EMBED_DIM:
        logger.warning("Unexpected embedding shape %s; skipping dense index.", vecs.shape)
        return None
    return vecs


def _build_faiss_index(chunks: list[ChunkRecord], openai_client) -> Optional[Any]:
    """Embed all chunks and build a FAISS FlatIP index (cosine via normalized IP)."""
    vecs = _embed_chunks(chunks, openai_client)
    if vecs is None:
        return None
    faiss.normalize_L2(vecs)
    index = faiss.IndexFlatIP(_EMBED_DIM)
    index.add(vecs)
    return index


def _tokenize(text: str) -> list[str]:
    """Tokenize text for BM25; removes stop words if NLTK data is available."""
    tokens = re.findall(r"\b[a-zA-Z]{2,}\b", text.lower())
    try:
        from nltk.corpus import stopwords  # type: ignore[import]

        stop = set(stopwords.words("english"))
        tokens = [t for t in tokens if t not in stop]
    except Exception:
        pass
    return tokens


def _save_chunks(chunks: list[ChunkRecord], path: Path) -> None:
    """Serialise a list of ChunkRecord objects to JSON."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump([c.model_dump() for c in chunks], f, ensure_ascii=False)


def _load_chunks(path: Path) -> list[ChunkRecord]:
    """Deserialise a list of ChunkRecord objects from JSON."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return [ChunkRecord.model_validate(item) for item in data]
