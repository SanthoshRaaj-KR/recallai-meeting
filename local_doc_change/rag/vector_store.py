"""Pinecone dense-vector backend for local-doc RAG (default when configured).

Pinecone is now the DEFAULT dense backend whenever ``PINECONE_API_KEY`` is set
and the ``pinecone`` package is installed. To force the in-process FAISS backend
instead, set ``LDOC_VECTOR_DB=faiss``. Any of these conditions silently falls
back to FAISS (logged once): the package is not installed, ``PINECONE_API_KEY``
is unset, or a Pinecone call fails — so the pipeline always works.

Relevant environment variables
------------------------------
LDOC_VECTOR_DB      "" (auto: pinecone if key present, else faiss) | "faiss" | "pinecone"
PINECONE_API_KEY    enables Pinecone by default when set
LDOC_PINECONE_INDEX index name (default "local-doc-rag")
PINECONE_CLOUD      serverless cloud (default "aws")
PINECONE_REGION     serverless region (default "us-east-1")

Each document folder is isolated in its own Pinecone *namespace* (the folder
content hash), mirroring the FAISS per-folder disk cache.
"""
from __future__ import annotations

import logging
import os
import time
from typing import Optional

logger = logging.getLogger(__name__)

_EMBED_DIM = 1536
_pc = None  # cached Pinecone client


def vector_db_choice() -> str:
    """Return the configured backend: 'pinecone' by default when an API key is
    present, else 'faiss'. Set LDOC_VECTOR_DB=faiss (or =pinecone) to force one."""
    explicit = os.getenv("LDOC_VECTOR_DB", "").strip().lower()
    if explicit in ("faiss", "pinecone"):
        return explicit
    # Auto: prefer Pinecone when a key is configured (install/connectivity are
    # checked in pinecone_enabled(), which falls back to FAISS on any problem).
    return "pinecone" if os.getenv("PINECONE_API_KEY", "").strip() else "faiss"


def _config() -> dict:
    return {
        "api_key": os.getenv("PINECONE_API_KEY", "").strip(),
        "index": os.getenv("LDOC_PINECONE_INDEX", "local-doc-rag").strip(),
        "cloud": os.getenv("PINECONE_CLOUD", "aws").strip(),
        "region": os.getenv("PINECONE_REGION", "us-east-1").strip(),
    }


def pinecone_enabled() -> bool:
    """True only when pinecone is selected, installed, and configured."""
    if vector_db_choice() != "pinecone":
        return False
    if not _config()["api_key"]:
        logger.warning(
            "LDOC_VECTOR_DB=pinecone but PINECONE_API_KEY is unset — falling back to FAISS."
        )
        return False
    try:
        import pinecone  # noqa: F401
    except ImportError:
        logger.warning(
            "LDOC_VECTOR_DB=pinecone but the 'pinecone' package is not installed "
            "(run: uv add pinecone) — falling back to FAISS."
        )
        return False
    return True


def _client():
    global _pc
    if _pc is None:
        from pinecone import Pinecone

        _pc = Pinecone(api_key=_config()["api_key"])
    return _pc


def _ensure_index(name: str):
    """Return a handle to the index, creating a serverless index if needed."""
    from pinecone import ServerlessSpec

    pc = _client()
    names = [i["name"] for i in pc.list_indexes()]
    if name not in names:
        cfg = _config()
        logger.info("Creating Pinecone serverless index %r (%s/%s)", name, cfg["cloud"], cfg["region"])
        pc.create_index(
            name=name,
            dimension=_EMBED_DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud=cfg["cloud"], region=cfg["region"]),
        )
        for _ in range(90):  # wait until the index is ready
            try:
                if pc.describe_index(name).status.get("ready"):
                    break
            except Exception:
                pass
            time.sleep(1)
    return pc.Index(name)


def upsert(namespace: str, ids: list[str], vectors) -> bool:
    """Upsert (id, vector) pairs into a namespace. Returns False on failure."""
    try:
        idx = _ensure_index(_config()["index"])
        items = [(str(i), [float(x) for x in v], {}) for i, v in zip(ids, vectors)]
        for k in range(0, len(items), 100):
            idx.upsert(vectors=items[k : k + 100], namespace=namespace)
        return True
    except Exception as exc:
        logger.warning("Pinecone upsert failed (%s) — dense retrieval disabled.", exc)
        return False


def query(namespace: str, vector, top_k: int) -> list[str]:
    """Return chunk ids ranked by similarity for a query vector."""
    try:
        idx = _ensure_index(_config()["index"])
        res = idx.query(
            vector=[float(x) for x in vector],
            top_k=top_k,
            namespace=namespace,
            include_values=False,
        )
        matches = res.get("matches", []) if isinstance(res, dict) else res.matches
        return [m["id"] if isinstance(m, dict) else m.id for m in matches]
    except Exception as exc:
        logger.warning("Pinecone query failed (%s).", exc)
        return []


def namespace_count(namespace: str) -> int:
    """Number of vectors already stored in a namespace (0 if none / error)."""
    try:
        idx = _ensure_index(_config()["index"])
        stats = idx.describe_index_stats()
        ns = (stats.get("namespaces", {}) if isinstance(stats, dict) else stats.namespaces) or {}
        entry = ns.get(namespace)
        if entry is None:
            return 0
        return entry["vector_count"] if isinstance(entry, dict) else entry.vector_count
    except Exception:
        return 0


def delete_namespace(namespace: str) -> None:
    """Best-effort delete of all vectors in a namespace (used by tests)."""
    try:
        idx = _ensure_index(_config()["index"])
        idx.delete(delete_all=True, namespace=namespace)
    except Exception as exc:
        logger.warning("Pinecone namespace delete failed (%s).", exc)
