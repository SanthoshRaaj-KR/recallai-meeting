"""Smoke-test the Pinecone default path: with PINECONE_API_KEY set and
LDOC_VECTOR_DB unset, build_index should store dense vectors in Pinecone and
HybridRetriever should query them. Tiny 3-doc footprint; cleans up its namespace.
"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / "../confluence_logic/.env")

import openai  # noqa: E402

from rag import vector_store  # noqa: E402
from rag.indexer import build_index  # noqa: E402
from rag.retriever import HybridRetriever  # noqa: E402

DOCS = {
    "alpha.md": "# Alpha Security Policy\n\n## Data Retention\nLogs are kept for 30 days.\n",
    "bravo.md": "# Bravo Finance Policy\n\n## Travel\nPer diem is 60 dollars.\n",
    "charlie.md": "# Charlie Engineering Handbook\n\n## On-Call\nAck within 5 minutes.\n",
}


def main():
    print("backend choice:", vector_store.vector_db_choice())
    print("pinecone_enabled():", vector_store.pinecone_enabled())
    if not vector_store.pinecone_enabled():
        print("RESULT: Pinecone not enabled (key/package). Pipeline would use FAISS. SKIP")
        return 0

    tmp = Path(tempfile.mkdtemp(prefix="pc_smoke_"))
    folder = tmp / "docs"
    folder.mkdir()
    for name, body in DOCS.items():
        (folder / name).write_text(body, encoding="utf-8")

    client = openai.AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    idx = build_index(str(folder), use_embeddings=True, contextual_retrieval=False,
                      openai_client=client)
    ns = idx.pinecone_namespace
    count = vector_store.namespace_count(ns) if ns else 0
    print(f"vector_db={idx.vector_db} namespace={ns} vectors_in_namespace={count} chunks={len(idx.chunks)}")

    r = HybridRetriever(idx, rerank=False)
    res = r.query("travel per diem in the Bravo finance policy", top_k=2)
    top = res[0].chunk if res else None
    routed = top is not None and "bravo" in os.path.basename(top.source_path).lower()
    print(f"query routed to: {os.path.basename(top.source_path) if top else '(none)'} "
          f"heading='{top.section_heading if top else ''}'  doc_title='{top.doc_title if top else ''}'")

    ok = idx.vector_db == "pinecone" and count >= len(idx.chunks) and routed
    if ns:
        vector_store.delete_namespace(ns)  # cleanup
    shutil.rmtree(tmp, ignore_errors=True)
    print(f"\nRESULT: {'PINECONE DEFAULT OK (stored + queried + routed)' if ok else 'CHECK'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
