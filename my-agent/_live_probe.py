"""Read-only probe of the LIVE integration surface via the Hybrid client (which
falls back to Rovo MCP when REST 403s) and the Confluence review RAG index."""
import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, "src")
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env.local")


async def main() -> None:
    print("=== HybridConfluenceClient (REST + Rovo MCP fallback) ===")
    try:
        from review_pipeline.confluence import HybridConfluenceClient
        hc = HybridConfluenceClient()
        try:
            pages = await hc.list_pages(limit=8)
            print(f"  list_pages -> {len(pages)} page(s)")
            for p in pages[:8]:
                print("   -", (p.get("title") or p.get("id")), "| id=", p.get("page_id") or p.get("id"))
        except Exception as exc:
            print(f"  list_pages FAILED: {type(exc).__name__}: {str(exc)[:200]}")
        for q in ("smarthub", "pricing", "SLA"):
            try:
                hits = await hc.search_pages(q, limit=5)
                print(f"  search {q!r} -> {len(hits)} hit(s): {[h.get('title') for h in hits[:5]]}")
            except Exception as exc:
                print(f"  search {q!r} FAILED: {type(exc).__name__}: {str(exc)[:150]}")
    except Exception as exc:
        print(f"  hybrid client init failed: {type(exc).__name__}: {exc}")

    print("\n=== Confluence review RAG (Pinecone) ===")
    try:
        from review_pipeline.rag import ConfluenceVectorIndex
        idx = ConfluenceVectorIndex()
        print(f"  backend={idx.backend} index={idx.index_name!r} ns={idx.namespace!r} enabled={idx.enabled}")
        from pinecone import Pinecone
        pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
        names = [i["name"] for i in pc.list_indexes()]
        print(f"  pinecone indexes: {names}")
        if idx.index_name in names:
            stats = pc.Index(idx.index_name).describe_index_stats()
            ns = (stats.get("namespaces", {}) if isinstance(stats, dict) else stats.namespaces) or {}
            total = stats.get("total_vector_count") if isinstance(stats, dict) else stats.total_vector_count
            print(f"  index {idx.index_name!r}: total_vectors={total} namespaces={ {k: (v['vector_count'] if isinstance(v,dict) else v.vector_count) for k,v in ns.items()} }")
        else:
            print(f"  index {idx.index_name!r} NOT in account -> review RAG empty for this index")
    except Exception as exc:
        print(f"  RAG probe failed: {type(exc).__name__}: {str(exc)[:200]}")


if __name__ == "__main__":
    asyncio.run(main())
