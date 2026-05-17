import os, sys, time
sys.path.insert(0, os.path.dirname(__file__))
from dotenv import load_dotenv
load_dotenv("confluence_logic/.env")

from confluence_logic.db.vector_store import PineconeStore

store = PineconeStore()
print(f"Index: {store.index_name}  Namespace: {repr(store.namespace)}\n")

queries = [
    "when is SOC2 coming?",
    "what is the product roadmap?",
    "who owns the security page?",
    "when is the product launch?",
    "what is the sprint plan?",
]

for q in queries:
    t0 = time.time()
    matches = store.search(q, top_k=3)
    elapsed_ms = (time.time() - t0) * 1000
    above_threshold = [m for m in matches if m.get("score", 0) >= 0.3]
    top_score = matches[0].get("score", 0) if matches else 0
    top_title = (matches[0].get("metadata") or {}).get("title", "—") if matches else "—"
    hit = "HIT" if above_threshold else "MISS"
    print(f"[{elapsed_ms:5.0f}ms] [{hit}] score={top_score:.3f}  top='{top_title}'")
    print(f"         query: {q!r}\n")
