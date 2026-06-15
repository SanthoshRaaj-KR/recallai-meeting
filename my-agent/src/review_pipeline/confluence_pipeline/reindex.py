"""Index a corpus folder into the Pinecone-native hybrid indexes.

Chunks every markdown/text file in a folder and upserts the chunks into BOTH the
dense (llama-text-embed-v2) and sparse (pinecone-sparse-english-v0) integrated
indexes. Each chunk carries its mapped Confluence ``page_id`` (from a
filename -> page_id map) so proposals can be attributed to the right page.

Usage:
    python -m review_pipeline.confluence_pipeline.reindex [corpus_dir] [page_map.json]

Defaults: corpus_dir = ../local_doc_change/stress_corpus/docs (combined
smarthub + stress corpus), page_map = ../local_doc_change/corpus_page_map.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from dotenv import load_dotenv

_HERE = Path(__file__).resolve()
_MY_AGENT = _HERE.parents[3]  # my-agent/
_REPO = _MY_AGENT.parent

load_dotenv(_MY_AGENT / ".env.local")

from review_pipeline.confluence_pipeline.chunker import chunk_folder  # noqa: E402
from review_pipeline.confluence_pipeline.retrieval import PineconeHybridIndex  # noqa: E402


def _default_corpus() -> str:
    return str(_REPO / "local_doc_change" / "stress_corpus" / "docs")


def _default_map() -> str:
    return str(_REPO / "local_doc_change" / "corpus_page_map.json")


def main(argv: list[str]) -> int:
    corpus = argv[1] if len(argv) > 1 else _default_corpus()
    map_path = argv[2] if len(argv) > 2 else _default_map()
    page_map: dict[str, dict] = {}
    if Path(map_path).exists():
        page_map = json.loads(Path(map_path).read_text(encoding="utf-8"))
        print(f"page_map: {len(page_map)} filename->page_id entries from {map_path}")
    else:
        print(f"page_map: {map_path} not found — proposals will lack page_id")

    print(f"chunking corpus: {corpus}")
    chunks = chunk_folder(corpus)
    mapped = sum(1 for c in chunks if Path(c.source_path).name in page_map)
    print(f"  {len(chunks)} chunks ({mapped} carry a Confluence page_id)")

    index = PineconeHybridIndex(create=True)
    print(f"upserting into dense={index.dense_index!r}, sparse={index.sparse_index!r}, "
          f"namespace={index.namespace!r} (server-side embedding)…")
    n = index.upsert_chunks(chunks, page_map=page_map)
    print(f"done: upserted {n} records into each index.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
