"""Index live Confluence pages (real storage XHTML) into the hybrid indexes.

Fetches every page via the REST connector, converts it to markdown with
``markdownify`` (preserving headings and tables), chunks it by section, and
upserts into the dense + sparse Pinecone indexes.  The index therefore
reflects the actual current content of each Confluence page — matching what
the pipeline edits and what the connector writes back.

Usage:
    python -m review_pipeline.confluence_pipeline.reindex_live

Requires ATLASSIAN_* credentials in my-agent/.env.local.
"""

from __future__ import annotations

import sys
from pathlib import Path

from dotenv import load_dotenv

_HERE = Path(__file__).resolve()
_MY_AGENT = _HERE.parents[3]  # my-agent/

load_dotenv(_MY_AGENT / ".env.local")

from review_pipeline.confluence import RestConfluenceClient  # noqa: E402
from review_pipeline.confluence_pipeline.chunker import chunk_markdown_text  # noqa: E402
from review_pipeline.confluence_pipeline.retrieval import PineconeHybridIndex  # noqa: E402
from review_pipeline.text_utils import storage_to_markdown  # noqa: E402


def _storage_to_markdown(html: str, title: str) -> str:
    """Convert Confluence storage XHTML to clean markdown (no tags / macro noise)."""
    return storage_to_markdown(html or "", title)


def main(argv: list[str]) -> int:
    client = RestConfluenceClient()
    listings = client.list_pages(500)
    print(f"Found {len(listings)} Confluence pages to index.")

    all_chunks = []
    fetched = 0
    for listing in listings:
        pid = str(listing.get("page_id") or "").strip()
        if not pid:
            continue
        try:
            page = client.fetch_page(pid)
        except Exception as exc:  # noqa: BLE001
            print(f"  ! page {pid}: fetch failed: {str(exc)[:120]}")
            continue
        markdown = _storage_to_markdown(page.html, page.title or listing.get("title", ""))
        # source_path is the page_id — no local file mapping needed
        chunks = chunk_markdown_text(markdown, source_path=pid, source_format="confluence")
        # Stamp the live Confluence version on every chunk. sync_index reads it back
        # off chunk :0 to decide whether a page is still fresh; without it the stored
        # version is 0, which reads back as "unknown" and makes every later sync
        # re-embed the entire corpus.
        for chunk in chunks:
            chunk.version = page.version
        all_chunks.extend(chunks)
        fetched += 1
        print(f"  {pid} ({page.title!r}): v{page.version} -> {len(chunks)} chunks")

    print(f"Fetched {fetched}/{len(listings)} pages; total {len(all_chunks)} chunks.")
    if not all_chunks:
        print("No chunks to index — aborting.")
        return 1

    index = PineconeHybridIndex(create=True)
    print(
        f"Upserting into dense={index.dense_index!r}, sparse={index.sparse_index!r}, "
        f"namespace={index.namespace!r}…"
    )
    n = index.upsert_chunks(all_chunks)
    print(f"Done: upserted {n} records into each index.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
