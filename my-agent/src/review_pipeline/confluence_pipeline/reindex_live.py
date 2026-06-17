"""Index LIVE Confluence pages (real storage XHTML) into the hybrid indexes.

Unlike ``reindex.py`` (which chunks local ``.md`` stand-ins), this fetches each
page in its real Confluence **storage format** via the REST connector, converts
it to markdown with ``markdownify`` (preserving headings and tables), chunks it
by section, and upserts. The index therefore reflects the actual current content
of each page — matching what the pipeline edits and what the connector writes
back — instead of hand-authored markdown copies that can drift from Confluence.

Usage:
    python -m review_pipeline.confluence_pipeline.reindex_live [page_map.json]

Default page_map = ../local_doc_change/corpus_page_map.json (filename -> page_id).
Requires ATLASSIAN_* credentials in my-agent/.env.local.
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

from markdownify import markdownify as _md  # noqa: E402
from review_pipeline.confluence import RestConfluenceClient  # noqa: E402
from review_pipeline.confluence_pipeline.chunker import chunk_markdown_text  # noqa: E402
from review_pipeline.confluence_pipeline.retrieval import PineconeHybridIndex  # noqa: E402


def _default_map() -> str:
    return str(_REPO / "local_doc_change" / "corpus_page_map.json")


def _storage_to_markdown(html: str, title: str) -> str:
    """Convert Confluence storage XHTML to markdown, guaranteeing a level-1 title.

    Escaping is disabled: markdownify by default backslash-escapes ``_``, ``*``,
    ``$`` etc. (``threat\\_type``, ``\\$14``), which corrupts table values, pollutes
    retrieval/eval tokens, and defeats the same-row dedup (an escaped vs unescaped
    before-row no longer matches). The indexed text must mirror the real page text.
    """
    markdown = _md(
        html or "",
        heading_style="ATX",
        strip=["span"],
        escape_asterisks=False,
        escape_underscores=False,
        escape_misc=False,
    ).strip()
    if not markdown.lstrip().startswith("# "):
        markdown = f"# {title}\n\n{markdown}"
    return markdown


def main(argv: list[str]) -> int:
    map_path = argv[1] if len(argv) > 1 else _default_map()
    page_map: dict[str, dict] = json.loads(Path(map_path).read_text(encoding="utf-8"))
    print(f"page_map: {len(page_map)} pages from {map_path}")

    client = RestConfluenceClient()
    all_chunks = []
    fetched = 0
    for filename, meta in page_map.items():
        pid = str(meta.get("page_id") or "").strip()
        if not pid:
            print(f"  - {filename}: no page_id, skipped")
            continue
        try:
            page = client.fetch_page(pid)
        except Exception as exc:  # noqa: BLE001 - report and continue
            print(f"  ! {filename}: fetch failed for page {pid}: {str(exc)[:120]}")
            continue
        markdown = _storage_to_markdown(page.html, page.title or meta.get("title", ""))
        chunks = chunk_markdown_text(markdown, source_path=filename, source_format="confluence")
        all_chunks.extend(chunks)
        fetched += 1
        print(f"  {filename}: page {pid} v{page.version} -> {len(chunks)} chunks")

    print(f"fetched {fetched}/{len(page_map)} pages; total {len(all_chunks)} chunks")
    if not all_chunks:
        print("no chunks to index — aborting.")
        return 1

    index = PineconeHybridIndex(create=True)
    print(f"upserting into dense={index.dense_index!r}, sparse={index.sparse_index!r}, "
          f"namespace={index.namespace!r}…")
    n = index.upsert_chunks(all_chunks, page_map=page_map)
    print(f"done: upserted {n} records into each index.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
