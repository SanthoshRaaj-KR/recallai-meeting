"""Force a full Confluence → Pinecone re-index.

Run this once after a chunking/metadata schema change to rewrite every chunk
in the index with the updated structure (e.g. heading_path, heading_level).

Usage:
    cd my-agent
    uv run python src/force_reindex.py

Requires the same env vars as the normal agent:
    PINECONE_API_KEY, ATLASSIAN_USER_EMAIL, ATLASSIAN_API_TOKEN, ATLASSIAN_DOMAIN
"""

from __future__ import annotations

import logging
import os
import sys

from dotenv import load_dotenv

load_dotenv(".env.local", override=True)
load_dotenv(".env", override=False)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("force_reindex")

# Add src/ to path so local imports resolve.
sys.path.insert(0, os.path.dirname(__file__))

from review_pipeline.confluence import RestConfluenceClient  # noqa: E402
from review_pipeline.rag import ConfluenceVectorIndex  # noqa: E402


def main() -> None:
    logger.info("Initialising Confluence REST client…")
    try:
        confluence = RestConfluenceClient()
    except ValueError as exc:
        logger.error("Confluence not configured: %s", exc)
        sys.exit(1)

    logger.info("Initialising Pinecone vector index…")
    rag = ConfluenceVectorIndex()
    if not rag.enabled:
        logger.error("Pinecone not configured (PINECONE_API_KEY missing or backend disabled).")
        sys.exit(1)

    logger.info("Fetching page listing from Confluence (up to 500 pages)…")
    listings = confluence.list_pages(500)
    logger.info("Found %d pages.", len(listings))

    processed = 0

    def progress_cb(done: int, total: int, title: str) -> None:
        nonlocal processed
        processed = done
        pct = int(100 * done / max(total, 1))
        logger.info("[%3d%%] %d/%d  %s", pct, done, total, title)

    logger.info("Starting force re-index (all pages will be rewritten)…")
    result = rag.sync_index(
        listings,
        confluence.fetch_page,
        progress_cb,
        force=True,
    )

    logger.info(
        "Done. checked=%d  changed=%d  skipped=%d  failed=%d  deleted=%d",
        result["checked"],
        result["changed"],
        result["skipped"],
        result["failed"],
        result.get("deleted", 0),
    )
    if result["failed"]:
        logger.warning("%d page(s) failed to re-index — check logs above.", result["failed"])
        sys.exit(1)


if __name__ == "__main__":
    main()
