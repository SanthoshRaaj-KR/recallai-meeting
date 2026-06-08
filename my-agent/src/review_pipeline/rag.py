from __future__ import annotations

import hashlib
import logging
import os
import re
from dataclasses import dataclass
from typing import Any

from openai import OpenAI

from .models import PageCandidate
from .text_utils import extract_sections, html_to_text, normalize_ws

logger = logging.getLogger(__name__)


@dataclass
class PageChunk:
    id: str
    page_id: str
    title: str
    space_key: str
    heading: str
    section_order: int
    chunk_order: int
    version: int | None
    text: str


@dataclass
class VectorSearchHit:
    page_id: str
    title: str
    space_key: str = ""
    heading: str = ""
    score: float = 0.0
    text: str = ""
    version: int | None = None


class ConfluenceVectorIndex:
    """Page-wise Confluence vector index for hybrid review retrieval.

    Pinecone is used when configured. The class is deliberately optional: local
    development and tests keep working without Pinecone credentials, while
    production can turn on scalable dense retrieval via environment variables.
    """

    def __init__(self) -> None:
        self.backend = (
            os.getenv("MY_AGENT_RAG_VECTOR_DB")
            or os.getenv("MY_AGENT_VECTOR_DB")
            or "pinecone"
        ).strip().lower()
        self.index_name = (
            os.getenv("MY_AGENT_RAG_INDEX")
            or os.getenv("PINECONE_INDEX_NAME")
            or "confluence-review-rag"
        ).strip()
        self.namespace = (
            os.getenv("MY_AGENT_RAG_NAMESPACE")
            or os.getenv("PINECONE_NAMESPACE")
            or "confluence-review"
        ).strip()
        self.embedding_model = (
            os.getenv("MY_AGENT_RAG_EMBEDDING_MODEL")
            or os.getenv("OPENAI_EMBEDDING_MODEL")
            or "text-embedding-3-small"
        ).strip()
        configured_dims = (
            os.getenv("MY_AGENT_RAG_EMBEDDING_DIMENSIONS")
            or os.getenv("OPENAI_EMBEDDING_DIMENSIONS")
            or os.getenv("PINECONE_INDEX_DIMENSION")
            or ""
        ).strip()
        self.embedding_dimensions = int(configured_dims) if configured_dims else None
        self.max_chunk_words = int(os.getenv("MY_AGENT_RAG_CHUNK_WORDS", "350"))
        self.max_metadata_chars = int(os.getenv("MY_AGENT_RAG_METADATA_CHARS", "5000"))
        self.create_index = os.getenv("MY_AGENT_RAG_CREATE_INDEX", "0").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        self.cloud = os.getenv("PINECONE_CLOUD", "aws").strip()
        self.region = os.getenv("PINECONE_REGION", "us-east-1").strip()
        self._openai: OpenAI | None = None
        self._pinecone_index: Any | None = None
        self._disabled_reason: str | None = None

    @property
    def enabled(self) -> bool:
        if self.backend in {"", "none", "off", "disabled"}:
            return False
        if self.backend != "pinecone":
            self._disabled_reason = f"Unsupported vector backend: {self.backend}"
            return False
        if not os.getenv("PINECONE_API_KEY", "").strip():
            self._disabled_reason = "PINECONE_API_KEY is unset"
            return False
        return True

    def upsert_page(self, page: PageCandidate) -> None:
        if not self.enabled or not page.page_id:
            return
        content_hash = self._page_content_hash(page)
        if self._indexed_page_is_fresh(page, content_hash):
            logger.debug("Vector RAG page %s is already fresh; skipping rechunk.", page.page_id)
            return
        chunks = chunk_page(page, max_words=self.max_chunk_words)
        if not chunks:
            return
        embeddings = self._embed([self._embedding_text(chunk) for chunk in chunks])
        vectors = []
        for chunk, embedding in zip(chunks, embeddings):
            vectors.append(
                {
                    "id": chunk.id,
                    "values": embedding,
                    "metadata": {
                        "page_id": chunk.page_id,
                        "title": chunk.title,
                        "space_key": chunk.space_key,
                        "heading": chunk.heading,
                        "section_order": chunk.section_order,
                        "chunk_order": chunk.chunk_order,
                        "version": chunk.version or 0,
                        "content_hash": content_hash,
                        "chunk_count": len(chunks),
                        "text": chunk.text[: self.max_metadata_chars],
                    },
                }
            )
        index = self._index()
        for start in range(0, len(vectors), 100):
            index.upsert(vectors=vectors[start : start + 100], namespace=self.namespace)
        self._delete_stale_chunks(page.page_id, len(chunks))

    def _indexed_page_is_fresh(self, page: PageCandidate, content_hash: str) -> bool:
        if not page.page_id:
            return False
        try:
            result = self._index().fetch(ids=[f"{page.page_id}:0"], namespace=self.namespace)
            vectors = result.get("vectors", {}) if isinstance(result, dict) else (result.vectors or {})
            first = vectors.get(f"{page.page_id}:0")
            if not first:
                return False
            metadata = first.get("metadata", {}) if isinstance(first, dict) else (first.metadata or {})
            indexed_hash = str(metadata.get("content_hash") or "")
            indexed_version = int(metadata.get("version") or 0) or None
            if page.version is not None and indexed_version == page.version and indexed_hash == content_hash:
                return True
            if page.version is None and indexed_hash == content_hash:
                return True
        except Exception as exc:
            logger.debug("Vector RAG freshness check failed for %s: %s", page.page_id, exc)
        return False

    def search(self, query: str, top_k: int = 8) -> list[VectorSearchHit]:
        if not self.enabled:
            if self._disabled_reason:
                logger.debug("Vector RAG disabled: %s", self._disabled_reason)
            return []
        clean_query = normalize_ws(query)
        if not clean_query:
            return []
        embedding = self._embed([clean_query])[0]
        result = self._index().query(
            vector=embedding,
            top_k=max(1, top_k),
            namespace=self.namespace,
            include_metadata=True,
            include_values=False,
        )
        matches = result.get("matches", []) if isinstance(result, dict) else result.matches
        hits: list[VectorSearchHit] = []
        for match in matches or []:
            metadata = match.get("metadata", {}) if isinstance(match, dict) else (match.metadata or {})
            page_id = str(metadata.get("page_id") or "")
            if not page_id:
                continue
            score = float(match.get("score", 0.0) if isinstance(match, dict) else (match.score or 0.0))
            hits.append(
                VectorSearchHit(
                    page_id=page_id,
                    title=str(metadata.get("title") or page_id),
                    space_key=str(metadata.get("space_key") or ""),
                    heading=str(metadata.get("heading") or ""),
                    score=score,
                    text=str(metadata.get("text") or ""),
                    version=int(metadata.get("version") or 0) or None,
                )
            )
        return hits

    def _index(self) -> Any:
        if self._pinecone_index is not None:
            return self._pinecone_index
        try:
            from pinecone import Pinecone
        except ImportError as exc:
            raise RuntimeError("Install the 'pinecone' package to enable vector RAG.") from exc

        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY", "").strip())
        if self.create_index:
            self._ensure_pinecone_index(pc)
        self._pinecone_index = pc.Index(self.index_name)
        return self._pinecone_index

    def _ensure_pinecone_index(self, pc: Any) -> None:
        try:
            existing = pc.list_indexes()
            names = [item["name"] if isinstance(item, dict) else item.name for item in existing]
            if self.index_name in names:
                return
            from pinecone import ServerlessSpec

            pc.create_index(
                name=self.index_name,
                dimension=self._embedding_dimension(),
                metric="cosine",
                spec=ServerlessSpec(cloud=self.cloud, region=self.region),
            )
        except Exception as exc:
            logger.warning("Could not ensure Pinecone index %s: %s", self.index_name, exc)
            raise

    def _embed(self, texts: list[str]) -> list[list[float]]:
        if self._openai is None:
            self._openai = OpenAI()
        request: dict[str, Any] = {"model": self.embedding_model, "input": texts}
        if self.embedding_dimensions and self.embedding_model.startswith("text-embedding-3"):
            request["dimensions"] = self.embedding_dimensions
        response = self._openai.embeddings.create(**request)
        return [list(item.embedding) for item in response.data]

    def _embedding_dimension(self) -> int:
        if self.embedding_dimensions:
            return self.embedding_dimensions
        if self.embedding_model == "text-embedding-3-large":
            return 3072
        return 1536

    def _embedding_text(self, chunk: PageChunk) -> str:
        return normalize_ws(
            "\n".join(
                [
                    f"Title: {chunk.title}",
                    f"Heading: {chunk.heading}",
                    f"Space: {chunk.space_key}",
                    chunk.text,
                ]
            )
        )

    def _delete_stale_chunks(self, page_id: str, live_count: int) -> None:
        stale_ids = [f"{page_id}:{idx}" for idx in range(live_count, live_count + 64)]
        try:
            self._index().delete(ids=stale_ids, namespace=self.namespace)
        except Exception as exc:
            logger.debug("Vector stale chunk cleanup failed for %s: %s", page_id, exc)

    def _page_content_hash(self, page: PageCandidate) -> str:
        value = "\n".join(
            [
                page.page_id or "",
                page.title or "",
                page.space_key or "",
                page.html or "",
                page.text or "",
            ]
        )
        return hashlib.sha256(value.encode("utf-8")).hexdigest()


    # ── Incremental sync ──────────────────────────────────────────────────────

    def sync_index(
        self,
        page_listings: list[dict[str, Any]],
        fetch_page_fn: Any,
        progress_cb: Any | None = None,
    ) -> dict[str, Any]:
        """Incrementally re-index only Confluence pages that changed.

        Algorithm (O(pages/100) Pinecone calls + O(changed) embed calls):

        1. Batch-fetch the first chunk ``{page_id}:0`` for every page from
           Pinecone in groups of 100.  The stored metadata holds the version
           number and content hash that were current at index time.
        2. Compare each page's live Confluence ``version.number`` (from the
           listing) against the stored version.  Pages whose version increased
           — or that aren't in Pinecone yet — are marked stale.
        3. Only for stale pages: call ``fetch_page_fn(page_id)`` to pull the
           full HTML, then rechunk + re-embed via the existing ``upsert_page``
           path.  Fresh pages are skipped entirely.

        Args:
            page_listings:  Lightweight dicts from Confluence REST —
                            at minimum ``{"page_id": str, "version": int|None}``.
            fetch_page_fn:  Callable ``(page_id: str) -> PageCandidate`` that
                            fetches the full page HTML.  Must be synchronous.
            progress_cb:    Optional callable ``(done, total, page_title)``
                            called after each page is processed.

        Returns:
            ``{"checked": int, "changed": int, "skipped": int, "failed": int}``
        """
        if not self.enabled:
            logger.warning("sync_index: vector RAG is disabled — nothing to sync.")
            return {"checked": 0, "changed": 0, "skipped": 0, "failed": 0}

        checked = changed = skipped = failed = 0
        total = len(page_listings)

        # ── Step 1: batch-fetch stored versions from Pinecone ──────────────
        # We only fetch chunk :0 per page — it carries content_hash and version
        # in its metadata, which is all we need for the freshness check.
        stored_versions: dict[str, int | None] = {}
        stored_hashes: dict[str, str] = {}
        chunk_ids = [f"{p['page_id']}:0" for p in page_listings if p.get("page_id")]
        index = self._index()
        for batch_start in range(0, len(chunk_ids), 100):
            batch = chunk_ids[batch_start : batch_start + 100]
            try:
                result = index.fetch(ids=batch, namespace=self.namespace)
                vectors = (
                    result.get("vectors", {})
                    if isinstance(result, dict)
                    else (result.vectors or {})
                )
                for vec_id, vec in (vectors or {}).items():
                    page_id = vec_id.rsplit(":", 1)[0]
                    metadata = (
                        vec.get("metadata", {})
                        if isinstance(vec, dict)
                        else (vec.metadata or {})
                    )
                    stored_versions[page_id] = int(metadata.get("version") or 0) or None
                    stored_hashes[page_id] = str(metadata.get("content_hash") or "")
            except Exception as exc:  # noqa: BLE001
                logger.warning("sync_index: Pinecone batch fetch failed: %s", exc)
                # Treat all pages in this batch as stale so they get re-indexed.
                for cid in batch:
                    pid = cid.rsplit(":", 1)[0]
                    stored_versions.setdefault(pid, None)

        # ── Step 2: compute the delta ──────────────────────────────────────
        stale: list[dict[str, Any]] = []
        for listing in page_listings:
            pid = listing.get("page_id")
            if not pid:
                continue
            live_version: int | None = listing.get("version")
            pinecone_version = stored_versions.get(pid)
            is_new = pid not in stored_versions
            is_changed = (
                live_version is not None
                and pinecone_version is not None
                and live_version > pinecone_version
            )
            if is_new or is_changed:
                stale.append(listing)
            else:
                skipped += 1

        logger.info(
            "sync_index: %d pages total — %d stale / %d fresh",
            total,
            len(stale),
            skipped,
        )

        # ── Step 3: re-embed only stale pages ─────────────────────────────
        for listing in stale:
            pid = listing["page_id"]
            title = listing.get("title", pid)
            try:
                page = fetch_page_fn(pid)
                self.upsert_page(page)
                changed += 1
                logger.debug("sync_index: re-indexed %r (page_id=%s)", title, pid)
            except Exception as exc:  # noqa: BLE001
                failed += 1
                logger.warning("sync_index: failed to re-index %r: %s", title, exc)
            checked += 1
            if progress_cb:
                try:
                    progress_cb(checked, len(stale), title)
                except Exception:  # noqa: BLE001
                    pass

        return {
            "checked": checked,
            "changed": changed,
            "skipped": skipped,
            "failed": failed,
        }


def chunk_page(page: PageCandidate, *, max_words: int = 350) -> list[PageChunk]:
    """Split one Confluence page into page-owned chunks.

    Chunks never combine multiple Confluence pages. Heading-delimited sections
    are preferred, then oversized sections are split by paragraph/word windows.
    """
    if not page.page_id:
        return []
    sections = page.sections or (extract_sections(page.html) if page.html else [])
    if not sections:
        text = page.text or html_to_text(page.html)
        sections = [{"heading": "", "level": "0", "text": text, "html": page.html}]

    chunks: list[PageChunk] = []
    for section_order, section in enumerate(sections):
        heading = normalize_ws(str(section.get("heading") or "Page intro"))
        text = normalize_ws(str(section.get("text") or html_to_text(str(section.get("html") or ""))))
        if not text:
            continue
        for part in _split_text(text, max_words=max_words):
            chunks.append(
                PageChunk(
                    id=f"{page.page_id}:{len(chunks)}",
                    page_id=page.page_id,
                    title=page.title,
                    space_key=page.space_key,
                    heading=heading,
                    section_order=section_order,
                    chunk_order=len(chunks),
                    version=page.version,
                    text=part,
                )
            )
    return chunks


def _split_text(text: str, *, max_words: int) -> list[str]:
    words = text.split()
    if len(words) <= max_words:
        return [text]
    paragraphs = [p.strip() for p in re.split(r"\n\n+", text) if p.strip()]
    if len(paragraphs) <= 1:
        return [" ".join(words[idx : idx + max_words]) for idx in range(0, len(words), max_words)]

    chunks: list[str] = []
    current: list[str] = []
    current_words = 0
    for paragraph in paragraphs:
        count = len(paragraph.split())
        if current and current_words + count > max_words:
            chunks.append("\n\n".join(current))
            current = [paragraph]
            current_words = count
        else:
            current.append(paragraph)
            current_words += count
    if current:
        chunks.append("\n\n".join(current))
    return chunks
