"""In-meeting Confluence RAG for Jarvis wake-word queries.

Reuses the same Pinecone integrated index as the post-meeting review pipeline
(MY_AGENT_RAG_INDEX / PINECONE_INDEX_NAME). Embedding is handled server-side by
Pinecone (llama-text-embed-v2) — no separate OpenAI call per query. A single
search() round-trip to Pinecone embeds the query and returns ranked chunks.

Index field schema expected per chunk (set during review-pipeline upsert):
  page_id, title, space_key, heading, section_order, chunk_order, version,
  content_hash, chunk_count, text (raw display content), chunk_text (embedded)
"""

from __future__ import annotations

import logging
import os
import re
import time
from collections import deque
from typing import Any

logger = logging.getLogger(__name__)

_TOP_K_DEFAULT = int(os.getenv("JARVIS_CONFLUENCE_RAG_TOP_K", "10"))
_SCORE_THRESHOLD = float(os.getenv("JARVIS_CONFLUENCE_RAG_SCORE_THRESHOLD", "0.1"))
# Characters kept from each chunk in the LLM context — keeps tokens tight.
_MAX_CHUNK_CHARS = int(os.getenv("JARVIS_CONFLUENCE_RAG_MAX_CHUNK_CHARS", "800"))

# How many recent transcript lines to harvest for query enrichment.
_TRANSCRIPT_CONTEXT_LINES = int(os.getenv("JARVIS_CONFLUENCE_RAG_CONTEXT_LINES", "12"))
# Max characters for the final enriched query sent to the embedding model.
_MAX_QUERY_CHARS = int(os.getenv("JARVIS_CONFLUENCE_RAG_MAX_QUERY_CHARS", "350"))

# Query-result cache: avoids repeat Pinecone calls for the same topic within a session.
_CACHE_TTL_S = int(os.getenv("JARVIS_CONFLUENCE_RAG_CACHE_TTL", "120"))  # 2 minutes
_CACHE_MAX = int(os.getenv("JARVIS_CONFLUENCE_RAG_CACHE_MAX", "20"))

# Spoken filler words that carry no topical signal — removed before embedding.
_FILLER_RE = re.compile(
    r"\b(um+|uh+|hmm+|mhm+|yeah|yep|yup|nope|okay|ok|right|like|so|just|"
    r"you know|actually|basically|literally|honestly|well|anyway|alright|"
    r"sure|great|good|nice|cool|got it|i see|i think|i mean|i guess|"
    r"kind of|sort of|you see|let me|let us|let's)\b",
    re.IGNORECASE,
)
# Speaker prefix patterns: "Alice:", "Bob (host):", "Jarvis:" etc.
_SPEAKER_PREFIX_RE = re.compile(r"^[^:]{1,40}:\s*")


class ConfluenceLiveRAG:
    """Pinecone-backed in-meeting Confluence knowledge retriever.

    Uses Pinecone integrated inference (llama-text-embed-v2). Embedding and
    search happen in a single server-side round-trip — no external OpenAI call
    is made per query. Lazy-initialises the Pinecone index client on first use.
    """

    def __init__(self) -> None:
        self.index_name = (
            os.getenv("MY_AGENT_RAG_INDEX")
            or os.getenv("PINECONE_INDEX_NAME")
            or "confluence-review-rag-v2"
        ).strip()
        self.namespace = (
            os.getenv("MY_AGENT_RAG_NAMESPACE")
            or os.getenv("PINECONE_NAMESPACE")
            or "confluence-review"
        ).strip()
        self._pinecone_index: Any | None = None
        # (query_str, hits, timestamp) — bounded by _CACHE_MAX entries, evicted by TTL.
        self._cache: deque[tuple[str, list[dict[str, Any]], float]] = deque(maxlen=_CACHE_MAX)

    @property
    def enabled(self) -> bool:
        return bool(os.getenv("PINECONE_API_KEY", "").strip())

    # ── Public API ──────────────────────────────────────────────────────────

    def search(self, query: str, top_k: int = _TOP_K_DEFAULT) -> list[dict[str, Any]]:
        """Retrieve top-k Confluence chunks relevant to *query*.

        Embedding and ANN search happen in one Pinecone round-trip via integrated
        inference (llama-text-embed-v2). Always call this via
        ``asyncio.to_thread(rag.search, query)`` from async code.

        Returns a list of dicts with keys:
          page_id, title, space_key, heading, section_order, text, score
        Sorted by descending score. Chunks below JARVIS_CONFLUENCE_RAG_SCORE_THRESHOLD
        are filtered out.
        """
        if not self.enabled:
            logger.debug("ConfluenceLiveRAG: PINECONE_API_KEY not set — skipping.")
            return []
        if not query.strip():
            return []
        now = time.monotonic()
        for cached_q, cached_hits, ts in self._cache:
            if cached_q == query and now - ts < _CACHE_TTL_S:
                logger.info("[Pinecone] cache hit — skipping search for: %.60r", query)
                return cached_hits
        try:
            index = self._get_index()
            # Pinecone 8.x integrated-inference search nests top_k/inputs under a
            # ``query`` dict; top-level top_k= raises TypeError on this SDK.
            result = index.search(
                namespace=self.namespace,
                query={"top_k": max(1, top_k), "inputs": {"text": query.strip()}},
                fields=["page_id", "title", "space_key", "heading", "section_order", "text"],
            )
            raw_hits = result.result.hits if hasattr(result, "result") else []
            hits: list[dict[str, Any]] = []
            for hit in raw_hits:
                fields = hit.fields if hasattr(hit, "fields") else {}
                # Pinecone 8.x Hit exposes the score as ``_score`` (``score`` is None).
                score = float(getattr(hit, "_score", None) or getattr(hit, "score", 0.0) or 0.0)
                if score < _SCORE_THRESHOLD:
                    continue
                hits.append(
                    {
                        "page_id": str(fields.get("page_id") or ""),
                        "title": str(fields.get("title") or ""),
                        "space_key": str(fields.get("space_key") or ""),
                        "heading": str(fields.get("heading") or ""),
                        "section_order": int(fields.get("section_order") or 0),
                        "text": str(fields.get("text") or "")[:_MAX_CHUNK_CHARS],
                        "score": score,
                    }
                )

            if hits:
                titles = ", ".join(
                    f"{h['title']} › {h['heading']}" if h.get("heading") and h["heading"] != "Page intro"
                    else h["title"]
                    for h in hits[:3]
                )
                logger.info(
                    "[Pinecone] %d chunk(s) retrieved — top matches: %s",
                    len(hits),
                    titles + (" …" if len(hits) > 3 else ""),
                )
            else:
                logger.info("[Pinecone] query returned 0 hits above threshold %.2f", _SCORE_THRESHOLD)
            self._cache.append((query, hits, now))
            return hits
        except Exception as exc:  # noqa: BLE001
            logger.warning("ConfluenceLiveRAG search failed: %s", exc)
            return []

    def format_context(self, hits: list[dict[str, Any]]) -> str:
        """Render search hits as a compact context block for the LLM prompt.

        Each chunk is labelled with its page title, section heading, and space
        key so the LLM can attribute its answer to a specific Confluence page.
        Duplicate page breadcrumbs are suppressed after the first appearance.
        """
        if not hits:
            return ""
        parts: list[str] = []
        seen_pages: set[str] = set()
        for hit in hits:
            text = (hit.get("text") or "").strip()
            if not text:
                continue
            page_id = hit.get("page_id") or ""
            # Hits are score-sorted — keep only the top-scoring chunk per page.
            if page_id and page_id in seen_pages:
                continue
            title = hit.get("title") or "Untitled"
            heading = (hit.get("heading") or "").strip()
            space_key = hit.get("space_key") or ""

            # Build a readable breadcrumb: "Page Title › Section Heading [SPACE]"
            label = title
            if heading and heading.lower() not in ("page intro", ""):
                label = f"{title} › {heading}"
            if space_key:
                label = f"{label} [{space_key}]"
            seen_pages.add(page_id)

            parts.append(f"[{label}]\n{text}")

        return "\n\n".join(parts)

    def build_search_query(
        self,
        question: str,
        recent_transcript: list[str],
        context_lines: int = _TRANSCRIPT_CONTEXT_LINES,
        topic_hint: str = "",
    ) -> str:
        """Build a noise-filtered, context-aware query for Pinecone search.

        The question carries the highest topical signal and always leads.
        Recent transcript lines are harvested for surrounding context — they
        tell the embedding model *what topic the meeting is currently on* even
        when the question itself is short (e.g. "what does that mean?").

        Noise removed from transcript lines before inclusion:
        - Speaker prefixes  ("Alice:", "Bob (host):", "Jarvis:")
        - Spoken filler words  (um, uh, yeah, basically, ...)
        - Lines under 4 words after cleaning  (back-channel noise like "Sure",
          "Exactly", or repeated name acknowledgements)
        - Jarvis's own replies (prefixed "Jarvis:") to avoid self-reference
          loops that pull irrelevant Confluence chunks

        The combined query is capped at JARVIS_CONFLUENCE_RAG_MAX_QUERY_CHARS
        (default 350) so the embedding model is never diluted by a wall of text.

        topic_hint is only used when the question is vague (fewer than 5 words).
        A specific question already carries enough topical signal — appending
        compacted memory from a previous topic would bias the embedding toward
        the old page instead of the one the user is now asking about.
        """
        q = re.sub(r"\s+", " ", question).strip()

        context_parts: list[str] = []
        # Walk the last N lines in reverse so we pick the most recent first,
        # then flip back for natural left-to-right reading order.
        candidates = recent_transcript[-context_lines:] if recent_transcript else []
        for line in candidates:
            # Drop Jarvis's own TTS replies — they describe previous answers,
            # not the topic being discussed.
            if re.match(r"(?i)^jarvis\s*:", line):
                continue
            # Strip speaker prefix.
            text = _SPEAKER_PREFIX_RE.sub("", line).strip()
            # Remove filler words, then collapse whitespace.
            text = _FILLER_RE.sub(" ", text)
            text = re.sub(r"\s+", " ", text).strip()
            # Skip back-channel noise (fewer than 4 meaningful words).
            if len(text.split()) < 4:
                continue
            context_parts.append(text)

        # Question first, then persistent topic hint (only for vague questions),
        # then recent transcript context.
        parts = [q]
        # Only append topic_hint when the question itself is too short to carry
        # enough topical signal (e.g. "what did we say?" or "elaborate on that").
        # For specific multi-word questions the hint would pull the embedding
        # toward the previous topic and cause the wrong page to be retrieved.
        if topic_hint and len(q.split()) < 5:
            parts.append(topic_hint.strip())
        if context_parts:
            parts.append(" ".join(context_parts))
        combined = " ".join(parts)
        return combined[:_MAX_QUERY_CHARS]

    def warmup(self) -> None:
        """Pre-warm the Pinecone connection.

        Performs a real (throwaway) search so the TCP connection and TLS
        handshake to Pinecone are established during the opening greeting —
        before the first wake-word query arrives. Cost: 1 read unit.
        Call via ``asyncio.to_thread(rag.warmup)`` from ``on_enter()``.
        """
        if not self.enabled:
            return
        try:
            index = self._get_index()
            index.search(namespace=self.namespace, query={"top_k": 1, "inputs": {"text": "warmup"}})
            logger.info("ConfluenceLiveRAG: connection pre-warmed")
        except Exception as exc:  # noqa: BLE001
            logger.warning("ConfluenceLiveRAG warmup failed: %s", exc)

    # ── Private helpers ──────────────────────────────────────────────────────

    def _get_index(self) -> Any:
        if self._pinecone_index is not None:
            return self._pinecone_index
        try:
            from pinecone import Pinecone
        except ImportError as exc:
            raise RuntimeError(
                "Install 'pinecone' to enable in-meeting Confluence RAG."
            ) from exc
        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY", "").strip())
        self._pinecone_index = pc.Index(self.index_name)
        return self._pinecone_index
