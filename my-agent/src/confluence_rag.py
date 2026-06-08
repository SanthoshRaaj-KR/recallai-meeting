"""In-meeting Confluence RAG for Jarvis wake-word queries.

Reuses the same Pinecone index as the post-meeting review pipeline
(MY_AGENT_RAG_INDEX / PINECONE_INDEX_NAME) but is purpose-built for
low-latency lookups during a live meeting. The search() method is
synchronous and should always be called via asyncio.to_thread() so
the event loop stays free.

Index metadata schema expected per chunk (set during review-pipeline upsert):
  page_id, title, space_key, heading, section_order, chunk_order, version,
  content_hash, chunk_count, text
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any

logger = logging.getLogger(__name__)

_TOP_K_DEFAULT = int(os.getenv("JARVIS_CONFLUENCE_RAG_TOP_K", "5"))
_SCORE_THRESHOLD = float(os.getenv("JARVIS_CONFLUENCE_RAG_SCORE_THRESHOLD", "0.25"))
# Characters kept from each chunk in the LLM context — keeps tokens tight.
_MAX_CHUNK_CHARS = int(os.getenv("JARVIS_CONFLUENCE_RAG_MAX_CHUNK_CHARS", "600"))

# How many recent transcript lines to harvest for query enrichment.
_TRANSCRIPT_CONTEXT_LINES = int(os.getenv("JARVIS_CONFLUENCE_RAG_CONTEXT_LINES", "5"))
# Max characters for the final enriched query sent to the embedding model.
_MAX_QUERY_CHARS = int(os.getenv("JARVIS_CONFLUENCE_RAG_MAX_QUERY_CHARS", "350"))

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

    Lazy-initialises both the OpenAI embedding client and the Pinecone index
    on first use so startup time is unaffected. Thread-safe for read access;
    write access is single-threaded (one asyncio.to_thread call at a time in
    practice).
    """

    def __init__(self) -> None:
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
        self._openai: Any | None = None
        self._pinecone_index: Any | None = None

    @property
    def enabled(self) -> bool:
        return bool(os.getenv("PINECONE_API_KEY", "").strip())

    # ── Public API ──────────────────────────────────────────────────────────

    def search(self, query: str, top_k: int = _TOP_K_DEFAULT) -> list[dict[str, Any]]:
        """Retrieve top-k Confluence chunks relevant to *query*.

        Always call this via ``asyncio.to_thread(rag.search, query)`` from
        async code to keep the event loop unblocked.

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
        try:
            embedding = self._embed(query)
            index = self._get_index()
            result = index.query(
                vector=embedding,
                top_k=max(1, top_k),
                namespace=self.namespace,
                include_metadata=True,
                include_values=False,
            )
            matches = (
                result.get("matches", []) if isinstance(result, dict) else result.matches
            ) or []
            hits: list[dict[str, Any]] = []
            for match in matches:
                metadata = (
                    match.get("metadata", {})
                    if isinstance(match, dict)
                    else (match.metadata or {})
                )
                score = float(
                    match.get("score", 0.0)
                    if isinstance(match, dict)
                    else (match.score or 0.0)
                )
                if score < _SCORE_THRESHOLD:
                    continue
                hits.append(
                    {
                        "page_id": str(metadata.get("page_id") or ""),
                        "title": str(metadata.get("title") or ""),
                        "space_key": str(metadata.get("space_key") or ""),
                        "heading": str(metadata.get("heading") or ""),
                        "section_order": int(metadata.get("section_order") or 0),
                        "text": str(metadata.get("text") or "")[:_MAX_CHUNK_CHARS],
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
            title = hit.get("title") or "Untitled"
            heading = (hit.get("heading") or "").strip()
            space_key = hit.get("space_key") or ""
            page_id = hit.get("page_id") or ""

            # Build a readable breadcrumb: "Page Title › Section Heading [SPACE]"
            label = title
            if heading and heading.lower() not in ("page intro", ""):
                label = f"{title} › {heading}"
            if space_key and page_id not in seen_pages:
                label = f"{label} [{space_key}]"
            seen_pages.add(page_id)

            parts.append(f"[{label}]\n{text}")

        return "\n\n".join(parts)

    def build_search_query(
        self,
        question: str,
        recent_transcript: list[str],
        context_lines: int = _TRANSCRIPT_CONTEXT_LINES,
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

        if not context_parts:
            return q[:_MAX_QUERY_CHARS]

        # Question first (dominant signal), then compact meeting context.
        combined = q + " " + " ".join(context_parts)
        return combined[:_MAX_QUERY_CHARS]

    def warmup(self) -> None:
        """Eagerly initialise the OpenAI and Pinecone clients.

        Call via ``asyncio.to_thread(rag.warmup)`` from ``on_enter()`` so the
        first wake-word query of a session doesn't pay the cold-init penalty.
        Only the client objects are created — no embedding or index query is
        made, so there is no token cost.
        """
        if not self.enabled:
            return
        try:
            if self._openai is None:
                from openai import OpenAI
                self._openai = OpenAI()
            self._get_index()
            logger.info("ConfluenceLiveRAG: clients pre-initialised")
        except Exception as exc:  # noqa: BLE001
            logger.warning("ConfluenceLiveRAG warmup failed: %s", exc)

    # ── Private helpers ──────────────────────────────────────────────────────

    def _embed(self, text: str) -> list[float]:
        if self._openai is None:
            from openai import OpenAI

            self._openai = OpenAI()
        request: dict[str, Any] = {
            "model": self.embedding_model,
            "input": [text.strip()],
        }
        if self.embedding_dimensions and self.embedding_model.startswith(
            "text-embedding-3"
        ):
            request["dimensions"] = self.embedding_dimensions
        response = self._openai.embeddings.create(**request)
        return list(response.data[0].embedding)

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
