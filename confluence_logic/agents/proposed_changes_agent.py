from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any, Callable, Dict, List, Optional

from openai import OpenAI

from .tools import fetch_live_page, list_workspace_pages, search_workspace_knowledge
from confluence_logic import confluence_page_graph

logger = logging.getLogger(__name__)

MAX_RETRIEVAL_QUERIES = int(os.getenv("JARVIS_PROPOSAL_RETRIEVAL_QUERIES", "6"))
MAX_RELEVANT_PAGES = int(os.getenv("JARVIS_PROPOSAL_RELEVANT_PAGES", "6"))
MAX_PAGE_CONTEXT_CHARS = int(os.getenv("JARVIS_PROPOSAL_PAGE_CONTEXT_CHARS", "3500"))


class ProposedChangesAgent:
    """Agent that turns meeting context into reviewable Confluence change proposals."""

    def __init__(
        self,
        model: str = "gpt-5-mini",
        client_factory: Optional[Callable[[], OpenAI]] = None,
        max_tokens: int = 1400,
    ):
        self.model = model
        self.client_factory = client_factory or OpenAI
        self.max_tokens = max_tokens

    @property
    def system_prompt(self) -> str:
        return (
            "You are Jarvis Confluence Proposal Agent, a specialist that proposes reviewable Confluence page updates "
            "from a meeting. You do not execute changes. You only produce a compact JSON list of proposed changes for "
            "a human to cross-check and approve later.\n\n"
            "Inputs include meeting context, optional user guidance, and ONLY the most relevant Confluence pages/sections "
            "retrieved by a separate search pipeline. Treat that retrieved page context as the entire workspace view you are allowed to use.\n\n"
            "Rules:\n"
            "- Return ONLY JSON shaped as {\"changes\": [...]}.\n"
            "- Each change must have: change_type, page_id, page_title, section_heading, before_content, after_content, rationale.\n"
            "- change_type must be one of create, edit, delete, title.\n"
            "- Prefer edit proposals for retrieved existing pages when a relevant page exists.\n"
            "- Use create only when the meeting clearly calls for a new page or no existing candidate fits.\n"
            "- Keep after_content concise but specific enough to paste into Confluence.\n"
            "- Never invent decisions, owners, dates, metrics, or page IDs.\n"
            "- For existing page edits, use page_id/page_title exactly from retrieved_page_context.\n"
            "- If page_id is unknown, set it to null and use the best inferred page_title.\n"
            "- If the optional user guidance is present, prioritize it over broad transcript coverage.\n"
            "- If no useful Confluence update is supported by the meeting context, return {\"changes\": []}."
        )

    @staticmethod
    def _truncate(value: str, max_chars: int = MAX_PAGE_CONTEXT_CHARS) -> str:
        text = (value or "").strip()
        if len(text) <= max_chars:
            return text
        return f"{text[: max_chars // 2]}\n[... omitted ...]\n{text[-(max_chars // 2):]}"

    @staticmethod
    def _meeting_search_queries(query: str, summary: Dict[str, Any]) -> List[str]:
        action_items = summary.get("action_items") or []
        action_text = " ".join(
            str(item.get("description") if isinstance(item, dict) else item)
            for item in action_items[:5]
        )
        candidates = [
            query,
            " ".join(summary.get("key_topics") or []),
            " ".join(summary.get("decisions") or []),
            action_text,
            summary.get("title") or "",
            summary.get("summary") or "",
        ]

        queries: List[str] = []
        seen: set[str] = set()
        for candidate in candidates:
            normalized = " ".join(str(candidate or "").split())[:500]
            key = normalized.lower()
            if normalized and key not in seen:
                seen.add(key)
                queries.append(normalized)
            if len(queries) >= MAX_RETRIEVAL_QUERIES:
                break
        return queries

    async def _graph_workspace_context(
        self,
        graph_user_id: str,
        search_queries: List[str],
    ) -> List[Dict[str, Any]]:
        if not graph_user_id or not search_queries:
            return []

        built = await confluence_page_graph.ensure_user_confluence_graph(graph_user_id)
        if not built:
            return []

        pages_by_key: Dict[str, Dict[str, Any]] = {}
        for search_query in search_queries:
            matches = await confluence_page_graph.query_user_confluence_graph(
                graph_user_id,
                search_query,
                limit=MAX_RELEVANT_PAGES,
            )
            for match in matches:
                key = match.get("page_id") or f"{match.get('title')}::{match.get('heading')}"
                if not key:
                    continue
                existing = pages_by_key.get(key)
                if existing is None or float(match.get("score") or 0) > float(existing.get("score") or 0):
                    pages_by_key[key] = match

        return sorted(
            pages_by_key.values(),
            key=lambda page: (-float(page.get("score") or 0), str(page.get("title") or "").lower()),
        )[:MAX_RELEVANT_PAGES]

    def _fallback_workspace_context(self, search_queries: List[str]) -> List[Dict[str, Any]]:
        pages: List[Dict[str, Any]] = []
        pages_by_id: Dict[str, Dict[str, Any]] = {}
        page_scores: Dict[str, int] = {}

        def add_candidates(response: Any, score: int) -> None:
            for candidate in getattr(response, "candidates", []) or []:
                page_id = getattr(candidate, "page_id", "")
                key = page_id or getattr(candidate, "title", "")
                if not key:
                    continue
                existing = pages_by_id.get(key)
                page_scores[key] = page_scores.get(key, 0) + score
                if existing:
                    if not existing.get("heading") and getattr(candidate, "heading", None):
                        existing["heading"] = getattr(candidate, "heading", None)
                    snippet = getattr(candidate, "snippet", "")
                    if snippet and snippet not in existing.get("snippet", ""):
                        existing["snippet"] = f"{existing.get('snippet', '')}\n{snippet}".strip()
                    continue

                pages_by_id[key] = {
                    "page_id": page_id or None,
                    "title": getattr(candidate, "title", ""),
                    "heading": getattr(candidate, "heading", None),
                    "space_key": getattr(candidate, "space_key", ""),
                    "snippet": getattr(candidate, "snippet", ""),
                }

        for index, search_query in enumerate(search_queries):
            try:
                add_candidates(search_workspace_knowledge(search_query), score=max(1, MAX_RETRIEVAL_QUERIES - index))
            except Exception as exc:
                logger.warning("Could not search workspace pages for proposal agent: %s", exc)

        if not pages_by_id:
            try:
                add_candidates(list_workspace_pages(limit=20), score=1)
            except Exception as exc:
                logger.warning("Could not list workspace pages for proposal agent: %s", exc)

        ranked_pages = sorted(
            pages_by_id.values(),
            key=lambda page: (
                -page_scores.get(page.get("page_id") or page.get("title") or "", 0),
                str(page.get("title") or "").lower(),
            ),
        )[:MAX_RELEVANT_PAGES]

        for page in ranked_pages:
            page_id = page.get("page_id")
            if not page_id:
                pages.append(page)
                continue

            try:
                live = fetch_live_page(page_id, page.get("heading"))
                page["available_headings"] = getattr(live, "available_headings", []) or []
                page["expected_version"] = getattr(live, "expected_version", None)
                section_html = getattr(live, "section_html", None)
                if section_html:
                    page["relevant_content"] = self._truncate(section_html)
                else:
                    page["relevant_content"] = self._truncate(page.get("snippet") or "")
            except Exception as exc:
                logger.warning("Could not fetch relevant page context for %s: %s", page_id, exc)
                page["relevant_content"] = self._truncate(page.get("snippet") or "")

            pages.append(page)

        return pages

    async def _workspace_context(self, query: str, summary: Dict[str, Any], graph_user_id: str = "") -> Dict[str, Any]:
        search_queries = self._meeting_search_queries(query, summary)
        pages = await self._graph_workspace_context(graph_user_id, search_queries)
        source = "neo4j_confluence_graph" if pages else "live_search_fallback"
        if not pages:
            pages = await asyncio.to_thread(self._fallback_workspace_context, search_queries)

        return {
            "retrieval_queries": search_queries,
            "retrieved_page_context": pages,
            "retrieval_source": source,
            "graph_user_id": graph_user_id or None,
        }

    @staticmethod
    def _normalize_changes(data: Dict[str, Any]) -> List[Dict[str, Any]]:
        raw_changes = data.get("changes")
        if not isinstance(raw_changes, list):
            return []

        normalized: List[Dict[str, Any]] = []
        for raw in raw_changes[:12]:
            if not isinstance(raw, dict):
                continue

            change_type = str(raw.get("change_type") or "edit").strip().lower()
            if change_type not in {"create", "edit", "delete", "title"}:
                change_type = "edit"

            page_title = str(raw.get("page_title") or "").strip()
            after_content = str(raw.get("after_content") or "").strip()
            before_content = str(raw.get("before_content") or "").strip()
            if not page_title or (change_type != "delete" and not after_content):
                continue

            normalized.append(
                {
                    "change_type": change_type,
                    "page_id": raw.get("page_id") or None,
                    "page_title": page_title,
                    "section_heading": raw.get("section_heading") or None,
                    "before_content": before_content or None,
                    "after_content": after_content or None,
                    "rationale": str(raw.get("rationale") or "").strip() or None,
                }
            )

        return normalized

    @staticmethod
    def _openai_completion_options(model: str, max_tokens: int) -> Dict[str, Any]:
        if model.startswith(("gpt-5", "o1", "o3", "o4")):
            return {"model": model, "max_completion_tokens": max_tokens}
        return {"model": model, "max_tokens": max_tokens, "temperature": 0.2}

    async def propose(
        self,
        *,
        transcript_text: str,
        summary: Dict[str, Any],
        query: str = "",
        graph_user_id: str = "",
    ) -> List[Dict[str, Any]]:
        workspace_context = await self._workspace_context(query, summary, graph_user_id=graph_user_id)
        payload = {
            "optional_user_guidance": query or None,
            "meeting_summary": {
                "title": summary.get("title"),
                "executive_summary": summary.get("summary"),
                "key_topics": summary.get("key_topics") or [],
                "decisions": summary.get("decisions") or [],
                "action_items": summary.get("action_items") or [],
                "minutes_of_meeting": summary.get("mom") or [],
                "participants": summary.get("participants") or [],
            },
            "workspace_context": workspace_context,
            "full_transcript": transcript_text,
        }

        response = await asyncio.to_thread(
            lambda: self.client_factory().chat.completions.create(
                **self._openai_completion_options(self.model, self.max_tokens),
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
                ],
                response_format={"type": "json_object"},
            )
        )
        raw = response.choices[0].message.content or "{}"
        data = json.loads(raw)
        return self._normalize_changes(data if isinstance(data, dict) else {})
