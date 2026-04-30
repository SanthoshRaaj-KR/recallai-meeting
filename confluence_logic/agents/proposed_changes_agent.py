from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Callable, Dict, List, Optional

from openai import OpenAI

from .tools import list_workspace_pages, search_workspace_knowledge

logger = logging.getLogger(__name__)


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
            "Inputs include the full meeting transcript, executive summary, minutes of meeting, decisions, action items, "
            "participants, optional user guidance, and available Confluence page candidates.\n\n"
            "Rules:\n"
            "- Return ONLY JSON shaped as {\"changes\": [...]}.\n"
            "- Each change must have: change_type, page_id, page_title, section_heading, before_content, after_content, rationale.\n"
            "- change_type must be one of create, edit, delete, title.\n"
            "- Prefer edit proposals for existing pages when a relevant page candidate exists.\n"
            "- Use create only when the meeting clearly calls for a new page or no existing candidate fits.\n"
            "- Keep after_content concise but specific enough to paste into Confluence.\n"
            "- Never invent decisions, owners, dates, metrics, or page IDs.\n"
            "- If page_id is unknown, set it to null and use the best inferred page_title.\n"
            "- If the optional user guidance is present, prioritize it over broad transcript coverage.\n"
            "- If no useful Confluence update is supported by the meeting context, return {\"changes\": []}."
        )

    def _workspace_context(self, query: str, summary: Dict[str, Any]) -> Dict[str, Any]:
        search_terms = [
            query,
            summary.get("title") or "",
            " ".join(summary.get("key_topics") or []),
            " ".join(summary.get("decisions") or []),
        ]
        search_query = " ".join(term for term in search_terms if term).strip()[:500]

        pages: List[Dict[str, Any]] = []
        seen: set[str] = set()

        def add_candidates(response: Any) -> None:
            for candidate in getattr(response, "candidates", []) or []:
                page_id = getattr(candidate, "page_id", "")
                key = page_id or getattr(candidate, "title", "")
                if not key or key in seen:
                    continue
                seen.add(key)
                pages.append(
                    {
                        "page_id": page_id or None,
                        "title": getattr(candidate, "title", ""),
                        "heading": getattr(candidate, "heading", None),
                        "space_key": getattr(candidate, "space_key", ""),
                        "snippet": getattr(candidate, "snippet", ""),
                    }
                )

        try:
            add_candidates(list_workspace_pages(limit=20))
        except Exception as exc:
            logger.warning("Could not list workspace pages for proposal agent: %s", exc)

        if search_query:
            try:
                add_candidates(search_workspace_knowledge(search_query))
            except Exception as exc:
                logger.warning("Could not search workspace pages for proposal agent: %s", exc)

        return {
            "available_pages": pages[:20],
            "search_query": search_query,
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

    async def propose(
        self,
        *,
        transcript_text: str,
        summary: Dict[str, Any],
        query: str = "",
    ) -> List[Dict[str, Any]]:
        workspace_context = await asyncio.to_thread(self._workspace_context, query, summary)
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
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
                ],
                max_tokens=self.max_tokens,
                temperature=0.2,
                response_format={"type": "json_object"},
            )
        )
        raw = response.choices[0].message.content or "{}"
        data = json.loads(raw)
        return self._normalize_changes(data if isinstance(data, dict) else {})
