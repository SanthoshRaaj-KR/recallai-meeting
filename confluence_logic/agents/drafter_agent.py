"""DrafterAgent — drafts one Confluence change proposal per candidate page (PIPE-02)."""

import json
import logging
import os
from typing import Any, Dict, List, Optional

from agents import Agent, Runner

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

DRAFTER_MODEL = os.getenv("JARVIS_AGENT_MODEL", "gpt-5-mini").strip()

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

DRAFTER_SYSTEM_PROMPT = (
    "You are a documentation drafter for a meeting intelligence system. "
    "Your job is to propose a single Confluence change based on what was discussed in a meeting.\n\n"
    "You receive a JSON payload with these keys:\n"
    "- page: {page_id, title, section_heading, relevant_content} — the target Confluence page\n"
    "- facts: {decisions, action_items, new_requirements, doc_worthy_updates, query_terms} — "
    "structured facts extracted from the meeting\n"
    "- transcript_excerpt: the last 3000 characters of the meeting transcript\n\n"
    "You MUST return a JSON object (not an array) with EXACTLY these keys:\n"
    "{\n"
    '  "change_type": "edit" | "title" | "delete",\n'
    '  "page_id": string | null,\n'
    '  "page_title": string,\n'
    '  "section_heading": string | null,\n'
    '  "before_content": string | null,\n'
    '  "after_content": string | null,\n'
    '  "rationale": string | null\n'
    "}\n\n"
    "CRITICAL CONSTRAINT: If the page already has a page_id (i.e. page_id is not null), "
    "you MUST only use change_type: edit, title, or delete. "
    "NEVER use change_type: create for an existing page. "
    "The create change_type is only valid when page_id is null (new page proposal).\n\n"
    "Guidelines:\n"
    "- Use 'edit' to update existing section content with new information from the meeting\n"
    "- Use 'title' to rename a page based on meeting decisions\n"
    "- Use 'delete' to remove content that is now outdated or explicitly deprecated in the meeting\n"
    "- Populate before_content with the existing content that would be changed (if known)\n"
    "- Populate after_content with the new content to add or replace\n"
    "- Write rationale explaining why this change is warranted based on meeting evidence\n"
    "- Be specific and use exact quotes from the transcript when possible\n"
    "- Return only valid JSON — no markdown, no explanation, no code blocks"
)

# ---------------------------------------------------------------------------
# Core drafting function
# ---------------------------------------------------------------------------


async def _run_drafter(
    page: Dict[str, Any],
    facts: Any,  # ExtractedFacts
    transcript_text: str,
) -> Dict[str, Any]:
    """Draft one change proposal for a candidate page."""
    page_id = page.get("page_id")
    page_title = page.get("title") or page.get("page_title") or "Unknown Page"

    drafter_input = json.dumps(
        {
            "page": {
                "page_id": page_id,
                "title": page_title,
                "section_heading": page.get("section_heading") or page.get("heading"),
                "relevant_content": (page.get("relevant_content") or "")[:2000],
            },
            "facts": {
                "decisions": getattr(facts, "decisions", []),
                "action_items": getattr(facts, "action_items", []),
                "new_requirements": getattr(facts, "new_requirements", []),
                "doc_worthy_updates": getattr(facts, "doc_worthy_updates", []),
                "query_terms": getattr(facts, "query_terms", []),
            },
            "transcript_excerpt": transcript_text[-3000:],
        },
        ensure_ascii=False,
    )

    agent = Agent(
        name=f"DrafterAgent-{page_id or 'new'}",
        model=DRAFTER_MODEL,
        instructions=DRAFTER_SYSTEM_PROMPT,
    )

    try:
        result = await Runner.run(agent, drafter_input)
        raw = result.final_output
        if isinstance(raw, str):
            data = json.loads(raw)
        elif isinstance(raw, dict):
            data = raw
        else:
            data = {}
        return _normalize_draft(data, page_id, page_title)
    except Exception as exc:
        logger.warning("Drafter failed for page %s: %s", page_id, exc)
        return {
            "change_type": "edit",
            "page_id": page_id,
            "page_title": page_title,
            "section_heading": None,
            "before_content": None,
            "after_content": None,
            "rationale": f"Drafter failed: {exc}",
        }


def _normalize_draft(
    data: Dict[str, Any],
    page_id: Optional[str],
    page_title: str,
) -> Dict[str, Any]:
    """Normalize and validate a raw draft dict from the LLM.

    Enforces the change_type constraint: existing pages (page_id not None)
    must only use edit/title/delete — never create (D-09).
    Zero-RAG fallback pages (page_id=None) may use create.
    """
    change_type = str(data.get("change_type") or "edit").strip().lower()

    # Enforce: existing pages must not produce create proposals (D-09)
    if page_id is not None and change_type not in {"edit", "title", "delete"}:
        change_type = "edit"

    # Zero-RAG fallback path (page_id=None) may produce create
    if page_id is None and change_type not in {"create", "edit"}:
        change_type = "create"

    return {
        "change_type": change_type,
        "page_id": page_id,
        "page_title": str(data.get("page_title") or page_title).strip(),
        "section_heading": data.get("section_heading") or None,
        "before_content": data.get("before_content") or None,
        "after_content": data.get("after_content") or None,
        "rationale": data.get("rationale") or None,
    }
