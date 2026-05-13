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
    "You are a Confluence documentation drafter. Given a meeting transcript and a Confluence page, "
    "propose one specific, formal change to that page.\n\n"
    "Input JSON keys:\n"
    "- page.relevant_content: the actual text currently on the Confluence page\n"
    "- facts: structured decisions/actions from the meeting\n"
    "- transcript_excerpt: recent meeting speech\n\n"
    "Return a JSON object with EXACTLY these keys:\n"
    "{\n"
    '  "change_type": "edit" | "title" | "delete" | "create",\n'
    '  "page_id": string | null,\n'
    '  "page_title": string,\n'
    '  "section_heading": string | null,\n'
    '  "before_content": string | null,\n'
    '  "after_content": string | null,\n'
    '  "rationale": string\n'
    "}\n\n"
    "CONTENT RULES — follow precisely:\n\n"
    "before_content:\n"
    "- Copy the EXACT 3-8 lines from page.relevant_content that will be changed.\n"
    "- Do NOT paste the entire page. Only the specific paragraph or bullet being modified.\n"
    "- Use markdown formatting (## headings, **bold**, bullet points) as it appears on the page.\n"
    "- If no existing content applies (new section or create), set to null.\n\n"
    "after_content:\n"
    "- Write the replacement content in formal, third-person English.\n"
    "- Maximum 10 lines. An HR professional must understand it immediately.\n"
    "- Use markdown: ## for headings, **bold** for key names/terms, - for bullet lists.\n"
    "- Be specific: include real names, page titles, decisions from the transcript.\n"
    "- No filler phrases like 'as discussed' or 'the team decided to'.\n\n"
    "section_heading:\n"
    "- The exact heading name of the section being changed on the page.\n\n"
    "rationale:\n"
    "- One sentence: why this change is needed, citing the meeting decision.\n\n"
    "CONSTRAINTS:\n"
    "- If page_id is not null, change_type must be edit, title, or delete — NEVER create.\n"
    "- If page_id is null, change_type should be create.\n"
    "- Return only valid JSON — no markdown wrapper, no explanation."
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
        logger.warning(
            "DrafterAgent: coercing change_type %r to 'create' for zero-RAG page (page_id=None)",
            change_type,
        )
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
