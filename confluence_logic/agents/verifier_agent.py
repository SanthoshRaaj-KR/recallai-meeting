"""VerifierAgent — enriches draft proposals with confidence, risk, and verifier_note. Never drops a card (PIPE-03)."""

import asyncio
import json
import logging
import os
from typing import Any, Dict, List, Optional

from openai import OpenAI

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

VERIFIER_MODEL = os.getenv("JARVIS_VERIFIER_MODEL", "gpt-5.4-mini").strip()
VERIFIER_MAX_TOKENS = int(os.getenv("JARVIS_VERIFIER_MAX_TOKENS", "700"))

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

VERIFIER_SYSTEM_PROMPT = (
    "You are a quality verification agent for a meeting intelligence system. "
    "Your job is to assess whether a proposed Confluence change is well-supported by the meeting transcript "
    "AND whether the proposed content is appropriate for the target page.\n\n"
    "You receive a JSON payload with:\n"
    "- draft: the proposed change (change_type, page_title, section_heading, before_content, after_content, rationale)\n"
    "- transcript_excerpt: the last 3000 characters of the meeting transcript\n"
    "- current_page_content: the current content of the target Confluence page section\n\n"
    "Return a JSON object with EXACTLY these keys:\n"
    "{\n"
    '  "confidence": "high" | "medium" | "low",\n'
    '  "risk": "safe" | "review" | "risky",\n'
    '  "verifier_note": string (1-2 sentences explaining your assessment),\n'
    '  "transcript_evidence": [string, ...] (1-3 verbatim quotes from the transcript that support this proposal),\n'
    '  "page_relevance": integer 0-10,\n'
    '  "content_type": "final_content" | "meta_instruction",\n'
    '  "should_drop": true | false\n'
    "}\n\n"
    "Confidence guidelines:\n"
    "- 'high': the change is directly and explicitly supported by multiple clear transcript statements\n"
    "- 'medium': the change is partially supported — some evidence but requires inference\n"
    "- 'low': weak or ambiguous support; the proposal may be extrapolating beyond what was discussed\n\n"
    "Risk guidelines:\n"
    "- 'safe': change is additive or clearly replaces outdated content; low chance of breaking anything\n"
    "- 'review': change modifies important content or could affect other pages; human review recommended\n"
    "- 'risky': change deletes content, renames a title that may be referenced externally, "
    "or the transcript evidence is ambiguous or contradictory\n\n"
    "Mark as 'risky' if:\n"
    "- The change_type is 'delete' (removes potentially important content)\n"
    "- The change_type is 'title' (renaming may break external references)\n"
    "- The transcript evidence is ambiguous or could support multiple interpretations\n\n"
    "Mark as 'review' if only partially supported by the transcript. "
    "Mark as 'safe' if clearly supported by direct, unambiguous transcript quotes.\n\n"
    "page_relevance (0-10): How appropriate is this target page for this change?\n"
    "- 10: The page is EXACTLY about this subject — title and content match the proposed change perfectly\n"
    "- 7-9: The page clearly covers this topic; the change clearly belongs here\n"
    "- 4-6: The page is loosely related; the change might belong here but it is uncertain\n"
    "- 1-3: The page is about a different subject; the change is a poor fit\n"
    "- 0: Completely wrong page — the change has nothing to do with this page's subject\n"
    "Compare draft.page_title and the proposed change subject against current_page_content to score this.\n\n"
    "content_type: Is draft.after_content actual page documentation or instructions to a writer?\n"
    "- 'final_content': after_content is publishable documentation — facts, descriptions, specs, bullet lists of real information\n"
    "  Examples: 'MS Dhoni is a former Indian cricket captain known for...', 'The team uses React for the frontend'\n"
    "- 'meta_instruction': after_content contains writing directives, editorial guidance, or planning notes — NOT real page content\n"
    "  Examples: 'Keep MS Dhoni as the primary subject', 'Add a clear differentiation section', "
    "'Include Virat Kohli only as comparison', 'Maintain a professional and neutral tone', "
    "'Use a structured format', 'This page should focus on...', 'Ensure the content covers...'\n"
    "If after_content is null (delete/title change), set content_type to 'final_content'.\n\n"
    "should_drop (true/false): Set to true if this proposal should be discarded entirely.\n"
    "Set should_drop=true when ANY of:\n"
    "- content_type is 'meta_instruction' (instruction text must never be written to Confluence)\n"
    "- page_relevance < 4 (wrong page — the change does not belong here)\n"
    "- confidence is 'low' AND page_relevance < 5 (weak transcript support AND weak page fit)\n"
    "Otherwise set should_drop=false.\n\n"
    "transcript_evidence must contain 1-3 verbatim quotes copied exactly from the transcript_excerpt. "
    "Return only valid JSON — no markdown, no explanation, no code blocks."
)

# ---------------------------------------------------------------------------
# OpenAI client singleton
# ---------------------------------------------------------------------------

_openai_client: Optional[OpenAI] = None


def _get_openai_client() -> OpenAI:
    global _openai_client
    if _openai_client is None:
        _openai_client = OpenAI()
    return _openai_client


def _openai_completion_options(model: str, max_tokens: int, temperature: float = 0.2) -> Dict[str, Any]:
    opts: Dict[str, Any] = {"model": model}
    if model.startswith(("gpt-5", "o1", "o3", "o4")):
        opts["max_completion_tokens"] = max_tokens
    else:
        opts["max_tokens"] = max_tokens
        opts["temperature"] = temperature
    return opts


# ---------------------------------------------------------------------------
# Core verification function
# ---------------------------------------------------------------------------


async def _run_verifier(
    draft: Dict[str, Any],
    transcript_text: str,
    page_content: str,
) -> Dict[str, Any]:
    """Enrich a draft card with confidence, risk, verifier_note, transcript_evidence.

    Never drops the card. On failure, sets safe defaults and returns (PIPE-03).
    """
    try:
        response = await asyncio.to_thread(
            lambda: _get_openai_client().chat.completions.create(
                **_openai_completion_options(VERIFIER_MODEL, VERIFIER_MAX_TOKENS),
                messages=[
                    {"role": "system", "content": VERIFIER_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": json.dumps(
                            {
                                "draft": draft,
                                "transcript_excerpt": transcript_text[-6000:],
                                "current_page_content": page_content[:2000],
                            },
                            ensure_ascii=False,
                        ),
                    },
                ],
                response_format={"type": "json_object"},
            )
        )
        raw = json.loads(response.choices[0].message.content or "{}")

        content_type = str(raw.get("content_type") or "final_content").strip().lower()
        if content_type not in ("final_content", "meta_instruction"):
            content_type = "final_content"

        page_relevance = raw.get("page_relevance")
        try:
            page_relevance = int(page_relevance)
            page_relevance = max(0, min(10, page_relevance))
        except (TypeError, ValueError):
            page_relevance = 5

        confidence = raw.get("confidence") or "low"
        risk = raw.get("risk") or "safe"
        verifier_note = raw.get("verifier_note") or ""

        # Downgrade confidence/risk based on content quality and page fit
        if content_type == "meta_instruction":
            confidence = "low"
            risk = "risky"
            verifier_note = f"[INSTRUCTION TEXT] after_content contains editorial directives, not documentation. {verifier_note}".strip()
        elif page_relevance < 4:
            confidence = "low"
            risk = "risky"
            verifier_note = f"[WRONG PAGE: relevance {page_relevance}/10] Change does not belong on this page. {verifier_note}".strip()

        should_drop = bool(raw.get("should_drop", False))
        # Enforce drop when critical conditions are met regardless of LLM output
        if content_type == "meta_instruction" or page_relevance < 4:
            should_drop = True

        draft["confidence"] = confidence
        draft["risk"] = risk
        draft["verifier_note"] = verifier_note
        draft["transcript_evidence"] = raw.get("transcript_evidence") or []
        draft["page_relevance"] = page_relevance
        draft["content_type"] = content_type
        draft["should_drop"] = should_drop
        return draft
    except Exception as exc:
        logger.warning("Verifier failed for page %s: %s", draft.get("page_id"), exc)
        draft.setdefault("confidence", "low")
        draft.setdefault("risk", "safe")
        draft.setdefault("verifier_note", "")
        draft.setdefault("transcript_evidence", [])
        draft.setdefault("page_relevance", 5)
        draft.setdefault("content_type", "final_content")
        draft.setdefault("should_drop", False)
        return draft
