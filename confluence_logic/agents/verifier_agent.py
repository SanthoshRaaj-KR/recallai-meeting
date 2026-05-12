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
VERIFIER_MAX_TOKENS = int(os.getenv("JARVIS_VERIFIER_MAX_TOKENS", "400"))

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

VERIFIER_SYSTEM_PROMPT = (
    "You are a quality verification agent for a meeting intelligence system. "
    "Your job is to assess whether a proposed Confluence change is well-supported by the meeting transcript.\n\n"
    "You receive a JSON payload with:\n"
    "- draft: the proposed change (change_type, page_title, section_heading, before_content, after_content, rationale)\n"
    "- transcript_excerpt: the last 3000 characters of the meeting transcript\n"
    "- current_page_content: the current content of the target Confluence page section\n\n"
    "Return a JSON object with EXACTLY these keys:\n"
    "{\n"
    '  "confidence": "high" | "medium" | "low",\n'
    '  "risk": "safe" | "review" | "risky",\n'
    '  "verifier_note": string (1-2 sentences explaining your assessment),\n'
    '  "transcript_evidence": [string, ...] (1-3 verbatim quotes from the transcript that support this proposal)\n'
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
                                "transcript_excerpt": transcript_text[-3000:],
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
        # Enrich — never drop the card (PIPE-03)
        draft["confidence"] = raw.get("confidence") or "low"
        draft["risk"] = raw.get("risk") or "safe"
        draft["verifier_note"] = raw.get("verifier_note") or ""
        draft["transcript_evidence"] = raw.get("transcript_evidence") or []
        return draft
    except Exception as exc:
        logger.warning("Verifier failed for page %s: %s", draft.get("page_id"), exc)
        draft.setdefault("confidence", "low")
        draft.setdefault("risk", "safe")
        draft.setdefault("verifier_note", "")
        draft.setdefault("transcript_evidence", [])
        return draft
