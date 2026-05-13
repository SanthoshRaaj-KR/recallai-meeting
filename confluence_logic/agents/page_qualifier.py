"""PageQualifier — gate between retrieval and drafting.

Before the (expensive) per-(intent, page) drafter call runs, the qualifier
decides whether the page is actually a good target for the intent. The goal
is to eliminate weak candidates before the drafter has a chance to produce
a destructive/wrong edit.

Two-pass design:
  1. Deterministic checks (free, no LLM): if intent.old_value is set, check
     whether it appears verbatim in page.full_content. This is the strongest
     possible signal — phrase-king behavior. When old_value is set but absent
     from the page, the page is disqualified for 'replace' actions.
  2. LLM judgment (one cheap call): when the deterministic check is inconclusive
     (no old_value, or action is additive), an LLM scores 0-10 how well the page
     fits the intent's subject. The threshold is strict (>=5 to qualify) and the
     prompt explicitly biases toward "no" when uncertain.

Returns:
    {
        "qualified": bool,
        "page_fit_score": int (0-10),
        "old_value_found": bool,
        "matched_phrase": str | None,
        "why": str
    }
"""
import asyncio
import json
import logging
import os
import re
from typing import Any, Dict, Optional

from openai import OpenAI

logger = logging.getLogger(__name__)

QUALIFIER_MODEL = os.getenv("JARVIS_QUALIFIER_MODEL", "gpt-5.4-mini").strip()
QUALIFIER_MAX_TOKENS = int(os.getenv("JARVIS_QUALIFIER_MAX_TOKENS", "400"))
QUALIFIER_FIT_THRESHOLD = int(os.getenv("JARVIS_QUALIFIER_FIT_THRESHOLD", "7"))

_openai_client: Optional[OpenAI] = None


def _get_openai_client() -> OpenAI:
    global _openai_client
    if _openai_client is None:
        _openai_client = OpenAI()
    return _openai_client


def _openai_completion_options(model: str, max_tokens: int) -> Dict[str, Any]:
    opts: Dict[str, Any] = {"model": model}
    if model.startswith(("gpt-5", "o1", "o3", "o4")):
        opts["max_completion_tokens"] = max_tokens
    else:
        opts["max_tokens"] = max_tokens
        opts["temperature"] = 0.1
    return opts


def _norm(text: str) -> str:
    """Whitespace-normalize and lowercase for verbatim presence checks."""
    return re.sub(r"\s+", " ", (text or "").strip().lower())


QUALIFIER_PROMPT = (
    "You are a strict page-fit qualifier for a Confluence change pipeline. "
    "Given ONE change intent and ONE candidate page, decide whether the page is the "
    "RIGHT target for this change.\n\n"

    "Input JSON:\n"
    "  intent: {subject, instruction, target_hint, old_value, new_value, action}\n"
    "  page: {title, headings, content_preview}\n\n"

    "SCORING RUBRIC (0-10):\n"
    "  10 — Page is dedicated to this exact subject. Title and content both confirm.\n"
    "  7-9 — Page covers this subject as a major section. Content clearly relates.\n"
    "  4-6 — Page mentions this subject as part of broader content. Maybe a target.\n"
    "  1-3 — Page is in the same general domain but covers a different specific subject.\n"
    "  0   — Page covers a completely different subject. Wrong target.\n\n"

    "STRICT GUIDANCE:\n"
    "- Title overlap is NOT enough. The page CONTENT must be about this subject.\n"
    "- 'PostgreSQL is mentioned somewhere' is NOT enough to say a page about ORM frameworks "
    "is the right target for a PostgreSQL version change.\n"
    "- A vague/generic page in the right domain (e.g. 'Engineering Overview' for a specific "
    "service change) scores 3 or below.\n"
    "- Be conservative — when uncertain, score LOWER. A missed page is recoverable via other "
    "retrieval paths; a wrong edit is not.\n\n"

    "Return JSON with EXACTLY these keys (no extras, no wrapper):\n"
    "{\n"
    "  \"page_fit_score\": <int 0-10>,\n"
    "  \"why\": \"<one sentence rationale citing specific page content vs intent subject>\"\n"
    "}\n"
    "Return only the JSON object. No markdown, no explanation."
)


async def _run_page_qualifier(
    intent: Any,  # ChangeIntent
    page: Dict[str, Any],
) -> Dict[str, Any]:
    """Decide whether (intent, page) should proceed to drafting.

    Phase 1 — deterministic checks. Phase 2 — LLM judgment when inconclusive.
    Always returns a fully-populated dict so downstream code can attach the
    fit_score / why / matched_phrase to the eventual proposal for audit.
    """
    subject = (getattr(intent, "subject", "") or "").strip()
    old_value = (getattr(intent, "old_value", "") or "").strip()
    action = (getattr(intent, "action", "replace") or "replace").strip().lower()
    instruction = (getattr(intent, "instruction", "") or "").strip()

    page_title = page.get("title") or page.get("page_title") or ""
    full_content = page.get("full_content") or page.get("relevant_content") or ""

    # ── Phase 1: deterministic checks ─────────────────────────────────────
    # Check if old_value is present verbatim on the page
    old_value_found = False
    if old_value and len(old_value) >= 3:
        norm_content = _norm(full_content)
        norm_old = _norm(old_value)
        if norm_old in norm_content:
            old_value_found = True
            
            # If the old_value is highly specific (>= 20 chars), it's a guaranteed match.
            # If it's shorter (e.g., 'Fat', '10 kg'), it might just be a common phrase appearing 
            # randomly, so we still require the LLM to verify the page's topical relevance.
            if len(old_value) >= 20:
                logger.info(
                    "Qualifier: '%s' QUALIFIED for intent '%s' — specific old_value '%s' present verbatim",
                    page_title, subject or instruction[:40], old_value,
                )
                return {
                    "qualified": True,
                    "page_fit_score": 10,
                    "old_value_found": True,
                    "matched_phrase": old_value,
                    "why": f"old_value '{old_value}' present verbatim on page",
                }
                
        # old_value set + NOT on page + action is replace → disqualify deterministically.
        # Replace cannot succeed without the old text being there.
        if action == "replace" and not old_value_found:
            logger.info(
                "Qualifier: '%s' REJECTED for intent '%s' — replace action but old_value '%s' "
                "not present on page (cannot replace what isn't there)",
                page_title, subject or instruction[:40], old_value,
            )
            return {
                "qualified": False,
                "page_fit_score": 1,
                "old_value_found": False,
                "matched_phrase": None,
                "why": f"old_value '{old_value}' not found on page; replace action cannot succeed here",
            }

    # ── Phase 2: LLM judgment ─────────────────────────────────────────────
    # Build a compact representation of the page for the LLM
    headings = page.get("available_headings") or []
    section_map = page.get("section_content_map") or {}
    # Trim the content_preview to a fixed budget
    content_preview = full_content[:2500]
    # If section_map exists, prefer it (more compact, structured)
    if section_map:
        section_summary = "\n".join(
            f"  ## {h}: {(section_map.get(h) or '')[:300]}"
            for h in list(section_map.keys())[:15]
        )
        page_repr_content = section_summary or content_preview
    else:
        page_repr_content = content_preview

    payload = {
        "intent": {
            "subject": subject,
            "instruction": instruction,
            "target_hint": (getattr(intent, "target_hint", "") or ""),
            "old_value": old_value,
            "new_value": (getattr(intent, "new_value", "") or ""),
            "action": action,
        },
        "page": {
            "title": page_title,
            "headings": headings[:20],
            "content_preview": page_repr_content,
        },
    }

    try:
        opts = _openai_completion_options(QUALIFIER_MODEL, QUALIFIER_MAX_TOKENS)
        response = await asyncio.to_thread(
            lambda: _get_openai_client().chat.completions.create(
                **opts,
                messages=[
                    {"role": "system", "content": QUALIFIER_PROMPT},
                    {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
                ],
                response_format={"type": "json_object"},
            )
        )
        data = json.loads(response.choices[0].message.content or "{}")
    except Exception as exc:
        logger.warning("Page qualifier LLM call failed for '%s': %s", page_title, exc)
        # Fail-safe: when the qualifier errors, default to LOW fit and qualified=False
        # so weak candidates can't slip through silently on transient failures.
        return {
            "qualified": False,
            "page_fit_score": 0,
            "old_value_found": False,
            "matched_phrase": None,
            "why": f"qualifier error: {exc}",
        }

    fit = data.get("page_fit_score")
    try:
        fit = int(fit)
        fit = max(0, min(10, fit))
    except (TypeError, ValueError):
        fit = 0

    why = (data.get("why") or "").strip() or "no rationale"
    qualified = fit >= QUALIFIER_FIT_THRESHOLD

    log_fn = logger.info if qualified else logger.debug
    log_fn(
        "Qualifier: '%s' %s for intent '%s' (fit=%d/10) — %s",
        page_title,
        "QUALIFIED" if qualified else "rejected",
        subject or instruction[:40],
        fit, why,
    )

    return {
        "qualified": qualified,
        "page_fit_score": fit,
        "old_value_found": old_value_found,
        "matched_phrase": old_value if old_value_found else None,
        "why": why,
    }
