"""PageQualifier — gate between retrieval and drafting.

Before the (expensive) per-(intent, page) drafter call runs, the qualifier
decides whether the page is actually a good target for the intent. The goal
is to eliminate weak candidates before the drafter has a chance to produce
a destructive/wrong edit.

Two-pass design:
  1. Deterministic checks (free, no LLM): if intent.old_value is set, check
     whether it appears verbatim in page.full_content (after normalizing ordinal
     suffixes, e.g. "30th"→"30"). This is the strongest possible signal —
     phrase-king behavior. A long specific old_value (≥20 chars) found verbatim
     qualifies immediately with score 10.
  2. LLM judgment (one cheap call): when the deterministic check is inconclusive
     (no old_value, short old_value, or old_value not found), an LLM scores 0-10
     how well the page fits the intent's subject. Threshold is 5 to qualify.
     When old_value was set but not found verbatim, the LLM score is capped at 6
     (reflecting uncertainty) and the prompt warns the LLM to score conservatively.
     The drafter is the true safety gate for replace operations.

Returns:
    {
        "qualified": bool,
        "page_fit_score": int (0-10),
        "old_value_found": bool,
        "matched_phrase": str | None,
        "why": str
    }
"""
import json
import logging
import os
import re
from typing import Any, Dict, Optional

from agents import Agent, Runner

logger = logging.getLogger(__name__)

QUALIFIER_MODEL = os.getenv("JARVIS_QUALIFIER_MODEL", "gpt-5.4-nano").strip()
QUALIFIER_MAX_TOKENS = int(os.getenv("JARVIS_QUALIFIER_MAX_TOKENS", "400"))
QUALIFIER_FIT_THRESHOLD = int(os.getenv("JARVIS_QUALIFIER_FIT_THRESHOLD", "5"))


def _norm(text: str) -> str:
    """Whitespace-normalize, lowercase, and strip ordinal suffixes for verbatim presence checks.

    Strips ordinal suffixes so "July 30th" matches "July 30" and "1st" matches "1".
    """
    t = (text or "").strip().lower()
    t = re.sub(r"\b(\d+)(st|nd|rd|th)\b", r"\1", t)
    return re.sub(r"\s+", " ", t).strip()


QUALIFIER_PROMPT = (
    "Score how well a Confluence page fits a change intent. 0-10.\n\n"
    "Input JSON: {intent: {subject, instruction, target_hint, old_value, new_value, action}, "
    "page: {title, headings, content_preview}}\n\n"
    "Scoring:\n"
    "  8-10: page is dedicated to this subject — title+content both confirm\n"
    "  5-7 : page covers this subject as a major section\n"
    "  2-4 : page mentions it in passing but is mainly about something else\n"
    "  0-1 : different subject entirely\n\n"
    "Rules: title overlap alone = max 3. Be conservative when unsure — score lower.\n\n"
    "Return JSON only: {\"page_fit_score\": <0-10>, \"why\": \"<one sentence>\"}"
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

    # ── Pre-check: structural rejects (no LLM needed) ─────────────────────
    # Template/scaffold pages must never be edited by the pipeline regardless of content match.
    if re.search(r"\btemplate\b", page_title, re.IGNORECASE):
        logger.info(
            "Qualifier: '%s' REJECTED — template/scaffold page (auto-reject, no LLM needed)",
            page_title,
        )
        return {
            "qualified": False,
            "page_fit_score": 0,
            "old_value_found": False,
            "matched_phrase": None,
            "why": "Template or scaffold page — not a valid edit target for the pipeline",
            "old_value_missing": False,
        }

    # ── Phase 1: deterministic checks ─────────────────────────────────────
    # Check if old_value is present verbatim on the page (after ordinal-suffix normalization).
    old_value_found = False
    _old_value_missing = False  # True when old_value was set but not found; caps LLM score to 6
    if old_value and len(old_value) >= 3:
        norm_content = _norm(full_content)
        norm_old = _norm(old_value)
        if norm_old in norm_content:
            old_value_found = True

            # A highly specific old_value (≥20 chars) found verbatim is an unambiguous match —
            # qualify immediately without spending an LLM call.
            # Shorter phrases (e.g. "Q4", "10 kg") might appear incidentally anywhere on the page,
            # so fall through to LLM scoring to confirm topical relevance.
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

        else:
            # old_value was specified but NOT found verbatim (even after ordinal normalization).
            # Do NOT auto-reject — fall through to LLM scoring so conceptually relevant pages
            # (e.g. page has "July 30" when intent has "July 30th" after another variation, or
            # the concept is present under different wording) can still reach the drafter.
            # The drafter is the true safety gate: it will say applies=false if unrelated, or
            # downgrade to append if replace anchor can't be found.
            # We cap the LLM score at 6 to signal this uncertainty.
            _old_value_missing = True
            logger.debug(
                "Qualifier: old_value '%s' not found verbatim on '%s' — falling to LLM (score capped at 6)",
                old_value, page_title,
            )

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

    # When old_value was expected but not found verbatim, tell the LLM to be extra conservative.
    # The drafter will decide the exact edit mode (replace vs append) if this page qualifies.
    if _old_value_missing:
        page_repr_content = (
            f"[QUALIFIER NOTE: intent.old_value='{old_value}' was NOT found verbatim in this page. "
            f"Score conservatively — only qualify if this page is clearly the right conceptual target "
            f"for this change. The drafter will handle finding the exact anchor text.]\n\n"
            + page_repr_content
        )

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

    agent = Agent(
        name="PageQualifier",
        model=QUALIFIER_MODEL,
        instructions=QUALIFIER_PROMPT,
    )

    try:
        result = await Runner.run(agent, json.dumps(payload, ensure_ascii=False))
        raw = result.final_output
        if isinstance(raw, str):
            data = json.loads(raw)
        elif isinstance(raw, dict):
            data = raw
        else:
            data = {}
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

    # Cap score at 6 when old_value was set but not found verbatim — the LLM cannot be
    # sure this is the right page without seeing the exact text to replace.
    if _old_value_missing:
        fit = min(fit, 6)

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
        "old_value_missing": _old_value_missing,
    }
