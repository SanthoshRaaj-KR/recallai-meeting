"""VerifierAgent — enriches draft proposals with confidence, risk, and verifier_note. Never drops a card (PIPE-03)."""

import json
import logging
import os
from typing import Any, Dict, List, Optional

from agents import Agent, Runner

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

VERIFIER_MODEL = os.getenv("JARVIS_VERIFIER_MODEL", "gpt-5.4-nano").strip()
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
    '  "content_type": "final_content" | "meta_instruction",\n'
    '  "change_summary": string (≤120 characters; plain-English headline of the change)\n'
    "}\n\n"
    "Phase 10 note: page-existence and token-level grounding are enforced by a "
    "downstream deterministic gate (grounding_gate.check_grounding + "
    "check_page_existence). Do NOT attempt those checks here. This prompt "
    "focuses on LLM judgments only: confidence, risk, evidence quotes, "
    "content type, and a plain-English change_summary for the ProposalCard.\n\n"
    "Confidence guidelines:\n"
    "- 'high': TWO conditions must BOTH be true: (1) the change is directly and explicitly supported "
    "by clear transcript statements, AND (2) after_content contains ONLY what was explicitly stated — "
    "no inferred technical details, no expanded specifics, no implementation details not spoken in the meeting. "
    "A Kafka migration proposal that adds consumer topology details not mentioned in the transcript is at most 'medium'.\n"
    "- 'medium': the change is partially supported — transcript mentions the topic but after_content "
    "includes reasonable inferences or additional context beyond exactly what was said\n"
    "- 'low': weak or ambiguous support; the proposal extrapolates significantly beyond what was discussed\n\n"
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
    "content_type: Is draft.after_content actual page documentation or instructions to a writer?\n"
    "- 'final_content': publishable facts, descriptions, specs, bullet lists of real information\n"
    "  Examples: 'The team uses React for the frontend and FastAPI for the backend', "
    "'- Encryption at rest required before beta', 'Beta release: **August 20**'\n"
    "- 'meta_instruction': writing directives, editorial guidance, or planning notes — NOT real page content\n"
    "  Examples: 'Add a clear differentiation section explaining why X is better', "
    "'Maintain a professional and neutral tone throughout', 'This page should focus on...', "
    "'Ensure the content covers all migration steps', 'Use a structured format with clear headings'\n"
    "If after_content is null (delete/title change), set content_type to 'final_content'.\n\n"
    "change_summary: a single ≤120-character plain-English headline for the ProposalCard. "
    "Examples: 'Reorder onboarding so Login (step 2) runs before Payment (step 3)', "
    "'Replace deprecated React reference with Vue on the Frameworks page'. "
    "Never include markdown, quotes around the whole string, or trailing periods.\n\n"
    "transcript_evidence: 1-3 DISTINCT verbatim quotes copied exactly from the transcript_excerpt. "
    "Never repeat the same quote. If the same statement appears multiple times in the transcript, "
    "include it only once. De-duplicate before returning.\n"
    "Return only valid JSON — no markdown, no explanation, no code blocks."
)

# ---------------------------------------------------------------------------
# Core verification function
# ---------------------------------------------------------------------------


async def _run_verifier(
    draft: Dict[str, Any],
    transcript_text: str,
    page_content: str,
) -> Dict[str, Any]:
    """Enrich a draft card with confidence, risk, verifier_note, transcript_evidence, change_summary.

    Phase 10 (Plan 10-07): page-existence and token-grounding checks have been
    MOVED to grounding_gate.check_page_existence + check_grounding (PROP-V2-01).
    The verifier no longer drops cards for low page_relevance; that
    responsibility lives in the deterministic gate run by _verify_and_persist
    just before persist. The verifier now focuses exclusively on LLM
    judgments (confidence / risk / evidence / content_type / change_summary).
    The ONE legacy drop preserved here is for content_type=="meta_instruction"
    — these are clearly-not-a-real-change cards (editorial directives like
    "make this clearer") that have nothing to do with grounding.

    Never raises. On failure, sets safe defaults and returns (PIPE-03).
    """
    try:
        user_input = json.dumps(
            {
                "draft": draft,
                "transcript_excerpt": transcript_text[-6000:],
                "current_page_content": page_content[:2000],
            },
            ensure_ascii=False,
        )

        agent = Agent(
            name="VerifierAgent",
            model=VERIFIER_MODEL,
            instructions=VERIFIER_SYSTEM_PROMPT,
        )

        result = await Runner.run(agent, user_input)
        raw = result.final_output
        if isinstance(raw, str):
            raw = json.loads(raw)
        elif not isinstance(raw, dict):
            raw = {}

        content_type = str(raw.get("content_type") or "final_content").strip().lower()
        if content_type not in ("final_content", "meta_instruction"):
            content_type = "final_content"

        confidence = raw.get("confidence") or "low"
        risk = raw.get("risk") or "safe"
        verifier_note = raw.get("verifier_note") or ""

        # Downgrade for meta-instruction cards. Per Plan 10-07 the
        # wrong-page downgrade (page_relevance < 4) is REMOVED because the
        # GroundingGate's page-existence check and per-op token-grounding
        # check together cover both "page does not exist" and "tokens
        # never spoken" cases more strictly than this LLM score did.
        if content_type == "meta_instruction":
            confidence = "low"
            risk = "risky"
            verifier_note = (
                "[INSTRUCTION TEXT] after_content contains editorial directives, "
                f"not documentation. {verifier_note}"
            ).strip()

        # Only meta_instruction triggers a verifier-side drop now —
        # everything else is left for GroundingGate to evaluate.
        should_drop = bool(raw.get("should_drop", False))
        if content_type == "meta_instruction":
            should_drop = True

        # Plan 10-07 (D-07 ProposalCard): the verifier emits a ≤120-char
        # change_summary. If the upstream drafter already populated one
        # (StructureAwareDrafter does), keep that and only fill from the
        # verifier when the slot is empty. The downstream
        # _verify_and_persist synthesis stays as a final fallback.
        llm_summary = (raw.get("change_summary") or "").strip()
        if llm_summary:
            llm_summary = llm_summary[:120]
        if not draft.get("change_summary") and llm_summary:
            draft["change_summary"] = llm_summary

        draft["confidence"] = confidence
        draft["risk"] = risk
        draft["verifier_note"] = verifier_note
        draft["transcript_evidence"] = raw.get("transcript_evidence") or []
        draft["content_type"] = content_type
        draft["should_drop"] = should_drop
        return draft
    except Exception as exc:
        logger.warning("Verifier failed for page %s: %s", draft.get("page_id"), exc)
        draft.setdefault("confidence", "low")
        draft.setdefault("risk", "safe")
        draft.setdefault("verifier_note", "")
        draft.setdefault("transcript_evidence", [])
        draft.setdefault("content_type", "final_content")
        draft.setdefault("should_drop", False)
        return draft
