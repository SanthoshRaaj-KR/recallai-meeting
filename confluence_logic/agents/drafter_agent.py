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
DRAFTER_TRANSCRIPT_BUDGET = int(os.getenv("JARVIS_DRAFTER_TRANSCRIPT_BUDGET", "8000"))


def _build_meeting_context(
    transcript_text: str,
    facts: Any = None,
    summary_json: Optional[Dict[str, Any]] = None,
) -> str:
    """Build a full meeting context for the drafter.

    Combines a structured meeting brief (from extracted facts that cover the
    ENTIRE transcript) with a transcript excerpt for raw quotes. This ensures
    the drafter never misses important decisions regardless of meeting length.
    """
    parts: list = []

    # ── 1. Structured meeting brief from extracted facts ──────────────────
    # The fact extraction agent already processed the FULL transcript (with
    # overlap chunking for long meetings), so these fields are complete.
    brief_lines: list = []
    if facts is not None:
        decisions = getattr(facts, "decisions", []) or []
        if decisions:
            brief_lines.append("DECISIONS:\n" + "\n".join(f"- {d}" for d in decisions[:15]))

        action_items = getattr(facts, "action_items", []) or []
        if action_items:
            brief_lines.append("ACTION ITEMS:\n" + "\n".join(f"- {a}" for a in action_items[:15]))

        new_reqs = getattr(facts, "new_requirements", []) or []
        if new_reqs:
            brief_lines.append("NEW REQUIREMENTS:\n" + "\n".join(f"- {r}" for r in new_reqs[:10]))

        doc_updates = getattr(facts, "doc_worthy_updates", []) or []
        if doc_updates:
            brief_lines.append("DOC-WORTHY UPDATES:\n" + "\n".join(f"- {u}" for u in doc_updates[:10]))

        # Include ALL change intents so the drafter understands the full scope
        change_intents = getattr(facts, "change_intents", []) or []
        if change_intents:
            intent_strs = []
            for ci in change_intents[:20]:
                subj = getattr(ci, "subject", "") or ""
                instr = getattr(ci, "instruction", "") or ""
                old = getattr(ci, "old_value", "") or ""
                new = getattr(ci, "new_value", "") or ""
                line = f"- {instr or subj}"
                if old and new:
                    line += f" ('{old}' → '{new}')"
                intent_strs.append(line)
            brief_lines.append("ALL CHANGES FROM MEETING:\n" + "\n".join(intent_strs))

    # Also pull from summary_json if available (pre-computed meeting summary)
    if summary_json and not brief_lines:
        if summary_json.get("decisions"):
            brief_lines.append("DECISIONS:\n" + "\n".join(f"- {d}" for d in summary_json["decisions"][:10]))
        if summary_json.get("action_items"):
            items = [
                (a.get("description") or str(a)) if isinstance(a, dict) else str(a)
                for a in summary_json["action_items"][:10]
            ]
            brief_lines.append("ACTION ITEMS:\n" + "\n".join(f"- {a}" for a in items))
        if summary_json.get("key_topics"):
            brief_lines.append("KEY TOPICS: " + ", ".join(summary_json["key_topics"][:10]))

    if brief_lines:
        parts.append("[MEETING BRIEF — extracted from full transcript]\n" + "\n\n".join(brief_lines))

    # ── 2. Transcript excerpt for raw quotes and nuance ───────────────────
    text = (transcript_text or "").strip()
    budget = DRAFTER_TRANSCRIPT_BUDGET
    if len(text) <= budget:
        parts.append("[FULL TRANSCRIPT]\n" + text)
    else:
        # head + tail so the drafter sees both opening context and recent discussion
        head_budget = budget // 4
        tail_budget = budget - head_budget
        excerpt = text[:head_budget] + "\n[... middle omitted ...]\n" + text[-tail_budget:]
        parts.append("[TRANSCRIPT EXCERPT]\n" + excerpt)

    return "\n\n".join(parts)

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

DRAFTER_SYSTEM_PROMPT = (
    "You are a Confluence documentation drafter. Given meeting context and a Confluence page, "
    "propose one specific, formal change to that page — but ONLY if the page is actually the right target.\n\n"
    "Input JSON keys:\n"
    "- page.relevant_content: the actual text currently on the Confluence page\n"
    "- facts: structured decisions/actions from the meeting\n"
    "- meeting_context: full meeting brief (decisions, action items, requirements) + transcript excerpt\n\n"

    "STEP 1 — SUITABILITY CHECK (do this before drafting):\n"
    "Read page.relevant_content and ask: 'Does this page cover the same subject as the meeting change?'\n"
    "If the page content is unrelated or only loosely related to the meeting facts, "
    "return {\"change_type\": \"skip\", \"page_id\": null, \"page_title\": \"\", "
    "\"section_heading\": null, \"before_content\": null, \"after_content\": null, "
    "\"rationale\": \"Page is not a relevant target for this change.\"}.\n"
    "Prefer no proposal over a wrong proposal.\n\n"

    "Return a JSON object with EXACTLY these keys:\n"
    "{\n"
    '  "change_type": "edit" | "title" | "delete" | "create" | "skip",\n'
    '  "page_id": string | null,\n'
    '  "page_title": string,\n'
    '  "section_heading": string | null,\n'
    '  "before_content": string | null,\n'
    '  "after_content": string | null,\n'
    '  "rationale": string\n'
    "}\n\n"

    "CONTENT RULES — follow precisely:\n\n"
    "before_content:\n"
    "- Copy the EXACT 1-5 lines from page.relevant_content that will be changed.\n"
    "- Keep short and specific — used to locate the exact wrong text.\n"
    "- Do NOT paste the entire page. Only the specific sentence or bullet being corrected.\n"
    "- Set to null for creates or title-only changes.\n\n"

    "after_content — CRITICAL CONTENT RULE:\n"
    "- Write FINAL, PUBLISHABLE page documentation. This text is written directly to Confluence.\n"
    "- Content must be factual, third-person, max 10 lines.\n"
    "- FORBIDDEN — never write these in after_content:\n"
    "  * 'Keep X as the primary subject'\n"
    "  * 'Include X only as comparison'\n"
    "  * 'Add a clear differentiation section'\n"
    "  * 'Maintain a professional tone'\n"
    "  * 'This page should focus on...'\n"
    "  * Any instruction to a writer rather than content for a reader\n"
    "- If facts are thin, write a short factual stub — never pad with instructions.\n"
    "- Use markdown: ## headings, **bold** for key terms, - for bullet lists.\n\n"

    "section_heading:\n"
    "- The exact heading name of the section being changed on the page.\n\n"
    "rationale:\n"
    "- One sentence: why this change is needed, citing the meeting decision.\n\n"

    "CONSTRAINTS:\n"
    "- If page_id is not null, change_type must be edit, title, delete, or skip — NEVER create.\n"
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
) -> Optional[Dict[str, Any]]:
    """Draft one change proposal for a candidate page.

    Returns None when the drafter determines the page is not a relevant target.
    Callers must check for None and skip verification/persistence.
    """
    page_id = page.get("page_id")
    page_title = page.get("title") or page.get("page_title") or "Unknown Page"

    drafter_input = json.dumps(
        {
            "page": {
                "page_id": page_id,
                "title": page_title,
                "section_heading": page.get("section_heading") or page.get("heading"),
                "relevant_content": (page.get("relevant_content") or "")[:2000],
                "selection_reason": page.get("source") or "rag_retrieval",
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
        return None


def _normalize_draft(
    data: Dict[str, Any],
    page_id: Optional[str],
    page_title: str,
) -> Optional[Dict[str, Any]]:
    """Normalize and validate a raw draft dict from the LLM.

    Returns None when the drafter signals the page is not a relevant target (skip).
    Enforces the change_type constraint: existing pages (page_id not None)
    must only use edit/title/delete — never create (D-09).
    Zero-RAG fallback pages (page_id=None) may use create.
    """
    change_type = str(data.get("change_type") or "edit").strip().lower()

    # Drafter explicitly skipped this page — not a relevant target
    if change_type == "skip":
        logger.info("DrafterAgent: skipping page '%s' — not a relevant target", page_title)
        return None

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


# ---------------------------------------------------------------------------
# Per-(intent, page) drafter — the new primary flow for the proposed-changes pipeline
# ---------------------------------------------------------------------------

INTENT_DRAFTER_PROMPT = (
    "You are a precise Confluence change drafter. You are given ONE structured change intent "
    "and ONE Confluence page. Decide if this page needs to be updated for this intent, and if "
    "yes, produce the EXACT edit.\n\n"

    "Inputs (JSON):\n"
    "- intent: {instruction, subject, target_hint, old_value, new_value, action, rationale}\n"
    "- page: {page_id, title, full_content, available_headings, section_content_map}\n"
    "    * full_content: the full live text of the page (read this carefully)\n"
    "    * available_headings: list of section heading strings actually on the page\n"
    "    * section_content_map: object mapping each heading to a short preview of the content under it. "
    "Use this to pick the section that ALREADY discusses the intent's subject, not just one whose name matches.\n"
    "- meeting_context: full meeting brief (all decisions, action items, requirements from the entire meeting) "
    "plus a transcript excerpt. Use this to understand the FULL scope of what was discussed.\n\n"

    "═══════════════════════════════════════════════\n"
    "STEP 1 — Does this intent apply to this page? (STRICT)\n"
    "═══════════════════════════════════════════════\n"
    "Read page.full_content carefully. Then ask:\n"
    "  a) Does this page contain content about intent.subject? (direct, not just topically adjacent)\n"
    "  b) If intent.old_value is set: does that exact string appear in page.full_content?\n"
    "  c) If intent.target_hint is set: does it match the page's title or any heading?\n\n"
    "Apply this STRICT rule (when in doubt, return applies=false):\n"
    "- (b) is true (old_value found verbatim on page) → applies=true. STRONGEST SIGNAL.\n"
    "- (b) is set but old_value is NOT on page:\n"
    "    • action is 'add'                                   → applies=true (additive, no anchor needed).\n"
    "    • action is 'replace' AND page is CLEARLY the right target (rule (a) strongly true —\n"
    "      page is dedicated to this subject, not merely mentioning it):\n"
    "      → applies=true, but you MUST use edit_mode='append' and set before_content=null.\n"
    "        You cannot replace text that isn't there verbatim, but the decision IS relevant.\n"
    "        Add the new information as a new bullet or sentence to the appropriate section.\n"
    "    • action is 'replace' AND page is only loosely related                → applies=false.\n"
    "- (b) is empty AND (a) is unambiguously true (page is dedicated to this subject) → applies=true.\n"
    "- (b) is empty AND only (a) is loosely true (page mentions subject in passing) → applies=false.\n"
    "- Only (c) loosely matches (title/heading word overlap only, content unrelated) → applies=false.\n"
    "- Page covers a different subject → applies=false.\n\n"
    "IMPORTANT: A topical match on title is NOT enough. The page CONTENT must be about this subject.\n"
    "Prefer applies=false when uncertain — a missed proposal is recoverable; a wrong edit is not.\n\n"

    "═══════════════════════════════════════════════\n"
    "STEP 2 — If applies=true, draft the precise edit\n"
    "═══════════════════════════════════════════════\n"
    "Decide change_type:\n"
    "- 'edit'   : modifying body content on this existing page\n"
    "- 'title'  : renaming this page (after_content = new title string only)\n"
    "- 'delete' : removing a section or the whole page (after_content = null)\n"
    "- 'create' : ONLY if page_id is null and action is 'create'\n\n"

    "For 'edit' change_type, ALSO classify the edit_mode (CRITICAL — execution dispatches on this):\n"
    "- 'replace'        : replace a SPECIFIC existing string on the page (you found verbatim old text). "
    "before_content MUST contain that exact text. The execution will find-and-replace surgically. "
    "Use this only when intent.old_value appears verbatim in page.full_content.\n"
    "- 'append'         : add a NEW sentence/bullet/paragraph to an existing section. before_content "
    "MUST be null. Use this for purely additive updates (e.g. adding a new fact to a runbook).\n"
    "- 'create_section' : add a brand-new section with its own heading. before_content MUST be null, "
    "section_heading MUST be the NEW heading to create. Use this when the topic belongs on the page "
    "but no existing section covers it.\n"
    "PROHIBITED: do NOT pick 'replace' if intent.old_value is empty or absent from the page. That would "
    "force a destructive section-replace at execution time and is the #1 cause of bad edits.\n\n"

    "before_content (CRITICAL — read every rule):\n"
    "- ONLY copy text that LITERALLY EXISTS in page.full_content. Search the full_content string "
    "character-by-character for the exact phrase you want to use. If you cannot find it verbatim, "
    "do NOT use it as before_content.\n"
    "- NEVER paraphrase, reword, or compose before_content — it must be a literal substring of full_content.\n"
    "- If intent.old_value is set AND appears verbatim in full_content: use that exact string.\n"
    "- If the text you want to replace is NOT in full_content: set before_content=null and "
    "edit_mode='append'. NEVER invent a plausible-sounding snippet.\n"
    "- For purely additive changes (new bullet/section): before_content=null.\n"
    "- For title-only/delete-page changes: null.\n\n"

    "═══════════════════════════════════════════════\n"
    "QUALITY RULES — read before writing after_content\n"
    "═══════════════════════════════════════════════\n"

    "RULE 1 — TRANSCRIPT GROUNDING (most critical):\n"
    "Every sentence in after_content must be DIRECTLY traceable to a specific statement made in the "
    "transcript. Do NOT add implementation details, technical specifics, methodology, or best practices "
    "that were not explicitly spoken in the meeting.\n"
    "  WRONG: transcript says 'add encryption at rest' → you write 'Enable AES-256 encryption, integrate "
    "with key management service, rotate keys quarterly, and encrypt all backups and replicas'\n"
    "  CORRECT: transcript says 'add encryption at rest' → you write '- Encryption at rest required for "
    "transcript storage (required before beta release)'\n"
    "If you cannot point to a direct quote in the transcript for a sentence, remove that sentence.\n\n"

    "RULE 2 — MINIMAL CHANGE:\n"
    "Make the smallest change that accurately captures the decision. Maximum 3 lines for new content. "
    "Match the scope of what was said, not what could theoretically be said about the topic.\n"
    "  WRONG: transcript says 'move to async Kafka pipeline' → you write a 5-line architecture paragraph "
    "describing consumer topology, event stages, partitioning strategy, and delivery guarantees\n"
    "  CORRECT: '- Transcript processing moving to async event-driven pipeline (Kafka)'\n\n"

    "RULE 3 — STATE CONSISTENCY:\n"
    "When a meeting decision cancels, deprioritizes, or supersedes an existing documented item, "
    "REPLACE the old entry — do not annotate alongside it. Adding a note next to the old item creates "
    "contradictory state in the document (the item is both listed and deprioritized).\n"
    "  WRONG: page has 'Mobile app' in Q4 roadmap → you append '- Mobile app deprioritized' below it\n"
    "  CORRECT: replace 'Mobile app' with 'Mobile app — deprioritized until next year'\n\n"

    "RULE 4 — SINGLE BEST SECTION:\n"
    "For this (intent, page) pair, update exactly ONE section — the one that best fits the change. "
    "Do not propose updates to multiple sections of the same page for the same intent.\n\n"

    "RULE 5 — SECTION SEMANTICS:\n"
    "'Known Limitations' and 'Known Issues' sections document CURRENT gaps or shortcomings — not "
    "requirements, mandates, or new features. If the change adds a requirement or prerequisite, "
    "write it under 'Requirements', 'Planned Work', or 'Prerequisites'. If no such section exists, "
    "use edit_mode='create_section' with an appropriate heading rather than misusing 'Known Limitations'.\n\n"

    "after_content (CRITICAL — FINAL PAGE CONTENT, NOT INSTRUCTIONS):\n"
    "- Write the EXACT replacement text that should appear on the page.\n"
    "- This text is written DIRECTLY to Confluence — it must read as published documentation.\n"
    "- FORBIDDEN — never write any of:\n"
    "  * 'Keep X as the primary subject'\n"
    "  * 'Include X only as comparison'\n"
    "  * 'Add a clear section about...'\n"
    "  * 'Maintain a professional tone'\n"
    "  * 'This page should focus on...'\n"
    "  * 'Ensure the content covers...'\n"
    "  * Anything that reads as direction to a writer rather than content for a reader\n"
    "- CORRECT examples:\n"
    "  * 'Beta release date: **August 20** (moved from July 30 to allow infrastructure stabilization)'\n"
    "  * '**Owner:** Priya (security compliance)'\n"
    "  * '- Encryption at rest required for transcript storage before beta'\n"
    "- Use markdown: ## for new section headings, **bold** for key terms, - for bullets.\n"
    "- Maximum 3 lines for new content; for replace, match the length of the old content.\n\n"

    "section_heading:\n"
    "- Use an EXACT name from page.available_headings if the change targets a specific section.\n"
    "- Look at page.section_content_map to pick the section whose content already covers the subject. "
    "Example: if the intent is about a database version and section_content_map shows that section "
    "'Infrastructure' contains the current version string, pick 'Infrastructure'.\n"
    "- If the edit touches the very first paragraph before any heading, use null.\n"
    "- For title changes: null.\n\n"

    "rationale:\n"
    "- One sentence citing the meeting decision and why this page needs this edit.\n\n"

    "Return JSON with EXACTLY these keys (no extras, no wrappers):\n"
    "{\n"
    '  "applies": true | false,\n'
    '  "reason": string (1 sentence — required when applies=false; brief justification when applies=true),\n'
    '  "change_type": "edit" | "title" | "delete" | "create" | null,\n'
    '  "edit_mode": "replace" | "append" | "create_section" | null,  // only meaningful when change_type=="edit"\n'
    '  "page_id": string | null,\n'
    '  "page_title": string,\n'
    '  "section_heading": string | null,\n'
    '  "before_content": string | null,\n'
    '  "after_content": string | null,\n'
    '  "rationale": string\n'
    "}\n"
    "Return only valid JSON — no markdown wrapper, no explanation."
)


async def _run_intent_drafter(
    intent: Any,  # ChangeIntent
    page: Dict[str, Any],
    transcript_text: str,
    *,
    max_page_chars: int = 8000,
    facts: Any = None,
    summary_json: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """Draft a precise change for ONE (intent, page) pair.

    Returns a normalized proposal dict when the page is relevant and a change is needed.
    Returns None when the page is not a relevant target for this intent.
    """
    page_id = page.get("page_id")
    page_title = page.get("title") or page.get("page_title") or "Unknown Page"
    full_content = (page.get("full_content") or page.get("relevant_content") or "")[:max_page_chars]
    available_headings = page.get("available_headings") or []

    drafter_input = json.dumps(
        {
            "intent": {
                "instruction": getattr(intent, "instruction", "") or "",
                "subject": getattr(intent, "subject", "") or "",
                "target_hint": getattr(intent, "target_hint", "") or "",
                "old_value": getattr(intent, "old_value", "") or "",
                "new_value": getattr(intent, "new_value", "") or "",
                "action": getattr(intent, "action", "replace") or "replace",
                "rationale": getattr(intent, "rationale", "") or "",
            },
            "page": {
                "page_id": page_id,
                "title": page_title,
                "full_content": full_content,
                "available_headings": available_headings[:30],
                "section_content_map": page.get("section_content_map") or {},
            },
            "meeting_context": _build_meeting_context(transcript_text, facts=facts, summary_json=summary_json),
        },
        ensure_ascii=False,
    )

    agent = Agent(
        name=f"IntentDrafter-{page_id or 'new'}",
        model=DRAFTER_MODEL,
        instructions=INTENT_DRAFTER_PROMPT,
    )

    try:
        result = await Runner.run(agent, drafter_input)
        raw = result.final_output
        if isinstance(raw, str):
            data = json.loads(raw)
        elif isinstance(raw, dict):
            data = raw
        else:
            return None
    except Exception as exc:
        logger.warning(
            "IntentDrafter failed for page %s / intent %r: %s",
            page_id, getattr(intent, "subject", "") or "?", exc,
        )
        return None

    return _normalize_intent_draft(data, page_id, page_title, intent=intent)


def _normalize_intent_draft(
    data: Dict[str, Any],
    page_id: Optional[str],
    page_title: str,
    *,
    intent: Any = None,
) -> Optional[Dict[str, Any]]:
    """Normalize an intent-drafter LLM output into a clean proposal dict.

    Returns None when applies=false (drafter rejected the page).
    Enforces the edit_mode safety contract:
      - replace requires before_content (otherwise downgrade to append)
      - append with before_content: strip before_content (it's extraneous context, not an anchor)
      - unknown edit_mode is inferred from before_content presence

    Extracted as a standalone function for unit testing.
    """
    if not isinstance(data, dict):
        return None

    applies = bool(data.get("applies"))
    if not applies:
        logger.info(
            "IntentDrafter: page '%s' does not apply to intent '%s' — %s",
            page_title, getattr(intent, "subject", "?") if intent else "?",
            data.get("reason") or "no reason given",
        )
        return None

    change_type = str(data.get("change_type") or "edit").strip().lower()
    if change_type not in {"edit", "title", "delete", "create"}:
        change_type = "edit"
    # Existing pages cannot produce 'create' proposals
    if page_id and change_type == "create":
        change_type = "edit"
    # Pages with no page_id should produce 'create' (zero-RAG path)
    if not page_id and change_type not in {"create"}:
        change_type = "create"

    before_content = data.get("before_content") or None
    after_content = data.get("after_content") or None

    # Normalize edit_mode. Only meaningful for change_type=="edit".
    edit_mode: Optional[str] = None
    if change_type == "edit":
        raw_mode = str(data.get("edit_mode") or "").strip().lower()
        if raw_mode in {"replace", "append", "create_section"}:
            edit_mode = raw_mode
        else:
            edit_mode = "replace" if before_content else "append"

        # SAFETY GUARD: replace requires before_content as an anchor. Without it,
        # downgrade to append so execution cannot fall into destructive section-replace.
        if edit_mode == "replace" and not before_content:
            logger.warning(
                "IntentDrafter for '%s': edit_mode='replace' but before_content is empty — "
                "downgrading to 'append' to prevent destructive section-replace",
                page_title,
            )
            edit_mode = "append"

        # SAFETY GUARD 2 (revised): if drafter says 'append' but accidentally included
        # before_content (e.g. existing section text as context), strip it.
        # Upgrading to 'replace' here caused execution failure because the section preview
        # doesn't match Confluence HTML exactly. Append has no anchor — keep it that way.
        if edit_mode == "append" and before_content:
            logger.debug(
                "IntentDrafter for '%s': edit_mode='append' with before_content — "
                "stripping before_content (append needs no anchor)",
                page_title,
            )
            before_content = None

    rationale = data.get("rationale")
    if not rationale and intent is not None:
        rationale = getattr(intent, "instruction", "")
    rationale = rationale or None

    return {
        "change_type": change_type,
        "edit_mode": edit_mode,
        "page_id": page_id,
        "page_title": str(data.get("page_title") or page_title).strip(),
        "section_heading": data.get("section_heading") or None,
        "before_content": before_content,
        "after_content": after_content,
        "rationale": rationale,
    }
