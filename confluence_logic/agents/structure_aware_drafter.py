"""Phase 10 structure-aware drafter (PROP-V2-02, PROP-V2-06, D-03/D-06).

Operates on the parsed AST emitted by :mod:`confluence_logic.agents.page_parser`;
emits one of six D-02 instruction shapes via a constrained JSON schema. Never
emits free-form prose for reorder ops — the ``StructuredOperation`` Literal
plus a post-validate guard make that structurally impossible.

This module REPLACES ``drafter_agent._run_intent_drafter`` for the
structure-aware path. The legacy ``_run_drafter`` / ``_run_intent_drafter``
are kept in ``drafter_agent.py`` for backward-compat fallback per
10-RESEARCH.md ("Deprecated/outdated" line 1058). The rewire lives in
Plan 10-07.

Public surface (imported by tests + the future Plan 10-07 orchestrator):

  * :class:`StructureAwareDrafterInput`  — input contract
  * :class:`StructuredOperation`         — output contract (constrained Literal)
  * :func:`draft_operation`              — async entrypoint
  * :data:`STRUCTURE_AWARE_DRAFTER_PROMPT` — module-level system prompt

Implementation notes:

* The agent is constructed once at import time as a module-level singleton —
  same pattern as ``fact_extraction_agent._fact_agent``. Constructing per call
  would re-fetch tool schemas on every draft.
* ``Runner.run`` is patched in tests via the module-level reference
  ``confluence_logic.agents.structure_aware_drafter.Runner`` — DO NOT rename
  this import without updating ``test_structure_aware_drafter.py``.
* Per D-10 model ceiling, the drafter defaults to ``gpt-5-mini`` and reads
  ``JARVIS_AGENT_MODEL`` for operator override.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

from agents import Agent, AgentOutputSchema, Runner

# Re-export the canonical schemas from ``core.schemas`` so callers can do
# ``from confluence_logic.agents.structure_aware_drafter import
# StructuredOperation`` and tests can patch this module directly.
from confluence_logic.core.schemas import (
    StructureAwareDrafterInput,
    StructuredOperation,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Module-level configuration
# ---------------------------------------------------------------------------

# D-10: GPT-5 family ceiling per CLAUDE.md; default to gpt-5-mini to keep the
# per-card cost low. Operators may override via JARVIS_AGENT_MODEL.
_AGENT_MODEL = os.getenv("JARVIS_AGENT_MODEL", "gpt-5-mini").strip()

# Truncate the AST summary that we feed into the prompt so we never blow past
# the model context window on very large pages. 4_000 chars covers ~1_000
# tokens of compact heading + ordered-list-item listing — empirically enough
# for the StructureAwareDrafter to pick the right action and indices.
_AST_SUMMARY_MAX_CHARS = int(os.getenv("JARVIS_DRAFTER_AST_SUMMARY_MAX_CHARS", "4000"))

# Truncate the transcript window similarly. The upstream pipeline already
# extracts a relevant slice via ``drafter_agent._find_relevant_transcript_window``;
# we apply a defensive secondary cap here.
_TRANSCRIPT_MAX_CHARS = int(os.getenv("JARVIS_DRAFTER_TRANSCRIPT_MAX_CHARS", "4000"))


# ---------------------------------------------------------------------------
# Module-level system prompt
# ---------------------------------------------------------------------------

STRUCTURE_AWARE_DRAFTER_PROMPT = (
    "You are the StructureAwareDrafter for Phase 10 of the meeting-to-Confluence "
    "pipeline. You receive ONE ChangeIntent + the typed AST of ONE qualified "
    "Confluence page + a transcript window. You MUST emit EXACTLY ONE "
    "StructuredOperation matching the JSON schema. You MUST NOT emit "
    "free-form prose, markdown, or any text outside the schema.\n\n"
    "═══════════════════════════════════════════════\n"
    "Action selection — pick exactly one\n"
    "═══════════════════════════════════════════════\n\n"
    "1. REORDER — emit action='reorder' when the intent describes moving an "
    "existing ordered-list item to a different position. Trigger phrases: "
    "'move X before/after Y', 'swap X and Y', 'X should come before Y', "
    "'X should be step N'. You MUST set:\n"
    "   * section_heading   = the existing Heading.text on the AST that contains the OrderedList\n"
    "   * from_index        = the source 0-based li.index from page_ast OrderedList\n"
    "   * to_index          = the destination 0-based index\n"
    "   You MUST NOT set new_text or new_content for a reorder — the dispatcher "
    "reconstructs the after-section by swapping the live HTML's <li>s. Any prose "
    "you emit here is silently discarded and may cause Pitfall 5 grounding "
    "violations downstream.\n\n"
    "2. REPLACE — emit action='replace' when the intent replaces specific text "
    "INSIDE an existing section. You MUST set:\n"
    "   * section_heading = the existing Heading.text\n"
    "   * old_text        = an EXACT substring from the AST's current section content\n"
    "   * new_text        = the replacement; every content-bearing token MUST be in "
    "the transcript_window OR current page content. Do NOT invent terminology.\n\n"
    "3. INSERT_AFTER — emit action='insert_after' when the intent ADDS new "
    "content under an existing heading. You MUST set:\n"
    "   * section_heading = the existing Heading.text\n"
    "   * anchor_text     = the last paragraph or list item already in that section\n"
    "   * new_text        = drawn from intent.verbatim_content when present; "
    "otherwise from the transcript_window verbatim.\n\n"
    "4. CREATE_SECTION — emit action='create_section' when the target section "
    "does NOT exist in the AST but the page is the right target. You MUST set:\n"
    "   * parent_heading = an existing top-level Heading.text from the AST (often the page H1)\n"
    "   * new_heading    = the heading to create\n"
    "   * new_content    = the body content (markdown or HTML), grounded in transcript_window\n"
    "   Do NOT invent a parent_heading — it must literally appear in the AST.\n\n"
    "5. DELETE_SECTION — emit action='delete_section' when the intent removes a "
    "whole section. You MUST set section_heading = the existing Heading.text.\n\n"
    "6. CREATE_PAGE — emit action='create_page' ONLY when the intent describes a "
    "brand-new page that has no candidate (page_meta is empty / page_meta.page_id "
    "is missing). You MUST set title and content; parent_page_id and space_key "
    "are optional and drawn from page_meta when present.\n\n"
    "7. SKIP — emit action='skip' with a reason when no operation can be grounded "
    "against the AST + transcript. Examples of reasons: 'no matching section', "
    "'intent.subject not present on this page', 'cannot identify anchor_text'.\n\n"
    "═══════════════════════════════════════════════\n"
    "Universal grounding rules\n"
    "═══════════════════════════════════════════════\n\n"
    " * ALWAYS populate change_summary with a ≤120-char plain-English sentence "
    "describing the change. Example: 'Reorder onboarding so Login runs before "
    "Payment'. Example: 'Replace React with Vue in the Frameworks section'.\n"
    " * page_id is always taken from page_meta — never invent. Leave it null in "
    "your output; the drafter post-stamps it.\n"
    " * section_heading / parent_heading / anchor_text MUST be literal strings "
    "from the AST (Heading.text or paragraph/list-item text). Never invent.\n"
    " * For any 'new content' field (new_text, new_content, content): every "
    "content-bearing token (numbers, dates, proper nouns, identifiers) MUST "
    "appear in {transcript_window ∪ AST current content}. Do NOT add "
    "implementation details, methodology, or best practices not stated in the meeting.\n"
    " * If intent.verbatim_content is non-empty, use it as the SOLE factual source "
    "for new_text/new_content for add/create operations — same rule as Phase 4.\n"
    " * Prefer SKIP over a low-confidence operation. A missed proposal is "
    "recoverable; a wrong edit is not.\n"
    " * PAGE SUBJECT GUARD: If the page title indicates a different primary subject "
    "than intent.subject (e.g. the page is about Person B but intent.subject is Person A, "
    "or the page covers Topic X but the change is specifically about Topic Y which does not "
    "appear as the page's main focus), emit action='skip' with reason='page subject mismatch — "
    "change belongs on a different page'. Do NOT write content about Subject A onto Subject B's page.\n"
    " * DELETE FIELD RULE: For action='delete_section', set ONLY section_heading. Do NOT "
    "set new_text, new_content, anchor_text, or old_text — the section is simply removed and "
    "any content you emit here will cause downstream duplication bugs.\n"
    " * FORMAT MATCH RULE: For action='replace', new_text must match the FORMAT of old_text. "
    "If old_text is a single date or short value, new_text must be just the new date or value — "
    "not a sentence explaining the change. Never expand a short value into a paragraph.\n\n"
    "Return EXACTLY one JSON object matching the StructuredOperation schema. No "
    "markdown wrapper, no explanation outside the JSON."
)


# ---------------------------------------------------------------------------
# Module-level singleton agent
# ---------------------------------------------------------------------------
# Constructed once at import time. ``output_type=AgentOutputSchema(...,
# strict_json_schema=False)`` matches the pattern used by
# ``fact_extraction_agent._fact_agent`` — strict JSON mode is incompatible
# with Pydantic Optional fields that lack ``additionalProperties=false``
# when the SDK auto-derives the schema.


_drafter_agent = Agent(
    name="StructureAwareDrafter",
    model=_AGENT_MODEL,
    instructions=STRUCTURE_AWARE_DRAFTER_PROMPT,
    output_type=AgentOutputSchema(StructuredOperation, strict_json_schema=False),
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _runs_to_text(runs: Any) -> str:
    """Flatten a list of TextRun objects (or dicts) to a plain string."""
    if not runs:
        return ""
    parts: List[str] = []
    for r in runs:
        if hasattr(r, "text"):
            parts.append(getattr(r, "text", "") or "")
        elif isinstance(r, dict):
            parts.append(r.get("text", "") or "")
        else:
            parts.append(str(r))
    return "".join(parts).strip()


def _summarize_ast(ast: Any, max_chars: int = _AST_SUMMARY_MAX_CHARS) -> str:
    """Produce a compact human-readable summary of an ASTRoot for the LLM prompt.

    Format:

        section[0] heading='Onboarding'
          ordered_list[0]:
            [0] Login
            [1] Payment
            [2] Dashboard
          paragraph[1]: 'After completing onboarding ...' (84 chars)
        section[1] heading='SLA'
          paragraph[0]: 'The SLA is ...' (40 chars)

    Truncates on a section boundary if the total length would exceed
    ``max_chars`` (T-10-20 DoS mitigation per the plan's threat model).
    """
    if ast is None:
        return "(empty AST)"
    sections = getattr(ast, "sections", None) or []
    lines: List[str] = []
    used = 0
    for sec in sections:
        heading = getattr(sec, "heading", None)
        sec_idx = getattr(sec, "section_index", "?")
        heading_text = getattr(heading, "text", None) if heading else None
        section_header = (
            f"section[{sec_idx}] heading={heading_text!r}"
            if heading_text is not None
            else f"section[{sec_idx}] heading=<none>"
        )
        lines.append(section_header)

        blocks = getattr(sec, "blocks", None) or []
        for b_idx, block in enumerate(blocks):
            kind = getattr(block, "kind", type(block).__name__)
            if kind in {"ordered_list", "unordered_list"}:
                lines.append(f"  {kind}[{b_idx}]:")
                items = getattr(block, "items", None) or []
                for item in items:
                    item_idx = getattr(item, "index", "?")
                    item_text = _runs_to_text(getattr(item, "runs", None))[:80]
                    lines.append(f"    [{item_idx}] {item_text}")
            elif kind == "paragraph":
                text = _runs_to_text(getattr(block, "runs", None))
                preview = text[:100]
                lines.append(
                    f"  paragraph[{b_idx}]: {preview!r}"
                    + ("" if len(text) <= 100 else f" ({len(text)} chars)")
                )
            else:
                # Macros, code blocks, tables — opaque per Phase 10 contract.
                lines.append(f"  {kind}[{b_idx}]: <opaque>")

        # Snapshot length; truncate at section boundary if we'd exceed budget.
        joined = "\n".join(lines)
        if len(joined) > max_chars:
            return joined[:max_chars].rstrip() + "\n... (AST truncated)"
        used = len(joined)

    summary = "\n".join(lines) if lines else "(no sections)"
    if len(summary) > max_chars:
        summary = summary[:max_chars].rstrip() + "\n... (AST truncated)"
    return summary


def _has_ordered_list(ast: Any) -> bool:
    """True iff *ast* contains at least one OrderedList block in any section."""
    for sec in getattr(ast, "sections", None) or []:
        for block in getattr(sec, "blocks", None) or []:
            if getattr(block, "kind", None) == "ordered_list":
                return True
    return False


def _intent_to_dict(intent: Any) -> Dict[str, Any]:
    """Extract the relevant ChangeIntent fields for the prompt — duck-typed so
    tests can pass mocks or dicts. Mirrors the field set the legacy drafter uses."""
    if isinstance(intent, dict):
        d = intent
        get = lambda k, default="": str(d.get(k, default) or default)  # noqa: E731
    else:
        get = lambda k, default="": str(getattr(intent, k, default) or default)  # noqa: E731
    return {
        "instruction": get("instruction"),
        "subject": get("subject"),
        "target_hint": get("target_hint"),
        "old_value": get("old_value"),
        "new_value": get("new_value"),
        "action": get("action", "replace"),
        "rationale": get("rationale"),
        "verbatim_content": get("verbatim_content"),
    }


def _build_user_message(inp: StructureAwareDrafterInput) -> str:
    """Compose the user message handed to ``Runner.run``."""
    import json

    intent_json = json.dumps(_intent_to_dict(inp.intent), ensure_ascii=False)
    page_meta_json = json.dumps(inp.page_meta or {}, ensure_ascii=False)
    ast_summary = _summarize_ast(inp.page_ast)
    transcript = (inp.transcript_window or "")[:_TRANSCRIPT_MAX_CHARS]

    return (
        "INTENT:\n"
        f"{intent_json}\n\n"
        "PAGE_META:\n"
        f"{page_meta_json}\n\n"
        "AST_SUMMARY:\n"
        f"{ast_summary}\n\n"
        "TRANSCRIPT_WINDOW:\n"
        f"{transcript}"
    )


def _coerce_to_structured_op(raw: Any) -> Optional[StructuredOperation]:
    """Best-effort coercion of Runner.final_output into a StructuredOperation.

    Returns None when coercion fails — caller produces a skip op.
    """
    if isinstance(raw, StructuredOperation):
        return raw
    if isinstance(raw, dict):
        try:
            return StructuredOperation.model_validate(raw)
        except Exception:
            return None
    # Strings (e.g. LLM emitted unwrapped JSON) — try to parse.
    if isinstance(raw, str):
        import json

        try:
            data = json.loads(raw)
        except Exception:
            return None
        if isinstance(data, dict):
            try:
                return StructuredOperation.model_validate(data)
            except Exception:
                return None
    # BaseModel from a different namespace — best-effort .model_dump round-trip.
    if hasattr(raw, "model_dump"):
        try:
            return StructuredOperation.model_validate(raw.model_dump())
        except Exception:
            return None
    return None


def _post_validate(op: StructuredOperation, inp: StructureAwareDrafterInput) -> StructuredOperation:
    """Stamp page_id from page_meta + defense-in-depth field stripping.

    Pitfall 5 (10-RESEARCH.md line 944): for reorder ops, the LLM occasionally
    rides along a regenerated <ol> in new_text / new_content even with a strict
    prompt. The dispatcher already ignores after_content for reorder, but a
    second defensive scrub at the drafter exit makes downstream code (and the
    GroundingGate token-subset check) easier to reason about.
    """
    page_id = (inp.page_meta or {}).get("page_id") if isinstance(inp.page_meta, dict) else None
    if page_id and not op.page_id and op.action != "create_page":
        op.page_id = str(page_id)

    # ── Reorder defense-in-depth (Failure Mode 2, Pitfall 5) ─────────────
    # The StructuredOperation Literal already prevents the LLM from returning
    # an action like "rewrite_section" — but a malicious / drifted LLM can
    # still ride along ``new_text`` or ``new_content`` carrying a freshly-
    # regenerated ``<ol>`` (the exact 'login -> username -> click mouse
    # button' failure the user reported). We strip both fields at the drafter
    # exit so:
    #   1. The dispatcher's reorder path (Plan 10-04) sees no LLM prose.
    #   2. The GroundingGate token-subset check (Plan 10-XX) cannot fail
    #      on hallucinated tokens because there's nothing to tokenize.
    #   3. Downstream UI rendering for the reorder card uses only the
    #      from_index/to_index swap visualisation — never an LLM-authored
    #      list.
    # This is independent of the dispatcher's own "ignore after_content for
    # reorder" rule; both layers must hold for the contract to be tight.
    if op.action == "reorder":
        if op.new_text is not None or op.new_content is not None:
            logger.warning(
                "StructureAwareDrafter: reorder op carried LLM-supplied "
                "new_text/new_content — stripping (Pitfall 5 defense-in-depth) "
                "page_id=%s section_heading=%r from_index=%s to_index=%s",
                op.page_id, op.section_heading, op.from_index, op.to_index,
            )
            op.new_text = None
            op.new_content = None

    # ── Delete-section defense-in-depth ──────────────────────────────────────
    # LLM sometimes emits new_text/new_content for delete_section ops (the
    # "above content duplicated" bug). Stripping here ensures apply_structured
    # never sees stray content that would cause commit_document_edit to write
    # the content back instead of just deleting the section.
    if op.action == "delete_section":
        if op.new_text is not None or op.new_content is not None or op.old_text is not None:
            logger.warning(
                "StructureAwareDrafter: delete_section op carried LLM-supplied content fields "
                "— stripping to prevent duplication (page_id=%s section_heading=%r)",
                op.page_id, op.section_heading,
            )
            op.new_text = None
            op.new_content = None
            op.old_text = None

    return op


# ---------------------------------------------------------------------------
# Public entrypoint
# ---------------------------------------------------------------------------


async def draft_operation(inp: StructureAwareDrafterInput) -> StructuredOperation:
    """Draft ONE structured operation for the (intent, page_ast, transcript) triple.

    Flow:
      1. Deterministic pre-flight grounding (no LLM):
         * reorder intent + no OrderedList anywhere in the AST -> skip immediately.
      2. Build a compact AST summary + structured prompt; call Runner.run.
      3. Coerce final_output to a StructuredOperation (or fall back to skip).
      4. Post-validate: stamp page_id, strip prose for reorder ops.

    Returns a fully-populated ``StructuredOperation``. Never returns None and
    never raises — failures degrade to ``action="skip"`` with a reason so the
    orchestrator can log and move on.
    """
    # Normalise to StructureAwareDrafterInput if a dict slipped through.
    if not isinstance(inp, StructureAwareDrafterInput):
        try:
            inp = StructureAwareDrafterInput.model_validate(inp)
        except Exception as exc:
            return StructuredOperation(
                action="skip",
                reason=f"invalid drafter input: {exc.__class__.__name__}",
            )

    intent_action = (
        (
            inp.intent.get("action")
            if isinstance(inp.intent, dict)
            else getattr(inp.intent, "action", "")
        )
        or ""
    ).strip().lower()

    # ── Pre-flight 1: reorder against an AST with no OrderedList -> skip. ──
    if intent_action == "reorder" and not _has_ordered_list(inp.page_ast):
        return StructuredOperation(
            action="skip",
            reason="reorder requested but no ordered list in page AST",
            page_id=(inp.page_meta or {}).get("page_id") if isinstance(inp.page_meta, dict) else None,
        )

    # ── LLM call ──
    user_message = _build_user_message(inp)
    try:
        result = await Runner.run(_drafter_agent, user_message)
    except Exception as exc:
        logger.warning(
            "StructureAwareDrafter Runner.run failed for page %s: %s",
            (inp.page_meta or {}).get("page_id"),
            exc,
        )
        return StructuredOperation(
            action="skip",
            reason=f"drafter LLM call failed: {exc.__class__.__name__}",
            page_id=(inp.page_meta or {}).get("page_id") if isinstance(inp.page_meta, dict) else None,
        )

    raw = getattr(result, "final_output", None)
    op = _coerce_to_structured_op(raw)
    if op is None:
        logger.warning(
            "StructureAwareDrafter: invalid Runner output (type=%s) — falling back to skip",
            type(raw).__name__,
        )
        return StructuredOperation(
            action="skip",
            reason="drafter output validation failed",
            page_id=(inp.page_meta or {}).get("page_id") if isinstance(inp.page_meta, dict) else None,
        )

    return _post_validate(op, inp)


__all__ = [
    "STRUCTURE_AWARE_DRAFTER_PROMPT",
    "StructureAwareDrafterInput",
    "StructuredOperation",
    "draft_operation",
]
