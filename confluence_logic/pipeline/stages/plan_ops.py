"""Stage 6: Operation planning — exactly one PlannedOperation per intent (OPS-V3-01).

For a pre-resolved ``(page_id, section_heading)`` candidate from the retrieval
stage, produces EXACTLY ONE ``PlannedOperation`` of one of the four shapes:

    edit_section | append | create_page | archive_deprecate

The section is NOT chosen here — it was resolved upstream (RETR-V3-02 /
anti-pattern §Architecture Patterns "don't let the drafter pick the section").

Algorithm (deterministic mapping first, light LLM only for content fill):

1. ``no_existing_target=True`` on the candidate → ``create_page``.
2. ``kind="new_workstream"`` with no old_value → ``create_page``.
3. ``kind="deprecation"`` → ``archive_deprecate``; field-strip (no new_text).
4. ``kind="fact_update"`` with old_value present in page_html → ``edit_section``.
5. ``kind="fact_update"`` with old_value NOT in page_html → ambiguous → ``None``.
6. ``kind in ("action_item", "decision")`` with section heading absent from page
   (or additive new content) → ``append``.
7. Fallback: any unresolvable case → ``None`` (not emitted).

Field-strip defense-in-depth: ``archive_deprecate`` ops never carry ``new_text``
or ``after_content``; other field constraints applied in ``_field_strip``.

Design notes:
  - Never returns None as a surprise — ambiguous → None is documented behavior.
  - Never raises — failures degrade to None with a logged reason.
  - Light LLM (MODEL_WORKER) fills ``after_content`` for ``edit_section`` /
    ``append`` / ``create_page`` when ``new_value`` / ``verbatim_content`` is
    insufficient.  In tests without API keys the LLM call degrades gracefully
    and ``new_value`` is used as the content fallback.
  - ``%``-style logging per CLAUDE.md convention.

OPS-V3-01 acceptance bar: test_plan_ops_v3.py GREEN, every input yields exactly
one op of the four shapes or None (ambiguous).
"""

from __future__ import annotations

import logging
import re
from typing import Any, Optional

from confluence_logic.pipeline.contracts import (
    ChangeIntentV3,
    PlannedOperation,
    SectionCandidate,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# HTML helpers (no new deps — stdlib re)
# ---------------------------------------------------------------------------


def _strip_html(html: str) -> str:
    """Strip HTML tags returning plain text."""
    return re.sub(r"<[^>]+>", " ", html or "")


def _extract_section_text(page_html: str, section_heading: Optional[str]) -> str:
    """Extract the text of a named heading section from page_html.

    Returns the text content between the heading tag and the next heading tag
    (or end of document).  Falls back to the full stripped page text when the
    heading is not found.
    """
    if not page_html:
        return ""
    if not section_heading:
        return _strip_html(page_html).strip()

    # Match <h1>...<hN> tags; content between opening heading and the next heading.
    # Simple extraction: find the heading text, grab text until next heading.
    stripped = _strip_html(page_html)
    pattern = re.compile(
        r"<h[1-6][^>]*>\s*" + re.escape(section_heading) + r"\s*</h[1-6]>",
        re.IGNORECASE,
    )
    match = pattern.search(page_html)
    if not match:
        # heading not found in raw HTML — return full stripped text
        return stripped.strip()

    # Extract text from end of heading tag to next heading
    remainder = page_html[match.end():]
    next_heading = re.search(r"<h[1-6]", remainder, re.IGNORECASE)
    section_html = remainder[: next_heading.start()] if next_heading else remainder
    return _strip_html(section_html).strip()


def _heading_present_in_page(page_html: str, section_heading: Optional[str]) -> bool:
    """Return True iff section_heading appears as an actual heading in page_html."""
    if not page_html or not section_heading:
        return False
    pattern = re.compile(
        r"<h[1-6][^>]*>\s*" + re.escape(section_heading) + r"\s*</h[1-6]>",
        re.IGNORECASE,
    )
    return bool(pattern.search(page_html))


def _old_value_in_page(page_html: str, old_value: str) -> bool:
    """Return True iff old_value appears (case-insensitive) in the stripped page text."""
    if not page_html or not old_value:
        return False
    stripped = _strip_html(page_html).lower()
    return old_value.strip().lower() in stripped


# ---------------------------------------------------------------------------
# Field-strip defense-in-depth
# ---------------------------------------------------------------------------


def _field_strip(op: PlannedOperation) -> PlannedOperation:
    """Remove fields that don't belong to the operation shape.

    ``archive_deprecate`` must not carry new_text / after_content.
    ``create_page`` must not carry section_heading / before_content.

    Mirrors the defense-in-depth strip in ``structure_aware_drafter._post_validate``
    (PATTERNS.md §plan_ops.py — reorder defense-in-depth model).
    """
    if op.operation == "archive_deprecate":
        if op.after_content is not None or op.before_content is not None:
            logger.warning(
                "plan_ops: stripping before/after_content from archive_deprecate op page_id=%s",
                op.page_id,
            )
            op = op.model_copy(update={"after_content": None, "before_content": None})
    if op.operation == "create_page":
        if op.section_heading is not None or op.before_content is not None:
            op = op.model_copy(update={"section_heading": None, "before_content": None})
    return op


# ---------------------------------------------------------------------------
# Content filler — light LLM or deterministic fallback
# ---------------------------------------------------------------------------


async def _fill_content(
    intent: ChangeIntentV3,
    candidate: SectionCandidate,
    page_html: Optional[str],
    operation: str,
) -> str:
    """Fill after_content for a resolved target using MODEL_WORKER.

    On any error (no API key, rate-limit, etc.) degrades to using
    ``intent.new_value`` or ``intent.verbatim_content`` as the content.
    This ensures tests without API keys still receive non-empty content.
    """
    # Deterministic fallback value
    fallback = (
        intent.new_value
        or intent.verbatim_content
        or intent.instruction
        or f"Updated value for {intent.subject}"
    )
    if not fallback or not fallback.strip():
        fallback = f"Updated: {intent.subject}"

    try:
        import openai  # lazy import
        from confluence_logic.pipeline.model_config import MODEL_WORKER

        client = openai.AsyncOpenAI()
        section_heading = candidate.section_heading or ""
        section_ctx = (
            _extract_section_text(page_html, section_heading) if page_html else ""
        )
        prompt = (
            f"Operation: {operation}\n"
            f"Subject: {intent.subject}\n"
            f"Old value: {intent.old_value or '(none)'}\n"
            f"New value: {intent.new_value or intent.verbatim_content or '(none)'}\n"
            f"Rationale: {intent.instruction}\n"
            f"Current section text: {section_ctx[:600]}\n\n"
            f"Write the updated content for the section. Be concise and factual."
        )
        response = await client.chat.completions.create(
            model=MODEL_WORKER,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
            temperature=0.1,
        )
        content = (response.choices[0].message.content or "").strip()
        return content if content else fallback
    except Exception as exc:
        logger.warning(
            "plan_ops: _fill_content LLM call failed (non-fatal): %s — using fallback",
            exc,
        )
        return fallback


# ---------------------------------------------------------------------------
# Public entrypoint
# ---------------------------------------------------------------------------


async def plan_operation(
    intent: ChangeIntentV3,
    candidate: SectionCandidate,
    page_html: Optional[str] = None,
) -> Optional[PlannedOperation]:
    """Plan exactly one operation for a pre-resolved (page_id, section_heading).

    Returns ``None`` for ambiguous/unfillable intents (never raises).

    Deterministic routing:
      1. no_existing_target=True  → create_page
      2. kind=new_workstream       → create_page
      3. kind=deprecation          → archive_deprecate (field-stripped)
      4. kind=fact_update + old_value in page → edit_section
      5. kind=fact_update + old_value NOT in page → None (ambiguous)
      6. kind=action_item/decision + heading absent → append
      7. kind=action_item/decision + heading present → edit_section
      8. Otherwise → None (unresolvable)

    Args:
        intent: The driving ChangeIntentV3 (kind, subject, old/new_value, etc.)
        candidate: Pre-resolved SectionCandidate with page_id and section_heading.
        page_html: Current page HTML content for grounding checks (may be None
            for create_page; treated as empty string otherwise).
    """
    try:
        return await _plan_operation_inner(intent, candidate, page_html)
    except Exception as exc:
        logger.warning(
            "plan_ops: unexpected error for subject=%r kind=%r: %s — returning None",
            intent.subject, intent.kind, exc,
        )
        return None


async def _plan_operation_inner(
    intent: ChangeIntentV3,
    candidate: SectionCandidate,
    page_html: Optional[str],
) -> Optional[PlannedOperation]:
    """Internal implementation — caller wraps in try/except."""
    page_id = candidate.page_id
    page_title = candidate.page_title or ""
    section_heading = candidate.section_heading
    space_key = candidate.space_key or ""

    # ------------------------------------------------------------------
    # Route 1: no_existing_target or new_workstream → create_page
    # ------------------------------------------------------------------
    if candidate.no_existing_target or intent.kind == "new_workstream":
        body = await _fill_content(intent, candidate, page_html, "create_page")
        op = PlannedOperation(
            operation="create_page",
            page_id=None,  # new page — no id yet
            page_title=intent.subject or "New Page",
            space_key=space_key or None,
            section_heading=None,
            before_content=None,
            after_content=body,
            rationale=intent.instruction or f"New workstream: {intent.subject}",
        )
        return _field_strip(op)

    # ------------------------------------------------------------------
    # Route 2: deprecation → archive_deprecate (no content fields)
    # ------------------------------------------------------------------
    if intent.kind == "deprecation":
        op = PlannedOperation(
            operation="archive_deprecate",
            page_id=page_id,
            page_title=page_title,
            space_key=space_key or None,
            section_heading=section_heading,
            before_content=None,
            after_content=None,
            rationale=intent.instruction or f"Deprecate: {intent.subject}",
        )
        return _field_strip(op)

    # ------------------------------------------------------------------
    # Route 3: fact_update → edit_section (only if old_value grounded)
    # ------------------------------------------------------------------
    if intent.kind == "fact_update":
        old_value = (intent.old_value or "").strip()
        if old_value and page_html:
            if not _old_value_in_page(page_html, old_value):
                # old_value not on the page → ambiguous, do not emit
                logger.warning(
                    "plan_ops: fact_update old_value=%r not found in page_id=%s — ambiguous, returning None",
                    old_value, page_id,
                )
                return None

        # Extract existing section text as before_content
        before_content = _extract_section_text(page_html, section_heading) if page_html else ""
        if not before_content:
            before_content = old_value or ""

        after_content = await _fill_content(intent, candidate, page_html, "edit_section")

        op = PlannedOperation(
            operation="edit_section",
            page_id=page_id,
            page_title=page_title,
            space_key=space_key or None,
            section_heading=section_heading,
            before_content=before_content,
            after_content=after_content,
            rationale=f"Update {intent.subject}: {old_value} → {intent.new_value}",
        )
        return _field_strip(op)

    # ------------------------------------------------------------------
    # Route 4: action_item / decision — append if heading absent, else edit
    # ------------------------------------------------------------------
    if intent.kind in ("action_item", "decision"):
        heading_present = page_html and _heading_present_in_page(page_html, section_heading)
        if heading_present:
            # Existing heading → edit_section
            before_content = _extract_section_text(page_html, section_heading)
            after_content = await _fill_content(intent, candidate, page_html, "edit_section")
            op = PlannedOperation(
                operation="edit_section",
                page_id=page_id,
                page_title=page_title,
                space_key=space_key or None,
                section_heading=section_heading,
                before_content=before_content,
                after_content=after_content,
                rationale=intent.instruction or f"Update section: {intent.subject}",
            )
        else:
            # Heading absent → append
            after_content = await _fill_content(intent, candidate, page_html, "append")
            op = PlannedOperation(
                operation="append",
                page_id=page_id,
                page_title=page_title,
                space_key=space_key or None,
                section_heading=section_heading,
                before_content=None,
                after_content=after_content,
                rationale=intent.instruction or f"Add section: {intent.subject}",
            )
        return _field_strip(op)

    # ------------------------------------------------------------------
    # Fallback: unresolvable kind → None
    # ------------------------------------------------------------------
    logger.warning(
        "plan_ops: unresolvable intent kind=%r subject=%r — returning None",
        intent.kind, intent.subject,
    )
    return None


__all__ = ["plan_operation"]
