"""pipeline/apply.py — safe-apply orchestration for v3 pipeline (SAFE-V3-01 / EDIT-V3-01).

Adapts _execute_pipeline_proposal from review/api.py into a clean, typed
stage that:
  * Runs a section-anchor + version preflight before every apply.
  * Defaults to the LLM EditorAgent path (JARVIS_V3_EDITOR_LLM on).
  * Falls back to apply_structured (deterministic) for reorder/archive when
    JARVIS_V3_EDITOR_LLM=0, behind the WR-01 version-advanced guard.
  * Reports success ONLY when the underlying tool returned success=True
    (EDIT-V3-01 / T-11-16).
  * Defaults archive_deprecate to label/banner — hard-delete gated on
    confirm_hard_delete=True (T-11-17).
  * Reindexes the page in Pinecone+Neo4j in-session after a successful apply
    (SAFE-V3-01).

Patchable internal helpers (_get_connector, _execute_via_editor_agent,
_archive_page, _hard_delete_page, _reindex_page) make the happy/failure paths
testable without live Confluence credentials.

SCOPE-LOCK: this module does NOT import or modify editor_agent.py or tools.py.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Optional

from pydantic import BaseModel

from confluence_logic.pipeline.contracts import PlannedOperation
from confluence_logic.pipeline.context import _parse_flag

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Killswitch — JARVIS_V3_EDITOR_LLM (default ON = use LLM EditorAgent)
# ---------------------------------------------------------------------------
# WR-07: empty env var must NOT flip the default.
JARVIS_V3_EDITOR_LLM: bool = _parse_flag("JARVIS_V3_EDITOR_LLM", default="1")


# ---------------------------------------------------------------------------
# ApplyResult
# ---------------------------------------------------------------------------

class ApplyResult(BaseModel):
    """Result of a safe-apply execution."""

    success: bool
    message: str = ""
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Patchable connector access (mirrors review/api.py _get_connector pattern)
# ---------------------------------------------------------------------------

def _get_connector():
    """Return the Confluence connector singleton.

    Lazy-imported so this module has no side-effects at import time.
    Thin wrapper so tests can patch ``confluence_logic.pipeline.apply._get_connector``
    and substitute a MagicMock.
    """
    from confluence_logic.connectors.confluence import ConfluenceConnector  # noqa: PLC0415
    # Mirror the lazy singleton pattern from review/api.py
    return ConfluenceConnector()


# ---------------------------------------------------------------------------
# Patchable sub-operations (allow tests to isolate each concern)
# ---------------------------------------------------------------------------

async def _execute_via_editor_agent(
    op: PlannedOperation,
    *,
    session_id: str = "",
    user_id: str = "",
) -> dict:
    """Drive the locked EditorAgent with a typed instruction.

    Returns a dict with at minimum {``success``: bool, ``message``: str}.
    Success is True ONLY when the editor tool returned success=True
    (EDIT-V3-01 / T-11-16).

    graph_user_id ContextVar is set/reset around the editor call so the
    editor's own tools can read it.  Stages must NEVER set it directly.
    """
    from confluence_logic.pipeline.editor.instruction import (  # noqa: PLC0415
        EditorInstruction,
        render_editor_prompt,
        _call_editor_agent,
    )
    import confluence_logic.confluence_page_graph as cpg  # noqa: PLC0415

    # Map PlannedOperation → EditorInstruction (field-name adaptor)
    instr = EditorInstruction(
        operation=op.operation,
        page_id=op.page_id,
        page_title=op.page_title,
        space_key=op.space_key,
        section_heading=op.section_heading,
        before_content=op.before_content,
        after_content=op.after_content,
        rationale=op.rationale,
    )

    # Set graph_user_id ContextVar around the editor call only.
    graph_token = cpg.set_current_graph_user_id(user_id or "")
    try:
        result = await _call_editor_agent(instr)
    except Exception as exc:
        logger.error(
            "_execute_via_editor_agent raised for op=%s page=%s: %s",
            op.operation, op.page_id or op.page_title, exc,
        )
        result = {"success": False, "message": f"ERROR: {exc}"}
    finally:
        cpg.reset_current_graph_user_id(graph_token)

    return result


async def _archive_page(page_id: str, page_title: str, rationale: str = "") -> bool:
    """Archive/deprecate a page via the label-banner mechanism (Task 1 decision).

    Adds a [DEPRECATED] prefix to the page title and prepends a deprecation
    banner via existing edit primitives. Does NOT hard-delete the page.

    Returns True on success, False on failure (graceful degradation).
    """
    try:
        from confluence_logic.pipeline.editor.instruction import (  # noqa: PLC0415
            EditorInstruction,
            execute_editor_instruction,
        )
        instr = EditorInstruction(
            operation="archive_deprecate",
            page_id=page_id,
            page_title=page_title,
            rationale=rationale,
            confirm_hard_delete=False,  # label-banner default
        )
        result = await execute_editor_instruction(instr)
        return result.success
    except Exception as exc:
        logger.warning(
            "_archive_page failed for page_id=%s (non-fatal): %s", page_id, exc
        )
        return False


async def _hard_delete_page(page_id: str, page_title: str, rationale: str = "") -> bool:
    """Permanently delete a page (requires confirm_hard_delete=True gate).

    This path is only reached when the caller explicitly passes
    confirm_hard_delete=True — the default is always the label-banner archive.

    Returns True on success, False on failure (graceful degradation).
    """
    try:
        from confluence_logic.pipeline.editor.instruction import (  # noqa: PLC0415
            EditorInstruction,
            execute_editor_instruction,
        )
        instr = EditorInstruction(
            operation="archive_deprecate",
            page_id=page_id,
            page_title=page_title,
            rationale=rationale,
            confirm_hard_delete=True,  # permanent delete
        )
        result = await execute_editor_instruction(instr)
        return result.success
    except Exception as exc:
        logger.warning(
            "_hard_delete_page failed for page_id=%s (non-fatal): %s", page_id, exc
        )
        return False


async def _reindex_page(page_id: str, *, user_id: str = "", session_id: str = "") -> None:
    """Reindex the page in Pinecone+Neo4j in-session after a successful apply.

    Fire-and-forget: failures are logged at WARNING level and swallowed so
    the caller's success response is unaffected (graceful-degradation pattern,
    SAFE-V3-01).

    ``user_id`` MUST be the graph-scoping key for this user — ``refresh_page_in_graph``
    returns False immediately on a falsy user_id, so passing "" silently skips
    the reindex (the page graph is per-user scoped).
    """
    try:
        import confluence_logic.confluence_page_graph as cpg  # noqa: PLC0415
        # refresh_page_in_graph is async; call it directly.
        await cpg.refresh_page_in_graph(user_id=user_id, page_id=page_id)
        logger.debug(
            "In-session reindex completed for page_id=%s user_id=%s", page_id, user_id
        )
    except Exception as exc:
        logger.warning(
            "_reindex_page failed for page_id=%s session=%s (non-fatal): %s",
            page_id, session_id, exc,
        )


# ---------------------------------------------------------------------------
# Section-anchor preflight helper
# ---------------------------------------------------------------------------

def _heading_present_in_html(html: str, heading: str) -> bool:
    """Return True if the heading string appears anywhere in the HTML body.

    Case-sensitive substring check — the heading name must appear verbatim
    (the EditorAgent fetches the live page; this guard prevents misplaced edits
    when the heading was renamed or removed since the card was drafted).
    """
    return heading in html


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

async def apply_proposal(
    op: PlannedOperation,
    *,
    session_id: str = "",
    user_id: str = "",
    confirm_hard_delete: bool = False,
) -> ApplyResult:
    """Execute one PlannedOperation safely with preflight + reindex.

    Steps:
      1. Version preflight: fetch live page metadata (captures starting_version).
      2. Section-anchor preflight: for edit_section ops, verify section_heading
         is present in the live HTML.  If absent → heading_not_found failure
         (SAFE-V3-01 / T-11-15).
      3. Execute via editor agent (default) or deterministic path (killswitch).
      4. For archive_deprecate: call _archive_page unless confirm_hard_delete=True
         (T-11-17 — hard-delete gated on explicit second confirm).
      5. Report success ONLY on tool success=True (EDIT-V3-01 / T-11-16).
      6. On success: reindex via _reindex_page (SAFE-V3-01).

    Args:
        op: The PlannedOperation to apply.
        session_id: Used for lock namespacing and meeting context retrieval.
        user_id: Used for graph_user_id threading into the editor call.
        confirm_hard_delete: Must be True to allow permanent page deletion.
            The default (False) always routes archive_deprecate to label/banner.

    Returns:
        ApplyResult with success=True only when the editor tool confirmed success.
    """
    page_id = op.page_id
    page_title = op.page_title or ""

    # ------------------------------------------------------------------
    # 1. Version preflight — capture starting_version before editor runs
    # ------------------------------------------------------------------
    starting_version: Optional[int] = None
    live_html: Optional[str] = None

    connector = _get_connector()

    if page_id:
        try:
            meta = connector.get_page_metadata(page_id)
            starting_version = (meta or {}).get("version", {}).get("number")
        except Exception as exc:
            logger.debug(
                "Could not read starting version for '%s' (non-fatal): %s",
                page_id, exc,
            )

        # Fetch live HTML for section-anchor preflight (edit_section only)
        if op.operation == "edit_section" and op.section_heading:
            try:
                live_html = connector.fetch_page_html(page_id)
            except Exception as exc:
                logger.debug(
                    "Could not fetch live HTML for '%s' (non-fatal): %s",
                    page_id, exc,
                )

    # ------------------------------------------------------------------
    # 2. Section-anchor preflight (SAFE-V3-01 / T-11-15)
    # ------------------------------------------------------------------
    if op.operation == "edit_section" and op.section_heading:
        if live_html is not None and not _heading_present_in_html(
            live_html, op.section_heading
        ):
            logger.warning(
                "Section-anchor preflight failed: heading '%s' not found in live page '%s'",
                op.section_heading, page_id,
            )
            return ApplyResult(
                success=False,
                message=f"Heading '{op.section_heading}' not found on live page — "
                        "apply aborted to prevent misplaced edit.",
                error="heading_not_found",
            )

    # ------------------------------------------------------------------
    # 3. Execute the operation
    # ------------------------------------------------------------------
    result: dict

    if op.operation == "archive_deprecate":
        # T-11-17: archive-default; hard-delete requires explicit second confirm.
        if confirm_hard_delete:
            success = await _hard_delete_page(page_id or "", page_title, op.rationale)
        else:
            success = await _archive_page(page_id or "", page_title, op.rationale)
        result = {"success": success, "message": "Archive/deprecate completed." if success else "Archive failed."}

    elif JARVIS_V3_EDITOR_LLM:
        # LLM EditorAgent path (default, EDIT-V3-01)
        result = await _execute_via_editor_agent(op, session_id=session_id, user_id=user_id)

        # WR-01: version-advanced guard — if the editor ran but the page version
        # advanced during the run, do NOT fall back to deterministic apply.
        editor_failed = not result.get("success", False)
        if editor_failed and starting_version is not None and page_id:
            try:
                meta_now = connector.get_page_metadata(page_id)
                current_version = (meta_now or {}).get("version", {}).get("number")
                if current_version is not None and current_version > starting_version:
                    logger.warning(
                        "EditorAgent reported failure for op=%s page=%s but page "
                        "version advanced (%s → %s) — refusing deterministic fallback "
                        "to prevent double-write (WR-01)",
                        op.operation, page_id, starting_version, current_version,
                    )
                    return ApplyResult(
                        success=False,
                        message=(
                            "The editor agent reported an error but the page was modified "
                            "during the run. Refusing to retry to avoid double-applying. "
                            "Please review the page in Confluence and regenerate if needed."
                        ),
                        error="partial_commit_detected",
                    )
            except Exception as exc:
                logger.debug(
                    "Could not read post-agent version for '%s' (non-fatal): %s",
                    page_id, exc,
                )

    else:
        # Deterministic fallback path (JARVIS_V3_EDITOR_LLM=0)
        # Only reorder/archive ops reach here in normal usage.
        try:
            from confluence_logic.agents.editor_dispatcher import (  # noqa: PLC0415
                apply_structured,
            )
            # Build a StructuredOperation-compatible dict from the PlannedOperation.
            # Map v3 operation names to the D-02 instruction shapes.
            _op_map = {
                "edit_section": "replace",
                "append": "insert_after",
                "create_page": "create_page",
                "archive_deprecate": "delete_section",
            }
            structured_op = {
                "action": _op_map.get(op.operation, op.operation),
                "page_id": page_id,
                "page_title": page_title,
                "heading": op.section_heading,
                "old_text": op.before_content,
                "new_text": op.after_content,
                "rationale": op.rationale,
            }
            apply_result = await asyncio.to_thread(apply_structured, structured_op)
            success_val = bool((apply_result or {}).get("success", False))
            result = {
                "success": success_val,
                "message": (apply_result or {}).get("message", ""),
            }
        except Exception as exc:
            logger.warning(
                "Deterministic apply_structured failed for op=%s page=%s: %s",
                op.operation, page_id, exc,
            )
            result = {"success": False, "message": f"ERROR: {exc}"}

    # ------------------------------------------------------------------
    # 4. Build ApplyResult and conditionally reindex (SAFE-V3-01)
    # ------------------------------------------------------------------
    success = bool(result.get("success", False))
    apply_result_obj = ApplyResult(
        success=success,
        message=result.get("message", ""),
        error=result.get("error"),
    )

    if success and page_id:
        # In-session reindex — fire-and-forget; failure must not affect caller.
        # Pass user_id so the per-user graph scope is honoured (empty user_id
        # makes refresh_page_in_graph a no-op).
        await _reindex_page(page_id, user_id=user_id, session_id=session_id)

    return apply_result_obj
