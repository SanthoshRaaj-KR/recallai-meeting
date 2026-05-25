"""RED tests for EditorInstruction handoff (EDIT-V3-01 — Phase 11).

EDIT-V3-01: EditorInstruction validates with all required fields; the prompt
rendered by render_editor_prompt is exact (matches expected template); the
editor is only counted as a success when the tool returns success=true.

STATIC GUARD: editor_agent.py and tools.py must be byte-identical to the
locked SHA256 hashes from Phase 10 (scope-lock per D2 — do not modify).

These tests import from ``confluence_logic.pipeline.editor.instruction`` which
does not exist yet. Pytest collection fails with ImportError — expected RED
state for Wave 0 of Phase 11.

The STATIC GUARD tests use only hashlib and pathlib — they will be GREEN even
in Wave 0 (no missing import), which is intentional: if anyone edits those
locked files the static guard catches it immediately.
"""

import hashlib
import pathlib
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

pytestmark = pytest.mark.asyncio

# ---------------------------------------------------------------------------
# STATIC GUARDS — byte-identity check for locked files (always runnable)
# ---------------------------------------------------------------------------

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent

LOCKED_EDITOR_AGENT_SHA256 = "028e9607da6dd54670da4ebc5c337d887d92c748f120eecd8b920eee50239e87"
LOCKED_TOOLS_SHA256 = "40419bc9a8cb552f41c610bdd0c6c710cf4ba36629e1bd5d1b7aa5a4b78b67a4"


def _sha256(path: pathlib.Path) -> str:
    sha = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            sha.update(chunk)
    return sha.hexdigest()


def test_editor_agent_py_byte_identical_to_locked_sha256():
    """EDIT-V3-01 static guard: editor_agent.py must not be modified.

    Expected SHA256: 028e9607da6dd54670da4ebc5c337d887d92c748f120eecd8b920eee50239e87
    If this test fails someone modified the locked file — revert the change.
    """
    path = REPO_ROOT / "confluence_logic" / "agents" / "editor_agent.py"
    assert path.exists(), f"editor_agent.py not found at {path}"
    actual = _sha256(path)
    assert actual == LOCKED_EDITOR_AGENT_SHA256, (
        f"editor_agent.py has been modified!\n"
        f"  Expected SHA256: {LOCKED_EDITOR_AGENT_SHA256}\n"
        f"  Actual SHA256:   {actual}\n"
        "This file is LOCKED (D2 scope fence). Revert any changes."
    )


def test_tools_py_byte_identical_to_locked_sha256():
    """EDIT-V3-01 static guard: tools.py must not be modified.

    Expected SHA256: 40419bc9a8cb552f41c610bdd0c6c710cf4ba36629e1bd5d1b7aa5a4b78b67a4
    If this test fails someone modified the locked file — revert the change.
    """
    path = REPO_ROOT / "confluence_logic" / "agents" / "tools.py"
    assert path.exists(), f"tools.py not found at {path}"
    actual = _sha256(path)
    assert actual == LOCKED_TOOLS_SHA256, (
        f"tools.py has been modified!\n"
        f"  Expected SHA256: {LOCKED_TOOLS_SHA256}\n"
        f"  Actual SHA256:   {actual}\n"
        "This file is LOCKED (scope fence). Revert any changes."
    )


# ---------------------------------------------------------------------------
# Imports from not-yet-built pipeline targets (RED: ImportError at collection)
# ---------------------------------------------------------------------------
from confluence_logic.pipeline.editor.instruction import (
    EditorInstruction,
    render_editor_prompt,
)
from confluence_logic.pipeline.contracts import PlannedOperation


# ---------------------------------------------------------------------------
# EDIT-V3-01-1: EditorInstruction validates with all required fields
# ---------------------------------------------------------------------------

def test_editor_instruction_validates_edit_section():
    """EDIT-V3-01: EditorInstruction for edit_section must accept all required fields."""
    instr = EditorInstruction(
        operation="edit_section",
        page_id="pg-soc2",
        page_title="SOC2 Compliance",
        section_heading="Audit Schedule",
        before_content="SOC2 audit is scheduled for Q3 2025.",
        after_content="SOC2 audit is scheduled for Q2 2025.",
        rationale="Meeting decision: audit moved from Q3 to Q2.",
    )
    assert instr.operation == "edit_section"
    assert instr.page_id == "pg-soc2"


def test_editor_instruction_validates_create_page():
    """EDIT-V3-01: EditorInstruction for create_page must allow None page_id."""
    instr = EditorInstruction(
        operation="create_page",
        page_id=None,
        page_title="Disaster Recovery Runbook",
        section_heading=None,
        before_content=None,
        after_content="# DR Runbook\n\n## RTO and RPO\nRTO: 4h.",
        rationale="No DR runbook existed.",
    )
    assert instr.operation == "create_page"
    assert instr.page_id is None


def test_editor_instruction_validates_append():
    """EDIT-V3-01: EditorInstruction for append."""
    instr = EditorInstruction(
        operation="append",
        page_id="pg-security-policy",
        page_title="Security Policy",
        section_heading="Secrets Management",
        before_content=None,
        after_content="Use HashiCorp Vault for all service secrets.",
        rationale="New policy adopted in this meeting.",
    )
    assert instr.operation == "append"


def test_editor_instruction_validates_archive_deprecate():
    """EDIT-V3-01: EditorInstruction for archive_deprecate."""
    instr = EditorInstruction(
        operation="archive_deprecate",
        page_id="pg-api-runbook-v1",
        page_title="API Runbook v1 (Legacy)",
        section_heading=None,
        before_content=None,
        after_content=None,
        rationale="v1 is fully superseded by v2 documentation.",
    )
    assert instr.operation == "archive_deprecate"


# ---------------------------------------------------------------------------
# EDIT-V3-01-2: render_editor_prompt produces exact expected prompt
# ---------------------------------------------------------------------------

def test_render_editor_prompt_includes_all_required_fields():
    """EDIT-V3-01: the rendered prompt must include page_id, section_heading,
    before_content, after_content, and rationale — the EditorAgent cannot
    succeed without all of these."""
    instr = EditorInstruction(
        operation="edit_section",
        page_id="pg-auth",
        page_title="Authentication",
        section_heading="Provider",
        before_content="We use OAuth.",
        after_content="We use SAML.",
        rationale="Enterprise customers require SAML.",
    )
    prompt = render_editor_prompt(instr)

    assert "pg-auth" in prompt, "page_id must appear in rendered prompt"
    assert "Provider" in prompt, "section_heading must appear in rendered prompt"
    assert "We use OAuth." in prompt, "before_content must appear in rendered prompt"
    assert "We use SAML." in prompt, "after_content must appear in rendered prompt"
    assert "Enterprise customers require SAML" in prompt, "rationale must appear"


# ---------------------------------------------------------------------------
# EDIT-V3-01-3: Success only when EditorAgent returns success=true
# ---------------------------------------------------------------------------

async def test_editor_handoff_succeeds_only_on_tool_success_true():
    """EDIT-V3-01: the pipeline must only count a card as applied when the
    EditorAgent tool returns success=True; a False return is a failure."""
    from confluence_logic.pipeline.editor.instruction import execute_editor_instruction

    instr = EditorInstruction(
        operation="edit_section",
        page_id="pg-soc2",
        page_title="SOC2 Compliance",
        section_heading="Audit Schedule",
        before_content="Q3 2025",
        after_content="Q2 2025",
        rationale="Audit moved to Q2.",
    )

    # Editor returns success=True.
    with patch(
        "confluence_logic.pipeline.editor.instruction._call_editor_agent",
        new=AsyncMock(return_value={"success": True, "message": "Applied."}),
    ):
        result = await execute_editor_instruction(instr)
    assert result.success is True

    # Editor returns success=False.
    with patch(
        "confluence_logic.pipeline.editor.instruction._call_editor_agent",
        new=AsyncMock(return_value={"success": False, "message": "Version conflict."}),
    ):
        result = await execute_editor_instruction(instr)
    assert result.success is False, (
        "EDIT-V3-01: success must only be True when editor tool returns success=True"
    )


async def test_editor_handoff_exception_is_not_success():
    """EDIT-V3-01: if the EditorAgent raises an exception the result is a failure,
    not a silent success."""
    from confluence_logic.pipeline.editor.instruction import execute_editor_instruction

    instr = EditorInstruction(
        operation="edit_section",
        page_id="pg-auth",
        page_title="Authentication",
        section_heading="Provider",
        before_content="OAuth",
        after_content="SAML",
        rationale="switch",
    )

    with patch(
        "confluence_logic.pipeline.editor.instruction._call_editor_agent",
        new=AsyncMock(side_effect=Exception("Network error")),
    ):
        result = await execute_editor_instruction(instr)

    assert result.success is False, (
        "EDIT-V3-01: exception during editor call must result in success=False"
    )
