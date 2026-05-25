"""RED tests for safe apply stage (SAFE-V3-01 — Phase 11).

SAFE-V3-01: Per-card HITL; section-anchor + version preflight before every
apply; regenerate-against-live if heading gone; archive-default (hard-delete
needs a second confirm); in-session reindex of Pinecone+Neo4j after apply.

Reuses Phase 5 fixture shapes from test_apply_hardening.py, extending them
with v3 requirements.

These tests import from ``confluence_logic.pipeline.apply`` which does not
exist yet. Pytest collection fails with ImportError — expected RED state for
Wave 0 of Phase 11.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch, call

pytestmark = pytest.mark.asyncio

# ---------------------------------------------------------------------------
# Import existing symbols (available from Phase 5)
# ---------------------------------------------------------------------------
from confluence_logic.review import api as review_api

# ---------------------------------------------------------------------------
# Import from not-yet-built pipeline target (RED: ImportError at collection)
# ---------------------------------------------------------------------------
from confluence_logic.pipeline.apply import (
    apply_proposal,
    ApplyResult,
)
from confluence_logic.pipeline.contracts import PlannedOperation


# ---------------------------------------------------------------------------
# Fixtures — reuse Phase 5 shapes
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_connector():
    connector = MagicMock()
    connector.fetch_page_html.return_value = (
        "<h2>Audit Schedule</h2><p>SOC2 audit scheduled for Q3 2025.</p>"
    )
    connector.get_page_metadata.return_value = {"version": {"number": 7}, "title": "SOC2 Compliance"}
    connector.push_update.return_value = True
    return connector


@pytest.fixture
def edit_op():
    return PlannedOperation(
        operation="edit_section",
        page_id="pg-soc2",
        page_title="SOC2 Compliance",
        section_heading="Audit Schedule",
        before_content="Q3 2025",
        after_content="Q2 2025",
        rationale="Audit moved to Q2.",
    )


@pytest.fixture
def create_op():
    return PlannedOperation(
        operation="create_page",
        page_id=None,
        page_title="Disaster Recovery Runbook",
        section_heading=None,
        before_content=None,
        after_content="## RTO and RPO\nRTO: 4h. RPO: 1h.",
        rationale="No DR page existed.",
    )


@pytest.fixture
def archive_op():
    return PlannedOperation(
        operation="archive_deprecate",
        page_id="pg-api-runbook-v1",
        page_title="API Runbook v1 (Legacy)",
        section_heading=None,
        before_content=None,
        after_content=None,
        rationale="v1 is superseded.",
    )


# ---------------------------------------------------------------------------
# SAFE-V3-01-1: Section-anchor preflight — heading must exist before apply
# ---------------------------------------------------------------------------

async def test_section_anchor_preflight_fails_when_heading_gone(
    mock_connector, edit_op
):
    """SAFE-V3-01: if the section heading is gone from the live page at apply
    time, apply must fail with error='heading_not_found'."""
    mock_connector.fetch_page_html.return_value = (
        "<h2>Other Section</h2><p>content</p>"  # heading absent
    )

    with patch(
        "confluence_logic.pipeline.apply._get_connector",
        return_value=mock_connector,
    ):
        result: ApplyResult = await apply_proposal(edit_op, session_id="sess-test")

    assert result.success is False
    assert result.error == "heading_not_found", (
        f"Expected error='heading_not_found', got {result.error!r}"
    )
    mock_connector.push_update.assert_not_called()


# ---------------------------------------------------------------------------
# SAFE-V3-01-2: Version preflight — check version before push_update
# ---------------------------------------------------------------------------

async def test_version_preflight_reads_page_version_before_apply(
    mock_connector, edit_op
):
    """SAFE-V3-01: apply must fetch the page version (get_page_metadata) before
    calling push_update — version conflict must be detectable."""
    with patch(
        "confluence_logic.pipeline.apply._get_connector",
        return_value=mock_connector,
    ), patch(
        "confluence_logic.pipeline.apply._execute_via_editor_agent",
        new=AsyncMock(return_value={"success": True}),
    ):
        await apply_proposal(edit_op, session_id="sess-test")

    mock_connector.get_page_metadata.assert_called(), (
        "SAFE-V3-01: get_page_metadata must be called before apply (version preflight)"
    )


# ---------------------------------------------------------------------------
# SAFE-V3-01-3: Archive-default (hard-delete requires second confirm)
# ---------------------------------------------------------------------------

async def test_archive_op_does_not_hard_delete_without_second_confirm(
    mock_connector, archive_op
):
    """SAFE-V3-01: archive_deprecate must archive/label the page, NOT hard-delete.
    Hard-delete requires an explicit second confirmation flag."""
    with patch(
        "confluence_logic.pipeline.apply._get_connector",
        return_value=mock_connector,
    ), patch(
        "confluence_logic.pipeline.apply._archive_page",
        new=AsyncMock(return_value=True),
    ) as mock_archive, patch(
        "confluence_logic.pipeline.apply._hard_delete_page",
        new=AsyncMock(return_value=True),
    ) as mock_delete:
        result = await apply_proposal(archive_op, session_id="sess-test")

    mock_archive.assert_called_once(), "archive_deprecate must call _archive_page"
    mock_delete.assert_not_called(), (
        "SAFE-V3-01: hard-delete must NOT be called without confirm_hard_delete=True"
    )


# ---------------------------------------------------------------------------
# SAFE-V3-01-4: In-session reindex after successful apply
# ---------------------------------------------------------------------------

async def test_in_session_reindex_triggered_after_successful_apply(
    mock_connector, edit_op
):
    """SAFE-V3-01: after a successful apply, the page must be reindexed in
    Pinecone+Neo4j in-session so the graph stays consistent."""
    with patch(
        "confluence_logic.pipeline.apply._get_connector",
        return_value=mock_connector,
    ), patch(
        "confluence_logic.pipeline.apply._execute_via_editor_agent",
        new=AsyncMock(return_value={"success": True}),
    ), patch(
        "confluence_logic.pipeline.apply._reindex_page",
        new=AsyncMock(),
    ) as mock_reindex:
        result = await apply_proposal(edit_op, session_id="sess-test")

    assert result.success is True
    mock_reindex.assert_called_once_with(
        edit_op.page_id,
        session_id="sess-test",
    ), "SAFE-V3-01: in-session reindex must be called after successful apply"


# ---------------------------------------------------------------------------
# SAFE-V3-01-5: No reindex on failed apply
# ---------------------------------------------------------------------------

async def test_no_reindex_on_failed_apply(mock_connector, edit_op):
    """SAFE-V3-01: if apply fails (e.g., heading gone), reindex must NOT run."""
    mock_connector.fetch_page_html.return_value = "<h2>Gone Section</h2><p>x</p>"

    with patch(
        "confluence_logic.pipeline.apply._get_connector",
        return_value=mock_connector,
    ), patch(
        "confluence_logic.pipeline.apply._reindex_page",
        new=AsyncMock(),
    ) as mock_reindex:
        result = await apply_proposal(edit_op, session_id="sess-test")

    assert result.success is False
    mock_reindex.assert_not_called(), (
        "SAFE-V3-01: reindex must not run when apply failed"
    )
