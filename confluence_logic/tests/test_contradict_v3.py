"""RED tests for contradiction and stale detection stage (CON-V3-01 — Phase 11).

CON-V3-01: For each fact-update intent, find EVERY location stating the old
value (transcript↔page and page↔page); group all affected pages under one
logical decision with a shared group_id; flag stale pages for archive.

The canonical test case is the SOC2 Q3→Q2 two-page fixture where both
pg-soc2 and pg-security-overview state "Q3" and both must appear in the
contradiction group with a single group_id.

These tests import from ``confluence_logic.pipeline.stages.contradict`` which
does not exist yet. Pytest collection fails with ImportError — expected RED
state for Wave 0 of Phase 11.
"""

import json
import pathlib
import pytest
from unittest.mock import AsyncMock, patch

pytestmark = pytest.mark.asyncio

# ---------------------------------------------------------------------------
# Imports from not-yet-built pipeline targets (RED: ImportError at collection)
# ---------------------------------------------------------------------------
from confluence_logic.pipeline.stages.contradict import detect_contradictions
from confluence_logic.pipeline.contracts import (
    ChangeIntentV3,
    EvidenceSpan,
    ContradictionGroup,
)

# ---------------------------------------------------------------------------
# Fixture loader
# ---------------------------------------------------------------------------

FIXTURE_DIR = (
    pathlib.Path(__file__).resolve().parent.parent.parent
    / "tests" / "fixtures" / "transcripts" / "phase11"
)


def _load_fixture(name: str) -> dict:
    path = FIXTURE_DIR / f"{name}.json"
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# CON-V3-01-1: SOC2 Q3→Q2 two-page case — both pages in one group
# ---------------------------------------------------------------------------

async def test_soc2_contradiction_yields_proposal_for_every_affected_page():
    """CON-V3-01: SOC2 Q3→Q2 transcript must produce one ContradictionGroup
    with BOTH pg-soc2 and pg-security-overview as affected pages, same group_id."""
    fixture = _load_fixture("contradiction_soc2_q3_to_q2_two_pages")
    workspace_pages = fixture["confluence_workspace_pages"]

    intent = ChangeIntentV3(
        kind="fact_update",
        subject="SOC2 audit schedule",
        old_value="Q3",
        new_value="Q2",
        dedup_key="soc2-audit-q3-to-q2",
        evidence=[EvidenceSpan(text="SOC2 moved from Q3 to Q2", start=0, end=25)],
    )

    with patch(
        "confluence_logic.pipeline.stages.contradict._find_pages_stating_old_value",
        new=AsyncMock(return_value=["pg-soc2", "pg-security-overview"]),
    ):
        groups = await detect_contradictions(
            intent,
            workspace_pages=workspace_pages,
        )

    assert len(groups) == 1, f"Expected 1 ContradictionGroup, got {len(groups)}"
    group = groups[0]
    assert isinstance(group, ContradictionGroup)
    affected_page_ids = {p.page_id for p in group.affected_pages}
    assert "pg-soc2" in affected_page_ids, (
        "pg-soc2 must be in the contradiction group (CON-V3-01)"
    )
    assert "pg-security-overview" in affected_page_ids, (
        "pg-security-overview must be in the contradiction group (CON-V3-01)"
    )


# ---------------------------------------------------------------------------
# CON-V3-01-2: All affected pages share one group_id
# ---------------------------------------------------------------------------

async def test_all_affected_pages_share_one_group_id():
    """CON-V3-01: all affected pages in a contradiction sweep must share a
    single group_id so the UI can group them as one logical decision."""
    intent = ChangeIntentV3(
        kind="fact_update",
        subject="database engine",
        old_value="MySQL",
        new_value="Postgres",
        dedup_key="db-mysql-postgres",
        evidence=[EvidenceSpan(text="migrated to Postgres", start=0, end=20)],
    )

    with patch(
        "confluence_logic.pipeline.stages.contradict._find_pages_stating_old_value",
        new=AsyncMock(return_value=["pg-db-runbook", "pg-arch", "pg-onboarding"]),
    ):
        groups = await detect_contradictions(
            intent,
            workspace_pages=[
                {"page_id": "pg-db-runbook", "title": "DB Runbook", "content_html": "<p>MySQL</p>"},
                {"page_id": "pg-arch", "title": "Architecture", "content_html": "<p>MySQL</p>"},
                {"page_id": "pg-onboarding", "title": "Onboarding", "content_html": "<p>MySQL</p>"},
            ],
        )

    assert len(groups) == 1
    group = groups[0]
    group_ids = {p.group_id for p in group.affected_pages}
    assert len(group_ids) == 1, (
        f"All affected pages must share one group_id, got {group_ids}"
    )


# ---------------------------------------------------------------------------
# CON-V3-01-3: Stale pages flagged for archive_deprecate
# ---------------------------------------------------------------------------

async def test_stale_pages_flagged_for_archive():
    """CON-V3-01: when a page's content is fully superseded and the transcript
    indicates it should be deprecated, the group marks it for archive_deprecate."""
    intent = ChangeIntentV3(
        kind="deprecation",
        subject="legacy billing service",
        old_value="legacy billing service",
        new_value=None,
        dedup_key="legacy-billing-deprecation",
        evidence=[EvidenceSpan(text="legacy billing service is deprecated", start=0, end=36)],
    )

    with patch(
        "confluence_logic.pipeline.stages.contradict._find_pages_stating_old_value",
        new=AsyncMock(return_value=["pg-billing-arch"]),
    ), patch(
        "confluence_logic.pipeline.stages.contradict._is_page_fully_superseded",
        new=AsyncMock(return_value=True),
    ):
        groups = await detect_contradictions(
            intent,
            workspace_pages=[
                {
                    "page_id": "pg-billing-arch",
                    "title": "Billing Architecture",
                    "content_html": "<p>legacy billing service handles renewals.</p>",
                }
            ],
        )

    assert len(groups) >= 1
    group = groups[0]
    stale = [p for p in group.affected_pages if p.recommended_op == "archive_deprecate"]
    assert len(stale) >= 1, (
        "Fully-superseded page must be flagged for archive_deprecate (CON-V3-01)"
    )


# ---------------------------------------------------------------------------
# CON-V3-01-4: No contradiction group when old_value not on any page
# ---------------------------------------------------------------------------

async def test_no_contradiction_when_old_value_absent():
    """CON-V3-01: if no workspace page contains the old value, no group is emitted."""
    intent = ChangeIntentV3(
        kind="fact_update",
        subject="deployment tool",
        old_value="Chef",
        new_value="Ansible",
        dedup_key="deploy-chef-ansible",
        evidence=[EvidenceSpan(text="switching from Chef", start=0, end=19)],
    )

    with patch(
        "confluence_logic.pipeline.stages.contradict._find_pages_stating_old_value",
        new=AsyncMock(return_value=[]),  # Chef not mentioned on any page
    ):
        groups = await detect_contradictions(
            intent,
            workspace_pages=[
                {"page_id": "pg-infra", "title": "Infra", "content_html": "<p>Ansible deploys.</p>"}
            ],
        )

    assert len(groups) == 0, (
        "No ContradictionGroup must be emitted when old_value is absent from all pages"
    )
