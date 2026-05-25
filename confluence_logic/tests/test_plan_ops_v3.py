"""RED tests for operation planning stage (OPS-V3-01 — Phase 11).

OPS-V3-01: For each (intent, section_candidate) pair the stage must emit
exactly one of: edit_section | append | create_page | archive_deprecate.
Ambiguous cases (insufficient content, conflicting signals) must NOT be emitted.

These tests import from ``confluence_logic.pipeline.stages.plan_ops`` which
does not exist yet. Pytest collection fails with ImportError — expected RED
state for Wave 0 of Phase 11.
"""

import pytest
from unittest.mock import AsyncMock, patch

pytestmark = pytest.mark.asyncio

# ---------------------------------------------------------------------------
# Imports from not-yet-built pipeline targets (RED: ImportError at collection)
# ---------------------------------------------------------------------------
from confluence_logic.pipeline.stages.plan_ops import plan_operation
from confluence_logic.pipeline.contracts import (
    ChangeIntentV3,
    EvidenceSpan,
    SectionCandidate,
    PlannedOperation,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _intent(
    kind: str = "fact_update",
    old_value: str = "old",
    new_value: str = "new",
) -> ChangeIntentV3:
    return ChangeIntentV3(
        kind=kind,
        subject="test subject",
        old_value=old_value,
        new_value=new_value,
        dedup_key="test-op-plan",
        evidence=[EvidenceSpan(text="some evidence", start=0, end=13)],
    )


def _candidate(
    page_id: str = "pg-test",
    section_heading: str = "Test Section",
    score: float = 0.85,
) -> SectionCandidate:
    return SectionCandidate(
        page_id=page_id,
        section_heading=section_heading,
        score=score,
        source="dense",
    )


# ---------------------------------------------------------------------------
# OPS-V3-01-1: Exactly one operation type per (intent, candidate) pair
# ---------------------------------------------------------------------------

async def test_fact_update_with_existing_section_produces_edit_section():
    """OPS-V3-01: fact_update intent with a high-scoring existing section →
    exactly one edit_section operation."""
    intent = _intent(kind="fact_update", old_value="Q3", new_value="Q2")
    candidate = _candidate(page_id="pg-soc2", section_heading="Audit Schedule")

    op = await plan_operation(intent, candidate, page_html="<h2>Audit Schedule</h2><p>Q3 audit scheduled.</p>")

    assert isinstance(op, PlannedOperation)
    assert op.operation == "edit_section", (
        f"Expected edit_section for fact_update with existing section, got {op.operation!r}"
    )


async def test_new_workstream_produces_create_page():
    """OPS-V3-01: new_workstream intent with no_existing_target=True → create_page."""
    intent = _intent(kind="new_workstream", old_value="", new_value="disaster recovery plan")
    candidate = SectionCandidate(
        page_id=None,
        section_heading=None,
        score=0.0,
        source="dense",
        no_existing_target=True,
    )

    op = await plan_operation(intent, candidate, page_html=None)

    assert op.operation == "create_page", (
        f"Expected create_page for new_workstream with no target, got {op.operation!r}"
    )


async def test_deprecation_intent_produces_archive_deprecate():
    """OPS-V3-01: deprecation intent → archive_deprecate."""
    intent = _intent(kind="deprecation", old_value="legacy service", new_value=None)
    candidate = _candidate(page_id="pg-legacy", section_heading="Overview")

    op = await plan_operation(
        intent,
        candidate,
        page_html="<h2>Overview</h2><p>Legacy service handles billing.</p>",
    )

    assert op.operation == "archive_deprecate", (
        f"Expected archive_deprecate for deprecation intent, got {op.operation!r}"
    )


async def test_additive_intent_with_no_existing_section_produces_append():
    """OPS-V3-01: action_item intent with new content and no matching section → append."""
    intent = ChangeIntentV3(
        kind="action_item",
        subject="secrets management",
        old_value="",
        new_value="Use HashiCorp Vault for all secrets.",
        dedup_key="secrets-vault",
        evidence=[EvidenceSpan(text="HashiCorp Vault", start=0, end=15)],
    )
    candidate = _candidate(page_id="pg-security-policy", section_heading="Secrets Management")

    op = await plan_operation(
        intent,
        candidate,
        page_html="<h2>Encryption</h2><p>AES-256 at rest.</p>",  # heading absent
    )

    assert op.operation == "append", (
        f"Expected append for new-section addition, got {op.operation!r}"
    )


# ---------------------------------------------------------------------------
# OPS-V3-01-2: Ambiguous operations must NOT be emitted
# ---------------------------------------------------------------------------

async def test_ambiguous_op_is_not_emitted():
    """OPS-V3-01: when the intent has conflicting signals (old_value not on page,
    new_value vague) the stage must return None — ambiguous ops not emitted."""
    intent = ChangeIntentV3(
        kind="fact_update",
        subject="deployment approach",
        old_value="bare-metal",
        new_value="something better",
        dedup_key="ambiguous-deploy",
        evidence=[EvidenceSpan(text="something better", start=0, end=16)],
    )
    candidate = _candidate(page_id="pg-arch", section_heading="Data Layer")

    op = await plan_operation(
        intent,
        candidate,
        page_html="<h2>Data Layer</h2><p>MySQL handles storage.</p>",
        # old_value "bare-metal" is NOT in the page_html → ambiguous
    )

    assert op is None, (
        "OPS-V3-01: ambiguous operation must not be emitted (return None)"
    )


# ---------------------------------------------------------------------------
# OPS-V3-01-3: Operation has resolved page + section + exact content
# ---------------------------------------------------------------------------

async def test_operation_has_fully_resolved_fields():
    """OPS-V3-01: a non-ambiguous edit_section op must have page_id, page_title,
    section_heading, before_content, and after_content all populated."""
    intent = _intent(kind="fact_update", old_value="Ansible", new_value="Terraform")
    candidate = _candidate(page_id="pg-deployment", section_heading="Topology")

    op = await plan_operation(
        intent,
        candidate,
        page_html="<h2>Topology</h2><p>We deploy via Ansible.</p>",
    )

    assert op is not None
    assert op.page_id == "pg-deployment"
    assert op.section_heading == "Topology"
    assert op.before_content is not None and op.before_content != "", (
        "before_content must be populated for edit_section"
    )
    assert op.after_content is not None and op.after_content != "", (
        "after_content must be populated for edit_section"
    )
