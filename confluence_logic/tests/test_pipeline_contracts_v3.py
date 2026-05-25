"""RED tests for pipeline modularization (ARCH-V3-01 — Phase 11).

Each stage must be importable in isolation (no side-effectful top-level
imports of live connectors), stage contracts must validate via Pydantic,
and killswitch env flags must toggle behavior without code changes.

These tests import from ``confluence_logic.pipeline.*`` which does not exist
yet.  Pytest collection will fail with an ImportError — that is the expected
RED state for Wave 0 of Phase 11.
"""

import pytest

pytestmark = pytest.mark.asyncio

# ---------------------------------------------------------------------------
# Imports from not-yet-built pipeline targets (RED: ImportError at collection)
# ---------------------------------------------------------------------------
from confluence_logic.pipeline.contracts import (
    EvidenceSpan,
    ChangeIntentV3,
    SectionCandidate,
    RetrievalResult,
    ContradictionGroup,
    PlannedOperation,
    ProposalCardV3,
    StageTrace,
)
from confluence_logic.pipeline.run import run as pipeline_run


# ---------------------------------------------------------------------------
# ARCH-V3-01-1: Stage isolation — each stage importable without live creds
# ---------------------------------------------------------------------------

async def test_extract_stage_importable_in_isolation():
    """ARCH-V3-01: extract stage imports without triggering OpenAI client init."""
    from confluence_logic.pipeline.stages import extract  # noqa: F401


async def test_retrieve_stage_importable_in_isolation():
    """ARCH-V3-01: retrieve stage imports without hitting Pinecone or Neo4j."""
    from confluence_logic.pipeline.stages import retrieve  # noqa: F401


async def test_rerank_stage_importable_in_isolation():
    """ARCH-V3-01: rerank stage imports without triggering LLM client."""
    from confluence_logic.pipeline.stages import rerank  # noqa: F401


async def test_iterate_stage_importable_in_isolation():
    """ARCH-V3-01: iterate stage imports without starting an Agent runner."""
    from confluence_logic.pipeline.stages import iterate  # noqa: F401


async def test_contradict_stage_importable_in_isolation():
    """ARCH-V3-01: contradict stage imports without Neo4j driver init."""
    from confluence_logic.pipeline.stages import contradict  # noqa: F401


async def test_gate_stage_importable_in_isolation():
    """ARCH-V3-01: gate stage imports without live Confluence connector."""
    from confluence_logic.pipeline.stages import gate  # noqa: F401


async def test_plan_ops_stage_importable_in_isolation():
    """ARCH-V3-01: plan_ops stage imports cleanly."""
    from confluence_logic.pipeline.stages import plan_ops  # noqa: F401


# ---------------------------------------------------------------------------
# ARCH-V3-01-2: Contract validation — Pydantic models accept valid payloads
# ---------------------------------------------------------------------------

def test_evidence_span_validates():
    """ARCH-V3-01: EvidenceSpan must validate with char offsets."""
    span = EvidenceSpan(
        text="SOC2 audit moved to Q2",
        start=10,
        end=32,
    )
    assert span.text == "SOC2 audit moved to Q2"
    assert span.start < span.end


def test_change_intent_v3_requires_evidence():
    """ARCH-V3-01: ChangeIntentV3 must have at least one EvidenceSpan."""
    with pytest.raises(Exception):
        # evidence must be non-empty list
        ChangeIntentV3(
            kind="fact_update",
            subject="SOC2 audit",
            old_value="Q3",
            new_value="Q2",
            dedup_key="soc2-audit-quarter",
            evidence=[],  # violates ≥1 constraint
        )


def test_change_intent_v3_valid():
    """ARCH-V3-01: ChangeIntentV3 validates with one span."""
    intent = ChangeIntentV3(
        kind="fact_update",
        subject="SOC2 audit schedule",
        old_value="Q3 2025",
        new_value="Q2 2025",
        dedup_key="soc2-audit-q3-q2",
        evidence=[
            EvidenceSpan(text="SOC2 moved from Q3 to Q2", start=0, end=24)
        ],
    )
    assert intent.kind == "fact_update"
    assert len(intent.evidence) >= 1


def test_planned_operation_valid_edit():
    """ARCH-V3-01: PlannedOperation with edit_section is valid."""
    op = PlannedOperation(
        operation="edit_section",
        page_id="pg-soc2",
        page_title="SOC2 Compliance",
        section_heading="Audit Schedule",
        before_content="Q3 2025",
        after_content="Q2 2025",
        rationale="Audit moved from Q3 to Q2 per meeting decision.",
    )
    assert op.operation == "edit_section"


def test_planned_operation_valid_create():
    """ARCH-V3-01: PlannedOperation with create_page is valid."""
    op = PlannedOperation(
        operation="create_page",
        page_id=None,
        page_title="Disaster Recovery Runbook",
        section_heading=None,
        before_content=None,
        after_content="RTO: 4h. RPO: 1h.",
        rationale="No DR runbook existed.",
    )
    assert op.operation == "create_page"
    assert op.page_id is None


def test_planned_operation_valid_archive():
    """ARCH-V3-01: PlannedOperation with archive_deprecate is valid."""
    op = PlannedOperation(
        operation="archive_deprecate",
        page_id="pg-api-runbook-v1",
        page_title="API Runbook v1 (Legacy)",
        section_heading=None,
        before_content=None,
        after_content=None,
        rationale="v1 is obsolete per meeting decision.",
    )
    assert op.operation == "archive_deprecate"


def test_stage_trace_validates():
    """ARCH-V3-01: StageTrace records latency and counts."""
    trace = StageTrace(
        stage="extract",
        phase="end",
        latency_ms=123.4,
        candidates_in=1,
        candidates_out=2,
        dropped=0,
        drop_reason=None,
        gate=None,
    )
    assert trace.stage == "extract"
    assert trace.latency_ms > 0


# ---------------------------------------------------------------------------
# ARCH-V3-01-3: Killswitch env flags toggle behavior
# ---------------------------------------------------------------------------

def test_killswitch_v3_enabled_default(monkeypatch):
    """ARCH-V3-01: JARVIS_PIPELINE_V3_ENABLED defaults to True when unset."""
    monkeypatch.delenv("JARVIS_PIPELINE_V3_ENABLED", raising=False)
    from importlib import reload
    import confluence_logic.pipeline.context as ctx_mod
    reload(ctx_mod)
    assert ctx_mod.JARVIS_PIPELINE_V3_ENABLED is True


def test_killswitch_v3_can_be_disabled(monkeypatch):
    """ARCH-V3-01: setting JARVIS_PIPELINE_V3_ENABLED=0 disables v3 pipeline."""
    monkeypatch.setenv("JARVIS_PIPELINE_V3_ENABLED", "0")
    from importlib import reload
    import confluence_logic.pipeline.context as ctx_mod
    reload(ctx_mod)
    assert ctx_mod.JARVIS_PIPELINE_V3_ENABLED is False


def test_killswitch_empty_string_does_not_disable(monkeypatch):
    """ARCH-V3-01: empty string env var must NOT flip the default to False (WR-07)."""
    monkeypatch.setenv("JARVIS_PIPELINE_V3_ENABLED", "")
    from importlib import reload
    import confluence_logic.pipeline.context as ctx_mod
    reload(ctx_mod)
    assert ctx_mod.JARVIS_PIPELINE_V3_ENABLED is True


def test_killswitch_rerank_can_be_disabled(monkeypatch):
    """ARCH-V3-01: JARVIS_V3_RERANK_ENABLED=0 disables reranking stage."""
    monkeypatch.setenv("JARVIS_V3_RERANK_ENABLED", "0")
    from importlib import reload
    import confluence_logic.pipeline.context as ctx_mod
    reload(ctx_mod)
    assert ctx_mod.JARVIS_V3_RERANK_ENABLED is False


def test_killswitch_contradiction_can_be_disabled(monkeypatch):
    """ARCH-V3-01: JARVIS_V3_CONTRADICTION_ENABLED=0 disables contradiction sweep."""
    monkeypatch.setenv("JARVIS_V3_CONTRADICTION_ENABLED", "0")
    from importlib import reload
    import confluence_logic.pipeline.context as ctx_mod
    reload(ctx_mod)
    assert ctx_mod.JARVIS_V3_CONTRADICTION_ENABLED is False
