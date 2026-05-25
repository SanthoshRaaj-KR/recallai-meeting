"""Phase 11 e2e proposal-quality scorecard (OBS-V3-01 — v3 pipeline).

Extends the Phase 10 harness shape (``tests/e2e_proposal_quality_v2_eval.py``)
with Phase 11-specific metrics:

    contradiction recall     = 100% (every affected page in a group is proposed)
    retrieval recall@k       = target page in top-k dense+BM25 fused results
    retrieval MRR            = mean reciprocal rank of target page
    retrieval nDCG           = normalized discounted cumulative gain
    hallucination rate       = 0% (GroundingGate is the safety net)
    targeting recall         >= 90% (target page hit)
    structure preservation   = 100% on reorder fixtures
    card render completeness = 100% (group_id + confidence + evidence present)

Deterministic by default: the v3 pipeline's LLM-bound stages are stubbed
with a fixture-specific CANNED_LLM dict keyed by fixture name; ``check_grounding``
and ``check_page_existence`` run for real so hallucinations are dropped on
the same code path production uses.

Run two ways:
    pytest tests/e2e_proposal_quality_v3_eval.py -x        # CI / unit tests
    python -m tests.e2e_proposal_quality_v3_eval           # human scorecard

RED until ``confluence_logic.pipeline.run`` exists (Wave 1+).
"""
from __future__ import annotations

import asyncio
import json
import pathlib
import sys
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import AsyncMock, MagicMock

import pytest

# ---------------------------------------------------------------------------
# Pipeline entrypoint (RED until Wave 1+ lands confluence_logic/pipeline/run.py)
# ---------------------------------------------------------------------------
from confluence_logic.pipeline.run import run as pipeline_run  # noqa: F401

# GroundingGate re-used for real hallucination detection (already GREEN).
from confluence_logic.agents.grounding_gate import content_bearing_tokens

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FIXTURE_DIR = (
    pathlib.Path(__file__).resolve().parent / "fixtures" / "transcripts" / "phase11"
)

HALLUCINATION_ALLOWED = 0
TARGETING_RECALL_MIN = 0.90
CONTRADICTION_RECALL_MIN = 1.00  # Phase 11 key requirement (CON-V3-01)
STRUCTURE_PRESERVATION_MIN = 1.00
RENDER_COMPLETENESS_MIN = 1.00


# ---------------------------------------------------------------------------
# Fixture loader — mirrors Phase 10 harness; extended with 'failure_mode'
# ---------------------------------------------------------------------------

def load_fixtures() -> List[Dict[str, Any]]:
    """Return every phase11 JSON fixture sorted by name."""
    paths = sorted(FIXTURE_DIR.glob("*.json"))
    fixtures: List[Dict[str, Any]] = []
    for path in paths:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        # Schema validation.
        assert "name" in data, f"{path.name}: missing 'name'"
        assert "failure_mode" in data, f"{path.name}: missing 'failure_mode'"
        assert "transcript_text" in data, f"{path.name}: missing 'transcript_text'"
        assert "confluence_workspace_pages" in data, (
            f"{path.name}: missing 'confluence_workspace_pages'"
        )
        assert "expected_proposals" in data, f"{path.name}: missing 'expected_proposals'"
        fixtures.append(data)
    return fixtures


def _fixture_target_page_ids(fixture: Dict[str, Any]) -> List[str]:
    """All expected target page_ids (may be multiple for contradiction fixtures)."""
    return [
        p.get("page_id")
        for p in fixture.get("expected_proposals") or []
        if p.get("page_id") is not None
    ]


def _fixture_group_ids(fixture: Dict[str, Any]) -> List[str]:
    """group_ids present in expected_proposals for contradiction fixtures."""
    return list({
        p.get("group_id")
        for p in fixture.get("expected_proposals") or []
        if p.get("group_id") is not None
    })


# ---------------------------------------------------------------------------
# CANNED_LLM — per-fixture deterministic stage outputs
# (stubbed responses for the v3 pipeline's LLM stages)
# ---------------------------------------------------------------------------
# Format per entry:
#   "change_intents": list of ChangeIntentV3-like dicts (extract stage output)
#   "rerank_order":   list of section ids in reranker preferred order
#   "plan_ops":       dict { page_id::section_heading -> PlannedOperation kwargs }

CANNED_LLM: Dict[str, Dict[str, Any]] = {
    # ── Contradiction fixtures ────────────────────────────────────────────────
    "contradiction_soc2_q3_to_q2_two_pages": {
        "change_intents": [
            dict(
                kind="fact_update",
                subject="SOC2 audit schedule",
                old_value="Q3",
                new_value="Q2",
                dedup_key="soc2-audit-q3-to-q2",
                verbatim_quote="SOC2 moved from Q3 to Q2",
            )
        ],
        "rerank_order": ["pg-soc2::Audit Schedule", "pg-security-overview::Audit Schedule"],
        "plan_ops": {
            "pg-soc2::Audit Schedule": dict(
                operation="edit_section",
                before_content="Q3 2025",
                after_content="Q2 2025",
                rationale="Audit moved to Q2 per meeting decision.",
            ),
            "pg-security-overview::Audit Schedule": dict(
                operation="edit_section",
                before_content="Q3",
                after_content="Q2",
                rationale="Stale Q3 reference on Security Overview page.",
            ),
        },
    },
    "contradiction_db_migration_three_pages": {
        "change_intents": [
            dict(
                kind="fact_update",
                subject="database engine",
                old_value="MySQL",
                new_value="Postgres",
                dedup_key="db-mysql-to-postgres",
                verbatim_quote="migrated from MySQL to Postgres",
            )
        ],
        "rerank_order": ["pg-db-runbook::Engine", "pg-arch::Data Layer", "pg-onboarding::Database Access"],
        "plan_ops": {
            "pg-db-runbook::Engine": dict(
                operation="edit_section",
                before_content="MySQL 8.0",
                after_content="Postgres",
                rationale="Database migrated to Postgres.",
            ),
            "pg-arch::Data Layer": dict(
                operation="edit_section",
                before_content="MySQL",
                after_content="Postgres",
                rationale="Architecture doc references old DB.",
            ),
            "pg-onboarding::Database Access": dict(
                operation="edit_section",
                before_content="MySQL",
                after_content="Postgres",
                rationale="Onboarding should reference new DB.",
            ),
        },
    },
    "contradiction_auth_provider_two_pages": {
        "change_intents": [
            dict(
                kind="fact_update",
                subject="authentication provider",
                old_value="OAuth",
                new_value="SAML",
                dedup_key="auth-oauth-to-saml",
                verbatim_quote="switching from OAuth to SAML",
            )
        ],
        "rerank_order": ["pg-auth::Provider", "pg-security-policy::Authentication Standard"],
        "plan_ops": {
            "pg-auth::Provider": dict(
                operation="edit_section",
                before_content="OAuth 2.0",
                after_content="SAML",
                rationale="Enterprise auth switched to SAML.",
            ),
            "pg-security-policy::Authentication Standard": dict(
                operation="edit_section",
                before_content="OAuth",
                after_content="SAML",
                rationale="Security policy must reference SAML.",
            ),
        },
    },
    "contradiction_deprecate_old_service": {
        "change_intents": [
            dict(
                kind="deprecation",
                subject="legacy billing service",
                old_value="legacy billing service",
                new_value=None,
                dedup_key="legacy-billing-deprecation",
                verbatim_quote="legacy billing service is fully deprecated",
            )
        ],
        "rerank_order": ["pg-billing-arch::Components", "pg-integration-guide::Billing Integration"],
        "plan_ops": {
            "pg-billing-arch::Components": dict(
                operation="archive_deprecate",
                before_content=None,
                after_content=None,
                rationale="Billing Architecture page fully superseded.",
            ),
            "pg-integration-guide::Billing Integration": dict(
                operation="edit_section",
                before_content="legacy billing service",
                after_content="payments platform",
                rationale="Update integration guide to new payments platform.",
            ),
        },
    },
    "contradiction_policy_version_bump": {
        "change_intents": [
            dict(
                kind="fact_update",
                subject="data retention window",
                old_value="90 days",
                new_value="180 days",
                dedup_key="data-retention-90-to-180",
                verbatim_quote="retention policy updated from 90 days to 180 days",
            )
        ],
        "rerank_order": ["pg-privacy::Data Retention", "pg-compliance-checklist::Data Retention"],
        "plan_ops": {
            "pg-privacy::Data Retention": dict(
                operation="edit_section",
                before_content="90 days",
                after_content="180 days",
                rationale="Retention updated per legal.",
            ),
            "pg-compliance-checklist::Data Retention": dict(
                operation="edit_section",
                before_content="90-day",
                after_content="180-day",
                rationale="Compliance checklist must reflect new window.",
            ),
        },
    },
    # ── Hallucinate fixtures (pipeline should produce 0 proposals) ────────────
    "hallucinate_jargon_not_in_transcript": {
        "change_intents": [],
        "rerank_order": [],
        "plan_ops": {},
    },
    "hallucinate_nonexistent_page": {
        "change_intents": [
            dict(
                kind="fact_update",
                subject="Quarterly Forecast Tracker",
                old_value="",
                new_value="Q2 numbers",
                dedup_key="hallucinate-qft",
                verbatim_quote="Quarterly Forecast Tracker",
            )
        ],
        "rerank_order": [],  # no matching section in workspace
        "plan_ops": {},
    },
    "hallucinate_new_token_in_after_content": {
        "change_intents": [],
        "rerank_order": [],
        "plan_ops": {},
    },
    "hallucinate_number_not_said": {
        "change_intents": [],
        "rerank_order": [],
        "plan_ops": {},
    },
    # ── Reorder fixtures ──────────────────────────────────────────────────────
    "reorder_login_before_signup": {
        "change_intents": [
            dict(
                kind="decision",
                subject="user flow step order",
                old_value="Signup before Login",
                new_value="Login before Signup",
                dedup_key="user-flow-step-order",
                verbatim_quote="Login should come before signup",
            )
        ],
        "rerank_order": ["pg-user-flow::Steps"],
        "plan_ops": {
            "pg-user-flow::Steps": dict(
                operation="edit_section",
                before_content=None,
                after_content=None,
                from_index=1,
                to_index=0,
                rationale="Login should precede Signup in user flow.",
            ),
        },
    },
    "reorder_deploy_steps": {
        "change_intents": [
            dict(
                kind="decision",
                subject="deploy step order",
                old_value="traffic shift before smoke test",
                new_value="smoke test before traffic shift",
                dedup_key="deploy-step-order",
                verbatim_quote="smoke test should happen before the traffic shift",
            )
        ],
        "rerank_order": ["pg-deploy-runbook::Deploy Steps"],
        "plan_ops": {
            "pg-deploy-runbook::Deploy Steps": dict(
                operation="edit_section",
                before_content=None,
                after_content=None,
                from_index=2,
                to_index=1,
                rationale="Smoke tests before traffic shift.",
            ),
        },
    },
    "reorder_incident_priority": {
        "change_intents": [
            dict(
                kind="decision",
                subject="escalation step order",
                old_value="on-call before lead",
                new_value="lead before on-call",
                dedup_key="escalation-order",
                verbatim_quote="Notifying the team lead should come before paging the on-call",
            )
        ],
        "rerank_order": ["pg-oncall-runbook::Escalation Steps"],
        "plan_ops": {
            "pg-oncall-runbook::Escalation Steps": dict(
                operation="edit_section",
                before_content=None,
                after_content=None,
                from_index=2,
                to_index=1,
                rationale="Lead notified before on-call page.",
            ),
        },
    },
    # ── Edit-section fixtures ─────────────────────────────────────────────────
    "edit_section_python_version": {
        "change_intents": [
            dict(
                kind="fact_update",
                subject="Python version",
                old_value="Python 2.7",
                new_value="Python 3.11",
                dedup_key="python-version",
                verbatim_quote="Python 3.11",
            )
        ],
        "rerank_order": ["pg-runbook::Setup"],
        "plan_ops": {
            "pg-runbook::Setup": dict(
                operation="edit_section",
                before_content="Python 2.7",
                after_content="Python 3.11",
                rationale="Runbook still referenced Python 2.7.",
            ),
        },
    },
    "edit_section_region_update": {
        "change_intents": [
            dict(
                kind="fact_update",
                subject="production regions",
                old_value="us-east-1 and eu-west-1",
                new_value="us-east-1, eu-west-1, and ap-southeast-1",
                dedup_key="region-singapore",
                verbatim_quote="ap-southeast-1 as a third production region",
            )
        ],
        "rerank_order": ["pg-infra::Regions"],
        "plan_ops": {
            "pg-infra::Regions": dict(
                operation="edit_section",
                before_content="us-east-1 and eu-west-1",
                after_content="us-east-1, eu-west-1, and ap-southeast-1",
                rationale="Singapore region added.",
            ),
        },
    },
    "edit_section_oncall_tool": {
        "change_intents": [
            dict(
                kind="fact_update",
                subject="on-call tool",
                old_value="PagerDuty",
                new_value="OpsGenie",
                dedup_key="oncall-tool",
                verbatim_quote="OpsGenie for on-call alerting",
            )
        ],
        "rerank_order": ["pg-runbook::On-call"],
        "plan_ops": {
            "pg-runbook::On-call": dict(
                operation="edit_section",
                before_content="PagerDuty",
                after_content="OpsGenie",
                rationale="On-call tool switched to OpsGenie.",
            ),
        },
    },
    # ── Append fixtures ───────────────────────────────────────────────────────
    "append_new_security_section": {
        "change_intents": [
            dict(
                kind="action_item",
                subject="secrets management",
                old_value="",
                new_value="HashiCorp Vault for secrets",
                dedup_key="secrets-vault",
                verbatim_quote="HashiCorp Vault",
            )
        ],
        "rerank_order": ["pg-security-policy::Secrets Management"],
        "plan_ops": {
            "pg-security-policy::Secrets Management": dict(
                operation="append",
                before_content=None,
                after_content="HashiCorp Vault is used for secrets management. All service credentials must be stored in Vault.",
                rationale="New Vault policy adopted.",
            ),
        },
    },
    "append_rate_limit_note": {
        "change_intents": [
            dict(
                kind="action_item",
                subject="API rate limit burst",
                old_value="",
                new_value="burst of 200 for first 10 seconds",
                dedup_key="rate-limit-burst",
                verbatim_quote="burst of up to 200",
            )
        ],
        "rerank_order": ["pg-api-docs::Rate Limits"],
        "plan_ops": {
            "pg-api-docs::Rate Limits": dict(
                operation="append",
                before_content=None,
                after_content="A burst allowance of up to 200 requests is permitted for the first 10 seconds of each minute.",
                rationale="Burst allowance documented.",
            ),
        },
    },
    "append_new_team_member": {
        "change_intents": [
            dict(
                kind="action_item",
                subject="team roster",
                old_value="",
                new_value="Jordan (Backend)",
                dedup_key="team-roster-jordan",
                verbatim_quote="Jordan",
            )
        ],
        "rerank_order": ["pg-team-roster::Members"],
        "plan_ops": {
            "pg-team-roster::Members": dict(
                operation="append",
                before_content=None,
                after_content="Jordan (Backend)",
                rationale="New hire Jordan joins backend team.",
            ),
        },
    },
    # ── Create-page fixtures ──────────────────────────────────────────────────
    "create_page_disaster_recovery": {
        "change_intents": [
            dict(
                kind="new_workstream",
                subject="disaster recovery runbook",
                old_value="",
                new_value="DR runbook with RTO, RPO, failover steps",
                dedup_key="create-dr-runbook",
                verbatim_quote="disaster recovery runbook",
            )
        ],
        "rerank_order": [],
        "plan_ops": {
            None: dict(
                operation="create_page",
                page_title="Disaster Recovery Runbook",
                before_content=None,
                after_content="# Disaster Recovery Runbook\n\n## RTO and RPO\nRTO: 4h. RPO: 1h.",
                rationale="No DR runbook existed.",
            ),
        },
    },
    "create_page_api_changelog": {
        "change_intents": [
            dict(
                kind="new_workstream",
                subject="API changelog",
                old_value="",
                new_value="v2 breaking changes: auth header rename, pagination cursor",
                dedup_key="create-api-changelog",
                verbatim_quote="API Changelog",
            )
        ],
        "rerank_order": [],
        "plan_ops": {
            None: dict(
                operation="create_page",
                page_title="API Changelog",
                before_content=None,
                after_content="# API Changelog\n\n## v2.0 Breaking Changes\n- Auth header renamed\n- Cursor-based pagination",
                rationale="No changelog page existed.",
            ),
        },
    },
    # ── Archive-deprecate fixtures ────────────────────────────────────────────
    "archive_deprecate_old_runbook": {
        "change_intents": [
            dict(
                kind="deprecation",
                subject="API Runbook v1",
                old_value="",
                new_value="",
                dedup_key="archive-api-runbook-v1",
                verbatim_quote="archive the old v1 runbook",
            )
        ],
        "rerank_order": ["pg-api-runbook-v1::Overview"],
        "plan_ops": {
            "pg-api-runbook-v1::Overview": dict(
                operation="archive_deprecate",
                before_content=None,
                after_content=None,
                rationale="v1 runbook superseded by v2.",
            ),
        },
    },
    "archive_deprecate_stale_soc2_page": {
        "change_intents": [
            dict(
                kind="deprecation",
                subject="SOC2 2023 summary",
                old_value="",
                new_value="",
                dedup_key="archive-soc2-2023",
                verbatim_quote="archive the 2023 page",
            )
        ],
        "rerank_order": ["pg-soc2-2023::Scope"],
        "plan_ops": {
            "pg-soc2-2023::Scope": dict(
                operation="archive_deprecate",
                before_content=None,
                after_content=None,
                rationale="SOC2 2023 is fully superseded by 2024.",
            ),
        },
    },
}


# ---------------------------------------------------------------------------
# Deterministic swap helpers (mirrors Phase 10 harness)
# ---------------------------------------------------------------------------

def _swap(module, name: str, new_value: Any) -> Any:
    """Replace module attribute, return old value for restoration."""
    old = getattr(module, name, None)
    setattr(module, name, new_value)
    return old


# ---------------------------------------------------------------------------
# Metric functions (placeholder — RED until pipeline.run exists)
# ---------------------------------------------------------------------------

def _measure_contradiction_recall(
    fixture: Dict[str, Any],
    proposals: List[Dict[str, Any]],
) -> float:
    """Return fraction of expected contradiction pages that appear in proposals.

    For contradiction fixtures: every page in expected_proposals with a
    group_id must be in the actual proposals list.
    """
    expected = [
        p for p in fixture.get("expected_proposals") or []
        if p.get("group_id") is not None
    ]
    if not expected:
        return 1.0  # non-contradiction fixture: not applicable

    expected_ids = {p["page_id"] for p in expected if p.get("page_id")}
    actual_ids = {p.get("page_id") for p in proposals}
    if not expected_ids:
        return 1.0
    return len(expected_ids & actual_ids) / len(expected_ids)


def _measure_retrieval_recall_at_k(
    fixture: Dict[str, Any],
    retrieval_result: Optional[Any],
    k: int = 5,
) -> float:
    """Return 1.0 if any expected page_id appears in top-k retrieved candidates."""
    # Placeholder: returns 0.0 until pipeline.run provides retrieval_result.
    if retrieval_result is None:
        return 0.0
    # TODO (Wave 1+): extract candidates from retrieval_result, check top-k.
    return 0.0


def _measure_mrr(fixture: Dict[str, Any], retrieval_result: Optional[Any]) -> float:
    """Mean Reciprocal Rank — placeholder until pipeline.run exists."""
    if retrieval_result is None:
        return 0.0
    # TODO (Wave 1+): compute from retrieval_result rank lists.
    return 0.0


# ---------------------------------------------------------------------------
# Scorecard runner — RED until pipeline.run exists
# ---------------------------------------------------------------------------

async def _run_v3_pipeline_for_fixture(
    fixture: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], Optional[Any]]:
    """Run the v3 pipeline stub against one fixture.

    Returns (proposals, retrieval_trace). Both are None-filled until
    pipeline.run exists.  This function is the integration point for Wave 1+.
    """
    # RED: pipeline_run import fails at collection; this body never executes.
    canned = CANNED_LLM.get(fixture["name"], {})
    _ = canned  # will be wired to pipeline stubs in Wave 1
    raise NotImplementedError(
        "v3 pipeline.run not built yet (Wave 0 RED) — this path is unreachable "
        "until Wave 1+ lands confluence_logic/pipeline/run.py"
    )


# ---------------------------------------------------------------------------
# pytest tests (RED until pipeline.run exists)
# ---------------------------------------------------------------------------

@pytest.fixture
def all_fixtures() -> List[Dict[str, Any]]:
    return load_fixtures()


def test_fixture_loader_finds_phase11_fixtures(all_fixtures):
    """OBS-V3-01: fixture loader must find all phase11 fixtures (≥20)."""
    assert len(all_fixtures) >= 20, (
        f"Expected ≥20 phase11 fixtures, found {len(all_fixtures)}"
    )


def test_fixture_loader_finds_contradiction_fixtures(all_fixtures):
    """OBS-V3-01: at least 5 contradiction fixtures must be present."""
    contradiction = [f for f in all_fixtures if f["failure_mode"] == "contradiction"]
    assert len(contradiction) >= 5, (
        f"Expected ≥5 contradiction fixtures, found {len(contradiction)}"
    )


def test_canned_llm_covers_all_fixtures(all_fixtures):
    """OBS-V3-01: every fixture must have a CANNED_LLM entry for deterministic runs."""
    missing = [f["name"] for f in all_fixtures if f["name"] not in CANNED_LLM]
    assert len(missing) == 0, (
        f"Missing CANNED_LLM entries for fixtures: {missing}"
    )


@pytest.mark.asyncio
async def test_contradiction_recall_metric_shape(all_fixtures):
    """OBS-V3-01: contradiction_recall metric must return float in [0, 1]."""
    for fixture in all_fixtures:
        recall = _measure_contradiction_recall(fixture, proposals=[])
        assert 0.0 <= recall <= 1.0, (
            f"contradiction_recall out of range for '{fixture['name']}': {recall}"
        )


@pytest.mark.asyncio
async def test_pipeline_run_is_importable():
    """OBS-V3-01 RED: pipeline_run must be importable (collection-level RED gate).

    This test FAILS at collection until confluence_logic/pipeline/run.py exists.
    """
    # If collection succeeded (import above), this body is reachable.
    # At Wave 0 it is never reached because the import at module level fails first.
    assert callable(pipeline_run), "pipeline_run must be a callable async function"


# ---------------------------------------------------------------------------
# Human scorecard CLI (mirrors Phase 10 __main__ block)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import textwrap

    fixtures = load_fixtures()
    print(f"\nPhase 11 v3 scorecard — {len(fixtures)} fixtures\n")
    print("(RED — pipeline.run not yet built; run after Wave 1+ to see real scores)\n")

    # Metric summary (placeholder).
    rows = []
    for fx in fixtures:
        mode = fx["failure_mode"]
        canned = CANNED_LLM.get(fx["name"])
        has_canned = "YES" if canned is not None else "MISSING"
        rows.append(f"  {fx['name']:<55} mode={mode:<15} canned={has_canned}")

    print("Fixture inventory:")
    print("\n".join(rows))
    print(f"\nTotal: {len(fixtures)} fixtures, {len(CANNED_LLM)} canned LLM entries")
    n_contradiction = sum(1 for f in fixtures if f["failure_mode"] == "contradiction")
    print(f"Contradiction fixtures: {n_contradiction} (target ≥5)")
    missing = [f["name"] for f in fixtures if f["name"] not in CANNED_LLM]
    if missing:
        print(f"WARNING: {len(missing)} fixtures missing CANNED_LLM entries: {missing}")
    else:
        print("All fixtures have CANNED_LLM entries.")
