"""Wave 0 RED — Phase 10 end-to-end proposal-quality scorecard.

Phase 10 — auto-propose-pipeline-quality-redesign-v2.

This pytest module is the acceptance gate for PROP-V2-07. It loads the
20 golden transcript fixtures in ``tests/fixtures/transcripts/phase10/``
and runs the Phase 10 pipeline (StructureAware drafter + PageRouter +
GroundingGate + EditorDispatcher) against each one, then asserts the
five quality metrics listed in 10-RESEARCH.md § Validation Architecture.

In Wave 0 every test ``pytest.fail`` with a Wave 0 marker because the
Phase 10 pipeline modules are not yet present. Later waves replace the
bodies with real assertions over the captured proposal payloads.

Run:
    conda activate ml && pytest tests/e2e_proposal_quality_v2_eval.py -v
"""
from __future__ import annotations

import glob
import json
import pathlib
from typing import Any, Dict, List

import pytest

# The pipeline entrypoint already exists — it is the Phase 8 _run_pipeline
# in confluence_logic.review.api. Wave 1+ will wrap it (or replace its
# inner stages) with the PageRouter / StructureAwareDrafter /
# GroundingGate / EditorDispatcher. For Wave 0 the import below is only
# used to demonstrate that the runner is wired against the existing
# module surface; the test bodies are RED stubs.
import confluence_logic.review.api as review_api  # noqa: F401


FIXTURE_DIR = (
    pathlib.Path(__file__).resolve().parent / "fixtures" / "transcripts" / "phase10"
)


def _load_fixtures() -> List[Dict[str, Any]]:
    """Return every JSON fixture under ``tests/fixtures/transcripts/phase10/``."""
    paths = sorted(glob.glob(str(FIXTURE_DIR / "*.json")))
    fixtures: List[Dict[str, Any]] = []
    for path in paths:
        with open(path, "r", encoding="utf-8") as fh:
            fixtures.append(json.load(fh))
    return fixtures


# ---------------------------------------------------------------------------
# PROP-V2-01 — hallucination rate must be exactly 0
# ---------------------------------------------------------------------------


def test_hallucination_rate_zero():
    """PROP-V2-07 / PROP-V2-01: zero cards may have page_id outside the fixture's workspace_pages OR after_content tokens outside ``{transcript ∪ page_content}``."""
    fixtures = _load_fixtures()
    assert len(fixtures) >= 20, f"need >=20 fixtures, found {len(fixtures)}"
    pytest.fail(
        "Wave 0 RED — Phase 10 pipeline (StructureAwareDrafter + GroundingGate) not yet implemented"
    )


# ---------------------------------------------------------------------------
# PROP-V2-03 — targeting recall ≥ 90% on the golden fixture set
# ---------------------------------------------------------------------------


def test_targeting_recall_ge_90():
    """PROP-V2-07 / PROP-V2-03: at least 18 of 20 fixtures hit the expected target page_id from ``expected_proposals``."""
    fixtures = _load_fixtures()
    assert len(fixtures) >= 20, f"need >=20 fixtures, found {len(fixtures)}"
    pytest.fail("Wave 0 RED — PageRouter not yet implemented (Wave 1)")


# ---------------------------------------------------------------------------
# PROP-V2-03 — targeting precision: zero wrong-page cards from qualifier-passing pages
# ---------------------------------------------------------------------------


def test_targeting_precision():
    """PROP-V2-07 / PROP-V2-03: no card from a qualifier-passing page is targeted at a page that is not in the fixture's ``expected_proposals``."""
    fixtures = _load_fixtures()
    assert len(fixtures) >= 20, f"need >=20 fixtures, found {len(fixtures)}"
    pytest.fail("Wave 0 RED — PageRouter + PageQualifier integration pending (Wave 1)")


# ---------------------------------------------------------------------------
# PROP-V2-02 — structure preservation 100% on the 5 reorder fixtures
# ---------------------------------------------------------------------------


def test_structure_preservation_100_on_ordered_procedures():
    """PROP-V2-07 / PROP-V2-02: each of the 5 reorder fixtures produces a ``reorder`` op whose sibling ``<li>`` elements are byte-identical to the before form."""
    fixtures = _load_fixtures()
    reorder_fixtures = [f for f in fixtures if f.get("failure_mode") == "reorder"]
    assert len(reorder_fixtures) == 5, (
        f"need exactly 5 reorder fixtures, found {len(reorder_fixtures)}"
    )
    pytest.fail(
        "Wave 0 RED — StructureAwareDrafter + EditorDispatcher reorder primitive not yet implemented"
    )


# ---------------------------------------------------------------------------
# PROP-V2-04 — card render completeness: every emitted card has the required default-visible fields
# ---------------------------------------------------------------------------


def test_card_render_completeness():
    """PROP-V2-07 / PROP-V2-04: every emitted card has ``change_summary``, breadcrumb, and ``section_heading`` set (the three header elements ProposalCardV2 renders default-visible)."""
    fixtures = _load_fixtures()
    assert len(fixtures) >= 20, f"need >=20 fixtures, found {len(fixtures)}"
    pytest.fail(
        "Wave 0 RED — Phase 10 pipeline does not yet attach change_summary/breadcrumb to verified cards (Wave 2)"
    )
