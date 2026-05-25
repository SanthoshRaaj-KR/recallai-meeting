"""RED tests for grounding gate + calibrated confidence (GND-V3-01 — Phase 11).

GND-V3-01: Hard grounding gate drops any card with an unverifiable page_id
or unsupported tokens in after_content; extends existing test_grounding_gate.py
with Phase 11 additions: calibrated confidence binning and sub-threshold
suppression/flagging.

These tests import from ``confluence_logic.pipeline.stages.gate`` for the v3
gate wrapper AND reuse the existing ``confluence_logic.agents.grounding_gate``
symbols (which already exist). The pipeline.stages.gate import is the RED
import that will fail at collection until Wave 1 lands it.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

pytestmark = pytest.mark.asyncio

# ---------------------------------------------------------------------------
# Import existing symbols (GREEN — already built in Phase 10)
# ---------------------------------------------------------------------------
from confluence_logic.agents.grounding_gate import (
    check_grounding,
    check_page_existence,
    content_bearing_tokens,
)

# ---------------------------------------------------------------------------
# Import from not-yet-built v3 pipeline gate (RED: ImportError at collection)
# ---------------------------------------------------------------------------
from confluence_logic.pipeline.stages.gate import (
    apply_grounding_gate_v3,
    calibrate_confidence,
)
from confluence_logic.pipeline.contracts import (
    PlannedOperation,
    ProposalCardV3,
)


# ---------------------------------------------------------------------------
# GND-V3-01-1: Hard gate drops card with unverifiable page_id (inherited)
# ---------------------------------------------------------------------------

async def test_hard_gate_drops_card_with_unverifiable_page_id():
    """GND-V3-01: a card whose page_id cannot be verified in the graph or
    via live REST is dropped (gate returns None / is_dropped=True)."""
    op = PlannedOperation(
        operation="edit_section",
        page_id="pg-does-not-exist",
        page_title="Ghost Page",
        section_heading="Background",
        before_content="old text",
        after_content="new text",
        rationale="test",
    )
    transcript = "We discussed updating the ghost page."

    with patch(
        "confluence_logic.pipeline.stages.gate._check_page_exists",
        new=AsyncMock(return_value=False),
    ):
        result = await apply_grounding_gate_v3(op, transcript_text=transcript)

    assert result.is_dropped, (
        "GND-V3-01: card with unverifiable page_id must be dropped by hard gate"
    )


# ---------------------------------------------------------------------------
# GND-V3-01-2: Hard gate drops card with unsupported tokens in after_content
# ---------------------------------------------------------------------------

async def test_hard_gate_drops_card_with_unsupported_tokens():
    """GND-V3-01: card whose after_content contains tokens absent from
    {transcript ∪ page_content} must be dropped."""
    op = PlannedOperation(
        operation="edit_section",
        page_id="pg-auth",
        page_title="Authentication",
        section_heading="Provider",
        before_content="We use OAuth.",
        after_content="We use Kerberos for enterprise federated identity management.",
        rationale="switch protocol",
    )
    transcript = "We are switching to SAML."
    page_content = "We use OAuth for all sign-ins."

    with patch(
        "confluence_logic.pipeline.stages.gate._check_page_exists",
        new=AsyncMock(return_value=True),
    ):
        result = await apply_grounding_gate_v3(
            op,
            transcript_text=transcript,
            current_page_content=page_content,
        )

    assert result.is_dropped, (
        "GND-V3-01: 'Kerberos' and 'federated' are not in transcript ∪ page — must drop"
    )
    assert len(result.drop_reasons) > 0


# ---------------------------------------------------------------------------
# GND-V3-01-3: Card with well-grounded content passes the hard gate
# ---------------------------------------------------------------------------

async def test_hard_gate_passes_grounded_card():
    """GND-V3-01: card grounded in both transcript and page must not be dropped."""
    op = PlannedOperation(
        operation="edit_section",
        page_id="pg-auth",
        page_title="Authentication",
        section_heading="Provider",
        before_content="We use OAuth.",
        after_content="We use SAML for all sign-ins.",
        rationale="switch to SAML as announced",
    )
    transcript = "We are switching authentication to SAML for all sign-ins."
    page_content = "We use OAuth for all sign-ins."

    with patch(
        "confluence_logic.pipeline.stages.gate._check_page_exists",
        new=AsyncMock(return_value=True),
    ):
        result = await apply_grounding_gate_v3(
            op,
            transcript_text=transcript,
            current_page_content=page_content,
        )

    assert not result.is_dropped, (
        "GND-V3-01: SAML is in transcript and sign-ins is in page — card must pass"
    )


# ---------------------------------------------------------------------------
# GND-V3-01-4: Calibrated confidence binning
# ---------------------------------------------------------------------------

def test_confidence_calibration_high():
    """GND-V3-01: >0.80 grounding score → confidence='high'."""
    confidence, label = calibrate_confidence(grounding_score=0.91, retrieval_score=0.88)
    assert label == "high", f"Expected 'high', got '{label}'"
    assert 0.0 <= confidence <= 1.0


def test_confidence_calibration_medium():
    """GND-V3-01: 0.60–0.80 grounding score → confidence='medium'."""
    confidence, label = calibrate_confidence(grounding_score=0.70, retrieval_score=0.65)
    assert label == "medium", f"Expected 'medium', got '{label}'"


def test_confidence_calibration_low():
    """GND-V3-01: <0.60 grounding score → confidence='low'."""
    confidence, label = calibrate_confidence(grounding_score=0.45, retrieval_score=0.40)
    assert label == "low", f"Expected 'low', got '{label}'"


# ---------------------------------------------------------------------------
# GND-V3-01-5: Sub-threshold suppression / flagging
# ---------------------------------------------------------------------------

async def test_sub_threshold_card_is_suppressed_not_emitted():
    """GND-V3-01: a card with confidence below the emit threshold must be
    suppressed (not included in the output proposal list) or flagged for review,
    not silently emitted as an approved proposal."""
    op = PlannedOperation(
        operation="edit_section",
        page_id="pg-infra",
        page_title="Infrastructure",
        section_heading="Regions",
        before_content="us-east-1",
        after_content="ap-southeast-2",
        rationale="vague region comment",
    )
    transcript = "Someone mentioned something about regions but wasn't specific."
    page_content = "Production runs in us-east-1 and eu-west-1."

    with patch(
        "confluence_logic.pipeline.stages.gate._check_page_exists",
        new=AsyncMock(return_value=True),
    ), patch(
        "confluence_logic.pipeline.stages.gate._compute_grounding_score",
        return_value=0.30,  # below threshold
    ):
        result = await apply_grounding_gate_v3(
            op,
            transcript_text=transcript,
            current_page_content=page_content,
        )

    # Sub-threshold: either dropped OR explicitly flagged (not silently emitted).
    assert result.is_dropped or result.is_flagged, (
        "GND-V3-01: sub-threshold card must be dropped or flagged, never silently emitted"
    )


# ---------------------------------------------------------------------------
# GND-V3-01-6: content_bearing_tokens reuse (inherited from Phase 10)
# ---------------------------------------------------------------------------

def test_content_bearing_tokens_filters_stopwords():
    """GND-V3-01: content_bearing_tokens must exclude common stopwords."""
    tokens = content_bearing_tokens("We are switching to SAML for all sign-ins.")
    # "we", "are", "to", "for", "all" should be filtered; "saml", "sign-ins" should remain.
    assert "saml" in tokens
    assert "we" not in tokens
    assert "are" not in tokens
