"""RED tests for structured extraction stage (EXT-V3-01 — Phase 11).

Requirement EXT-V3-01: Every ChangeIntentV3 must have ≥1 verbatim evidence
span with char offsets computed by Python str.find; final-state collapse
deduplicates contradictory intents; no intent is emitted without evidence.

These tests import from ``confluence_logic.pipeline.stages.extract`` which
does not exist yet. Pytest collection fails with ImportError — expected RED
state for Wave 0 of Phase 11.
"""

import pytest
from unittest.mock import AsyncMock, patch

pytestmark = pytest.mark.asyncio

# ---------------------------------------------------------------------------
# Import from not-yet-built pipeline target (RED: ImportError at collection)
# ---------------------------------------------------------------------------
from confluence_logic.pipeline.stages.extract import extract_intents
from confluence_logic.pipeline.contracts import ChangeIntentV3, EvidenceSpan


# ---------------------------------------------------------------------------
# EXT-V3-01-1: Each intent has ≥1 verbatim evidence span
# ---------------------------------------------------------------------------

async def test_every_intent_has_at_least_one_evidence_span():
    """EXT-V3-01: extract_intents must not emit an intent with empty evidence."""
    transcript = (
        "Alice: We are moving the SOC2 audit from Q3 to Q2 this year.\n"
        "Bob: Confirmed, Q2 is the new target."
    )
    with patch(
        "confluence_logic.pipeline.stages.extract._run_llm_extraction",
        new=AsyncMock(return_value=[
            {
                "kind": "fact_update",
                "subject": "SOC2 audit schedule",
                "old_value": "Q3",
                "new_value": "Q2",
                "dedup_key": "soc2-audit-quarter",
            }
        ]),
    ):
        intents = await extract_intents(transcript)

    assert len(intents) > 0, "expected at least one intent"
    for intent in intents:
        assert isinstance(intent, ChangeIntentV3), f"expected ChangeIntentV3, got {type(intent)}"
        assert len(intent.evidence) >= 1, (
            f"intent '{intent.subject}' has no evidence spans — EXT-V3-01 violation"
        )


# ---------------------------------------------------------------------------
# EXT-V3-01-2: Evidence spans are verbatim from transcript (str.find offsets)
# ---------------------------------------------------------------------------

async def test_evidence_span_offsets_match_str_find():
    """EXT-V3-01: span.start and span.end must equal str.find(span.text) offsets."""
    transcript = "We switched authentication from OAuth to SAML for all sign-ins."
    with patch(
        "confluence_logic.pipeline.stages.extract._run_llm_extraction",
        new=AsyncMock(return_value=[
            {
                "kind": "fact_update",
                "subject": "authentication provider",
                "old_value": "OAuth",
                "new_value": "SAML",
                "dedup_key": "auth-provider",
                "verbatim_quote": "OAuth to SAML",
            }
        ]),
    ):
        intents = await extract_intents(transcript)

    assert len(intents) > 0
    for intent in intents:
        for span in intent.evidence:
            start = transcript.find(span.text)
            assert start != -1, (
                f"Evidence text '{span.text}' not found verbatim in transcript — "
                "EXT-V3-01 requires verbatim span"
            )
            assert span.start == start, (
                f"span.start={span.start} != str.find result {start}"
            )
            assert span.end == start + len(span.text), (
                f"span.end={span.end} != start+len {start + len(span.text)}"
            )


# ---------------------------------------------------------------------------
# EXT-V3-01-3: Final-state collapse deduplicates contradictory intents
# ---------------------------------------------------------------------------

async def test_final_state_collapse_deduplicates_same_dedup_key():
    """EXT-V3-01: two intents with the same dedup_key collapse to the last one."""
    transcript = (
        "First we thought Q3. Actually wait, Q2. No — definitely Q2."
    )
    with patch(
        "confluence_logic.pipeline.stages.extract._run_llm_extraction",
        new=AsyncMock(return_value=[
            {
                "kind": "fact_update",
                "subject": "audit quarter",
                "old_value": "current",
                "new_value": "Q3",
                "dedup_key": "audit-quarter-final",
                "verbatim_quote": "thought Q3",
            },
            {
                "kind": "fact_update",
                "subject": "audit quarter",
                "old_value": "Q3",
                "new_value": "Q2",
                "dedup_key": "audit-quarter-final",
                "verbatim_quote": "definitely Q2",
            },
        ]),
    ):
        intents = await extract_intents(transcript)

    dedup_keys = [i.dedup_key for i in intents]
    assert dedup_keys.count("audit-quarter-final") == 1, (
        "Final-state collapse must deduplicate to one intent per dedup_key"
    )
    surviving = next(i for i in intents if i.dedup_key == "audit-quarter-final")
    assert surviving.new_value == "Q2", (
        "Final-state collapse must keep the LAST stated value"
    )


# ---------------------------------------------------------------------------
# EXT-V3-01-4: No intent emitted when LLM returns empty evidence
# ---------------------------------------------------------------------------

async def test_intent_without_verbatim_quote_is_dropped():
    """EXT-V3-01: if no verbatim span can be located in transcript, intent is dropped."""
    transcript = "Just a general discussion with no specific change decisions."
    with patch(
        "confluence_logic.pipeline.stages.extract._run_llm_extraction",
        new=AsyncMock(return_value=[
            {
                "kind": "new_workstream",
                "subject": "some imagined project",
                "old_value": "",
                "new_value": "launch it",
                "dedup_key": "imagined-project",
                "verbatim_quote": "unicorn project that is not in the transcript",
            }
        ]),
    ):
        intents = await extract_intents(transcript)

    # The intent whose verbatim quote cannot be found in transcript must be dropped.
    assert len(intents) == 0, (
        "Intents with unresolvable evidence spans must be dropped (EXT-V3-01)"
    )


# ---------------------------------------------------------------------------
# EXT-V3-01-5: Kind discriminator accepts all valid kinds
# ---------------------------------------------------------------------------

def test_change_intent_v3_kinds():
    """EXT-V3-01: ChangeIntentV3.kind accepts all documented discriminator values."""
    valid_kinds = [
        "decision",
        "fact_update",
        "action_item",
        "new_workstream",
        "deprecation",
    ]
    span = EvidenceSpan(text="some evidence", start=0, end=13)
    for kind in valid_kinds:
        intent = ChangeIntentV3(
            kind=kind,
            subject="test",
            old_value="old",
            new_value="new",
            dedup_key=f"test-{kind}",
            evidence=[span],
        )
        assert intent.kind == kind
