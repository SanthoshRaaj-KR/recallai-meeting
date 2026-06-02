"""Tests for IntentExtractionAgent and VerifierAgent I/O contracts.

All tests fail with ImportError until Wave 2 implements agents_local/ modules.
The import is deferred into each test body so pytest can collect without errors.
"""

from __future__ import annotations

import os

import pytest

from models import LocalDocIntent, LocalDocProposal

# These tests make live OpenAI Agents calls. Skip (don't fail) when no key is
# configured so the suite stays deterministic offline / in CI.
requires_openai = pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set — live agent call skipped",
)

TRANSCRIPT = (
    "We need to update the data retention policy. "
    "Previously we kept records for 7 years but new regulations require 5 years. "
    "Also, the access control process should now require two manager approvals instead of one."
)


@requires_openai
@pytest.mark.asyncio
async def test_intent_extraction_returns_intents():
    """IntentExtractionAgent returns a non-empty list of LocalDocIntent."""
    from agents_local.intent_extraction import IntentExtractionAgent  # ImportError until Wave 2

    agent = IntentExtractionAgent()
    intents = await agent.extract(TRANSCRIPT)
    assert isinstance(intents, list)
    assert len(intents) >= 1
    for intent in intents:
        assert isinstance(intent, LocalDocIntent)
        assert 0.0 <= intent.confidence <= 1.0
        assert intent.verbatim_snippets


@pytest.mark.skip(reason="Requires mocked LLM to inject low-confidence intents — deferred to integration tests")
def test_intent_extraction_confidence_threshold():
    """Intents with confidence < 0.5 are filtered from output."""
    pytest.fail("NOT IMPLEMENTED — IntentExtractionAgent not yet built")


@requires_openai
@pytest.mark.asyncio
async def test_verifier_scores_proposal():
    """VerifierAgent returns scores in [0,1] and a non-empty verifier_note."""
    from agents_local.verifier import VerifierAgent  # ImportError until Wave 2

    agent = VerifierAgent()
    result = await agent.verify(
        before_content="Records retained for 7 years.",
        after_content="Records retained for 5 years.",
        intent_description="Update retention period from 7 to 5 years per new regulations.",
    )
    assert 0.0 <= result.factual_consistency <= 1.0
    assert 0.0 <= result.formatting_integrity <= 1.0
    assert 0.0 <= result.intent_fulfillment <= 1.0
    assert result.verifier_note
