"""Explicit-create phrase recognition — Phase 8 / D-02.

Asserts each of the 6 EXPLICIT CREATE phrases in FACT_EXTRACTION_PROMPT yields
at least one ChangeIntent with action='create'. Defaults to mocking
agents.Runner.run so the test is deterministic and cheap; if env var
JARVIS_TEST_USE_REAL_LLM=1 is set the real LLM is exercised instead.
"""
from __future__ import annotations

import os
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest


EXPLICIT_PHRASES = [
    ("create a page", "Create a page for the new Auth service. It will document the OAuth flow."),
    ("create a new page", "We should create a new page about the SOC2 audit and its evidence requirements."),
    ("make a page", "Make a page that covers the API rate limits and our throttling strategy."),
    ("make a new page", "Bob, please make a new page for the rollout checklist with deployment, rollback, and verification steps."),
    ("new page for", "We need a new page for the Inventory Service launch — endpoints, owners, SLO."),
    ("set up a page", "Set up a page for the Q1 OKRs covering revenue, product velocity, and reliability."),
]

_USE_REAL_LLM = os.getenv("JARVIS_TEST_USE_REAL_LLM") == "1"


def _make_create_facts(subject: str, instruction: str):
    """Construct a real ExtractedFacts with one explicit create intent."""
    from confluence_logic.agents.fact_extraction_agent import ChangeIntent, ExtractedFacts

    return ExtractedFacts(
        change_intents=[
            ChangeIntent(
                instruction=instruction,
                subject=subject,
                target_hint=subject,
                old_value="",
                new_value="",
                action="create",
                rationale=f"User explicitly asked: {instruction}",
                verbatim_content="",
            )
        ]
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "phrase,transcript",
    EXPLICIT_PHRASES,
    ids=[p[0].replace(" ", "_") for p in EXPLICIT_PHRASES],
)
async def test_explicit_create_phrase_yields_create_intent(
    phrase: str, transcript: str, monkeypatch
):
    """When the transcript contains an explicit create phrase, FactExtraction
    must produce at least one ChangeIntent with action='create'."""
    from confluence_logic.agents.fact_extraction_agent import _run_fact_extraction

    if not _USE_REAL_LLM:
        # Patch the Runner.run that _extract_chunk uses so the stubbed result
        # mirrors what the live LLM would return for an explicit create phrase.
        subject = (
            transcript.split(phrase, 1)[-1].strip().split(".")[0].strip(" ,;:")
            or "Documentation"
        )

        async def _fake_run(_agent, _text, **_kwargs):
            r = MagicMock()
            r.final_output = _make_create_facts(subject, transcript)
            return r

        monkeypatch.setattr("agents.Runner.run", _fake_run)

    facts = await _run_fact_extraction(transcript)
    intents = list(getattr(facts, "change_intents", []) or [])
    create_intents = [i for i in intents if (getattr(i, "action", "") or "").lower() == "create"]

    assert len(create_intents) >= 1, (
        f"Phrase '{phrase}': fact extraction produced no create intents.\n"
        f"  transcript: {transcript}\n"
        f"  intents: {[(i.action, i.subject) for i in intents]}"
    )


