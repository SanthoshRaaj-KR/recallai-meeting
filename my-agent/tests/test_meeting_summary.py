"""Tests for generate_meeting_summary — schema validation and LLM response parsing."""
import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from review_pipeline.pipeline import (
    ProposalPipeline,
    _ACTION_ITEMS_SCHEMA,
    _DECISIONS_SCHEMA,
    _human_only,
)


# ── Schema validation ─────────────────────────────────────────────────────────


def test_action_items_schema_has_no_any_of():
    """anyOf is not supported by Cerebras strict mode — schema must not use it."""
    schema_str = json.dumps(_ACTION_ITEMS_SCHEMA)
    assert "anyOf" not in schema_str, "anyOf found in _ACTION_ITEMS_SCHEMA — Cerebras will reject it"


def test_action_items_schema_owner_and_due_are_plain_string():
    items_schema = _ACTION_ITEMS_SCHEMA["json_schema"]["schema"]["properties"]["action_items"]["items"]
    assert items_schema["properties"]["owner"] == {"type": "string"}
    assert items_schema["properties"]["due"] == {"type": "string"}


def test_decisions_schema_is_valid():
    schema = _DECISIONS_SCHEMA["json_schema"]["schema"]
    assert schema["properties"]["decisions"]["items"]["type"] == "string"
    assert "additionalProperties" in schema


# ── _human_only filter ────────────────────────────────────────────────────────


def test_human_only_strips_jarvis_entries():
    transcript = [
        {"participant": "Alice", "text": "We should move the launch."},
        {"participant": "Jarvis", "text": "Based on Confluence, the launch was originally Q3."},
        {"participant": "Bob", "text": "Agreed, let's push to Q4."},
        {"speaker": "JARVIS", "text": "Noted."},
    ]
    result = _human_only(transcript)
    assert len(result) == 2
    assert all(
        (e.get("participant") or e.get("speaker") or "").lower() != "jarvis"
        for e in result
    )


def test_human_only_keeps_all_humans():
    transcript = [
        {"participant": "Alice", "text": "Decision A."},
        {"participant": "Bob", "text": "Decision B."},
    ]
    assert _human_only(transcript) == transcript


# ── Cerebras response parsing ─────────────────────────────────────────────────


def _make_cerebras_response(content: str):
    choice = SimpleNamespace(message=SimpleNamespace(content=content))
    return SimpleNamespace(choices=[choice])


def _fake_cerebras(**responses):
    """Returns an AsyncOpenAI-like mock whose chat.completions.create echoes
    a pre-set response keyed by schema name in response_format."""
    async def create(**kwargs):
        fmt = kwargs.get("response_format", {})
        name = (fmt.get("json_schema") or {}).get("name", "")
        content = responses.get(name, "{}")
        return _make_cerebras_response(content)

    client = MagicMock()
    client.chat = MagicMock()
    client.chat.completions = MagicMock()
    client.chat.completions.create = create
    return client


def test_decisions_parsed_correctly():
    pipeline = ProposalPipeline()
    cerebras = _fake_cerebras(
        decisions_response=json.dumps({"decisions": ["Launch moved to Q4", "Precision threshold set to 0.95"]}),
        summary_response=json.dumps({"title": "T", "summary": "S", "key_topics": []}),
        action_items_response=json.dumps({"action_items": []}),
        mom_response=json.dumps({"mom": []}),
    )
    openai_fallback = _fake_cerebras()  # never called if Cerebras succeeds

    result = asyncio.run(pipeline._decisions_llm(cerebras, openai_fallback, "transcript text"))
    assert result == ["Launch moved to Q4", "Precision threshold set to 0.95"]


def test_action_items_parsed_with_owner_and_due():
    pipeline = ProposalPipeline()
    cerebras = _fake_cerebras(
        action_items_response=json.dumps({"action_items": [
            {"description": "Update the runbook", "owner": "Alice", "due": "Friday"},
            {"description": "Fix the deployment script", "owner": "", "due": ""},
        ]}),
    )
    openai_fallback = _fake_cerebras()

    result = asyncio.run(pipeline._action_items_llm(cerebras, openai_fallback, "transcript text"))
    assert len(result) == 2
    assert result[0] == {"description": "Update the runbook", "owner": "Alice", "due": "Friday"}
    # empty strings normalised to None
    assert result[1] == {"description": "Fix the deployment script", "owner": None, "due": None}


def test_action_items_empty_string_owner_normalised_to_none():
    pipeline = ProposalPipeline()
    cerebras = _fake_cerebras(
        action_items_response=json.dumps({"action_items": [
            {"description": "Write docs", "owner": "", "due": ""},
        ]}),
    )
    result = asyncio.run(pipeline._action_items_llm(cerebras, MagicMock(), "text"))
    assert result[0]["owner"] is None
    assert result[0]["due"] is None


def test_decisions_falls_back_to_openai_on_cerebras_error():
    pipeline = ProposalPipeline()

    async def cerebras_fail(**kwargs):
        raise RuntimeError("Cerebras unavailable")

    cerebras = MagicMock()
    cerebras.chat = MagicMock()
    cerebras.chat.completions = MagicMock()
    cerebras.chat.completions.create = cerebras_fail

    openai_fallback = _fake_cerebras(
        # OpenAI fallback uses json_object — no schema name, just raw key
        **{"": json.dumps({"decisions": ["Decision from OpenAI fallback"]})}
    )

    result = asyncio.run(pipeline._decisions_llm(cerebras, openai_fallback, "transcript text"))
    assert result == ["Decision from OpenAI fallback"]


def test_action_items_falls_back_to_openai_on_cerebras_error():
    pipeline = ProposalPipeline()

    async def cerebras_fail(**kwargs):
        raise RuntimeError("schema rejected")

    cerebras = MagicMock()
    cerebras.chat = MagicMock()
    cerebras.chat.completions = MagicMock()
    cerebras.chat.completions.create = cerebras_fail

    openai_fallback = _fake_cerebras(
        **{"": json.dumps({"action_items": [
            {"description": "Deploy to prod", "owner": "Bob", "due": "Monday"}
        ]})}
    )

    result = asyncio.run(pipeline._action_items_llm(cerebras, openai_fallback, "transcript text"))
    assert result == [{"description": "Deploy to prod", "owner": "Bob", "due": "Monday"}]


# ── Full generate_meeting_summary integration ─────────────────────────────────


def test_generate_meeting_summary_returns_all_sections(monkeypatch):
    pipeline = ProposalPipeline()

    cerebras = _fake_cerebras(
        summary_response=json.dumps({
            "title": "Q3 Planning",
            "summary": "The team reviewed metrics.\n\nLaunch moved to Q4.",
            "key_topics": ["metrics", "launch date"],
        }),
        decisions_response=json.dumps({"decisions": ["Launch moved to Q4"]}),
        action_items_response=json.dumps({"action_items": [
            {"description": "Update roadmap", "owner": "Alice", "due": "Friday"},
        ]}),
        mom_response=json.dumps({"mom": [
            {"topic": "Metrics review", "summary": "Precision threshold raised to 0.95."},
        ]}),
    )

    monkeypatch.setenv("CEREBRAS_API_KEY", "fake-key")

    with patch("review_pipeline.pipeline.AsyncOpenAI") as mock_openai_cls:
        mock_openai_cls.side_effect = [cerebras, MagicMock()]  # cerebras, then openai fallback
        result = asyncio.run(pipeline.generate_meeting_summary(
            session_id="test-session",
            transcript=[
                {"participant": "Alice", "text": "We should raise the precision threshold."},
                {"participant": "Jarvis", "text": "According to Confluence, the threshold was 0.90."},
                {"participant": "Bob", "text": "Launch should move to Q4."},
            ],
        ))

    assert result["title"] == "Q3 Planning"
    assert result["decisions"] == ["Launch moved to Q4"]
    assert len(result["action_items"]) == 1
    assert result["action_items"][0]["owner"] == "Alice"
    assert len(result["mom"]) == 1
    assert result["mom"][0]["topic"] == "Metrics review"
    # Jarvis entry should not appear in participants list? No - Jarvis is filtered
    # from transcript_text but participants are extracted from the raw transcript.
    assert "Alice" in result["participants"]
    assert "Bob" in result["participants"]
