"""
Tests for confluence_logic/meeting_responder.py

TDD tests for summarize_meeting() and generate_opinion() functions.
"""
import asyncio
import os
import sys
import pytest
from unittest.mock import MagicMock, patch

# Add parent dir to path if needed
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from confluence_logic.meeting_responder import (
    summarize_meeting,
    generate_opinion,
    _EMPTY_TRANSCRIPT_FALLBACK,
    JARVIS_SUMMARY_MAX_TOKENS,
    JARVIS_OPINION_MAX_TOKENS,
    MEETING_RESPONDER_MODEL,
)


SAMPLE_TRANSCRIPT = [
    {"participant": "Alice", "text": "We should use GPT-4", "timestamp": 0},
]

SAMPLE_TRANSCRIPT_BOB = [
    {"participant": "Bob", "text": "Option A is faster", "timestamp": 0},
]

GROUNDING_PHRASES = [
    "Based on what I heard",
    "From the discussion",
    "Given what the team discussed",
]

EMPTY_TRANSCRIPT_FALLBACK = "I haven't heard anything in the meeting yet."


# ----- Empty transcript tests -----

def test_summarize_empty_transcript_returns_fallback():
    """summarize_meeting([]) returns the empty-transcript fallback string."""
    result = asyncio.run(summarize_meeting([]))
    assert result == EMPTY_TRANSCRIPT_FALLBACK


def test_generate_opinion_empty_transcript_returns_fallback():
    """generate_opinion([]) returns the empty-transcript fallback string."""
    result = asyncio.run(generate_opinion([]))
    assert result == EMPTY_TRANSCRIPT_FALLBACK


def test_fallback_constant_matches_expected():
    """_EMPTY_TRANSCRIPT_FALLBACK constant has the exact expected string."""
    assert _EMPTY_TRANSCRIPT_FALLBACK == EMPTY_TRANSCRIPT_FALLBACK


# ----- Token cap / env var tests -----

def test_summary_max_tokens_default():
    """JARVIS_SUMMARY_MAX_TOKENS defaults to 400."""
    # If env var is not set, default is 400
    if "JARVIS_SUMMARY_MAX_TOKENS" not in os.environ:
        assert JARVIS_SUMMARY_MAX_TOKENS == 400


def test_opinion_max_tokens_default():
    """JARVIS_OPINION_MAX_TOKENS defaults to 200."""
    if "JARVIS_OPINION_MAX_TOKENS" not in os.environ:
        assert JARVIS_OPINION_MAX_TOKENS == 200


def test_model_env_var_used():
    """MEETING_RESPONDER_MODEL reads from JARVIS_GENERAL_MODEL env var."""
    # Default should be gpt-4o-mini if env var not set
    if "JARVIS_GENERAL_MODEL" not in os.environ:
        assert MEETING_RESPONDER_MODEL == "gpt-4o-mini"


# ----- API call tests with mock -----

def _make_mock_response(content: str):
    """Build a mock OpenAI chat completion response."""
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = content
    return mock_response


def test_summarize_meeting_calls_api_with_transcript():
    """summarize_meeting with a non-empty transcript calls OpenAI API and returns non-empty string."""
    fake_summary = "Alice proposed using GPT-4 for the project."

    with patch("confluence_logic.meeting_responder._get_client") as mock_get_client:
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_client.chat.completions.create.return_value = _make_mock_response(fake_summary)

        result = asyncio.run(summarize_meeting(SAMPLE_TRANSCRIPT))

    assert result != EMPTY_TRANSCRIPT_FALLBACK
    assert result == fake_summary
    mock_client.chat.completions.create.assert_called_once()
    call_kwargs = mock_client.chat.completions.create.call_args[1]
    assert call_kwargs["max_tokens"] == JARVIS_SUMMARY_MAX_TOKENS


def test_generate_opinion_starts_with_grounding_phrase():
    """generate_opinion returns a string starting with a recognized grounding phrase."""
    fake_opinion = "Based on what I heard, Option A looks like the better path."

    with patch("confluence_logic.meeting_responder._get_client") as mock_get_client:
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_client.chat.completions.create.return_value = _make_mock_response(fake_opinion)

        result = asyncio.run(generate_opinion(SAMPLE_TRANSCRIPT_BOB))

    assert result != EMPTY_TRANSCRIPT_FALLBACK
    starts_with_grounding = any(result.startswith(phrase) for phrase in GROUNDING_PHRASES)
    assert starts_with_grounding, f"Expected grounding phrase at start; got: {result!r}"
    call_kwargs = mock_client.chat.completions.create.call_args[1]
    assert call_kwargs["max_tokens"] == JARVIS_OPINION_MAX_TOKENS


def test_generate_opinion_with_query():
    """generate_opinion with a query argument includes it in the API call."""
    fake_opinion = "From the discussion so far, I'd go with GPT-4."

    with patch("confluence_logic.meeting_responder._get_client") as mock_get_client:
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_client.chat.completions.create.return_value = _make_mock_response(fake_opinion)

        result = asyncio.run(generate_opinion(SAMPLE_TRANSCRIPT, query="Which model should we use?"))

    assert result == fake_opinion
    call_kwargs = mock_client.chat.completions.create.call_args[1]
    messages = call_kwargs["messages"]
    user_message = next(m for m in messages if m["role"] == "user")
    assert "Which model should we use?" in user_message["content"]


def test_summarize_meeting_api_error_returns_fallback():
    """summarize_meeting returns error fallback string when API raises exception."""
    with patch("confluence_logic.meeting_responder._get_client") as mock_get_client:
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_client.chat.completions.create.side_effect = Exception("API error")

        result = asyncio.run(summarize_meeting(SAMPLE_TRANSCRIPT))

    assert result == "Sorry, I couldn't generate a summary right now."


def test_generate_opinion_api_error_returns_fallback():
    """generate_opinion returns error fallback string when API raises exception."""
    with patch("confluence_logic.meeting_responder._get_client") as mock_get_client:
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_client.chat.completions.create.side_effect = Exception("API error")

        result = asyncio.run(generate_opinion(SAMPLE_TRANSCRIPT_BOB))

    assert result == "Sorry, I couldn't form an opinion right now."


def test_summarize_meeting_uses_asyncio_to_thread():
    """Both summarize_meeting and generate_opinion use asyncio.to_thread (checked via source)."""
    import inspect
    import confluence_logic.meeting_responder as mod
    src = inspect.getsource(mod)
    assert "asyncio.to_thread" in src
    occurrences = src.count("asyncio.to_thread")
    assert occurrences >= 2, f"Expected at least 2 asyncio.to_thread calls, found {occurrences}"
