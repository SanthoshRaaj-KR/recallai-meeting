"""Tests for general_responder — web search LLM router (WEBSEARCH-01)."""
import asyncio
import pytest
from types import SimpleNamespace
from unittest.mock import patch, MagicMock

from confluence_logic import general_responder as gr


def _mock_openai_response(content: str):
    """Build a mock OpenAI ChatCompletion response."""
    choice = SimpleNamespace(message=SimpleNamespace(content=content))
    return SimpleNamespace(choices=[choice])


@pytest.mark.asyncio
async def test_web_search_router_yes_for_weather():
    """WEBSEARCH-01: weather question triggers web search."""
    with patch.object(gr, "_get_client") as mock_client:
        mock_client.return_value.chat.completions.create.return_value = _mock_openai_response("yes")
        result = await gr._needs_web_search("what's the weather in Paris")
    assert result is True


@pytest.mark.asyncio
async def test_web_search_router_no_for_factual():
    """WEBSEARCH-01: factual question does not trigger web search."""
    with patch.object(gr, "_get_client") as mock_client:
        mock_client.return_value.chat.completions.create.return_value = _mock_openai_response("no")
        result = await gr._needs_web_search("what is a variable in Python")
    assert result is False


@pytest.mark.asyncio
async def test_web_search_router_yes_for_sports():
    """WEBSEARCH-01: sports score question triggers web search."""
    with patch.object(gr, "_get_client") as mock_client:
        mock_client.return_value.chat.completions.create.return_value = _mock_openai_response("yes")
        result = await gr._needs_web_search("who won the F1 race today")
    assert result is True


@pytest.mark.asyncio
async def test_web_search_router_fallback_on_error():
    """WEBSEARCH-01: returns False when OpenAI call fails."""
    with patch.object(gr, "_get_client") as mock_client:
        mock_client.return_value.chat.completions.create.side_effect = Exception("API down")
        result = await gr._needs_web_search("what's the weather in Paris")
    assert result is False


@pytest.mark.asyncio
async def test_general_answer_prompt_allows_independent_reasoning_with_meeting_context():
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _mock_openai_response(
        "I would challenge the current plan and add a phased rollout."
    )

    with patch.object(gr, "_get_client", return_value=mock_client), \
         patch.object(gr, "_needs_web_search", return_value=False):
        answer = await gr.answer_general_question(
            "What do you think about the plan?",
            graph_context="The team discussed a Friday launch.",
        )

    assert answer == "I would challenge the current plan and add a phased rollout."
    messages = mock_client.chat.completions.create.call_args.kwargs["messages"]
    system_text = messages[0]["content"]
    assert "do not let it constrain your reasoning" in system_text
    assert "respectfully disagree" in system_text
    assert "combine the meeting context with your own knowledge" in system_text
