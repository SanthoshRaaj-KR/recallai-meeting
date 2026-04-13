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
