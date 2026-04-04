"""Tests for _split_sentences() and speak_chunked() in jarvis.py."""

import asyncio
import pytest
from unittest.mock import AsyncMock, patch, call

from jarvis import _split_sentences, speak_chunked


class TestSplitSentences:
    def test_three_sentences(self):
        result = _split_sentences("Hello world. How are you? I am fine!")
        assert result == ["Hello world.", "How are you?", "I am fine!"]

    def test_single_sentence(self):
        result = _split_sentences("Hello world.")
        assert result == ["Hello world."]

    def test_empty_string(self):
        result = _split_sentences("")
        assert result == []

    def test_no_terminal_punctuation(self):
        result = _split_sentences("Hello world")
        assert result == ["Hello world"]

    def test_extra_whitespace_between_sentences(self):
        result = _split_sentences("Hello world.   How are you?")
        assert len(result) == 2
        assert result[0] == "Hello world."
        assert result[1] == "How are you?"

    def test_abbreviation_not_split(self):
        result = _split_sentences("Dr. Smith said hello. We agreed.")
        # "Dr. Smith said hello." should remain together — "Dr." followed by space+lowercase
        # The key assertion: result has exactly 2 items (abbreviation not treated as sentence end)
        assert len(result) == 2
        assert "Dr. Smith said hello." in result[0]


class TestSpeakChunked:
    @pytest.mark.asyncio
    async def test_calls_speak_once_per_sentence(self):
        with patch("jarvis.speak") as mock_speak:
            mock_speak.return_value = True
            await speak_chunked("Hello. World.", "bot-123")
            assert mock_speak.call_count == 2

    @pytest.mark.asyncio
    async def test_empty_string_no_speak_calls(self):
        with patch("jarvis.speak") as mock_speak:
            await speak_chunked("", "bot-123")
            mock_speak.assert_not_called()

    @pytest.mark.asyncio
    async def test_single_sentence_one_speak_call(self):
        with patch("jarvis.speak") as mock_speak:
            mock_speak.return_value = True
            await speak_chunked("Hello there.", "bot-123")
            assert mock_speak.call_count == 1
            mock_speak.assert_called_once_with("Hello there.", "bot-123")

    @pytest.mark.asyncio
    async def test_order_preserved(self):
        calls = []
        def record_speak(text, bot_id):
            calls.append(text)
            return True

        with patch("jarvis.speak", side_effect=record_speak):
            await speak_chunked("First sentence. Second sentence. Third.", "bot-123")

        assert calls[0].startswith("First")
        assert calls[1].startswith("Second")
        assert calls[2].startswith("Third")
