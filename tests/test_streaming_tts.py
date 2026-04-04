"""Tests for the streaming TTS pipeline in jarvis.py.

Tests _stream_llm_and_speak() and the updated handle_query() call sites.
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch, call


class TestStreamLlmAndSpeak:
    @pytest.mark.asyncio
    async def test_speaks_first_sentence_before_stream_ends(self):
        """Sentence spoken when sentence boundary detected mid-stream."""
        speak_calls = []

        def fake_speak(text, bot_id):
            speak_calls.append(text)
            return True

        with patch("jarvis.speak", side_effect=fake_speak):
            with patch("jarvis.client") as mock_client:
                chunk1 = MagicMock()
                chunk1.choices[0].delta.content = "Hello. "
                chunk2 = MagicMock()
                chunk2.choices[0].delta.content = "World!"
                mock_client.chat.completions.create.return_value = iter([chunk1, chunk2])

                from jarvis import _stream_llm_and_speak
                await _stream_llm_and_speak(
                    [{"role": "user", "content": "hi"}],
                    "bot-123"
                )

        # "Hello." is a complete sentence; should be spoken before end of stream
        assert any("Hello" in c for c in speak_calls), f"Expected 'Hello' in speak_calls: {speak_calls}"

    @pytest.mark.asyncio
    async def test_each_sentence_becomes_one_speak_call(self):
        """Two complete sentences produce two speak() calls."""
        speak_calls = []

        def fake_speak(text, bot_id):
            speak_calls.append(text)
            return True

        with patch("jarvis.speak", side_effect=fake_speak):
            with patch("jarvis.client") as mock_client:
                chunk1 = MagicMock()
                chunk1.choices[0].delta.content = "First sentence. "
                chunk2 = MagicMock()
                chunk2.choices[0].delta.content = "Second sentence."
                mock_client.chat.completions.create.return_value = iter([chunk1, chunk2])

                from jarvis import _stream_llm_and_speak
                await _stream_llm_and_speak(
                    [{"role": "user", "content": "hi"}],
                    "bot-123"
                )

        assert len(speak_calls) == 2, f"Expected 2 speak calls, got {len(speak_calls)}: {speak_calls}"
        assert "First sentence" in speak_calls[0]
        assert "Second sentence" in speak_calls[1]

    @pytest.mark.asyncio
    async def test_remaining_buffer_spoken_after_stream_ends(self):
        """Any remaining buffered text after stream ends is spoken as final chunk."""
        speak_calls = []

        def fake_speak(text, bot_id):
            speak_calls.append(text)
            return True

        with patch("jarvis.speak", side_effect=fake_speak):
            with patch("jarvis.client") as mock_client:
                # Single chunk with no sentence-ending punctuation followed by space
                chunk = MagicMock()
                chunk.choices[0].delta.content = "Hello world"
                mock_client.chat.completions.create.return_value = iter([chunk])

                from jarvis import _stream_llm_and_speak
                await _stream_llm_and_speak(
                    [{"role": "user", "content": "hi"}],
                    "bot-123"
                )

        assert len(speak_calls) == 1, f"Expected 1 speak call for remainder, got {len(speak_calls)}"
        assert "Hello world" in speak_calls[0]

    @pytest.mark.asyncio
    async def test_empty_stream_no_speak_calls(self):
        """No speak() calls when stream produces no content."""
        with patch("jarvis.speak") as mock_speak:
            with patch("jarvis.client") as mock_client:
                empty_chunk = MagicMock()
                empty_chunk.choices[0].delta.content = None
                mock_client.chat.completions.create.return_value = iter([empty_chunk])

                from jarvis import _stream_llm_and_speak
                await _stream_llm_and_speak([], "bot-123")

        mock_speak.assert_not_called()

    @pytest.mark.asyncio
    async def test_uses_asyncio_to_thread_for_speak(self):
        """_stream_llm_and_speak uses asyncio.to_thread for speak() calls (non-blocking)."""
        to_thread_calls = []

        async def fake_to_thread(fn, *args):
            to_thread_calls.append((fn, args))
            # Simulate the speak call returning True
            if callable(fn) and fn.__name__ in ("speak", "_run_stream"):
                if fn.__name__ == "_run_stream":
                    # Return a list of tokens
                    return ["Hello. "]
                return True
            return fn(*args)

        with patch("jarvis.asyncio.to_thread", side_effect=fake_to_thread):
            with patch("jarvis.client") as mock_client:
                mock_client.chat.completions.create.return_value = iter([])

                from jarvis import _stream_llm_and_speak
                # We patch asyncio.to_thread to capture calls
                await _stream_llm_and_speak(
                    [{"role": "user", "content": "hi"}],
                    "bot-123"
                )

        # At minimum, asyncio.to_thread was used (for _run_stream)
        assert len(to_thread_calls) >= 1, "Expected at least one asyncio.to_thread call"
        fn_names = [c[0].__name__ for c in to_thread_calls if callable(c[0])]
        assert "_run_stream" in fn_names or any("stream" in n for n in fn_names), \
            f"Expected _run_stream in to_thread calls: {fn_names}"


class TestHandleQueryStreamingWiring:
    @pytest.mark.asyncio
    async def test_direct_query_uses_stream_llm_and_speak(self):
        """When finish_reason is 'stop' on final answer round, _stream_llm_and_speak is invoked."""
        with patch("jarvis._is_memory_query", return_value=False):
            with patch("jarvis._stream_llm_and_speak") as mock_stream:
                mock_stream.return_value = None  # coroutine that does nothing
                mock_stream.side_effect = None

                with patch("jarvis.client") as mock_client:
                    msg = MagicMock()
                    msg.content = "The weather is sunny."
                    msg.tool_calls = None
                    response = MagicMock()
                    response.choices[0].message = msg
                    response.choices[0].finish_reason = "stop"
                    mock_client.chat.completions.create.return_value = response

                    # _stream_llm_and_speak needs to be awaitable
                    async def fake_stream(messages, bot_id):
                        pass

                    mock_stream.side_effect = fake_stream

                    import jarvis
                    await jarvis.handle_query("What's the weather?", "bot-123")

        mock_stream.assert_called_once()

    @pytest.mark.asyncio
    async def test_memory_query_uses_speak_chunked(self):
        """Memory query path calls speak_chunked, not bare speak()."""
        from agents.orchestrator import OrchestratorResult

        mock_result = OrchestratorResult(
            query="what did we decide?",
            query_type="memory_query",
            answer="First sentence. Second sentence.",
            source_meeting_ids=["mtg-1"],
            confidence="high",
        )

        with patch("jarvis.orchestrator") as mock_orch:
            mock_orch.run = AsyncMock(return_value=mock_result)
            with patch("jarvis.speak_chunked") as mock_speak_chunked:
                async def fake_speak_chunked(text, bot_id):
                    pass
                mock_speak_chunked.side_effect = fake_speak_chunked

                import jarvis
                await jarvis.handle_query("what did we decide?", "bot-123")

        mock_speak_chunked.assert_called_once_with(mock_result.answer, "bot-123")

    @pytest.mark.asyncio
    async def test_disambiguation_path_uses_speak_chunked(self):
        """Disambiguation voice path calls speak_chunked for the options text."""
        from agents.orchestrator import OrchestratorResult

        mock_result = OrchestratorResult(
            query="what happened last Monday?",
            query_type="memory_query",
            answer="Multiple meetings found.",
            source_meeting_ids=["mtg-1", "mtg-2"],
            confidence="low",
            needs_disambiguation=True,
            disambiguation_options=[
                {"index": 1, "meeting_id": "mtg-1", "title": "Standup", "channel": "eng", "date": "2025-01-06"},
                {"index": 2, "meeting_id": "mtg-2", "title": "Planning", "channel": "eng", "date": "2025-01-06"},
            ],
        )

        with patch("jarvis.orchestrator") as mock_orch:
            mock_orch.run = AsyncMock(return_value=mock_result)
            with patch("jarvis.speak_chunked") as mock_speak_chunked:
                async def fake_speak_chunked(text, bot_id):
                    pass
                mock_speak_chunked.side_effect = fake_speak_chunked

                import jarvis
                await jarvis.handle_query("what happened last Monday?", "bot-123")

        mock_speak_chunked.assert_called_once()
        # Verify it was called with the disambiguation text (not result.answer)
        spoken_text = mock_speak_chunked.call_args[0][0]
        assert "Multiple meetings" in spoken_text or "Option" in spoken_text
