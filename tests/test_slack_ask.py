"""
TDD tests for /ask Slack command, disambiguation wiring, and memory query routing.

All tests in this file are written BEFORE the implementation is added to jarvis.py.
They are expected to FAIL (ImportError or AttributeError) until Task 2 is complete.

Test groups:
- TestIsMemoryQuery       — _is_memory_query() heuristic function
- TestHandleMemoryQuery   — _handle_memory_query() orchestrator delegation
- TestHandleAskCommand    — _handle_ask() Slack slash command handler
- TestHandleMessageDisambig — _handle_message_disambig() message event handler
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

# These imports will FAIL (ImportError) until Task 2 adds the symbols to jarvis.py
import jarvis
from jarvis import _is_memory_query, _handle_memory_query, _handle_ask, _handle_message_disambig
from agents.orchestrator import OrchestratorResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_orchestrator_result(**kwargs):
    defaults = dict(
        query="what did we decide?",
        query_type="memory_query",
        answer="We decided to use Postgres. (Meeting: eng-standup, 2025-03-29)",
        source_meeting_ids=["mtg-abc"],
        confidence="high",
        needs_disambiguation=False,
        disambiguation_options=[],
    )
    defaults.update(kwargs)
    return OrchestratorResult(**defaults)


async def make_slack_context(text="what did we decide about the API?"):
    ack = AsyncMock()
    say = AsyncMock()
    client = AsyncMock()
    command = {
        "text": text,
        "user_id": "U123",
        "channel_id": "C456",
    }
    return ack, say, client, command


OPTION_1 = {
    "index": 1,
    "meeting_id": "mtg-1",
    "title": "Sprint review",
    "channel": "eng-standup",
    "date": "2025-03-29",
}
OPTION_2 = {
    "index": 2,
    "meeting_id": "mtg-2",
    "title": "Incident retro",
    "channel": "eng-standup",
    "date": "2025-03-30",
}


# ---------------------------------------------------------------------------
# TestIsMemoryQuery
# ---------------------------------------------------------------------------

class TestIsMemoryQuery:
    def test_is_memory_query_true_for_what_did_we(self):
        assert _is_memory_query("what did we decide?") is True

    def test_is_memory_query_true_for_last_week(self):
        assert _is_memory_query("what happened last week?") is True

    def test_is_memory_query_true_for_yesterday(self):
        assert _is_memory_query("summarize yesterday's standup") is True

    def test_is_memory_query_false_for_weather(self):
        assert _is_memory_query("what's the weather in Paris?") is False

    def test_is_memory_query_false_for_general_question(self):
        assert _is_memory_query("who is the CEO of Apple?") is False


# ---------------------------------------------------------------------------
# TestHandleMemoryQuery
# ---------------------------------------------------------------------------

class TestHandleMemoryQuery:
    @pytest.mark.asyncio
    async def test_handle_memory_query_calls_orchestrator_run(self):
        mock_result = make_orchestrator_result()
        mock_orchestrator = AsyncMock()
        mock_orchestrator.run = AsyncMock(return_value=mock_result)

        with patch.object(jarvis, "orchestrator", mock_orchestrator):
            await _handle_memory_query("what did we decide?", "U123", "C456")

        mock_orchestrator.run.assert_called_once_with(
            query="what did we decide?",
            user_id="U123",
            channel_id="C456",
        )

    @pytest.mark.asyncio
    async def test_handle_memory_query_returns_orchestrator_result(self):
        mock_result = make_orchestrator_result()
        mock_orchestrator = AsyncMock()
        mock_orchestrator.run = AsyncMock(return_value=mock_result)

        with patch.object(jarvis, "orchestrator", mock_orchestrator):
            result = await _handle_memory_query("what did we decide?", "U123", "C456")

        assert isinstance(result, OrchestratorResult)

    @pytest.mark.asyncio
    async def test_handle_memory_query_when_orchestrator_none_raises_or_returns_error(self):
        """
        When orchestrator is None (no PINECONE_API_KEY), _handle_memory_query
        should return an OrchestratorResult with answer containing "not configured"
        OR raise a RuntimeError.

        Implementation choice (Task 2): returns OrchestratorResult with "not configured"
        in the answer (no exception raised) — callers do not need try/except for this case.
        """
        with patch.object(jarvis, "orchestrator", None):
            result = await _handle_memory_query("what did we decide?", "U123", "C456")

        # Either: returns an OrchestratorResult with "not configured" in answer
        # OR: raises RuntimeError — this test accepts either approach
        assert isinstance(result, OrchestratorResult)
        assert "not configured" in result.answer.lower() or "pinecone" in result.answer.lower()


# ---------------------------------------------------------------------------
# TestHandleAskCommand
# ---------------------------------------------------------------------------

class TestHandleAskCommand:
    @pytest.mark.asyncio
    async def test_ask_calls_ack_immediately(self):
        ack, say, client, command = await make_slack_context()
        mock_result = make_orchestrator_result()
        mock_orchestrator = AsyncMock()
        mock_orchestrator.run = AsyncMock(return_value=mock_result)

        with patch.object(jarvis, "orchestrator", mock_orchestrator):
            await _handle_ask(ack, say, client, command)

        ack.assert_called_once()

    @pytest.mark.asyncio
    async def test_ask_posts_answer_to_channel(self):
        ack, say, client, command = await make_slack_context()
        mock_result = make_orchestrator_result(
            answer="We decided to use Postgres.",
            needs_disambiguation=False,
        )
        mock_orchestrator = AsyncMock()
        mock_orchestrator.run = AsyncMock(return_value=mock_result)

        with patch.object(jarvis, "orchestrator", mock_orchestrator):
            await _handle_ask(ack, say, client, command)

        # Either say() or client.chat_postMessage() called with the answer text
        say_calls = [str(c) for c in say.call_args_list]
        post_calls = [str(c) for c in client.chat_postMessage.call_args_list]
        all_text = " ".join(say_calls + post_calls)
        assert "We decided to use Postgres" in all_text

    @pytest.mark.asyncio
    async def test_ask_empty_text_returns_usage_hint(self):
        ack, say, client, command = await make_slack_context(text="")

        with patch.object(jarvis, "orchestrator", AsyncMock()):
            await _handle_ask(ack, say, client, command)

        say_calls = " ".join(str(c) for c in say.call_args_list)
        assert "usage" in say_calls.lower() or "please provide" in say_calls.lower()

    @pytest.mark.asyncio
    async def test_ask_orchestrator_not_configured_posts_error(self):
        ack, say, client, command = await make_slack_context()

        with patch.object(jarvis, "orchestrator", None):
            await _handle_ask(ack, say, client, command)

        say_calls = " ".join(str(c) for c in say.call_args_list)
        assert "not configured" in say_calls.lower() or "pinecone" in say_calls.lower()

    @pytest.mark.asyncio
    async def test_ask_disambiguation_posts_numbered_list(self):
        ack, say, client, command = await make_slack_context()
        mock_result = make_orchestrator_result(
            needs_disambiguation=True,
            disambiguation_options=[OPTION_1, OPTION_2],
        )
        mock_orchestrator = AsyncMock()
        mock_orchestrator.run = AsyncMock(return_value=mock_result)

        # Clear any stale pending state
        jarvis._pending_disambig.pop("U123", None)

        with patch.object(jarvis, "orchestrator", mock_orchestrator):
            await _handle_ask(ack, say, client, command)

        say_calls = " ".join(str(c) for c in say.call_args_list)
        assert "1." in say_calls or "1)" in say_calls
        assert "2." in say_calls or "2)" in say_calls

    @pytest.mark.asyncio
    async def test_ask_disambiguation_stores_pending_for_user(self):
        ack, say, client, command = await make_slack_context()
        mock_result = make_orchestrator_result(
            needs_disambiguation=True,
            disambiguation_options=[OPTION_1, OPTION_2],
        )
        mock_orchestrator = AsyncMock()
        mock_orchestrator.run = AsyncMock(return_value=mock_result)

        # Clear any stale pending state
        jarvis._pending_disambig.pop("U123", None)

        with patch.object(jarvis, "orchestrator", mock_orchestrator):
            await _handle_ask(ack, say, client, command)

        assert jarvis._pending_disambig.get("U123") is not None


# ---------------------------------------------------------------------------
# TestHandleMessageDisambig
# ---------------------------------------------------------------------------

class TestHandleMessageDisambig:
    @pytest.mark.asyncio
    async def test_message_with_valid_number_resolves_disambig(self):
        jarvis._pending_disambig["U123"] = [OPTION_1, OPTION_2]

        message = {"user": "U123", "text": "1", "channel": "C456"}
        say = AsyncMock()
        client = AsyncMock()

        scoped_result = make_orchestrator_result(
            query="Tell me about meeting mtg-1",
            source_meeting_ids=["mtg-1"],
        )
        mock_orchestrator = AsyncMock()
        mock_orchestrator.run = AsyncMock(return_value=scoped_result)

        with patch.object(jarvis, "orchestrator", mock_orchestrator):
            await _handle_message_disambig(message, say, client)

        # Orchestrator was called
        mock_orchestrator.run.assert_called_once()
        # Pending state cleared
        assert jarvis._pending_disambig.get("U123") is None

    @pytest.mark.asyncio
    async def test_message_without_pending_disambig_does_nothing(self):
        jarvis._pending_disambig.pop("U888", None)

        message = {"user": "U888", "text": "1", "channel": "C456"}
        say = AsyncMock()
        client = AsyncMock()

        mock_orchestrator = AsyncMock()
        mock_orchestrator.run = AsyncMock()

        with patch.object(jarvis, "orchestrator", mock_orchestrator):
            await _handle_message_disambig(message, say, client)

        mock_orchestrator.run.assert_not_called()

    @pytest.mark.asyncio
    async def test_message_with_non_numeric_text_ignored_when_pending(self):
        jarvis._pending_disambig["U123"] = [OPTION_1]

        message = {"user": "U123", "text": "hello", "channel": "C456"}
        say = AsyncMock()
        client = AsyncMock()

        mock_orchestrator = AsyncMock()
        mock_orchestrator.run = AsyncMock()

        with patch.object(jarvis, "orchestrator", mock_orchestrator):
            await _handle_message_disambig(message, say, client)

        mock_orchestrator.run.assert_not_called()
        # Pending state preserved
        assert jarvis._pending_disambig.get("U123") is not None

    @pytest.mark.asyncio
    async def test_message_out_of_range_number_posts_error(self):
        jarvis._pending_disambig["U123"] = [OPTION_1, OPTION_2]

        message = {"user": "U123", "text": "5", "channel": "C456"}
        say = AsyncMock()
        client = AsyncMock()

        mock_orchestrator = AsyncMock()
        mock_orchestrator.run = AsyncMock()

        with patch.object(jarvis, "orchestrator", mock_orchestrator):
            await _handle_message_disambig(message, say, client)

        # Error message posted
        say.assert_called_once()
        # Pending state preserved
        assert jarvis._pending_disambig.get("U123") is not None
        # Orchestrator NOT called
        mock_orchestrator.run.assert_not_called()
