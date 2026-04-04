"""
Tests for OrchestratorAgent — TDD RED phase.

Tests cover:
- OrchestratorResult model validation
- OrchestratorAgent initialization
- classify() method with mocked OpenAI
- run() method with mocked retriever, date_resolver, answer_agent, and OpenAI
"""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch

from agents.orchestrator import OrchestratorAgent, OrchestratorResult
from agents.answer_agent import AnswerOutput
from agents.retriever import RetrievalResult


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


def make_retrieval_result(meeting_ids=None, with_date=True):
    """Return a RetrievalResult with N meetings, each on a different day."""
    meeting_ids = meeting_ids or ["mtg-001"]
    # Each meeting gets a start_ts 86400 seconds apart to simulate different days
    results = []
    for i, mid in enumerate(meeting_ids):
        results.append({
            "id": mid,
            "score": 0.9,
            "metadata": {
                "channel_name": "eng-standup",
                "channel_id": "C_ENG",
                "start_ts": 1743292800 + (i * 86400),  # distinct days
                "summary_text": f"Meeting {mid} summary.",
                "decisions": ["Some decision"],
                "participants": ["Alice"],
                "action_items": [],
                "topics_covered": ["planning"],
            },
            "rerank_score": 0.9 - (i * 0.1),
        })
    return RetrievalResult(
        query="test query",
        results=results,
        total_candidates=20,
        returned_count=len(results),
    )


def make_answer_output(answer="Test answer.", meeting_ids=None):
    return AnswerOutput(
        answer=answer,
        source_meeting_ids=meeting_ids or ["mtg-001"],
        confidence="high",
    )


def make_openai_mock(classification: str):
    """Return an AsyncMock that simulates openai response returning the given classification."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = classification
    mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
    return mock_client


# ---------------------------------------------------------------------------
# TestOrchestratorResult
# ---------------------------------------------------------------------------


class TestOrchestratorResult:
    def test_orchestrator_result_required_fields(self):
        """OrchestratorResult can be constructed with required fields."""
        result = OrchestratorResult(
            query="q",
            query_type="memory_query",
            answer="a",
            source_meeting_ids=["m1"],
            confidence="high",
        )
        assert result.query == "q"
        assert result.query_type == "memory_query"
        assert result.answer == "a"
        assert result.source_meeting_ids == ["m1"]
        assert result.confidence == "high"

    def test_orchestrator_result_disambiguation_defaults_false(self):
        """needs_disambiguation defaults to False."""
        result = OrchestratorResult(
            query="q",
            query_type="memory_query",
            answer="a",
            source_meeting_ids=[],
            confidence="low",
        )
        assert result.needs_disambiguation is False

    def test_orchestrator_result_disambiguation_options_defaults_empty(self):
        """disambiguation_options defaults to empty list."""
        result = OrchestratorResult(
            query="q",
            query_type="memory_query",
            answer="a",
            source_meeting_ids=[],
            confidence="low",
        )
        assert result.disambiguation_options == []


# ---------------------------------------------------------------------------
# TestOrchestratorAgentInit
# ---------------------------------------------------------------------------


class TestOrchestratorAgentInit:
    def test_init_requires_retriever_date_resolver_answer_agent(self):
        """OrchestratorAgent can be constructed with required dependencies."""
        OrchestratorAgent(
            retriever=MagicMock(),
            date_resolver=MagicMock(),
            answer_agent=MagicMock(),
        )
        # Should not raise


# ---------------------------------------------------------------------------
# TestOrchestratorAgentClassify
# ---------------------------------------------------------------------------


class TestOrchestratorAgentClassify:
    @pytest.mark.asyncio
    async def test_classify_returns_memory_query_for_decision_question(self):
        """classify() returns 'memory_query' when GPT responds with 'memory_query'."""
        mock_client = make_openai_mock("memory_query")
        with patch("agents.orchestrator.AsyncOpenAI", return_value=mock_client):
            agent = OrchestratorAgent(
                retriever=MagicMock(),
                date_resolver=MagicMock(),
                answer_agent=MagicMock(),
            )
            result = await agent.classify("what did we decide about the API?")
        assert result == "memory_query"

    @pytest.mark.asyncio
    async def test_classify_returns_action_item_query(self):
        """classify() returns 'action_item_query' when GPT responds accordingly."""
        mock_client = make_openai_mock("action_item_query")
        with patch("agents.orchestrator.AsyncOpenAI", return_value=mock_client):
            agent = OrchestratorAgent(
                retriever=MagicMock(),
                date_resolver=MagicMock(),
                answer_agent=MagicMock(),
            )
            result = await agent.classify("what are my action items?")
        assert result == "action_item_query"

    @pytest.mark.asyncio
    async def test_classify_returns_live_meeting(self):
        """classify() returns 'live_meeting' when GPT responds accordingly."""
        mock_client = make_openai_mock("live_meeting")
        with patch("agents.orchestrator.AsyncOpenAI", return_value=mock_client):
            agent = OrchestratorAgent(
                retriever=MagicMock(),
                date_resolver=MagicMock(),
                answer_agent=MagicMock(),
            )
            result = await agent.classify("what are we discussing right now?")
        assert result == "live_meeting"

    @pytest.mark.asyncio
    async def test_classify_unknown_response_defaults_to_memory_query(self):
        """classify() returns 'memory_query' as safe default for unrecognized responses."""
        mock_client = make_openai_mock("garbled_text")
        with patch("agents.orchestrator.AsyncOpenAI", return_value=mock_client):
            agent = OrchestratorAgent(
                retriever=MagicMock(),
                date_resolver=MagicMock(),
                answer_agent=MagicMock(),
            )
            result = await agent.classify("some ambiguous query")
        assert result == "memory_query"


# ---------------------------------------------------------------------------
# TestOrchestratorAgentRun
# ---------------------------------------------------------------------------


class TestOrchestratorAgentRun:
    @pytest.mark.asyncio
    async def test_run_returns_orchestrator_result(self):
        """run() returns an OrchestratorResult instance."""
        mock_client = make_openai_mock("memory_query")
        mock_retriever = MagicMock()
        mock_retriever.retrieve = AsyncMock(return_value=make_retrieval_result())
        mock_date_resolver = MagicMock()
        mock_answer_agent = MagicMock()
        mock_answer_agent.run = AsyncMock(return_value=make_answer_output())

        with patch("agents.orchestrator.AsyncOpenAI", return_value=mock_client):
            agent = OrchestratorAgent(
                retriever=mock_retriever,
                date_resolver=mock_date_resolver,
                answer_agent=mock_answer_agent,
            )
            result = await agent.run(
                query="what did we decide about the API?",
                user_id="U123",
                channel_id="C_ENG",
            )
        assert isinstance(result, OrchestratorResult)

    @pytest.mark.asyncio
    async def test_run_memory_query_calls_retriever(self):
        """run() calls retriever.retrieve once for memory_query without date expression."""
        mock_client = make_openai_mock("memory_query")
        mock_retriever = MagicMock()
        mock_retriever.retrieve = AsyncMock(return_value=make_retrieval_result())
        mock_date_resolver = MagicMock()
        mock_answer_agent = MagicMock()
        mock_answer_agent.run = AsyncMock(return_value=make_answer_output())

        with patch("agents.orchestrator.AsyncOpenAI", return_value=mock_client):
            agent = OrchestratorAgent(
                retriever=mock_retriever,
                date_resolver=mock_date_resolver,
                answer_agent=mock_answer_agent,
            )
            await agent.run(
                query="what did we decide about the API?",
                user_id="U123",
                channel_id="C_ENG",
            )
        mock_retriever.retrieve.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_action_item_query_routes_to_answer_agent_with_action_items_type(self):
        """run() calls answer_agent.run with query_type='action_items' for action_item_query."""
        mock_client = make_openai_mock("action_item_query")
        mock_retriever = MagicMock()
        mock_retriever.retrieve = AsyncMock(return_value=make_retrieval_result())
        mock_date_resolver = MagicMock()
        mock_answer_agent = MagicMock()
        mock_answer_agent.run = AsyncMock(return_value=make_answer_output())

        with patch("agents.orchestrator.AsyncOpenAI", return_value=mock_client):
            agent = OrchestratorAgent(
                retriever=mock_retriever,
                date_resolver=mock_date_resolver,
                answer_agent=mock_answer_agent,
            )
            await agent.run(
                query="what are my action items?",
                user_id="U123",
                channel_id="C_ENG",
            )
        call_kwargs = mock_answer_agent.run.call_args
        assert call_kwargs.kwargs.get("query_type") == "action_items" or (
            len(call_kwargs.args) >= 3 and call_kwargs.args[2] == "action_items"
        )

    @pytest.mark.asyncio
    async def test_run_live_meeting_does_not_call_retriever(self):
        """run() does NOT call retriever.retrieve for live_meeting queries."""
        mock_client = make_openai_mock("live_meeting")
        mock_retriever = MagicMock()
        mock_retriever.retrieve = AsyncMock(return_value=make_retrieval_result())
        mock_date_resolver = MagicMock()
        mock_answer_agent = MagicMock()
        mock_answer_agent.run = AsyncMock(return_value=make_answer_output())

        with patch("agents.orchestrator.AsyncOpenAI", return_value=mock_client):
            agent = OrchestratorAgent(
                retriever=mock_retriever,
                date_resolver=mock_date_resolver,
                answer_agent=mock_answer_agent,
            )
            result = await agent.run(
                query="what are we discussing right now?",
                user_id="U123",
                channel_id="C_ENG",
            )
        mock_retriever.retrieve.assert_not_called()
        assert "past meetings" in result.answer.lower() or "only answer" in result.answer.lower()

    @pytest.mark.asyncio
    async def test_run_with_date_expression_calls_date_resolver(self):
        """run() calls date_resolver.resolve when query has a date expression."""
        from agents.date_resolver import DateResolutionResult

        mock_client = make_openai_mock("memory_query")
        mock_retriever = MagicMock()
        mock_retriever.retrieve = AsyncMock(return_value=make_retrieval_result())
        mock_date_resolver = MagicMock()
        mock_date_resolver.resolve = MagicMock(return_value=DateResolutionResult(
            start_ts=1743206400,
            end_ts=1743292799,
            expression="last Monday",
            is_relative=True,
        ))
        mock_answer_agent = MagicMock()
        mock_answer_agent.run = AsyncMock(return_value=make_answer_output())

        with patch("agents.orchestrator.AsyncOpenAI", return_value=mock_client):
            agent = OrchestratorAgent(
                retriever=mock_retriever,
                date_resolver=mock_date_resolver,
                answer_agent=mock_answer_agent,
            )
            await agent.run(
                query="what happened last Monday?",
                user_id="U123",
                channel_id="C_ENG",
            )
        mock_date_resolver.resolve.assert_called_once()
        # date_resolver.resolve should have been called with "last Monday" (extracted sub-expression)
        call_arg = mock_date_resolver.resolve.call_args.args[0]
        assert "last" in call_arg.lower()

        # retriever should have been called with start_ts and end_ts
        retrieve_call = mock_retriever.retrieve.call_args
        assert retrieve_call.kwargs.get("start_ts") == 1743206400
        assert retrieve_call.kwargs.get("end_ts") == 1743292799

    @pytest.mark.asyncio
    async def test_run_without_date_expression_does_not_call_date_resolver(self):
        """run() does NOT call date_resolver when query has no date expression."""
        mock_client = make_openai_mock("memory_query")
        mock_retriever = MagicMock()
        mock_retriever.retrieve = AsyncMock(return_value=make_retrieval_result())
        mock_date_resolver = MagicMock()
        mock_answer_agent = MagicMock()
        mock_answer_agent.run = AsyncMock(return_value=make_answer_output())

        with patch("agents.orchestrator.AsyncOpenAI", return_value=mock_client):
            agent = OrchestratorAgent(
                retriever=mock_retriever,
                date_resolver=mock_date_resolver,
                answer_agent=mock_answer_agent,
            )
            await agent.run(
                query="what did we decide about the API?",
                user_id="U123",
                channel_id="C_ENG",
            )
        mock_date_resolver.resolve.assert_not_called()
        # retriever should have been called with start_ts=None, end_ts=None
        retrieve_call = mock_retriever.retrieve.call_args
        assert retrieve_call.kwargs.get("start_ts") is None
        assert retrieve_call.kwargs.get("end_ts") is None

    @pytest.mark.asyncio
    async def test_run_disambiguation_triggered_when_date_in_query_and_multiple_results(self):
        """needs_disambiguation=True when date in query AND retriever returns >1 meeting."""
        from agents.date_resolver import DateResolutionResult

        mock_client = make_openai_mock("memory_query")
        mock_retriever = MagicMock()
        mock_retriever.retrieve = AsyncMock(
            return_value=make_retrieval_result(meeting_ids=["mtg-001", "mtg-002"])
        )
        mock_date_resolver = MagicMock()
        mock_date_resolver.resolve = MagicMock(return_value=DateResolutionResult(
            start_ts=1743206400,
            end_ts=1743292799,
            expression="last Monday",
            is_relative=True,
        ))
        mock_answer_agent = MagicMock()
        mock_answer_agent.run = AsyncMock(return_value=make_answer_output())

        with patch("agents.orchestrator.AsyncOpenAI", return_value=mock_client):
            agent = OrchestratorAgent(
                retriever=mock_retriever,
                date_resolver=mock_date_resolver,
                answer_agent=mock_answer_agent,
            )
            result = await agent.run(
                query="what happened last Monday?",
                user_id="U123",
                channel_id="C_ENG",
            )
        assert result.needs_disambiguation is True
        assert len(result.disambiguation_options) == 2

    @pytest.mark.asyncio
    async def test_run_no_disambiguation_when_no_date_in_query_even_with_multiple_results(self):
        """needs_disambiguation=False when no date in query, even with multiple results."""
        mock_client = make_openai_mock("memory_query")
        mock_retriever = MagicMock()
        mock_retriever.retrieve = AsyncMock(
            return_value=make_retrieval_result(meeting_ids=["mtg-001", "mtg-002"])
        )
        mock_date_resolver = MagicMock()
        mock_answer_agent = MagicMock()
        mock_answer_agent.run = AsyncMock(return_value=make_answer_output())

        with patch("agents.orchestrator.AsyncOpenAI", return_value=mock_client):
            agent = OrchestratorAgent(
                retriever=mock_retriever,
                date_resolver=mock_date_resolver,
                answer_agent=mock_answer_agent,
            )
            result = await agent.run(
                query="what have we discussed recently?",
                user_id="U123",
                channel_id="C_ENG",
            )
        assert result.needs_disambiguation is False

    @pytest.mark.asyncio
    async def test_run_no_disambiguation_when_single_result_with_date(self):
        """needs_disambiguation=False when date in query but only 1 meeting returned."""
        from agents.date_resolver import DateResolutionResult

        mock_client = make_openai_mock("memory_query")
        mock_retriever = MagicMock()
        mock_retriever.retrieve = AsyncMock(
            return_value=make_retrieval_result(meeting_ids=["mtg-001"])
        )
        mock_date_resolver = MagicMock()
        mock_date_resolver.resolve = MagicMock(return_value=DateResolutionResult(
            start_ts=1743206400,
            end_ts=1743292799,
            expression="last Monday",
            is_relative=True,
        ))
        mock_answer_agent = MagicMock()
        mock_answer_agent.run = AsyncMock(return_value=make_answer_output())

        with patch("agents.orchestrator.AsyncOpenAI", return_value=mock_client):
            agent = OrchestratorAgent(
                retriever=mock_retriever,
                date_resolver=mock_date_resolver,
                answer_agent=mock_answer_agent,
            )
            result = await agent.run(
                query="what happened last Monday?",
                user_id="U123",
                channel_id="C_ENG",
            )
        assert result.needs_disambiguation is False

    @pytest.mark.asyncio
    async def test_run_disambiguation_options_have_required_fields(self):
        """Each disambiguation option contains all required fields."""
        from agents.date_resolver import DateResolutionResult

        mock_client = make_openai_mock("memory_query")
        mock_retriever = MagicMock()
        mock_retriever.retrieve = AsyncMock(
            return_value=make_retrieval_result(meeting_ids=["mtg-001", "mtg-002"])
        )
        mock_date_resolver = MagicMock()
        mock_date_resolver.resolve = MagicMock(return_value=DateResolutionResult(
            start_ts=1743206400,
            end_ts=1743292799,
            expression="last Monday",
            is_relative=True,
        ))
        mock_answer_agent = MagicMock()
        mock_answer_agent.run = AsyncMock(return_value=make_answer_output())

        with patch("agents.orchestrator.AsyncOpenAI", return_value=mock_client):
            agent = OrchestratorAgent(
                retriever=mock_retriever,
                date_resolver=mock_date_resolver,
                answer_agent=mock_answer_agent,
            )
            result = await agent.run(
                query="what happened last Monday?",
                user_id="U123",
                channel_id="C_ENG",
            )
        assert result.needs_disambiguation is True
        for option in result.disambiguation_options:
            assert "index" in option
            assert "meeting_id" in option
            assert "title" in option
            assert "channel" in option
            assert "date" in option

    @pytest.mark.asyncio
    async def test_run_result_contains_answer_from_answer_agent(self):
        """result.answer comes from answer_agent when no disambiguation."""
        mock_client = make_openai_mock("memory_query")
        mock_retriever = MagicMock()
        mock_retriever.retrieve = AsyncMock(return_value=make_retrieval_result())
        mock_date_resolver = MagicMock()
        mock_answer_agent = MagicMock()
        mock_answer_agent.run = AsyncMock(
            return_value=make_answer_output(answer="Specific answer.")
        )

        with patch("agents.orchestrator.AsyncOpenAI", return_value=mock_client):
            agent = OrchestratorAgent(
                retriever=mock_retriever,
                date_resolver=mock_date_resolver,
                answer_agent=mock_answer_agent,
            )
            result = await agent.run(
                query="what did we decide about the API?",
                user_id="U123",
                channel_id="C_ENG",
            )
        assert result.answer == "Specific answer."
