"""
TDD RED phase — tests for AnswerAgent.

All tests in this file will FAIL before agents/answer_agent.py is created.
Run: pytest tests/test_answer_agent.py
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from agents.answer_agent import AnswerAgent, AnswerOutput
from agents.retriever import RetrievalResult


def make_retrieval_result(results=None):
    """Helper to build a RetrievalResult with default fixture data."""
    return RetrievalResult(
        query="what did we decide?",
        results=results or [
            {
                "id": "mtg-abc",
                "score": 0.9,
                "metadata": {
                    "channel_name": "eng-standup",
                    "start_ts": 1743292800,  # 2025-03-29 00:00:00 UTC
                    "decisions": ["Use Postgres for storage"],
                    "summary_text": "Team decided to use Postgres.",
                    "participants": ["Alice", "Bob"],
                    "action_items": [],
                },
                "rerank_score": 0.99,
            }
        ],
        total_candidates=20,
        returned_count=1,
    )


class TestAnswerOutput:
    def test_answer_output_has_required_fields(self):
        output = AnswerOutput(
            answer="hello",
            source_meeting_ids=["mtg-1"],
            confidence="high",
        )
        assert output.answer == "hello"
        assert output.source_meeting_ids == ["mtg-1"]
        assert output.confidence == "high"

    def test_confidence_is_one_of_three_values(self):
        for confidence in ("high", "medium", "low"):
            output = AnswerOutput(
                answer="test",
                source_meeting_ids=[],
                confidence=confidence,
            )
            assert output.confidence == confidence

    def test_source_meeting_ids_is_list(self):
        output = AnswerOutput(
            answer="test",
            source_meeting_ids=["mtg-1", "mtg-2"],
            confidence="medium",
        )
        assert isinstance(output.source_meeting_ids, list)


class TestAnswerAgentInit:
    def test_init_creates_instance(self):
        agent = AnswerAgent()
        assert agent is not None

    def test_init_accepts_model_override(self):
        agent = AnswerAgent(model="gpt-4o")
        assert agent is not None


class TestAnswerAgentRun:
    def _make_mock_run_return(self):
        """Return value object that mimics Runner.run() result."""
        mock_result = MagicMock()
        mock_result.final_output = AnswerOutput(
            answer="We decided to use Postgres. (Meeting: eng-standup, 2025-03-29)",
            source_meeting_ids=["mtg-abc"],
            confidence="high",
        )
        return mock_result

    @pytest.mark.asyncio
    async def test_run_returns_answer_output(self):
        with patch("agents.answer_agent.Runner.run", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = self._make_mock_run_return()
            agent = AnswerAgent()
            result = await agent.run(
                query="what did we decide?",
                retrieval_result=make_retrieval_result(),
                query_type="decision",
            )
            assert isinstance(result, AnswerOutput)

    @pytest.mark.asyncio
    async def test_run_calls_runner_run(self):
        with patch("agents.answer_agent.Runner.run", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = self._make_mock_run_return()
            agent = AnswerAgent()
            await agent.run(
                query="what did we decide?",
                retrieval_result=make_retrieval_result(),
                query_type="decision",
            )
            mock_run.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_accepts_all_query_types(self):
        for query_type in ("decision", "summary", "cross_meeting", "action_items"):
            with patch("agents.answer_agent.Runner.run", new_callable=AsyncMock) as mock_run:
                mock_run.return_value = self._make_mock_run_return()
                agent = AnswerAgent()
                # Should not raise
                await agent.run(
                    query="test query",
                    retrieval_result=make_retrieval_result(),
                    query_type=query_type,
                )

    @pytest.mark.asyncio
    async def test_run_passes_context_in_input(self):
        with patch("agents.answer_agent.Runner.run", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = self._make_mock_run_return()
            agent = AnswerAgent()
            await agent.run(
                query="what did we decide?",
                retrieval_result=make_retrieval_result(),
                query_type="decision",
            )
            # Extract the input argument passed to Runner.run
            call_args = mock_run.call_args
            # Runner.run(self._agent, input=prompt) → args[1] or kwargs['input']
            input_arg = call_args.kwargs.get("input") or call_args.args[1]
            assert "eng-standup" in input_arg

    @pytest.mark.asyncio
    async def test_run_result_has_answer_string(self):
        with patch("agents.answer_agent.Runner.run", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = self._make_mock_run_return()
            agent = AnswerAgent()
            result = await agent.run(
                query="what did we decide?",
                retrieval_result=make_retrieval_result(),
                query_type="decision",
            )
            assert isinstance(result.answer, str)
            assert len(result.answer) > 0

    @pytest.mark.asyncio
    async def test_run_result_source_meeting_ids_is_list_of_strings(self):
        with patch("agents.answer_agent.Runner.run", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = self._make_mock_run_return()
            agent = AnswerAgent()
            result = await agent.run(
                query="what did we decide?",
                retrieval_result=make_retrieval_result(),
                query_type="decision",
            )
            assert all(isinstance(mid, str) for mid in result.source_meeting_ids)

    @pytest.mark.asyncio
    async def test_run_result_confidence_is_valid(self):
        with patch("agents.answer_agent.Runner.run", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = self._make_mock_run_return()
            agent = AnswerAgent()
            result = await agent.run(
                query="what did we decide?",
                retrieval_result=make_retrieval_result(),
                query_type="decision",
            )
            assert result.confidence in ("high", "medium", "low")

    @pytest.mark.asyncio
    async def test_run_empty_retrieval_still_returns_answer_output(self):
        with patch("agents.answer_agent.Runner.run", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = self._make_mock_run_return()
            agent = AnswerAgent()
            result = await agent.run(
                query="what did we decide?",
                retrieval_result=make_retrieval_result(results=[]),
                query_type="decision",
            )
            assert isinstance(result, AnswerOutput)
