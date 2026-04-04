"""
Tests for HistoryManagerAgent.

All tests mock AsyncOpenAI to prevent real API calls.
All tests mock MeetingWriterAgent and AnswerAgent to be fully unit-isolated.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from agents.answer_agent import AnswerOutput
from agents.history_manager import HistoryManagerAgent
from agents.orchestrator import OrchestratorResult
from agents.retriever import RetrievalResult
from storage.models import MeetingIndexEntry


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_entry(meeting_id: str, title: str = "Standup", channel: str = "general") -> MeetingIndexEntry:
    """Build a minimal MeetingIndexEntry for testing."""
    return MeetingIndexEntry(
        meeting_id=meeting_id,
        title=title,
        date="2026-04-01",
        channel_id="C123",
        channel_name=channel,
        overview="A test meeting overview.",
        md_path=f"meetings/C123/2026-04-01_{meeting_id}.md",
        participants=["Alice", "Bob"],
        start_ts=1743465600,
    )


def _make_answer_output(
    answer: str = "The decision was X.",
    meeting_id: str = "mtg-001",
    confidence: str = "high",
) -> AnswerOutput:
    return AnswerOutput(
        answer=answer,
        source_meeting_ids=[meeting_id],
        confidence=confidence,
    )


def _make_agent(
    entries,
    md_content: str = "# Meeting content",
    retriever=None,
):
    """Build a HistoryManagerAgent with mocked dependencies."""
    writer = MagicMock()
    writer.read_index = AsyncMock(return_value=entries)
    writer.read_md = AsyncMock(return_value=md_content)

    answer_agent = MagicMock()
    answer_agent.run = AsyncMock(return_value=_make_answer_output())

    with patch("agents.history_manager.AsyncOpenAI"):
        agent = HistoryManagerAgent(
            meeting_writer=writer,
            answer_agent=answer_agent,
            retriever=retriever,
        )

    agent._writer = writer
    agent._answer = answer_agent
    return agent


# ---------------------------------------------------------------------------
# Test 1: Single matching meeting returns synthesized answer
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_run_single_match_returns_answer():
    """Single LLM-selected meeting loads .md and returns AnswerAgent output."""
    entry = _make_entry("mtg-001")
    agent = _make_agent(entries=[entry])

    # Patch _select_meeting to simulate confident single match
    agent._select_meeting = AsyncMock(return_value={"selected": "mtg-001", "candidates": []})
    agent._answer.run = AsyncMock(return_value=_make_answer_output("The decision was X.", "mtg-001", "high"))

    result = await agent.run(query="What did we decide?", user_id="U1", channel_id="C123")

    assert isinstance(result, OrchestratorResult)
    assert result.answer == "The decision was X."
    assert result.needs_disambiguation is False
    assert result.source_meeting_ids == ["mtg-001"]


# ---------------------------------------------------------------------------
# Test 2: Multiple candidates triggers disambiguation
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_run_multiple_candidates_triggers_disambiguation():
    """Two equally plausible meetings produce a disambiguation OrchestratorResult."""
    entry1 = _make_entry("mtg-001", title="Standup Monday")
    entry2 = _make_entry("mtg-002", title="Standup Tuesday")
    agent = _make_agent(entries=[entry1, entry2])

    agent._select_meeting = AsyncMock(
        return_value={"selected": "", "candidates": ["mtg-001", "mtg-002"]}
    )

    result = await agent.run(query="What happened in the standup?", user_id="U1", channel_id="C123")

    assert result.needs_disambiguation is True
    assert len(result.disambiguation_options) == 2
    for opt in result.disambiguation_options:
        assert "index" in opt
        assert "meeting_id" in opt
        assert "title" in opt
        assert "channel" in opt
        assert "date" in opt


# ---------------------------------------------------------------------------
# Test 3: Empty index returns "no history" message without calling _select_meeting
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_run_empty_index_returns_no_history():
    """Empty meeting index returns graceful 'no history' without LLM call."""
    agent = _make_agent(entries=[])
    agent._select_meeting = AsyncMock()

    result = await agent.run(query="Anything?", user_id="U1", channel_id="C123")

    assert "No meeting history" in result.answer
    assert result.confidence == "low"
    agent._select_meeting.assert_not_called()


# ---------------------------------------------------------------------------
# Test 4: No index match falls back to Pinecone when retriever is injected
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_run_no_match_falls_back_to_pinecone():
    """When LLM finds no match and retriever is set, Pinecone retrieval is used."""
    entry = _make_entry("mtg-001")

    # Build a mock retriever that returns one result
    mock_retriever = MagicMock()
    mock_retriever.retrieve = AsyncMock(
        return_value=RetrievalResult(
            query="past meeting topic",
            results=[{
                "id": "mtg-001",
                "score": 0.85,
                "metadata": {
                    "channel_name": "general",
                    "start_ts": 1743465600,
                    "summary_text": "Pinecone content here",
                    "decisions": [],
                    "topics_covered": [],
                    "participants": [],
                    "action_items": [],
                },
            }],
            total_candidates=20,
            returned_count=1,
        )
    )

    agent = _make_agent(entries=[entry], retriever=mock_retriever)
    agent._select_meeting = AsyncMock(return_value={"selected": "", "candidates": []})
    agent._answer.run = AsyncMock(
        return_value=AnswerOutput(answer="Found via Pinecone.", source_meeting_ids=["mtg-001"], confidence="medium")
    )

    result = await agent.run(query="past meeting topic", user_id="U1", channel_id="C123")

    mock_retriever.retrieve.assert_called_once()
    assert result.answer == "Found via Pinecone."
    assert result.confidence != "low"


# ---------------------------------------------------------------------------
# Test 5: No match and no retriever returns graceful "no match" message
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_run_no_match_no_retriever_returns_graceful_message():
    """No LLM match and no retriever injected returns a polite 'no match' response."""
    entry = _make_entry("mtg-001")
    agent = _make_agent(entries=[entry], retriever=None)
    agent._select_meeting = AsyncMock(return_value={"selected": "", "candidates": []})

    result = await agent.run(query="something unrelated", user_id="U1", channel_id="C123")

    assert "couldn't find" in result.answer.lower() or "no match" in result.answer.lower() or "couldn't" in result.answer.lower()
    assert result.source_meeting_ids == []


# ---------------------------------------------------------------------------
# Test 6: _select_meeting parse error returns safe fallback without raising
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_select_meeting_parse_error_returns_safe_fallback():
    """Invalid JSON response from LLM is caught; safe fallback dict returned."""
    entry = _make_entry("mtg-001")

    # Build agent with real _select_meeting but mocked OpenAI that returns bad JSON
    writer = MagicMock()
    writer.read_index = AsyncMock(return_value=[entry])
    writer.read_md = AsyncMock(return_value="# content")
    answer_agent = MagicMock()
    answer_agent.run = AsyncMock(return_value=_make_answer_output())

    with patch("agents.history_manager.AsyncOpenAI") as mock_openai_cls:
        # Simulate the OpenAI response returning invalid JSON content
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "oops"

        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
        mock_openai_cls.return_value = mock_client

        agent = HistoryManagerAgent(
            meeting_writer=writer,
            answer_agent=answer_agent,
        )

    result = await agent._select_meeting("any query", [entry])

    assert result == {"selected": "", "candidates": []}
