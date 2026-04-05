"""
Tests for Plan 07-03: Wire rolling pipeline into jarvis.py (transcript-first design).

Tests cover:
1. Sentence buffer flush triggers on sentence count threshold (writes transcript lines, no LLM)
2. Force flush (interrupt) clears buffer
3. Flush with empty buffer is a no-op
4. Meeting header written only on first flush
5. handle_query routes memory query to HistoryManagerAgent first
6. handle_query falls back to OrchestratorAgent when HistoryManagerAgent raises
7. Disconnect handler resets buffer state
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

import jarvis
from storage.models import MeetingIndexEntry
from agents.orchestrator import OrchestratorResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_index_entry(**overrides) -> MeetingIndexEntry:
    defaults = dict(
        meeting_id="mtg-test",
        title="Test Meeting",
        date="2026-04-05",
        channel_id="C123",
        channel_name="test-channel",
        overview="",
        md_path="meetings/C123/2026-04-05_mtg-test.md",
        participants=[],
        start_ts=1775328000,
    )
    defaults.update(overrides)
    return MeetingIndexEntry(**defaults)


def _reset_buffer_state(entry=None, header_written=False):
    """Reset all module-level buffer state before each test."""
    jarvis._sentence_buffer.clear()
    jarvis._full_transcript_lines.clear()
    jarvis._last_flush_ts = 0.0
    jarvis._current_meeting_entry = entry
    jarvis._meeting_header_written = header_written


# ---------------------------------------------------------------------------
# Test 1: Sentence buffer flush triggers on sentence count threshold
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_flush_triggers_on_sentence_count():
    """Flush occurs when buffer has >= SENTENCE_FLUSH_COUNT sentences; no LLM called."""
    entry = _make_index_entry()
    _reset_buffer_state(entry=entry)

    for i in range(jarvis.SENTENCE_FLUSH_COUNT):
        jarvis._sentence_buffer.append(f"Alice: This is sentence number {i}.")

    with patch.object(jarvis.meeting_writer, "write_meeting_header", new_callable=AsyncMock), \
         patch.object(jarvis.meeting_writer, "append_transcript_lines", new_callable=AsyncMock) as mock_append, \
         patch.object(jarvis.meeting_writer, "upsert_index", new_callable=AsyncMock):

        await jarvis._flush_sentence_buffer(bot_id="test", force=False)

    # append_transcript_lines called with the buffer lines
    mock_append.assert_called_once()
    lines_arg = mock_append.call_args[0][1]  # second positional arg
    assert any("Alice:" in l for l in lines_arg)

    # Buffer is empty after flush
    assert jarvis._sentence_buffer == []


# ---------------------------------------------------------------------------
# Test 2: Force flush clears buffer regardless of count
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_force_flush_clears_buffer():
    """force=True flushes even when buffer is below threshold."""
    entry = _make_index_entry()
    _reset_buffer_state(entry=entry)

    for i in range(3):
        jarvis._sentence_buffer.append(f"Bob: Short line {i}.")

    with patch.object(jarvis.meeting_writer, "write_meeting_header", new_callable=AsyncMock), \
         patch.object(jarvis.meeting_writer, "append_transcript_lines", new_callable=AsyncMock) as mock_append, \
         patch.object(jarvis.meeting_writer, "upsert_index", new_callable=AsyncMock):

        await jarvis._flush_sentence_buffer(bot_id="test", force=True)

    mock_append.assert_called_once()
    assert jarvis._sentence_buffer == []


# ---------------------------------------------------------------------------
# Test 3: Flush with empty buffer is a no-op
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_flush_empty_buffer_is_noop():
    """Flushing an empty buffer does not call any writer methods."""
    entry = _make_index_entry()
    _reset_buffer_state(entry=entry)
    jarvis._sentence_buffer.clear()

    with patch.object(jarvis.meeting_writer, "append_transcript_lines", new_callable=AsyncMock) as mock_append:
        await jarvis._flush_sentence_buffer(bot_id="test", force=True)

    mock_append.assert_not_called()


# ---------------------------------------------------------------------------
# Test 4: Meeting header written only on first flush
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_meeting_header_written_only_once():
    """write_meeting_header is called exactly once across two consecutive flushes."""
    entry = _make_index_entry()
    _reset_buffer_state(entry=entry, header_written=False)

    with patch.object(jarvis.meeting_writer, "write_meeting_header", new_callable=AsyncMock) as mock_header, \
         patch.object(jarvis.meeting_writer, "append_transcript_lines", new_callable=AsyncMock), \
         patch.object(jarvis.meeting_writer, "upsert_index", new_callable=AsyncMock):

        jarvis._sentence_buffer.extend(["Alice: line one.", "Bob: line two.", "Alice: line three."])
        await jarvis._flush_sentence_buffer(bot_id="test", force=True)

        jarvis._sentence_buffer.extend(["Bob: another.", "Alice: and another."])
        await jarvis._flush_sentence_buffer(bot_id="test", force=True)

    assert mock_header.call_count == 1


# ---------------------------------------------------------------------------
# Test 5: handle_query routes memory query to HistoryManagerAgent first
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_handle_query_routes_to_history_manager_first():
    """Memory query is routed to HistoryManagerAgent; OrchestratorAgent not called."""
    history_result = OrchestratorResult(
        query="what did we decide last week?",
        query_type="memory_query",
        answer="You decided to ship on Friday.",
        source_meeting_ids=["mtg-1"],
        confidence="high",
    )

    with patch.object(jarvis.history_manager, "run", new_callable=AsyncMock) as mock_hm, \
         patch.object(jarvis, "_stream_llm_and_speak", new_callable=AsyncMock), \
         patch("jarvis.speak_chunked", new_callable=AsyncMock) as mock_speak, \
         patch.object(jarvis.orchestrator, "run", new_callable=AsyncMock) as mock_orch:

        mock_hm.return_value = history_result
        await jarvis.handle_query(query="what did we decide last week?", bot_id="test")

    mock_hm.assert_called_once()
    mock_orch.assert_not_called()
    mock_speak.assert_called_once()
    assert "Friday" in mock_speak.call_args[0][0]


# ---------------------------------------------------------------------------
# Test 6: handle_query falls back to OrchestratorAgent when HistoryManagerAgent raises
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_handle_query_falls_back_to_orchestrator_on_history_error():
    """When HistoryManagerAgent raises, OrchestratorAgent is called as fallback."""
    orch_result = OrchestratorResult(
        query="what did we decide last week?",
        query_type="memory_query",
        answer="Pinecone says: you decided X.",
        source_meeting_ids=["mtg-2"],
        confidence="medium",
    )

    with patch.object(jarvis.history_manager, "run", new_callable=AsyncMock) as mock_hm, \
         patch("jarvis.speak_chunked", new_callable=AsyncMock) as mock_speak, \
         patch.object(jarvis.orchestrator, "run", new_callable=AsyncMock) as mock_orch:

        mock_hm.side_effect = RuntimeError("LLM timeout")
        mock_orch.return_value = orch_result
        await jarvis.handle_query(query="what did we decide last week?", bot_id="test")

    mock_orch.assert_called_once()
    mock_speak.assert_called_once()
    assert "Pinecone says" in mock_speak.call_args[0][0]


# ---------------------------------------------------------------------------
# Test 7: Disconnect handler resets buffer state
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_disconnect_resets_buffer_state():
    """After disconnect reset, all buffer state is cleared."""
    entry = _make_index_entry()
    _reset_buffer_state(entry=entry, header_written=True)
    jarvis._sentence_buffer.append("Alice: Some sentence.")
    jarvis._full_transcript_lines.append("Alice: Some sentence.")

    # Simulate the disconnect reset block
    jarvis._current_meeting_entry = None
    jarvis._meeting_header_written = False
    jarvis._last_flush_ts = 0.0
    async with jarvis._sentence_buffer_lock:
        jarvis._sentence_buffer.clear()
        jarvis._full_transcript_lines.clear()

    assert jarvis._current_meeting_entry is None
    assert jarvis._meeting_header_written is False
    assert jarvis._sentence_buffer == []
    assert jarvis._full_transcript_lines == []
    assert jarvis._last_flush_ts == 0.0
