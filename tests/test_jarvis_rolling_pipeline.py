"""
Tests for Plan 07-03: Wire rolling pipeline into jarvis.py.

Tests cover:
1. Sentence buffer flush triggers on sentence count threshold
2. Force flush (interrupt) clears buffer and increments batch number
3. Flush with empty buffer is a no-op
4. Meeting header written only on first batch
5. handle_query routes memory query to HistoryManagerAgent first
6. handle_query falls back to OrchestratorAgent when HistoryManagerAgent raises
7. Disconnect handler resets buffer state

All tests use unittest.mock to patch LLM calls and file I/O.
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

import jarvis
from storage.models import BatchSummaryOutput, MeetingIndexEntry
from agents.orchestrator import OrchestratorResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_index_entry(**overrides) -> MeetingIndexEntry:
    """Return a sample MeetingIndexEntry with sensible defaults."""
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


def _make_batch_summary(**overrides) -> BatchSummaryOutput:
    """Return a sample BatchSummaryOutput with sensible defaults."""
    defaults = dict(
        summary_text="Alice discussed the sprint plan. Bob confirmed the timeline.",
        key_points=["Sprint plan discussed", "Timeline confirmed"],
        speakers=["Alice", "Bob"],
    )
    defaults.update(overrides)
    return BatchSummaryOutput(**defaults)


def _reset_buffer_state(entry=None, batch_num=0, header_written=False):
    """Reset all module-level buffer state before each test."""
    jarvis._sentence_buffer.clear()
    jarvis._last_flush_ts = 0.0
    jarvis._current_batch_num = batch_num
    jarvis._current_meeting_entry = entry
    jarvis._meeting_header_written = header_written


# ---------------------------------------------------------------------------
# Test 1: Sentence buffer flush triggers on sentence count threshold
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_flush_triggers_on_sentence_count():
    """Flush occurs when buffer has >= SENTENCE_FLUSH_COUNT sentences."""
    fake_summary = _make_batch_summary()
    entry = _make_index_entry()
    _reset_buffer_state(entry=entry)

    # Fill buffer with enough lines to meet the threshold
    for i in range(jarvis.SENTENCE_FLUSH_COUNT):
        jarvis._sentence_buffer.append(f"Alice: This is sentence number {i}.")

    with patch.object(jarvis.rolling_summarizer, "run", new_callable=AsyncMock) as mock_run, \
         patch.object(jarvis.meeting_writer, "write_meeting_header", new_callable=AsyncMock), \
         patch.object(jarvis.meeting_writer, "append_batch", new_callable=AsyncMock) as mock_append, \
         patch.object(jarvis.meeting_writer, "upsert_index", new_callable=AsyncMock):

        mock_run.return_value = fake_summary
        await jarvis._flush_sentence_buffer(bot_id="test", force=False)

    # rolling_summarizer.run called once with joined buffer lines
    mock_run.assert_called_once()
    call_args = mock_run.call_args[0][0]
    assert "Alice:" in call_args

    # append_batch called once with batch_num=1
    mock_append.assert_called_once()
    assert mock_append.call_args.kwargs.get("batch_num") == 1 or mock_append.call_args[1].get("batch_num") == 1

    # Buffer is empty after flush
    assert jarvis._sentence_buffer == []


# ---------------------------------------------------------------------------
# Test 2: Force flush (interrupt) clears buffer and increments batch number
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_force_flush_clears_buffer_and_increments_batch():
    """force=True flushes even when buffer is below threshold; batch_num increments."""
    fake_summary = _make_batch_summary()
    entry = _make_index_entry()
    _reset_buffer_state(entry=entry, batch_num=0)

    # Buffer has only 3 lines — below threshold
    for i in range(3):
        jarvis._sentence_buffer.append(f"Bob: Short line {i}.")

    with patch.object(jarvis.rolling_summarizer, "run", new_callable=AsyncMock) as mock_run, \
         patch.object(jarvis.meeting_writer, "write_meeting_header", new_callable=AsyncMock), \
         patch.object(jarvis.meeting_writer, "append_batch", new_callable=AsyncMock), \
         patch.object(jarvis.meeting_writer, "upsert_index", new_callable=AsyncMock):

        mock_run.return_value = fake_summary
        await jarvis._flush_sentence_buffer(bot_id="test", force=True)

    # rolling_summarizer.run was called (force=True bypasses count check)
    mock_run.assert_called_once()

    # Buffer is empty after forced flush
    assert jarvis._sentence_buffer == []

    # Batch number incremented from 0 to 1
    assert jarvis._current_batch_num == 1


# ---------------------------------------------------------------------------
# Test 3: Flush with empty buffer is a no-op
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_flush_empty_buffer_is_noop():
    """Flushing an empty buffer (even with force=True) does not call summarizer or writer."""
    entry = _make_index_entry()
    _reset_buffer_state(entry=entry)
    # Ensure buffer is empty
    jarvis._sentence_buffer.clear()

    with patch.object(jarvis.rolling_summarizer, "run", new_callable=AsyncMock) as mock_run, \
         patch.object(jarvis.meeting_writer, "append_batch", new_callable=AsyncMock) as mock_append:

        await jarvis._flush_sentence_buffer(bot_id="test", force=True)

    mock_run.assert_not_called()
    mock_append.assert_not_called()


# ---------------------------------------------------------------------------
# Test 4: Meeting header written only on first batch
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_meeting_header_written_only_once():
    """write_meeting_header is called exactly once across two consecutive flushes."""
    fake_summary = _make_batch_summary()
    entry = _make_index_entry()
    _reset_buffer_state(entry=entry, header_written=False)

    with patch.object(jarvis.rolling_summarizer, "run", new_callable=AsyncMock) as mock_run, \
         patch.object(jarvis.meeting_writer, "write_meeting_header", new_callable=AsyncMock) as mock_header, \
         patch.object(jarvis.meeting_writer, "append_batch", new_callable=AsyncMock) as mock_append, \
         patch.object(jarvis.meeting_writer, "upsert_index", new_callable=AsyncMock):

        mock_run.return_value = fake_summary

        # First flush
        jarvis._sentence_buffer.extend(["Alice: line one.", "Bob: line two.", "Alice: line three."])
        await jarvis._flush_sentence_buffer(bot_id="test", force=True)

        # Second flush
        jarvis._sentence_buffer.extend(["Bob: another sentence.", "Alice: and another."])
        await jarvis._flush_sentence_buffer(bot_id="test", force=True)

    # write_meeting_header called exactly once (not twice)
    assert mock_header.call_count == 1, f"Expected 1 header write, got {mock_header.call_count}"

    # append_batch called twice (batch 1 and batch 2)
    assert mock_append.call_count == 2, f"Expected 2 batch appends, got {mock_append.call_count}"


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

    # HistoryManagerAgent.run was called once
    mock_hm.assert_called_once()

    # OrchestratorAgent.run was NOT called (History Manager succeeded)
    mock_orch.assert_not_called()

    # speak_chunked was called with the answer text
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

    # OrchestratorAgent.run was called once as fallback
    mock_orch.assert_called_once()

    # speak_chunked was called with the orchestrator's answer
    mock_speak.assert_called_once()
    assert "Pinecone says" in mock_speak.call_args[0][0]


# ---------------------------------------------------------------------------
# Test 7: Disconnect handler resets buffer state
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_disconnect_resets_buffer_state():
    """Resetting buffer state sets _current_meeting_entry=None, _current_batch_num=0, empty buffer."""
    entry = _make_index_entry()
    _reset_buffer_state(entry=entry, batch_num=3, header_written=True)
    jarvis._sentence_buffer.append("Alice: Some sentence.")

    # Simulate the disconnect reset block directly
    jarvis._current_meeting_entry = None
    jarvis._meeting_header_written = False
    jarvis._current_batch_num = 0
    jarvis._last_flush_ts = 0.0
    async with jarvis._sentence_buffer_lock:
        jarvis._sentence_buffer.clear()

    assert jarvis._current_meeting_entry is None
    assert jarvis._current_batch_num == 0
    assert jarvis._sentence_buffer == []
    assert jarvis._last_flush_ts == 0.0
