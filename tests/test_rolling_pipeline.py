"""
Tests for Plan 07-01: Rolling Summarizer + Meeting Writer pipeline.

Tests cover:
1. BatchSummaryOutput model validation
2. MeetingIndexEntry model validation
3. RollingSummarizerAgent.run() — mocked Runner.run()
4. MeetingWriterAgent.write_meeting_header() — real filesystem under tmp_path
5. MeetingWriterAgent.append_batch() — real filesystem, append-only verification
6. MeetingWriterAgent.upsert_index() — create and update idempotency
7. MeetingWriterAgent.read_index() — empty list when no index exists

No mocking of filesystem for writer tests — tmp_path fixture throughout (Plan spec).
"""

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from storage.models import BatchSummaryOutput, MeetingIndexEntry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_index_entry(**overrides) -> MeetingIndexEntry:
    """Return a sample MeetingIndexEntry with sensible defaults."""
    defaults = dict(
        meeting_id="mtg-abc",
        title="Engineering Standup",
        date="2026-04-05",
        channel_id="C123",
        channel_name="eng-standup",
        overview="Discussed sprint blockers and deployment timeline.",
        md_path="meetings/C123/2026-04-05_mtg-abc.md",
        participants=["Alice", "Bob"],
        start_ts=1775328000,
    )
    defaults.update(overrides)
    return MeetingIndexEntry(**defaults)


def _make_batch_summary(**overrides) -> BatchSummaryOutput:
    """Return a sample BatchSummaryOutput with sensible defaults."""
    defaults = dict(
        summary_text="Alice outlined the deployment plan. Bob raised a concern about CI.",
        key_points=["Deployment on Friday", "CI pipeline issue"],
        speakers=["Alice", "Bob"],
    )
    defaults.update(overrides)
    return BatchSummaryOutput(**defaults)


# ---------------------------------------------------------------------------
# Test 1: BatchSummaryOutput model validation
# ---------------------------------------------------------------------------

def test_batch_summary_output_valid():
    """BatchSummaryOutput instantiates with valid data and fields are correct types."""
    output = BatchSummaryOutput(
        summary_text="s",
        key_points=["a"],
        speakers=["Alice"],
    )
    assert isinstance(output.speakers, list)
    assert output.key_points == ["a"]
    assert output.summary_text == "s"


# ---------------------------------------------------------------------------
# Test 2: MeetingIndexEntry model validation
# ---------------------------------------------------------------------------

def test_meeting_index_entry_valid():
    """MeetingIndexEntry instantiates and all required fields are accessible."""
    entry = _make_index_entry()
    assert entry.meeting_id == "mtg-abc"
    assert entry.title == "Engineering Standup"
    assert entry.date == "2026-04-05"
    assert entry.channel_id == "C123"
    assert entry.channel_name == "eng-standup"
    assert entry.overview == "Discussed sprint blockers and deployment timeline."
    assert entry.md_path == "meetings/C123/2026-04-05_mtg-abc.md"
    assert entry.participants == ["Alice", "Bob"]
    assert isinstance(entry.start_ts, int)


# ---------------------------------------------------------------------------
# Test 3: RollingSummarizerAgent returns BatchSummaryOutput
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_rolling_summarizer_agent_run():
    """RollingSummarizerAgent.run() delegates to Runner.run() and returns BatchSummaryOutput."""
    from agents.rolling_summarizer import RollingSummarizerAgent

    fake_output = _make_batch_summary()
    mock_result = MagicMock()
    mock_result.final_output = fake_output

    with patch("agents.rolling_summarizer.Runner.run", new_callable=AsyncMock) as mock_run:
        mock_run.return_value = mock_result
        agent = RollingSummarizerAgent()
        transcript = "Alice: Hello\nBob: Hi there"
        result = await agent.run(transcript)

    assert isinstance(result, BatchSummaryOutput)
    assert result.summary_text == fake_output.summary_text
    mock_run.assert_called_once()
    call_kwargs = mock_run.call_args
    # Verify the transcript string was passed as input
    assert call_kwargs.kwargs.get("input") == transcript or transcript in str(call_kwargs)


# ---------------------------------------------------------------------------
# Test 4: write_meeting_header creates .md with correct header
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_write_meeting_header_creates_file(tmp_path):
    """write_meeting_header creates a .md file with the correct header block."""
    from agents.meeting_writer import MeetingWriterAgent

    writer = MeetingWriterAgent(base_dir=str(tmp_path))
    entry = _make_index_entry()

    await writer.write_meeting_header(entry)

    expected_path = tmp_path / "C123" / "2026-04-05_mtg-abc.md"
    assert expected_path.exists(), f"Expected .md file at {expected_path}"

    content = expected_path.read_text()
    assert "# Meeting:" in content
    assert "**Channel:**" in content
    assert "**Date:**" in content
    assert "Engineering Standup" in content
    assert "2026-04-05" in content


# ---------------------------------------------------------------------------
# Test 5: append_batch appends sections without rewriting
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_append_batch_appends_two_sections(tmp_path):
    """append_batch appends two batch sections and preserves the original header."""
    from agents.meeting_writer import MeetingWriterAgent

    writer = MeetingWriterAgent(base_dir=str(tmp_path))
    entry = _make_index_entry()

    # Write header first
    await writer.write_meeting_header(entry)

    batch1 = _make_batch_summary(
        summary_text="First batch content.",
        key_points=["Point A"],
        speakers=["Alice"],
    )
    batch2 = _make_batch_summary(
        summary_text="Second batch content.",
        key_points=["Point B"],
        speakers=["Bob"],
    )

    await writer.append_batch(entry, batch_num=1, batch_summary=batch1, timestamp_str="10:00 AM")
    await writer.append_batch(entry, batch_num=2, batch_summary=batch2, timestamp_str="10:05 AM")

    expected_path = tmp_path / "C123" / "2026-04-05_mtg-abc.md"
    content = expected_path.read_text()

    # Original header is preserved
    assert "# Meeting:" in content

    # Both batch sections are present
    assert "Batch 1" in content
    assert "Batch 2" in content

    # Speakers line present in each section
    assert "**Speakers:** Alice" in content
    assert "**Speakers:** Bob" in content


# ---------------------------------------------------------------------------
# Test 6: upsert_index create and update
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_upsert_index_creates_and_updates(tmp_path):
    """upsert_index creates index on first call; second call with same meeting_id replaces entry."""
    from agents.meeting_writer import MeetingWriterAgent

    writer = MeetingWriterAgent(base_dir=str(tmp_path))
    entry = _make_index_entry(overview="Initial overview.")

    # First upsert — creates the index
    await writer.upsert_index(entry)

    # Second upsert — same meeting_id, updated overview
    updated_entry = _make_index_entry(overview="Updated overview after batch 2.")
    await writer.upsert_index(updated_entry)

    # read_index should return exactly one entry (no duplicates)
    entries = await writer.read_index()
    assert len(entries) == 1, f"Expected 1 entry, got {len(entries)}"

    # The overview should reflect the second upsert
    assert entries[0].overview == "Updated overview after batch 2."


# ---------------------------------------------------------------------------
# Test 7: read_index returns empty list when no index exists
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_read_index_returns_empty_when_no_file(tmp_path):
    """read_index returns [] without raising an exception when the index file is absent."""
    from agents.meeting_writer import MeetingWriterAgent

    writer = MeetingWriterAgent(base_dir=str(tmp_path))
    result = await writer.read_index()

    assert result == [], f"Expected empty list, got {result}"
