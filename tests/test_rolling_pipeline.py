"""
Tests for Plan 07-01: Meeting Writer pipeline (updated for transcript-first design).

Tests cover:
1. MeetingMetadataOutput model validation
2. MeetingIndexEntry model validation (including new goals/key_decisions/conclusions fields)
3. MeetingMetadataAgent.run() — mocked Runner.run()
4. MeetingWriterAgent.write_meeting_header() — real filesystem under tmp_path
5. MeetingWriterAgent.append_transcript_lines() — real filesystem, append-only
6. MeetingWriterAgent.finalize_meeting() — appends Meeting Summary section
7. MeetingWriterAgent.upsert_index() — create and update idempotency
"""

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from storage.models import MeetingIndexEntry, MeetingMetadataOutput


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_index_entry(**overrides) -> MeetingIndexEntry:
    defaults = dict(
        meeting_id="mtg-abc",
        title="Engineering Standup",
        date="2026-04-05",
        channel_id="C123",
        channel_name="eng-standup",
        overview="",
        md_path="meetings/C123/2026-04-05_mtg-abc.md",
        participants=["Alice", "Bob"],
        start_ts=1775328000,
    )
    defaults.update(overrides)
    return MeetingIndexEntry(**defaults)


def _make_metadata(**overrides) -> MeetingMetadataOutput:
    defaults = dict(
        overview="Team discussed deployment and CI issues.",
        goals=["Unblock deployment", "Resolve CI failures"],
        key_decisions=["Deploy on Friday", "Bob owns CI fix"],
        conclusions=["All blockers assigned", "Next sync Thursday"],
    )
    defaults.update(overrides)
    return MeetingMetadataOutput(**defaults)


# ---------------------------------------------------------------------------
# Test 1: MeetingMetadataOutput model validation
# ---------------------------------------------------------------------------

def test_meeting_metadata_output_valid():
    """MeetingMetadataOutput instantiates with valid data."""
    output = MeetingMetadataOutput(
        overview="Short overview.",
        goals=["Goal A"],
        key_decisions=["Decision X"],
        conclusions=["Conclusion Z"],
    )
    assert output.overview == "Short overview."
    assert isinstance(output.goals, list)
    assert isinstance(output.key_decisions, list)
    assert isinstance(output.conclusions, list)


# ---------------------------------------------------------------------------
# Test 2: MeetingIndexEntry model validation
# ---------------------------------------------------------------------------

def test_meeting_index_entry_valid():
    """MeetingIndexEntry includes new metadata fields with correct defaults."""
    entry = _make_index_entry()
    assert entry.meeting_id == "mtg-abc"
    assert entry.channel_id == "C123"
    assert entry.goals == []
    assert entry.key_decisions == []
    assert entry.conclusions == []
    assert isinstance(entry.start_ts, int)


# ---------------------------------------------------------------------------
# Test 3: MeetingMetadataAgent.run() delegates to Runner.run()
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_meeting_metadata_agent_run():
    """MeetingMetadataAgent.run() calls Runner.run() and returns MeetingMetadataOutput."""
    from agents.rolling_summarizer import MeetingMetadataAgent

    fake_output = _make_metadata()
    mock_result = MagicMock()
    mock_result.final_output = fake_output

    with patch("agents.rolling_summarizer.Runner.run", new_callable=AsyncMock) as mock_run:
        mock_run.return_value = mock_result
        agent = MeetingMetadataAgent()
        transcript = "Alice: Hello\nBob: Hi there"
        result = await agent.run(transcript)

    assert isinstance(result, MeetingMetadataOutput)
    assert result.overview == fake_output.overview
    mock_run.assert_called_once()


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
    assert expected_path.exists()
    content = expected_path.read_text()
    assert "# Meeting:" in content
    assert "Engineering Standup" in content
    assert "2026-04-05" in content
    assert "**Participants:**" in content


# ---------------------------------------------------------------------------
# Test 5: append_transcript_lines appends raw lines, preserves header
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_append_transcript_lines_appends_and_preserves_header(tmp_path):
    """append_transcript_lines appends raw lines without rewriting the header."""
    from agents.meeting_writer import MeetingWriterAgent

    writer = MeetingWriterAgent(base_dir=str(tmp_path))
    entry = _make_index_entry()

    await writer.write_meeting_header(entry)
    await writer.append_transcript_lines(entry, ["Alice: Good morning.", "Bob: Let's start."])
    await writer.append_transcript_lines(entry, ["Alice: Any blockers?"])

    content = (tmp_path / "C123" / "2026-04-05_mtg-abc.md").read_text()
    assert "# Meeting:" in content
    assert "Alice: Good morning." in content
    assert "Bob: Let's start." in content
    assert "Alice: Any blockers?" in content


# ---------------------------------------------------------------------------
# Test 6: finalize_meeting appends Meeting Summary section
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_finalize_meeting_appends_summary_section(tmp_path):
    """finalize_meeting appends a structured Meeting Summary after the transcript."""
    from agents.meeting_writer import MeetingWriterAgent

    writer = MeetingWriterAgent(base_dir=str(tmp_path))
    entry = _make_index_entry()
    metadata = _make_metadata()

    await writer.write_meeting_header(entry)
    await writer.append_transcript_lines(entry, ["Alice: Hello."])
    await writer.finalize_meeting(entry, metadata)

    content = (tmp_path / "C123" / "2026-04-05_mtg-abc.md").read_text()
    assert "## Meeting Summary" in content
    assert "**Goals:**" in content
    assert "**Key Decisions:**" in content
    assert "**Conclusions:**" in content
    assert "Unblock deployment" in content
    assert "Deploy on Friday" in content
    # Transcript still present above the summary
    assert "Alice: Hello." in content


# ---------------------------------------------------------------------------
# Test 7: upsert_index create and update idempotency
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_upsert_index_creates_and_updates(tmp_path):
    """upsert_index creates index; second call with same meeting_id replaces entry."""
    from agents.meeting_writer import MeetingWriterAgent

    writer = MeetingWriterAgent(base_dir=str(tmp_path))
    entry = _make_index_entry(overview="Initial.")
    await writer.upsert_index(entry)

    updated = _make_index_entry(overview="Updated.", goals=["Goal X"])
    await writer.upsert_index(updated)

    entries = await writer.read_index()
    assert len(entries) == 1
    assert entries[0].overview == "Updated."
    assert entries[0].goals == ["Goal X"]
