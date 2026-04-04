"""
Tests for SummarizerAgent.

RED phase: All tests fail with ImportError before agents/summarizer.py exists.
GREEN phase: All tests pass after implementation.

Tests cover:
- Structure: returned MeetingRecord type and required fields
- Attribution: action item owner is always a participant name, never blank
- Partial detection: status="partial" when transcript < 500 chars
"""

import time
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from storage.models import ActionItem, MeetingRecord

# This import fails until agents/summarizer.py is created (RED phase)
from agents.summarizer import SummarizerAgent

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_TRANSCRIPT = """Alice: We should finalize the API contract by end of week.
Bob: I'll own the RFC. Can I get it reviewed by Thursday?
Alice: Sure. I'll do the review. Also, we decided to drop the v1 endpoints.
Carol: Agreed. Let's move forward with v2 only.
Bob: One more thing — Carol, can you update the changelog?
Carol: Yes, I'll handle that.
Alice: Great, so the plan is: Bob writes the RFC, I review it, and Carol updates the changelog.
Bob: Correct. And let's make sure the API contract is finalized before we start the migration.
Carol: Agreed. I'll also draft a migration guide for the v1 to v2 transition.
Alice: Good idea. Let's set a deadline of next Friday for the full migration guide.
Carol: Works for me. I'll have a draft ready by Wednesday for review.
Bob: I'll review Carol's draft on Thursday then.
Alice: Perfect. We are aligned on everything."""

# Verify SAMPLE_TRANSCRIPT >= 500 chars for long-transcript tests
assert len(SAMPLE_TRANSCRIPT) >= 500, (
    f"SAMPLE_TRANSCRIPT must be >= 500 chars, got {len(SAMPLE_TRANSCRIPT)}"
)

SHORT_TRANSCRIPT = "Alice: Quick sync. Bob: Sounds good. Alice: Let's wrap."

# Verify SHORT_TRANSCRIPT < 500 chars for partial detection tests
assert len(SHORT_TRANSCRIPT) < 500, (
    f"SHORT_TRANSCRIPT must be < 500 chars, got {len(SHORT_TRANSCRIPT)}"
)

SAMPLE_META = {
    "meeting_id": "mtg-test-001",
    "channel_id": "C99999",
    "channel_name": "eng-platform",
    "start_ts": 1711900000,
    "end_ts": 1711903600,
    "duration_seconds": 3600,
    "participants": ["Alice", "Bob", "Carol"],
}

# Pre-built MeetingRecord that mocked runner will return via final_output
MOCK_MEETING_RECORD = MeetingRecord(
    meeting_id="mtg-test-001",
    channel_id="C99999",
    channel_name="eng-platform",
    start_ts=1711900000,
    end_ts=1711903600,
    duration_seconds=3600,
    participants=["Alice", "Bob", "Carol"],
    summary_text="The team discussed finalizing the API contract and agreed to drop v1 endpoints.",
    topics_covered=["API contract", "changelog update", "v2 migration"],
    decisions=["Drop v1 endpoints", "Move forward with v2 only"],
    action_items=[
        ActionItem(owner="Bob", task="Write the RFC", due="Thursday"),
        ActionItem(owner="Alice", task="Review the RFC by Thursday"),
        ActionItem(owner="Carol", task="Update the changelog"),
    ],
    status="complete",
    raw_transcript_chars=len(SAMPLE_TRANSCRIPT),
    summarized_at=int(time.time()),
)


def _make_mock_result(record: MeetingRecord) -> MagicMock:
    """Build a mock RunResult where .final_output returns the given MeetingRecord."""
    mock_result = MagicMock()
    mock_result.final_output = record
    return mock_result


# ---------------------------------------------------------------------------
# TestSummarizerAgentStructure
# ---------------------------------------------------------------------------


class TestSummarizerAgentStructure:
    """Tests verifying returned type and required field presence."""

    async def test_returns_meeting_record_instance(self):
        """run() must return a MeetingRecord instance."""
        mock_result = _make_mock_result(MOCK_MEETING_RECORD)
        with patch("agents.summarizer.Runner") as MockRunner:
            MockRunner.run = AsyncMock(return_value=mock_result)
            agent = SummarizerAgent()
            result = await agent.run(
                transcript=SAMPLE_TRANSCRIPT, meeting_meta=SAMPLE_META
            )
        assert isinstance(result, MeetingRecord)

    async def test_structured_output_has_all_required_fields(self):
        """Returned record has non-empty decisions, topics_covered, participants, action_items."""
        mock_result = _make_mock_result(MOCK_MEETING_RECORD)
        with patch("agents.summarizer.Runner") as MockRunner:
            MockRunner.run = AsyncMock(return_value=mock_result)
            agent = SummarizerAgent()
            result = await agent.run(
                transcript=SAMPLE_TRANSCRIPT, meeting_meta=SAMPLE_META
            )
        assert len(result.decisions) > 0, "decisions must be non-empty"
        assert len(result.topics_covered) > 0, "topics_covered must be non-empty"
        assert len(result.participants) > 0, "participants must be non-empty"
        assert len(result.action_items) > 0, "action_items must be non-empty"

    async def test_passes_transcript_to_agent(self):
        """The raw transcript string must appear in the input passed to Runner.run."""
        mock_result = _make_mock_result(MOCK_MEETING_RECORD)
        captured_input = []

        async def capture_run(agent_obj, input, **kwargs):
            captured_input.append(input)
            return mock_result

        with patch("agents.summarizer.Runner") as MockRunner:
            MockRunner.run = capture_run
            agent = SummarizerAgent()
            await agent.run(transcript=SAMPLE_TRANSCRIPT, meeting_meta=SAMPLE_META)

        assert len(captured_input) == 1
        assert SAMPLE_TRANSCRIPT in captured_input[0], (
            "Transcript text must appear in the prompt sent to Runner"
        )


# ---------------------------------------------------------------------------
# TestSummarizerAgentAttribution
# ---------------------------------------------------------------------------


class TestSummarizerAgentAttribution:
    """Tests verifying action item attribution to specific participant names."""

    async def test_action_item_owner_is_participant_name(self):
        """Action item owner must be a participant name (e.g. 'Bob')."""
        mock_result = _make_mock_result(MOCK_MEETING_RECORD)
        with patch("agents.summarizer.Runner") as MockRunner:
            MockRunner.run = AsyncMock(return_value=mock_result)
            agent = SummarizerAgent()
            result = await agent.run(
                transcript=SAMPLE_TRANSCRIPT, meeting_meta=SAMPLE_META
            )
        assert result.action_items[0].owner == "Bob"

    async def test_action_item_owner_never_blank(self):
        """SummarizerAgent must raise ValueError if any action item has a blank owner."""
        blank_owner_record = MeetingRecord(
            meeting_id="mtg-test-001",
            channel_id="C99999",
            channel_name="eng-platform",
            start_ts=1711900000,
            summary_text="Summary.",
            participants=["Alice", "Bob"],
            topics_covered=["Topic"],
            decisions=["Decision"],
            action_items=[
                ActionItem(owner="", task="Do something"),
            ],
            status="complete",
            raw_transcript_chars=len(SAMPLE_TRANSCRIPT),
            summarized_at=int(time.time()),
        )
        mock_result = _make_mock_result(blank_owner_record)
        with patch("agents.summarizer.Runner") as MockRunner:
            MockRunner.run = AsyncMock(return_value=mock_result)
            agent = SummarizerAgent()
            with pytest.raises(ValueError, match="owner must not be blank"):
                await agent.run(
                    transcript=SAMPLE_TRANSCRIPT, meeting_meta=SAMPLE_META
                )

    async def test_action_item_has_task_field(self):
        """Each action item must have a non-empty task field."""
        mock_result = _make_mock_result(MOCK_MEETING_RECORD)
        with patch("agents.summarizer.Runner") as MockRunner:
            MockRunner.run = AsyncMock(return_value=mock_result)
            agent = SummarizerAgent()
            result = await agent.run(
                transcript=SAMPLE_TRANSCRIPT, meeting_meta=SAMPLE_META
            )
        for item in result.action_items:
            assert item.task and len(item.task) > 0, (
                f"action_items task must be non-empty, got: {item.task!r}"
            )


# ---------------------------------------------------------------------------
# TestSummarizerAgentPartialDetection
# ---------------------------------------------------------------------------


class TestSummarizerAgentPartialDetection:
    """Tests verifying status tagging based on transcript length."""

    async def test_short_transcript_produces_partial_status(self):
        """Transcript < 500 chars must produce status='partial'."""
        # Build a mock record that the LLM would return (status doesn't matter — agent overrides)
        short_record = MeetingRecord(
            meeting_id="mtg-test-001",
            channel_id="C99999",
            channel_name="eng-platform",
            start_ts=1711900000,
            summary_text="Quick sync.",
            participants=["Alice", "Bob"],
            topics_covered=["Sync"],
            decisions=[],
            action_items=[],
            status="complete",  # LLM may say complete — agent must override to partial
            raw_transcript_chars=len(SHORT_TRANSCRIPT),
            summarized_at=int(time.time()),
        )
        mock_result = _make_mock_result(short_record)
        with patch("agents.summarizer.Runner") as MockRunner:
            MockRunner.run = AsyncMock(return_value=mock_result)
            agent = SummarizerAgent()
            result = await agent.run(
                transcript=SHORT_TRANSCRIPT, meeting_meta=SAMPLE_META
            )
        assert result.status == "partial", (
            f"Short transcript must produce status='partial', got: {result.status!r}"
        )

    async def test_long_transcript_preserves_complete_status(self):
        """Transcript >= 500 chars must produce status='complete'."""
        mock_result = _make_mock_result(MOCK_MEETING_RECORD)
        with patch("agents.summarizer.Runner") as MockRunner:
            MockRunner.run = AsyncMock(return_value=mock_result)
            agent = SummarizerAgent()
            result = await agent.run(
                transcript=SAMPLE_TRANSCRIPT, meeting_meta=SAMPLE_META
            )
        assert result.status == "complete", (
            f"Long transcript must preserve status='complete', got: {result.status!r}"
        )

    async def test_raw_transcript_chars_is_set(self):
        """raw_transcript_chars must equal len(transcript) for both short and long."""
        mock_result_long = _make_mock_result(MOCK_MEETING_RECORD)
        with patch("agents.summarizer.Runner") as MockRunner:
            MockRunner.run = AsyncMock(return_value=mock_result_long)
            agent = SummarizerAgent()
            result = await agent.run(
                transcript=SAMPLE_TRANSCRIPT, meeting_meta=SAMPLE_META
            )
        assert result.raw_transcript_chars == len(SAMPLE_TRANSCRIPT), (
            "raw_transcript_chars must equal len(transcript)"
        )

        # Also test for short transcript
        short_record = MeetingRecord(
            meeting_id="mtg-test-001",
            channel_id="C99999",
            channel_name="eng-platform",
            start_ts=1711900000,
            summary_text="Quick sync.",
            participants=["Alice", "Bob"],
            topics_covered=["Sync"],
            decisions=[],
            action_items=[],
            status="complete",
            raw_transcript_chars=len(SHORT_TRANSCRIPT),
            summarized_at=int(time.time()),
        )
        mock_result_short = _make_mock_result(short_record)
        with patch("agents.summarizer.Runner") as MockRunner:
            MockRunner.run = AsyncMock(return_value=mock_result_short)
            agent = SummarizerAgent()
            result_short = await agent.run(
                transcript=SHORT_TRANSCRIPT, meeting_meta=SAMPLE_META
            )
        assert result_short.raw_transcript_chars == len(SHORT_TRANSCRIPT), (
            "raw_transcript_chars must equal len(SHORT_TRANSCRIPT)"
        )

    async def test_summarized_at_is_unix_epoch_integer(self):
        """summarized_at must be a positive integer (Unix epoch)."""
        mock_result = _make_mock_result(MOCK_MEETING_RECORD)
        with patch("agents.summarizer.Runner") as MockRunner:
            MockRunner.run = AsyncMock(return_value=mock_result)
            agent = SummarizerAgent()
            result = await agent.run(
                transcript=SAMPLE_TRANSCRIPT, meeting_meta=SAMPLE_META
            )
        assert isinstance(result.summarized_at, int), (
            f"summarized_at must be an int, got: {type(result.summarized_at)}"
        )
        assert result.summarized_at > 0, (
            f"summarized_at must be positive, got: {result.summarized_at}"
        )
