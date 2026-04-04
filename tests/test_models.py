"""
Tests for storage.models — MeetingRecord and ActionItem Pydantic models.

Verifies round-trip serialization, field types (timestamps are int, not datetime),
and validation with minimal and full field sets.
"""

import pytest
from storage.models import ActionItem, MeetingRecord


class TestActionItem:
    """Tests for the ActionItem model."""

    def test_action_item_required_fields(self):
        """ActionItem has owner (str) and task (str) fields."""
        item = ActionItem(owner="Alice", task="Write RFC for v2.1")
        assert item.owner == "Alice"
        assert item.task == "Write RFC for v2.1"
        assert item.due is None

    def test_action_item_with_due(self):
        """ActionItem due is Optional[str]."""
        item = ActionItem(owner="Bob", task="Review PR", due="2024-04-10")
        assert item.due == "2024-04-10"

    def test_action_item_round_trip(self):
        """ActionItem serializes and deserializes without data loss."""
        item = ActionItem(owner="Charlie", task="Deploy hotfix", due="2024-04-15")
        json_str = item.model_dump_json()
        restored = ActionItem.model_validate_json(json_str)
        assert restored.owner == item.owner
        assert restored.task == item.task
        assert restored.due == item.due


class TestMeetingRecordMinimal:
    """Tests for MeetingRecord with only required fields."""

    def test_minimal_record_validates(self):
        """MeetingRecord with only required fields validates successfully."""
        record = MeetingRecord(
            meeting_id="test-meeting-001",
            channel_id="C01234567",
            channel_name="weekly-eng-sync",
            start_ts=1712000000,
            summary_text="A brief meeting about the roadmap.",
        )
        assert record.meeting_id == "test-meeting-001"
        assert record.channel_id == "C01234567"
        assert record.channel_name == "weekly-eng-sync"
        assert record.start_ts == 1712000000
        assert record.summary_text == "A brief meeting about the roadmap."

    def test_minimal_record_defaults(self):
        """MeetingRecord defaults are applied correctly."""
        record = MeetingRecord(
            meeting_id="m1",
            channel_id="C1",
            channel_name="general",
            start_ts=1000000,
            summary_text="Short meeting.",
        )
        assert record.end_ts is None
        assert record.duration_seconds is None
        assert record.participants == []
        assert record.topics_covered == []
        assert record.action_items == []
        assert record.decisions == []
        assert record.series_name is None
        assert record.recurrence_pattern is None
        assert record.status == "complete"
        assert record.raw_transcript_chars is None
        assert record.summarized_at is None

    def test_start_ts_is_int(self):
        """start_ts must be stored as int (Unix epoch), not datetime."""
        record = MeetingRecord(
            meeting_id="m1",
            channel_id="C1",
            channel_name="general",
            start_ts=1712000000,
            summary_text="Test",
        )
        assert isinstance(record.start_ts, int)

    def test_end_ts_is_int_or_none(self):
        """end_ts must be int (Unix epoch) when set, not datetime."""
        record = MeetingRecord(
            meeting_id="m1",
            channel_id="C1",
            channel_name="general",
            start_ts=1712000000,
            end_ts=1712003600,
            summary_text="Test",
        )
        assert isinstance(record.end_ts, int)

    def test_summarized_at_is_int_or_none(self):
        """summarized_at must be int (Unix epoch) when set."""
        record = MeetingRecord(
            meeting_id="m1",
            channel_id="C1",
            channel_name="general",
            start_ts=1712000000,
            summarized_at=1712003700,
            summary_text="Test",
        )
        assert isinstance(record.summarized_at, int)


class TestMeetingRecordFull:
    """Tests for MeetingRecord with all optional fields set."""

    def _full_record(self) -> MeetingRecord:
        return MeetingRecord(
            meeting_id="uuid-001",
            channel_id="C01234567",
            channel_name="weekly-eng-sync",
            start_ts=1712000000,
            end_ts=1712003600,
            duration_seconds=3600,
            participants=["Alice", "Bob", "Charlie"],
            summary_text="Free-form paragraph summary of the meeting.",
            topics_covered=["roadmap Q2", "Jenkins pipeline", "hiring"],
            action_items=[
                ActionItem(owner="Alice", task="Write RFC for v2.1", due="2024-04-10")
            ],
            decisions=["Ship v2.1 by April 15", "Defer Jenkins migration to Q3"],
            series_name="Weekly Engineering Sync",
            recurrence_pattern="weekly",
            status="complete",
            raw_transcript_chars=14200,
            summarized_at=1712003700,
        )

    def test_full_record_validates(self):
        """MeetingRecord with all optional fields set validates successfully."""
        record = self._full_record()
        assert record.meeting_id == "uuid-001"
        assert len(record.participants) == 3
        assert len(record.action_items) == 1
        assert record.action_items[0].owner == "Alice"
        assert record.status == "complete"

    def test_full_record_round_trip_json(self):
        """MeetingRecord round-trips through model_dump_json() and model_validate_json() with all fields intact."""
        record = self._full_record()
        json_str = record.model_dump_json(indent=2)
        restored = MeetingRecord.model_validate_json(json_str)

        assert restored.meeting_id == record.meeting_id
        assert restored.channel_id == record.channel_id
        assert restored.channel_name == record.channel_name
        assert restored.start_ts == record.start_ts
        assert restored.end_ts == record.end_ts
        assert restored.duration_seconds == record.duration_seconds
        assert restored.participants == record.participants
        assert restored.summary_text == record.summary_text
        assert restored.topics_covered == record.topics_covered
        assert restored.decisions == record.decisions
        assert restored.series_name == record.series_name
        assert restored.recurrence_pattern == record.recurrence_pattern
        assert restored.status == record.status
        assert restored.raw_transcript_chars == record.raw_transcript_chars
        assert restored.summarized_at == record.summarized_at

    def test_full_record_action_items_round_trip(self):
        """Action items survive round-trip serialization."""
        record = self._full_record()
        json_str = record.model_dump_json(indent=2)
        restored = MeetingRecord.model_validate_json(json_str)

        assert len(restored.action_items) == 1
        item = restored.action_items[0]
        assert item.owner == "Alice"
        assert item.task == "Write RFC for v2.1"
        assert item.due == "2024-04-10"

    def test_timestamps_are_ints_in_serialized_json(self):
        """start_ts and end_ts are stored as int in the serialized JSON, not ISO strings."""
        import json

        record = self._full_record()
        data = json.loads(record.model_dump_json())

        assert isinstance(data["start_ts"], int), "start_ts must be int in JSON"
        assert isinstance(data["end_ts"], int), "end_ts must be int in JSON"
        assert isinstance(data["summarized_at"], int), "summarized_at must be int in JSON"

    def test_status_field_defaults_to_complete(self):
        """status defaults to 'complete' for standard meeting records."""
        record = MeetingRecord(
            meeting_id="m1",
            channel_id="C1",
            channel_name="general",
            start_ts=1000000,
            summary_text="Test",
        )
        assert record.status == "complete"

    def test_status_field_accepts_partial(self):
        """status accepts 'partial' for mid-meeting summaries."""
        record = MeetingRecord(
            meeting_id="m1",
            channel_id="C1",
            channel_name="general",
            start_ts=1000000,
            summary_text="Partial summary so far.",
            status="partial",
        )
        assert record.status == "partial"
