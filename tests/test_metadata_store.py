"""
Tests for storage.metadata_store — MetadataStore async file I/O.

Uses tmp_path fixture to avoid polluting the real filesystem.
All tests are async (asyncio_mode=auto from pytest.ini).
"""

import pytest
from pathlib import Path
from storage.metadata_store import MetadataStore
from storage.models import ActionItem, MeetingRecord


def _make_record(
    meeting_id: str = "meeting-001",
    channel_id: str = "C01234567",
) -> MeetingRecord:
    """Create a test MeetingRecord with sensible defaults."""
    return MeetingRecord(
        meeting_id=meeting_id,
        channel_id=channel_id,
        channel_name="weekly-eng-sync",
        start_ts=1712000000,
        end_ts=1712003600,
        duration_seconds=3600,
        participants=["Alice", "Bob"],
        summary_text="Discussed Q2 roadmap and Jenkins pipeline.",
        topics_covered=["roadmap Q2", "Jenkins pipeline"],
        action_items=[
            ActionItem(owner="Alice", task="Write RFC", due="2024-04-10")
        ],
        decisions=["Ship v2.1 by April 15"],
        series_name="Weekly Engineering Sync",
        recurrence_pattern="weekly",
        status="complete",
        raw_transcript_chars=5400,
        summarized_at=1712003700,
    )


class TestMetadataStoreWrite:
    """Tests for MetadataStore.write()."""

    async def test_write_creates_file_at_correct_path(self, tmp_path):
        """write() creates file at {base_dir}/{channel_id}/{meeting_id}.json."""
        store = MetadataStore(base_dir=str(tmp_path))
        record = _make_record()
        file_path = await store.write(record)

        expected = tmp_path / record.channel_id / f"{record.meeting_id}.json"
        assert file_path == expected
        assert expected.exists()

    async def test_write_creates_parent_directories(self, tmp_path):
        """write() creates intermediate directories if they don't exist."""
        store = MetadataStore(base_dir=str(tmp_path / "deep" / "path"))
        record = _make_record()
        file_path = await store.write(record)

        assert file_path.exists()

    async def test_write_file_contains_valid_json(self, tmp_path):
        """write() produces a valid JSON file that can be parsed."""
        import json

        store = MetadataStore(base_dir=str(tmp_path))
        record = _make_record()
        file_path = await store.write(record)

        content = file_path.read_text()
        data = json.loads(content)
        assert data["meeting_id"] == record.meeting_id
        assert data["channel_id"] == record.channel_id
        assert isinstance(data["start_ts"], int)

    async def test_write_returns_path_object(self, tmp_path):
        """write() returns a Path object."""
        store = MetadataStore(base_dir=str(tmp_path))
        record = _make_record()
        result = await store.write(record)
        assert isinstance(result, Path)


class TestMetadataStoreRead:
    """Tests for MetadataStore.read()."""

    async def test_read_returns_identical_record(self, tmp_path):
        """read() returns MeetingRecord identical to what was written."""
        store = MetadataStore(base_dir=str(tmp_path))
        record = _make_record()
        await store.write(record)

        restored = await store.read(record.channel_id, record.meeting_id)

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
        assert restored.status == record.status

    async def test_read_restores_action_items(self, tmp_path):
        """read() correctly deserializes nested action items."""
        store = MetadataStore(base_dir=str(tmp_path))
        record = _make_record()
        await store.write(record)

        restored = await store.read(record.channel_id, record.meeting_id)

        assert len(restored.action_items) == 1
        item = restored.action_items[0]
        assert item.owner == "Alice"
        assert item.task == "Write RFC"
        assert item.due == "2024-04-10"

    async def test_read_raises_file_not_found_for_nonexistent_meeting(self, tmp_path):
        """read() raises FileNotFoundError if meeting JSON doesn't exist."""
        store = MetadataStore(base_dir=str(tmp_path))

        with pytest.raises(FileNotFoundError):
            await store.read("C_NONEXISTENT", "nonexistent-meeting-id")

    async def test_read_timestamps_are_ints(self, tmp_path):
        """read() returns timestamps as int (Unix epoch), not datetime objects."""
        store = MetadataStore(base_dir=str(tmp_path))
        record = _make_record()
        await store.write(record)

        restored = await store.read(record.channel_id, record.meeting_id)

        assert isinstance(restored.start_ts, int)
        assert isinstance(restored.end_ts, int)
        assert isinstance(restored.summarized_at, int)


class TestMetadataStoreListByChannel:
    """Tests for MetadataStore.list_by_channel()."""

    async def test_list_returns_meeting_ids(self, tmp_path):
        """list_by_channel() returns meeting_ids for a given channel_id."""
        store = MetadataStore(base_dir=str(tmp_path))
        record1 = _make_record(meeting_id="m-001", channel_id="C01")
        record2 = _make_record(meeting_id="m-002", channel_id="C01")
        await store.write(record1)
        await store.write(record2)

        meeting_ids = await store.list_by_channel("C01")

        assert set(meeting_ids) == {"m-001", "m-002"}

    async def test_list_empty_for_nonexistent_channel(self, tmp_path):
        """list_by_channel() returns empty list if channel dir doesn't exist."""
        store = MetadataStore(base_dir=str(tmp_path))
        meeting_ids = await store.list_by_channel("C_NONEXISTENT")
        assert meeting_ids == []

    async def test_list_only_returns_own_channel_meetings(self, tmp_path):
        """list_by_channel() doesn't return meetings from other channels."""
        store = MetadataStore(base_dir=str(tmp_path))
        record_c1 = _make_record(meeting_id="m-c1", channel_id="C01")
        record_c2 = _make_record(meeting_id="m-c2", channel_id="C02")
        await store.write(record_c1)
        await store.write(record_c2)

        c1_meetings = await store.list_by_channel("C01")
        c2_meetings = await store.list_by_channel("C02")

        assert c1_meetings == ["m-c1"]
        assert c2_meetings == ["m-c2"]
