"""Integration tests for concurrent WebSocket transcript writes and agent reads.

These tests prove that INFRA-01 is satisfied: simultaneous transcript writes
and agent reads under asyncio.Lock produce no state corruption.
"""

import asyncio
import pytest

from meeting_state import MeetingState


# ---------------------------------------------------------------------------
# Test 1: Concurrent transcript writes and reads
# ---------------------------------------------------------------------------

async def test_concurrent_transcript_writes_and_reads():
    """50 concurrent writes and 10 concurrent reads produce no corruption."""
    ms = MeetingState()
    errors = []

    async def write(i):
        await ms.add_transcript(f"user-{i}", f"message-{i}", float(i))

    async def read():
        try:
            result = await ms.get_transcript()
            assert isinstance(result, str)
        except Exception as e:
            errors.append(e)

    writes = [write(i) for i in range(50)]
    reads = [read() for _ in range(10)]
    await asyncio.gather(*writes, *reads)

    assert not errors, f"Read errors during concurrent access: {errors}"
    count = await ms.get_transcript_count()
    assert count == 50, f"Expected 50 entries, got {count}"


# ---------------------------------------------------------------------------
# Test 2: Concurrent bot_id and listening toggles
# ---------------------------------------------------------------------------

async def test_concurrent_bot_id_and_listening():
    """Simultaneous listening toggles and bot_id set/gets produce no corruption."""
    ms = MeetingState()
    errors = []

    async def toggle_listening(i):
        try:
            await ms.set_listening(i % 2 == 0)
        except Exception as e:
            errors.append(e)

    async def set_bot(i):
        try:
            await ms.set_bot_id(f"id-{i:03d}")
        except Exception as e:
            errors.append(e)

    async def get_bot():
        try:
            val = await ms.get_bot_id()
            # Must be None or a properly formatted id string
            if val is not None:
                assert val.startswith("id-"), f"Corrupted bot_id: {val!r}"
        except Exception as e:
            errors.append(e)

    tasks = []
    for i in range(20):
        tasks.append(toggle_listening(i))
        tasks.append(set_bot(i))
        tasks.append(get_bot())

    await asyncio.gather(*tasks)

    assert not errors, f"Errors during concurrent bot_id/listening ops: {errors}"

    # Final state must be consistent
    final_bot_id = await ms.get_bot_id()
    assert final_bot_id is not None, "bot_id should not be None after setters ran"
    assert final_bot_id.startswith("id-"), f"Final bot_id corrupted: {final_bot_id!r}"

    final_listening = await ms.is_listening()
    assert isinstance(final_listening, bool), f"is_listening must return bool, got {type(final_listening)}"


# ---------------------------------------------------------------------------
# Test 3: Concurrent writes and health snapshots
# ---------------------------------------------------------------------------

async def test_concurrent_writes_and_health_snapshots():
    """30 concurrent writes and 10 health snapshots produce consistent results."""
    ms = MeetingState()
    snapshot_results = []

    async def write(i):
        await ms.add_transcript(f"speaker-{i}", f"line-{i}", float(i))

    async def snapshot():
        result = await ms.get_health_snapshot()
        snapshot_results.append(result)

    writes = [write(i) for i in range(30)]
    snapshots = [snapshot() for _ in range(10)]
    await asyncio.gather(*writes, *snapshots)

    # Every snapshot must have the required keys and valid values
    assert len(snapshot_results) == 10, f"Expected 10 snapshots, got {len(snapshot_results)}"
    for snap in snapshot_results:
        assert "bot_id" in snap, f"Missing 'bot_id' key in snapshot: {snap}"
        assert "active" in snap, f"Missing 'active' key in snapshot: {snap}"
        assert "transcript_lines" in snap, f"Missing 'transcript_lines' key in snapshot: {snap}"
        assert isinstance(snap["transcript_lines"], int), (
            f"transcript_lines must be int, got {type(snap['transcript_lines'])}"
        )
        assert snap["transcript_lines"] >= 0, (
            f"transcript_lines must be non-negative, got {snap['transcript_lines']}"
        )

    # Final count must equal the number of writes
    final_count = await ms.get_transcript_count()
    assert final_count == 30, f"Expected 30 entries after writes, got {final_count}"
