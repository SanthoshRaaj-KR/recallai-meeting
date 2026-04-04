"""Tests for MeetingState class — thread-safe asyncio-based meeting state."""

import asyncio
import time
import pytest

from meeting_state import MeetingState


# Test 1: Initial state values
async def test_initial_state():
    state = MeetingState()
    assert await state.get_bot_id() is None
    assert await state.is_active() is False
    assert await state.is_listening() is False
    assert await state.get_transcript_count() == 0


# Test 2: add_transcript appends entry, get_transcript returns formatted string
async def test_add_and_get_transcript():
    state = MeetingState()
    await state.add_transcript("Alice", "Hello everyone", 1000.0)
    await state.add_transcript("Bob", "Hi Alice", 1001.0)
    transcript = await state.get_transcript()
    assert "Alice: Hello everyone" in transcript
    assert "Bob: Hi Alice" in transcript


# Test 3: get_transcript with empty log returns "[No transcript yet]"
async def test_get_transcript_empty():
    state = MeetingState()
    result = await state.get_transcript()
    assert result == "[No transcript yet]"


# Test 4: set_bot_id stores value, get_bot_id returns it
async def test_set_get_bot_id():
    state = MeetingState()
    await state.set_bot_id("bot-abc-123")
    assert await state.get_bot_id() == "bot-abc-123"


# Test 5: set_active stores value, is_active returns it
async def test_set_active():
    state = MeetingState()
    await state.set_active(True)
    assert await state.is_active() is True
    await state.set_active(False)
    assert await state.is_active() is False


# Test 6: set_listening stores value, is_listening returns it
async def test_set_listening():
    state = MeetingState()
    await state.set_listening(True)
    assert await state.is_listening() is True
    await state.set_listening(False)
    assert await state.is_listening() is False


# Test 7: get_transcript_count returns correct count
async def test_get_transcript_count():
    state = MeetingState()
    assert await state.get_transcript_count() == 0
    await state.add_transcript("Alice", "First", 1000.0)
    assert await state.get_transcript_count() == 1
    await state.add_transcript("Bob", "Second", 1001.0)
    assert await state.get_transcript_count() == 2


# Test 8: Concurrent — 50 simultaneous add_transcript calls produce exactly 50 entries
async def test_concurrent_add_transcript():
    state = MeetingState()
    tasks = [
        state.add_transcript(f"Participant-{i}", f"Message {i}", float(i))
        for i in range(50)
    ]
    await asyncio.gather(*tasks)
    count = await state.get_transcript_count()
    assert count == 50, f"Expected 50 entries, got {count}"


# Test 9: Concurrent — simultaneous set_bot_id and get_bot_id never returns partial/corrupted value
async def test_concurrent_set_get_bot_id():
    state = MeetingState()

    async def setter(i: int):
        await state.set_bot_id(f"bot-{i:04d}")

    async def getter():
        val = await state.get_bot_id()
        # Value must be either None or a properly formatted bot id
        if val is not None:
            assert val.startswith("bot-"), f"Corrupted bot_id: {val!r}"

    tasks = []
    for i in range(20):
        tasks.append(setter(i))
        tasks.append(getter())

    await asyncio.gather(*tasks)
    # Final value must be a valid bot id (not None, since setters ran)
    final = await state.get_bot_id()
    assert final is not None
    assert final.startswith("bot-"), f"Final bot_id corrupted: {final!r}"


# Test 10: get_health_snapshot returns correct structure
async def test_get_health_snapshot():
    state = MeetingState()
    await state.set_bot_id("bot-xyz")
    await state.set_active(True)
    await state.add_transcript("Alice", "Test", 1000.0)

    snapshot = await state.get_health_snapshot()
    assert snapshot["bot_id"] == "bot-xyz"
    assert snapshot["active"] is True
    assert snapshot["transcript_lines"] == 1
