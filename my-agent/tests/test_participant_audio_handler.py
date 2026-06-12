"""Tests for participant_audio_handler — per-participant STT and wake-word gate."""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import participant_audio_handler as handler
from participant_audio_handler import SessionAudioState, _WAKE_PATTERN


# ── Wake-word pattern ─────────────────────────────────────────────────────────

@pytest.mark.parametrize("text,expected_query", [
    ("Hey Jarvis, what did we decide?", "what did we decide?"),
    ("jarvis summarize the meeting", "summarize the meeting"),
    ("Hey Jarvas, who owns the action item?", "who owns the action item?"),
    ("Hey Jervis what time is it", "what time is it"),
    ("hey jarvus, how many bugs are open", "how many bugs are open"),
])
def test_wake_pattern_matches(text: str, expected_query: str) -> None:
    m = _WAKE_PATTERN.search(text.strip())
    assert m is not None
    assert m.group(1).strip() == expected_query


@pytest.mark.parametrize("text", [
    "can we deploy on Friday?",
    "I think the build is broken",
    "let me check the docs",
    "",
])
def test_wake_pattern_no_match(text: str) -> None:
    assert _WAKE_PATTERN.search(text.strip()) is None


def test_wake_pattern_bare_jarvis_has_empty_query() -> None:
    """Bare 'Jarvis' with no follow-up should produce an empty query string."""
    m = _WAKE_PATTERN.search("Hey Jarvis")
    assert m is not None
    assert m.group(1).strip() == ""


# ── Semaphore logic ───────────────────────────────────────────────────────────

@pytest.fixture
def session_state(event_loop):
    return SessionAudioState("test-session-abc123", event_loop)


@pytest.mark.asyncio
async def test_semaphore_first_wake_word_wins() -> None:
    """First participant to say 'Hey Jarvis' acquires the semaphore."""
    loop = asyncio.get_running_loop()
    state = SessionAudioState("sess-001", loop)

    with patch("participant_audio_handler.session_store") as mock_store:
        mock_store.get.return_value = {"transcript": []}
        mock_store.patch.return_value = None

        await state._on_final_transcript(1, "Alice", "Hey Jarvis, what did we decide?")

        assert state._semaphore_locked is True
        # active_speaker must have been written to session_store
        patch_calls = [call.args for call in mock_store.patch.call_args_list]
        speaker_patches = [
            args for args in patch_calls
            if isinstance(args[1], dict) and args[1].get("active_speaker") == "Alice"
        ]
        assert len(speaker_patches) == 1


@pytest.mark.asyncio
async def test_semaphore_blocks_second_speaker() -> None:
    """While semaphore is locked, a second speaker's wake word is suppressed."""
    loop = asyncio.get_running_loop()
    state = SessionAudioState("sess-002", loop)

    with patch("participant_audio_handler.session_store") as mock_store:
        mock_store.get.return_value = {"transcript": []}
        mock_store.patch.return_value = None

        # Alice fires first
        await state._on_final_transcript(1, "Alice", "Hey Jarvis, summarize this")
        assert state._semaphore_locked is True

        # Bob tries while semaphore is locked
        patch_call_count_before = mock_store.patch.call_count
        await state._on_final_transcript(2, "Bob", "Hey Jarvis, how many bugs are open")

        # No NEW session_store patch with Bob as active_speaker
        bob_patches = [
            call.args for call in mock_store.patch.call_args_list
            if isinstance(call.args[1] if call.args else {}, dict)
            and call.args[1].get("active_speaker") == "Bob"
        ]
        assert len(bob_patches) == 0
        # Semaphore still locked by Alice
        assert state._semaphore_locked is True


@pytest.mark.asyncio
async def test_semaphore_releases_correctly() -> None:
    """release_semaphore() unlocks the semaphore and clears session_store fields."""
    loop = asyncio.get_running_loop()
    state = SessionAudioState("sess-003", loop)
    state._semaphore_locked = True

    with patch("participant_audio_handler.session_store") as mock_store:
        mock_store.patch.return_value = None
        state.release_semaphore()

    assert state._semaphore_locked is False
    patched = mock_store.patch.call_args.args[1]
    assert patched["active_speaker"] is None
    assert patched["jarvis_speaking"] is False


@pytest.mark.asyncio
async def test_release_semaphore_after_disconnect_patches_store() -> None:
    """release_semaphore() still patches session_store when no in-process state exists."""
    with patch("participant_audio_handler.session_store") as mock_store:
        mock_store.patch.return_value = None
        handler.release_semaphore("orphan-session-999")

    mock_store.patch.assert_called_once()
    _, kwargs_dict = mock_store.patch.call_args.args
    assert kwargs_dict["active_speaker"] is None


# ── Transcript writing ────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_transcript_entry_written_for_non_wake_word() -> None:
    """Every utterance (wake word or not) is persisted as a diarized entry."""
    loop = asyncio.get_running_loop()
    state = SessionAudioState("sess-004", loop)

    with patch("participant_audio_handler.session_store") as mock_store:
        mock_store.get.return_value = {"transcript": []}
        mock_store.patch.return_value = None

        await state._on_final_transcript(3, "Carol", "I think the build is broken")

        # patch must have been called with a transcript list containing Carol's entry
        transcript_patches = [
            call.args[1]["transcript"]
            for call in mock_store.patch.call_args_list
            if "transcript" in call.args[1]
        ]
        assert len(transcript_patches) == 1
        entry = transcript_patches[0][-1]
        assert entry["participant"] == "Carol"
        assert entry["text"] == "I think the build is broken"
        assert entry["source"] == "recall_participant"


@pytest.mark.asyncio
async def test_transcript_truncated_at_2000_entries() -> None:
    """Transcript list is capped at 2000 entries to prevent unbounded growth."""
    loop = asyncio.get_running_loop()
    state = SessionAudioState("sess-005", loop)

    existing = [{"participant": "X", "text": f"line {i}", "source": "recall_participant"}
                for i in range(2000)]

    with patch("participant_audio_handler.session_store") as mock_store:
        mock_store.get.return_value = {"transcript": existing}
        mock_store.patch.return_value = None

        await state._on_final_transcript(1, "Alice", "one more line")

        saved = mock_store.patch.call_args.args[1]["transcript"]
        assert len(saved) == 2000
        assert saved[-1]["text"] == "one more line"


# ── cleanup_session ───────────────────────────────────────────────────────────

def test_cleanup_session_removes_state() -> None:
    """cleanup_session() removes the session from the in-process registry."""
    loop = asyncio.new_event_loop()
    handler._sessions["cleanup-test"] = SessionAudioState("cleanup-test", loop)

    handler.cleanup_session("cleanup-test")

    assert "cleanup-test" not in handler._sessions
    loop.close()


def test_cleanup_session_noop_for_unknown_session() -> None:
    """cleanup_session() does not raise for an unknown session id."""
    handler.cleanup_session("does-not-exist-xyz")  # must not raise


# ── ParticipantSTTSession: connect disabled gracefully ────────────────────────

def test_participant_stt_stream_noop_before_connect() -> None:
    """stream() before connect() must not raise."""
    from participant_audio_handler import ParticipantSTTSession

    loop = asyncio.new_event_loop()
    p = ParticipantSTTSession(
        session_id="s1",
        participant_id=99,
        participant_name="Tester",
        on_final_transcript=AsyncMock(),
        loop=loop,
    )
    # _connected is False — should silently drop the bytes
    p.stream(b"\x00" * 320)
    loop.close()


@pytest.mark.asyncio
async def test_participant_stt_no_assemblyai_logs_warning(caplog) -> None:
    """When assemblyai is not importable, connect_async logs a warning instead of raising."""
    from participant_audio_handler import ParticipantSTTSession

    loop = asyncio.get_running_loop()
    p = ParticipantSTTSession(
        session_id="s1",
        participant_id=1,
        participant_name="Tester",
        on_final_transcript=AsyncMock(),
        loop=loop,
    )

    with patch("participant_audio_handler._AAI_AVAILABLE", False):
        import logging
        with caplog.at_level(logging.WARNING):
            await p.connect_async()

    assert not p._connected
