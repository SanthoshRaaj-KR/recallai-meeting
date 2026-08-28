"""Listen-only mode: a participant can mute Jarvis mid-meeting from the live
status page. When enabled, Jarvis keeps transcribing for context but never
speaks — not even to acknowledge the wake word.

The flag lives on session_store (written by bot_service) and is polled into
``Assistant._listen_only``. These tests drive the two response gates directly
with the cached flag pre-set, so no network/session_store access is needed.
"""

import asyncio
import collections
import os
import sys
from unittest.mock import AsyncMock, MagicMock, PropertyMock, patch

import pytest
from livekit.agents import StopResponse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from agent import Assistant

# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_assistant(listen_only: bool) -> Assistant:
    """Minimal Assistant with just the attributes the response gates touch."""
    a = Assistant.__new__(Assistant)
    a._session_id = "sess-listen"
    a._confluence_enabled = False
    a._transcript = collections.deque(maxlen=500)
    a._transcript_memory = MagicMock()
    a._confluence_rag = MagicMock()
    a._last_compacted_memory = ""
    a._last_rag_context = ""
    a._partial_wake_fired = False
    a._listen_only = listen_only
    return a


def _msg(text: str):
    m = MagicMock()
    m.text_content = text
    return m


# ── on_user_turn_completed: the wake-word gate ────────────────────────────────


@pytest.mark.asyncio
async def test_listen_only_suppresses_wake_query() -> None:
    """With listen-only ON, a real wake-word query is suppressed (StopResponse)
    and never reaches the RAG/LLM dispatch — even though it would normally answer."""
    assistant = _make_assistant(listen_only=True)

    with (
        patch.object(assistant, "_post_transcript"),
        patch.object(assistant, "_run_in_compactor", new=AsyncMock()),
        pytest.raises(StopResponse),
    ):
        await assistant.on_user_turn_completed(
            MagicMock(), _msg("Hey Jarvis, what did we decide?")
        )

    # Still listens: the utterance was buffered for meeting context.
    assert any("what did we decide" in line for line in assistant._transcript)
    # Never reached the RAG dispatch path.
    assistant._confluence_rag.search.assert_not_called()


@pytest.mark.asyncio
async def test_listen_only_still_buffers_plain_speech() -> None:
    """Non-wake speech is buffered (as always) and suppressed under listen-only."""
    assistant = _make_assistant(listen_only=True)

    with (
        patch.object(assistant, "_post_transcript"),
        patch.object(assistant, "_run_in_compactor", new=AsyncMock()),
        pytest.raises(StopResponse),
    ):
        await assistant.on_user_turn_completed(
            MagicMock(), _msg("the deadline is next friday")
        )

    assert any("deadline is next friday" in line for line in assistant._transcript)


@pytest.mark.asyncio
async def test_wake_query_dispatches_when_listen_only_off() -> None:
    """Control: with listen-only OFF, the same wake query is NOT suppressed by the
    listen-only gate — it proceeds past it to the normal dispatch path."""
    assistant = _make_assistant(listen_only=False)
    assistant._confluence_rag.enabled = (
        False  # skip the RAG network call, still dispatches
    )

    with (
        patch.object(assistant, "_post_transcript"),
        patch.object(assistant, "_run_in_compactor", new=AsyncMock(return_value="")),
        patch.object(assistant, "update_chat_ctx", new=AsyncMock()),
        patch.object(assistant, "_refresh_transcript_in_ctx"),
    ):
        # Should NOT raise StopResponse — the turn is allowed to generate a reply.
        await assistant.on_user_turn_completed(
            MagicMock(), _msg("Hey Jarvis, what did we decide?")
        )


# ── stt_node: the early "Yes?" acknowledgement ────────────────────────────────


@pytest.mark.asyncio
async def test_listen_only_suppresses_partial_ack() -> None:
    """The early "Yes?" ack fired from interim transcripts is suppressed under
    listen-only, so the wake word draws no audible response at all."""
    from livekit.agents import stt as lk_stt

    assistant = _make_assistant(listen_only=True)
    mock_session = MagicMock()
    mock_session.say = AsyncMock()

    # A fake interim event containing the wake word.
    alt = MagicMock()
    alt.text = "hey jarvis"
    event = MagicMock(spec=lk_stt.SpeechEvent)
    event.type = lk_stt.SpeechEventType.INTERIM_TRANSCRIPT
    event.alternatives = [alt]

    async def _fake_default_stt_node(self, audio, model_settings):
        yield event

    async def _empty_audio():
        return
        yield  # pragma: no cover

    with (
        patch("agent.Agent.default.stt_node", _fake_default_stt_node),
        patch.object(
            Assistant, "session", new_callable=PropertyMock, return_value=mock_session
        ),
    ):
        async for _ in assistant.stt_node(_empty_audio(), MagicMock()):
            pass

    await asyncio.sleep(0)  # let any scheduled ack task run
    mock_session.say.assert_not_called()
    assert assistant._partial_wake_fired is False


# ── no-interrupt (hold-the-floor) toggle ──────────────────────────────────────


def test_no_interrupt_toggles_agent_allow_interruptions() -> None:
    """Enabling no-interrupt makes replies uninterruptible (allow_interruptions
    False); disabling restores normal barge-in (allow_interruptions True). The
    runtime reads allow_interruptions per reply, so this applies live."""
    a = Assistant.__new__(Assistant)
    a._no_interrupt = False
    a._allow_interruptions = None  # stand-in for the SDK's NOT_GIVEN default

    a._apply_no_interrupt(True)
    assert a._no_interrupt is True
    assert a._allow_interruptions is False  # replies now hold the floor

    a._apply_no_interrupt(False)
    assert a._no_interrupt is False
    assert a._allow_interruptions is True  # barge-in allowed again
