"""Phase 4 — Wave 0 RED stubs for agent_worker.py changes.

These tests use source-text introspection rather than runtime construction because
AgentSession requires live LiveKit credentials. Source-grep is sufficient to verify
the structural changes mandated by D-02, D-06, D-08, D-09, and Pitfall 1.
"""
from pathlib import Path

import pytest


_AGENT_WORKER_PATH = (
    Path(__file__).resolve().parent.parent / "agent_worker.py"
)


@pytest.fixture(scope="module")
def agent_worker_source() -> str:
    assert _AGENT_WORKER_PATH.exists(), f"agent_worker.py not found at {_AGENT_WORKER_PATH}"
    return _AGENT_WORKER_PATH.read_text(encoding="utf-8")


def test_stt_enabled_with_deepgram_nova3(agent_worker_source: str) -> None:
    """D-02/D-09: stt=None must be replaced with inference.STT(model='deepgram/nova-3'...)."""
    assert "stt=None" not in agent_worker_source, (
        "stt=None must be removed — replace with inference.STT(model='deepgram/nova-3', language='multi') per D-09"
    )
    assert 'inference.STT(model="deepgram/nova-3"' in agent_worker_source, (
        "agent_worker.py must instantiate inference.STT(model=\"deepgram/nova-3\", language=\"multi\") per D-02/D-09"
    )
    assert 'language="multi"' in agent_worker_source, (
        "language='multi' is required per D-09 (multilingual detection)"
    )


# Alias matching the shorter selector in VALIDATION.md.
test_stt_enabled = test_stt_enabled_with_deepgram_nova3


def test_no_data_received_handler(agent_worker_source: str) -> None:
    """D-06: The data_received IPC handler must be removed from agent_worker.py."""
    assert "data_received" not in agent_worker_source, (
        "data_received handler must be removed per D-06 — STT now drives generate_reply"
    )
    assert "def _on_data(" not in agent_worker_source, (
        "_on_data callback must be removed per D-06"
    )
    assert "publish_data(" not in agent_worker_source, (
        "publish_data must not appear in agent_worker.py per D-06"
    )


def test_session_start_uses_recall_browser_participant_identity(agent_worker_source: str) -> None:
    """Phase 6: identity must be recall-browser-{session_id} (was recall-relay-)."""
    assert "room_io.RoomOptions(" in agent_worker_source, (
        "session.start() must pass room_io.RoomOptions(...) per Pitfall 1 (avoid feedback loop)"
    )
    assert 'participant_identity=f"recall-relay-' not in agent_worker_source, \
        "recall-relay- identity must be removed (Phase 6)"
    assert 'participant_identity=f"recall-browser-' in agent_worker_source, \
        "identity must be recall-browser-{session_id} per Phase 6 / D-05"
    # Sanity: session_id should come from ctx.job.metadata (verified at runtime; source must reference it).
    assert "ctx.job.metadata" in agent_worker_source or "job.metadata" in agent_worker_source, (
        "session_id must be resolved from ctx.job.metadata per RESEARCH §participant_identity coordination"
    )


def test_agent_worker_subscribes_to_recall_browser(agent_worker_source: str) -> None:
    """R6-05: agent_worker.py must use recall-browser- prefix, not recall-relay-.

    RED until Wave 3 updates agent_worker.py.
    """
    assert "recall-browser-" in agent_worker_source, \
        "agent_worker.py must reference recall-browser- identity (Phase 6 / R6-05)"
    assert "recall-relay-" not in agent_worker_source, \
        "recall-relay- prefix must be fully removed from agent_worker.py"


def test_wake_word_gate_present(agent_worker_source: str) -> None:
    """Wake-word gate: llm_node must check _extract_query before dispatching to LLM."""
    assert "def llm_node(" in agent_worker_source, \
        "JarvisAgent must override llm_node() to gate on wake word"
    assert "_extract_query(" in agent_worker_source, \
        "llm_node must call _extract_query() to detect wake word"
    assert "_WAKE_PATTERN" in agent_worker_source, \
        "_WAKE_PATTERN regex must be defined locally in agent_worker.py"


def test_ack_audio_playback_present(agent_worker_source: str) -> None:
    """Ack audio: tts_node prepends random ack before LLM TTS; bare wake uses say().

    Design: _ack_q (asyncio.Queue) carries a single True from on_user_turn_completed
    Case 3 (wake word confirmed) to tts_node. This avoids the preemptive_generation
    race where llm_node fires speculatively before the wake gate runs.
    """
    assert "on_user_turn_completed" in agent_worker_source, \
        "JarvisAgent must override on_user_turn_completed (bare wake handler)"
    assert "get_random_query_ack_audio" in agent_worker_source, \
        "agent_worker must use get_random_query_ack_audio() for ack selection in tts_node"
    assert "tts_node" in agent_worker_source, \
        "JarvisAgent must override tts_node to prepend ack before LLM TTS"
    assert "_ack_q" in agent_worker_source, \
        "JarvisAgent must use asyncio.Queue _ack_q to signal ack between on_user_turn_completed and tts_node"
    assert "put_nowait(True)" in agent_worker_source, \
        "on_user_turn_completed Case 3 must put True into _ack_q (after wake word confirmed)"
    assert "get_nowait()" in agent_worker_source, \
        "tts_node must call _ack_q.get_nowait() to decide whether to play ack"


def test_extract_query_logic() -> None:
    """Unit test for _extract_query wake-word detection logic."""
    from confluence_logic.agent_worker import _extract_query

    assert _extract_query("hey jarvis what time is it") == "what time is it"
    assert _extract_query("Hey Jarvis, summarize the meeting") == "summarize the meeting"
    assert _extract_query("ok jarvis") == ""          # bare wake, empty query
    assert _extract_query("hi jarv, stop") == "stop"
    assert _extract_query("the meeting is running long") is None   # no wake word
    assert _extract_query("") is None
