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


def test_stt_switched_to_assemblyai(agent_worker_source: str) -> None:
    """R7-01 (D-01): STT must use assemblyai.STT(u3-rt-pro, keyterms_prompt=[...], language_detection=False).

    RED until Wave 1 (plan 002) lands D-01.
    """
    assert "from livekit.plugins import assemblyai" in agent_worker_source, (
        "agent_worker.py must import assemblyai plugin (NOT via inference.STT) per D-01"
    )
    assert "assemblyai.STT(" in agent_worker_source, (
        "agent_worker.py must instantiate assemblyai.STT(...) per D-01"
    )
    assert '"u3-rt-pro"' in agent_worker_source, (
        'model must be "u3-rt-pro" (canonical Literal name per livekit-plugins-assemblyai==1.5.9 — NOT "universal-3-rt-pro")'
    )
    assert "keyterms_prompt=" in agent_worker_source, (
        "keyterms_prompt=[...] required for wake-word reliability (NOT word_boost — that name does not exist in v1.5.9)"
    )
    assert '"Jarvis"' in agent_worker_source, (
        'keyterms_prompt list must contain "Jarvis"'
    )
    assert '"Hey Jarvis"' in agent_worker_source, (
        'keyterms_prompt list must contain "Hey Jarvis"'
    )
    assert "language_detection=False" in agent_worker_source, (
        "language_detection=False required (NOT language_code — that param does not exist in v1.5.9)"
    )
    # Old Deepgram strings must be gone — otherwise the switch is incomplete.
    assert 'inference.STT(model="deepgram/nova-3"' not in agent_worker_source, (
        "Deepgram Nova-3 STT must be removed per D-01"
    )
    assert 'language="multi"' not in agent_worker_source, (
        'language="multi" was Deepgram-only — must be removed per D-01'
    )


# Alias matching the shorter selector in VALIDATION.md.
test_stt_enabled = test_stt_switched_to_assemblyai


def test_llm_upgraded_to_gpt41_mini(agent_worker_source: str) -> None:
    """R7-02 (D-02): JARVIS_LK_LLM default must be openai/gpt-4.1-mini.

    RED until Wave 1 (plan 002) lands D-02.
    """
    assert "openai/gpt-4.1-mini" in agent_worker_source, (
        "JARVIS_LK_LLM default must be 'openai/gpt-4.1-mini' per D-02"
    )
    assert "openai/gpt-4o-mini" not in agent_worker_source, (
        "Old 'openai/gpt-4o-mini' default must be replaced per D-02"
    )


def test_tts_upgraded_to_sonic_turbo(agent_worker_source: str) -> None:
    """R7-03 (D-03): TTS model string must be sonic-turbo, not sonic-3.

    RED until Wave 1 (plan 002) lands D-03.
    """
    assert "sonic-turbo" in agent_worker_source, (
        "TTS model must be 'sonic-turbo' per D-03 (40ms TTFA vs 90ms)"
    )
    assert "sonic-3" not in agent_worker_source, (
        "Old 'sonic-3' model string must be replaced per D-03"
    )


def test_endpointing_tightened(agent_worker_source: str) -> None:
    """R7-04 (D-04): min_delay=0.15 and false_interruption_timeout=0.6.

    RED until Wave 1 (plan 002) lands D-04.
    """
    assert '"min_delay": 0.15' in agent_worker_source, (
        "endpointing.min_delay must be 0.15 per D-04 (was 0.3)"
    )
    assert '"min_delay": 0.3' not in agent_worker_source, (
        "Old min_delay=0.3 must be replaced per D-04"
    )
    assert '"false_interruption_timeout": 0.6' in agent_worker_source, (
        "interruption.false_interruption_timeout must be 0.6 per D-04 (was 1.2)"
    )
    assert '"false_interruption_timeout": 1.2' not in agent_worker_source, (
        "Old false_interruption_timeout=1.2 must be replaced per D-04"
    )


def test_ack_uses_play_ack_frames(agent_worker_source: str) -> None:
    """R7-05 (D-05): bare wake must call _play_ack_frames, not session.say('Yes?').

    RED until Wave 2 (plan 003) lands D-05.
    """
    assert "_play_ack_frames(" in agent_worker_source, (
        "_play_ack_frames(...) must be called from on_user_turn_completed bare-wake case per D-05"
    )
    assert "async def _play_ack_frames" in agent_worker_source, (
        "_play_ack_frames must be defined as an async function in agent_worker.py per D-05"
    )
    assert "AudioStreamDecoder" in agent_worker_source, (
        "MP3→PCM decode via AudioStreamDecoder is required for _play_ack_frames per RESEARCH §3"
    )
    assert 'self.session.say("Yes?"' not in agent_worker_source, (
        "Old self.session.say(\"Yes?\", ...) bare-wake call site must be removed per D-05. "
        "Note: _play_ack_frames fallback uses session.say (no self.) — that is correct and expected."
    )


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
