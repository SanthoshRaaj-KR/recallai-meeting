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


def test_session_start_uses_recall_relay_participant_identity(agent_worker_source: str) -> None:
    """Pitfall 1: AgentSession must link to recall-relay-{session_id}, not the first participant
    (which would be jarvis-publisher and cause a feedback loop)."""
    assert "room_io.RoomOptions(" in agent_worker_source, (
        "session.start() must pass room_io.RoomOptions(...) per Pitfall 1 (avoid feedback loop)"
    )
    assert 'participant_identity=f"recall-relay-' in agent_worker_source, (
        "RoomOptions must set participant_identity=f\"recall-relay-{session_id}\" per Pitfall 1"
    )
    # Sanity: session_id should come from ctx.job.metadata (verified at runtime; source must reference it).
    assert "ctx.job.metadata" in agent_worker_source or "job.metadata" in agent_worker_source, (
        "session_id must be resolved from ctx.job.metadata per RESEARCH §participant_identity coordination"
    )
