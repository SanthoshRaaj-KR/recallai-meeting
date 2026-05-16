"""Tests for teardown lifecycle + env var migration — REQ-19, REQ-20 (final).

Phase 03 — populated by Plan 06.
"""
import os
import pathlib
import sys
from unittest.mock import AsyncMock, MagicMock

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

_ENV_EXAMPLE = pathlib.Path(__file__).parent.parent / ".env.example"


def test_env_example_has_no_legacy_tts_vars():
    content = _ENV_EXAMPLE.read_text(encoding="utf-8")
    assert "JARVIS_TTS_PROVIDER=" not in content, "legacy JARVIS_TTS_PROVIDER= still in .env.example"
    assert "JARVIS_TTS_MODEL=" not in content, "legacy JARVIS_TTS_MODEL= still in .env.example"
    assert "JARVIS_TTS_VOICE=" not in content, "legacy JARVIS_TTS_VOICE= still in .env.example"
    assert "JARVIS_TTS_SPEED=" not in content, "legacy JARVIS_TTS_SPEED= still in .env.example"


def test_env_example_has_lk_block():
    content = _ENV_EXAMPLE.read_text(encoding="utf-8")
    assert "JARVIS_LK_TTS_PROVIDER=" in content, "JARVIS_LK_TTS_PROVIDER= missing from .env.example"
    assert "JARVIS_LK_TTS_VOICE=" in content, "JARVIS_LK_TTS_VOICE= missing from .env.example"
    assert "JARVIS_LK_LLM=" in content, "JARVIS_LK_LLM= missing from .env.example"


@pytest.mark.asyncio
async def test_teardown_closes_agent_session(monkeypatch):
    from confluence_logic.jarvis_agentic import _teardown_livekit_room

    mock_session = MagicMock()
    mock_session.aclose = AsyncMock()
    mock_room = MagicMock()
    mock_room.disconnect = AsyncMock()

    from confluence_logic import jarvis_agentic as ja
    monkeypatch.setattr(ja, "_meeting_sessions", {
        "sid-tear": {"agent_session": mock_session, "livekit_room": mock_room}
    })

    await _teardown_livekit_room("sid-tear")

    assert mock_session.aclose.called, "AgentSession.aclose() was not called during teardown"
    assert ja._meeting_sessions.get("sid-tear", {}).get("agent_session") is None, \
        "agent_session was not popped from state during teardown"
