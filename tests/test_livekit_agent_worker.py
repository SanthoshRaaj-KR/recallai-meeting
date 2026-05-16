"""Tests for confluence_logic/agent_worker.py — REQ-11 (entrypoint), REQ-13 (no STT), REQ-14 (Cartesia TTS), REQ-20 (env vars).

Phase 03 — LiveKit Native Agent Framework. Populated incrementally:
  - Plan 02 adds test_worker_entrypoint_imports, test_agent_session_no_stt, test_cartesia_tts_config, test_env_var_tts_config
  - Plan 04 adds test_interrupt_handling_config
"""
import asyncio
import importlib
import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch, sentinel

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import confluence_logic.agent_worker as w


def test_worker_entrypoint_imports():
    assert callable(w.prewarm)
    assert asyncio.iscoroutinefunction(w.entrypoint)


def test_agent_name_constant():
    assert w.JARVIS_AGENT_WORKER_NAME == "jarvis-agent"
    assert w.server._agent_name == "jarvis-agent"


def test_prewarm_loads_vad():
    with patch("confluence_logic.agent_worker.silero.VAD.load", return_value=sentinel.VAD):
        proc = MagicMock()
        proc.userdata = {}
        w.prewarm(proc)
        assert proc.userdata["vad"] is sentinel.VAD


@pytest.mark.asyncio
async def test_env_var_tts_config(monkeypatch):
    monkeypatch.setenv("JARVIS_LK_TTS_PROVIDER", "elevenlabs")
    import confluence_logic.agent_worker as worker_mod
    importlib.reload(worker_mod)

    captured_args = []

    with patch.object(worker_mod, "AgentSession") as MockSession, \
         patch.object(worker_mod, "inference") as mock_inf, \
         patch.object(worker_mod, "MultilingualModel"), \
         patch.object(worker_mod, "JarvisAgent"):
        MockSession.return_value.start = AsyncMock()
        mock_inf.TTS.side_effect = lambda *a, **kw: captured_args.append(a) or MagicMock()
        mock_inf.LLM.return_value = MagicMock()

        ctx = MagicMock()
        ctx.proc.userdata = {"vad": MagicMock()}
        ctx.room = MagicMock()
        await worker_mod.entrypoint(ctx)

    assert len(captured_args) > 0
    assert captured_args[0][0].startswith("elevenlabs/")

    # Restore: delete env var first so the reload reads the default "cartesia"
    monkeypatch.delenv("JARVIS_LK_TTS_PROVIDER", raising=False)
    importlib.reload(worker_mod)
