"""Tests for AgentSession composition — REQ-13 (STT=None pattern), REQ-14 (Cartesia Sonic-3 wiring).

Phase 03 — populated by Plan 02. Distinct from test_livekit_agent_worker.py: this file
isolates AgentSession construction so tests can mock livekit.agents.AgentSession.
"""
import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import confluence_logic.agent_worker as w


def _make_mock_ctx():
    ctx = MagicMock()
    ctx.proc.userdata = {"vad": MagicMock(name="silero_vad_instance")}
    ctx.room = MagicMock()
    return ctx


@pytest.mark.asyncio
async def test_agent_session_no_stt():
    with patch("confluence_logic.agent_worker.AgentSession") as MockSession, \
         patch("confluence_logic.agent_worker.inference") as mock_inf, \
         patch("confluence_logic.agent_worker.MultilingualModel"), \
         patch("confluence_logic.agent_worker.JarvisAgent"):
        MockSession.return_value.start = AsyncMock()
        mock_inf.TTS.return_value = MagicMock()
        mock_inf.LLM.return_value = MagicMock()

        ctx = _make_mock_ctx()
        await w.entrypoint(ctx)

        assert MockSession.call_args.kwargs["stt"] is None


@pytest.mark.asyncio
async def test_cartesia_tts_config():
    with patch("confluence_logic.agent_worker.AgentSession") as MockSession, \
         patch("confluence_logic.agent_worker.inference") as mock_inf, \
         patch("confluence_logic.agent_worker.MultilingualModel"), \
         patch("confluence_logic.agent_worker.JarvisAgent"):
        MockSession.return_value.start = AsyncMock()
        mock_inf.TTS.return_value = MagicMock()
        mock_inf.LLM.return_value = MagicMock()

        ctx = _make_mock_ctx()
        await w.entrypoint(ctx)

        assert mock_inf.TTS.call_args.args[0] == "cartesia/sonic-3"
        assert mock_inf.TTS.call_args.kwargs["voice"] == "9626c31c-bec5-4cca-baa8-f8ba9e84c8bc"


@pytest.mark.asyncio
async def test_vad_passed_from_userdata():
    with patch("confluence_logic.agent_worker.AgentSession") as MockSession, \
         patch("confluence_logic.agent_worker.inference") as mock_inf, \
         patch("confluence_logic.agent_worker.MultilingualModel"), \
         patch("confluence_logic.agent_worker.JarvisAgent"):
        MockSession.return_value.start = AsyncMock()
        mock_inf.TTS.return_value = MagicMock()
        mock_inf.LLM.return_value = MagicMock()

        ctx = _make_mock_ctx()
        await w.entrypoint(ctx)

        assert MockSession.call_args.kwargs["vad"] is ctx.proc.userdata["vad"]


@pytest.mark.asyncio
async def test_preemptive_generation_disabled():
    with patch("confluence_logic.agent_worker.AgentSession") as MockSession, \
         patch("confluence_logic.agent_worker.inference") as mock_inf, \
         patch("confluence_logic.agent_worker.MultilingualModel"), \
         patch("confluence_logic.agent_worker.JarvisAgent"):
        MockSession.return_value.start = AsyncMock()
        mock_inf.TTS.return_value = MagicMock()
        mock_inf.LLM.return_value = MagicMock()

        ctx = _make_mock_ctx()
        await w.entrypoint(ctx)

        assert MockSession.call_args.kwargs["preemptive_generation"] is False
