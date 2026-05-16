"""Tests for agent_bridge.py @function_tool wrappers — REQ-15 (tool bridge), REQ-16 (interrupt config).

Phase 03 — populated by Plan 03 (bridge tests) and Plan 04 (interrupt config tests).
"""
import os
import sys
from unittest.mock import AsyncMock, MagicMock

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from confluence_logic import agent_bridge


def test_jarvis_tools_list_nonempty():
    assert len(agent_bridge.JARVIS_TOOLS) == 5


def test_tool_names():
    names = {t.info.name for t in agent_bridge.JARVIS_TOOLS}
    assert names == {
        "summarize_meeting_tool",
        "generate_opinion_tool",
        "extract_action_items_tool",
        "summarize_speaker_tool",
        "answer_general_question_tool",
    }


@pytest.mark.asyncio
async def test_summarize_meeting_tool_delegates(monkeypatch):
    fake = AsyncMock(return_value="FAKE_SUMMARY")
    monkeypatch.setattr(agent_bridge, "summarize_meeting", fake)
    monkeypatch.setattr(
        agent_bridge, "get_transcript_log_for_session",
        lambda sid: [{"participant": "A", "text": "hi", "timestamp": 0}],
    )
    ctx = MagicMock()
    ctx.job.metadata = '{"session_id": "abc"}'
    result = await agent_bridge.summarize_meeting_tool(ctx, detail_level="brief")
    assert result == "FAKE_SUMMARY"
    fake.assert_called_once()
    assert fake.call_args.kwargs.get("detail_level") == "brief"


@pytest.mark.asyncio
async def test_generate_opinion_tool_delegates(monkeypatch):
    fake = AsyncMock(return_value="FAKE_OPINION")
    monkeypatch.setattr(agent_bridge, "generate_opinion", fake)
    monkeypatch.setattr(agent_bridge, "get_transcript_log_for_session", lambda sid: [])
    ctx = MagicMock()
    ctx.job.metadata = '{"session_id": "abc"}'
    result = await agent_bridge.generate_opinion_tool(ctx, query="x")
    assert result == "FAKE_OPINION"
    fake.assert_called_once()
    assert fake.call_args.kwargs.get("query") == "x"


@pytest.mark.asyncio
async def test_extract_action_items_tool_delegates(monkeypatch):
    fake = AsyncMock(return_value="FAKE_ITEMS")
    monkeypatch.setattr(agent_bridge, "extract_action_items", fake)
    monkeypatch.setattr(agent_bridge, "get_transcript_log_for_session", lambda sid: [])
    ctx = MagicMock()
    ctx.job.metadata = '{"session_id": "abc"}'
    result = await agent_bridge.extract_action_items_tool(ctx)
    assert result == "FAKE_ITEMS"
    fake.assert_called_once()


@pytest.mark.asyncio
async def test_summarize_speaker_tool_delegates(monkeypatch):
    fake = AsyncMock(return_value="FAKE_SPEAKER_SUMMARY")
    monkeypatch.setattr(agent_bridge, "summarize_speaker", fake)
    monkeypatch.setattr(agent_bridge, "get_transcript_log_for_session", lambda sid: [])
    ctx = MagicMock()
    ctx.job.metadata = '{"session_id": "abc"}'
    result = await agent_bridge.summarize_speaker_tool(ctx, speaker_name="alice")
    assert result == "FAKE_SPEAKER_SUMMARY"
    fake.assert_called_once()
    assert fake.call_args.kwargs.get("speaker_name") == "alice"


@pytest.mark.asyncio
async def test_answer_general_question_tool_delegates(monkeypatch):
    fake = AsyncMock(return_value="FAKE_ANSWER")
    monkeypatch.setattr(agent_bridge, "answer_general_question", fake)
    ctx = MagicMock()
    ctx.job.metadata = "{}"
    result = await agent_bridge.answer_general_question_tool(ctx, question="hello", force_web_search=False)
    assert result == "FAKE_ANSWER"
    fake.assert_called_once()
    assert fake.call_args.args[0] == "hello"
    assert fake.call_args.kwargs.get("force_web_search") is False


def test_transcript_log_missing_session_returns_empty():
    assert agent_bridge.get_transcript_log_for_session("not-a-real-id") == []


def test_transcript_log_existing_session(monkeypatch):
    from confluence_logic import jarvis_agentic
    fake_state = {"sid-1": {"transcript_log": [{"participant": "A", "text": "hi", "timestamp": 0}]}}
    monkeypatch.setattr(jarvis_agentic, "_meeting_sessions", fake_state)
    result = agent_bridge.get_transcript_log_for_session("sid-1")
    assert len(result) == 1
    assert result[0]["participant"] == "A"


def test_session_id_extraction_from_context_metadata():
    ctx_good = MagicMock()
    ctx_good.job.metadata = '{"session_id": "sid-42"}'
    assert agent_bridge._session_id_from_context(ctx_good) == "sid-42"

    ctx_bad = MagicMock()
    ctx_bad.job.metadata = "not-valid-json{{{"
    assert agent_bridge._session_id_from_context(ctx_bad) == ""


# --- Plan 04: tool wiring + interrupt config tests ---

def test_jarvis_agent_receives_tools(monkeypatch):
    from confluence_logic import agent_worker as w
    from confluence_logic.agent_bridge import JARVIS_TOOLS

    captured = {}
    original_init = w.Agent.__init__

    def spy_init(self, *args, **kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(w.Agent, "__init__", spy_init)
    w.JarvisAgent()
    assert captured.get("tools") is JARVIS_TOOLS
    assert "instructions" in captured


@pytest.mark.asyncio
async def test_interrupt_handling_config():
    from confluence_logic import agent_worker as w
    from unittest.mock import patch, AsyncMock

    with patch.object(w, "AgentSession") as MockSession, \
         patch.object(w, "inference") as mock_inf, \
         patch.object(w, "MultilingualModel"), \
         patch.object(w, "JarvisAgent"):
        MockSession.return_value.start = AsyncMock()
        mock_inf.TTS.return_value = MagicMock()
        mock_inf.LLM.return_value = MagicMock()

        ctx = MagicMock()
        ctx.proc.userdata = {"vad": MagicMock()}
        ctx.room = MagicMock()
        ctx.room.on = lambda event: (lambda fn: fn)
        await w.entrypoint(ctx)

    turn_handling = MockSession.call_args.kwargs["turn_handling"]
    interruption = turn_handling.get("interruption") or turn_handling["interruption"]
    assert interruption["resume_false_interruption"] is True
    assert interruption["false_interruption_timeout"] == 1.0
