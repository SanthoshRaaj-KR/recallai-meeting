"""Tests for IPC data channel + push_audio_to_livekit removal — REQ-17, REQ-18.

Phase 03 — populated by Plan 04 (worker-side data_received handler),
Plan 05 (FastAPI-side _send_query_to_agent), Plan 06 (push_audio_to_livekit caller removal).
"""
import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from confluence_logic import agent_worker as w


def _make_entrypoint_ctx():
    ctx = MagicMock()
    ctx.proc.userdata = {"vad": MagicMock()}
    ctx.room = MagicMock()
    return ctx


def _capture_room_handler(ctx):
    """Replace ctx.room.on with a recorder that captures the registered function."""
    captured = {}

    def fake_on(event_name):
        def decorator(fn):
            captured[event_name] = fn
            return fn
        return decorator

    ctx.room.on = fake_on
    return captured


@pytest.mark.asyncio
async def test_data_received_handler_registered():
    ctx = _make_entrypoint_ctx()
    captured = _capture_room_handler(ctx)

    with patch.object(w, "AgentSession") as MockSession, \
         patch.object(w, "inference"), \
         patch.object(w, "MultilingualModel"), \
         patch.object(w, "JarvisAgent"):
        MockSession.return_value.start = AsyncMock()
        await w.entrypoint(ctx)

    assert "data_received" in captured, "data_received handler was not registered"


@pytest.mark.asyncio
async def test_data_received_dispatches_user_query():
    ctx = _make_entrypoint_ctx()
    captured = _capture_room_handler(ctx)

    with patch.object(w, "AgentSession") as MockSession, \
         patch.object(w, "inference"), \
         patch.object(w, "MultilingualModel"), \
         patch.object(w, "JarvisAgent"):
        session_instance = MockSession.return_value
        session_instance.start = AsyncMock()
        session_instance.generate_reply = MagicMock(return_value=MagicMock())
        await w.entrypoint(ctx)

    handler = captured["data_received"]
    packet = MagicMock()
    packet.data = b'{"type":"user_query","query":"hello"}'

    with patch.object(w.asyncio, "create_task") as mock_task:
        handler(packet)
        assert mock_task.call_count == 1
        session_instance.generate_reply.assert_called_once_with(user_input="hello")


@pytest.mark.asyncio
async def test_data_received_drops_bad_json():
    ctx = _make_entrypoint_ctx()
    captured = _capture_room_handler(ctx)

    with patch.object(w, "AgentSession") as MockSession, \
         patch.object(w, "inference"), \
         patch.object(w, "MultilingualModel"), \
         patch.object(w, "JarvisAgent"):
        MockSession.return_value.start = AsyncMock()
        await w.entrypoint(ctx)

    handler = captured["data_received"]
    packet = MagicMock()
    packet.data = b"\xff\xfe not json"

    with patch.object(w.asyncio, "create_task") as mock_task:
        handler(packet)
        assert mock_task.call_count == 0


@pytest.mark.asyncio
async def test_data_received_drops_wrong_type():
    ctx = _make_entrypoint_ctx()
    captured = _capture_room_handler(ctx)

    with patch.object(w, "AgentSession") as MockSession, \
         patch.object(w, "inference"), \
         patch.object(w, "MultilingualModel"), \
         patch.object(w, "JarvisAgent"):
        MockSession.return_value.start = AsyncMock()
        await w.entrypoint(ctx)

    handler = captured["data_received"]
    packet = MagicMock()
    packet.data = b'{"type":"some_other","query":"x"}'

    with patch.object(w.asyncio, "create_task") as mock_task:
        handler(packet)
        assert mock_task.call_count == 0


# ---------------------------------------------------------------------------
# Plan 05: in-process AgentSession + queue dispatch tests (REQ-13/14/18)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_in_process_session_stored_in_meeting_sessions(monkeypatch):
    from unittest.mock import AsyncMock, MagicMock, patch
    from confluence_logic import jarvis_agentic as ja

    mock_room = MagicMock()
    monkeypatch.setattr(ja, "_meeting_sessions", {"sid-1": {"livekit_room": mock_room}})

    mock_session = MagicMock()
    mock_session.start = AsyncMock()
    mock_session.agent_state = "listening"

    mock_http_ctx = AsyncMock()
    mock_http_ctx.__aenter__ = AsyncMock(return_value=MagicMock())
    mock_http_ctx.__aexit__ = AsyncMock(return_value=False)

    with patch("confluence_logic.jarvis_agentic.AgentSession", return_value=mock_session), \
         patch("confluence_logic.jarvis_agentic._OpenAITTS", return_value=MagicMock()), \
         patch("confluence_logic.jarvis_agentic._OpenAILLM", return_value=MagicMock()), \
         patch("confluence_logic.jarvis_agentic._lk_http_context") as mock_lk_http, \
         patch("confluence_logic.jarvis_agentic.asyncio.create_task", return_value=MagicMock()):
        mock_lk_http.open.return_value = mock_http_ctx
        await ja._start_in_process_agent_session("sid-1")

    state = ja._meeting_sessions["sid-1"]
    assert state.get("agent_session") is mock_session
    assert state.get("_agent_query_queue") is not None
    assert state.get("_agent_consumer_task") is not None


@pytest.mark.asyncio
async def test_in_process_session_returns_none_without_room(monkeypatch):
    from confluence_logic import jarvis_agentic as ja
    monkeypatch.setattr(ja, "_meeting_sessions", {"sid-1": {}})
    result = await ja._start_in_process_agent_session("sid-1")
    assert result is None
    assert "agent_session" not in ja._meeting_sessions["sid-1"]


@pytest.mark.asyncio
async def test_debounced_dispatch_puts_query_in_queue(monkeypatch):
    import asyncio
    from unittest.mock import MagicMock
    from confluence_logic import jarvis_agentic as ja

    queue = asyncio.Queue(maxsize=8)
    mock_session = MagicMock()
    mock_session.agent_state = "listening"

    monkeypatch.setattr(ja, "_meeting_sessions", {
        "sid-q": {"agent_session": mock_session, "_agent_query_queue": queue}
    })
    monkeypatch.setattr(ja, "JARVIS_DEBOUNCE_SECONDS", 0)
    monkeypatch.setattr(ja, "_resolve_session_id", lambda bot_id: "sid-q")
    monkeypatch.setattr(ja, "meeting_state", ja.MeetingStateProxy())

    await ja._debounced_dispatch("what did we discuss", "bot-1")

    assert not queue.empty()
    queued = queue.get_nowait()
    assert queued == "what did we discuss"


@pytest.mark.asyncio
async def test_debounced_dispatch_falls_back_when_initializing(monkeypatch):
    import asyncio
    from unittest.mock import MagicMock, AsyncMock, patch
    from confluence_logic import jarvis_agentic as ja

    queue = asyncio.Queue(maxsize=8)
    mock_session = MagicMock()
    mock_session.agent_state = "initializing"  # not ready yet

    monkeypatch.setattr(ja, "_meeting_sessions", {
        "sid-i": {"agent_session": mock_session, "_agent_query_queue": queue}
    })
    monkeypatch.setattr(ja, "JARVIS_DEBOUNCE_SECONDS", 0)
    monkeypatch.setattr(ja, "_resolve_session_id", lambda bot_id: "sid-i")
    monkeypatch.setattr(ja, "meeting_state", ja.MeetingStateProxy())

    # Plan 06: handle_spoken_request removed from _debounced_dispatch; initializing → drop query
    await ja._debounced_dispatch("help", "bot-1")

    # Queue should be empty — session was initializing so query was dropped (not queued)
    assert queue.empty(), "Initializing session should not receive queries"


@pytest.mark.asyncio
async def test_query_consumer_serializes_via_await_speech_handle():
    import asyncio
    from unittest.mock import MagicMock
    from confluence_logic import jarvis_agentic as ja

    call_order = []

    class FakeSpeechHandle:
        def __init__(self, label):
            self.label = label

        def __await__(self):
            async def _inner():
                call_order.append(f"await_{self.label}")
            return _inner().__await__()

    mock_session = MagicMock()
    mock_session.generate_reply = lambda user_input: (
        call_order.append(f"gen_{user_input[:3]}") or FakeSpeechHandle(user_input[:3])
    )

    queue: asyncio.Queue = asyncio.Queue()
    await queue.put("query_A")
    await queue.put("query_B")
    await queue.put(None)  # shutdown sentinel

    await ja._query_consumer(mock_session, queue)

    gen_positions = [i for i, v in enumerate(call_order) if v.startswith("gen_")]
    await_positions = [i for i, v in enumerate(call_order) if v.startswith("await_")]
    # generate_reply for first query must come before its await, which comes before second generate_reply
    assert gen_positions[0] < await_positions[0] < gen_positions[1], \
        "Consumer must await first speech before generating second"


# ---------------------------------------------------------------------------
# Plan 06: push_audio caller removal verification (REQ-17)
# ---------------------------------------------------------------------------

def test_no_push_audio_callers_in_voice_path():
    from confluence_logic import jarvis_agentic as ja
    import inspect
    for fn_name in ("_emit_micro_ack", "_handle_interruption", "_speak_gap_filler"):
        src = inspect.getsource(getattr(ja, fn_name))
        assert "push_audio_to_livekit" not in src, f"{fn_name} still references push_audio_to_livekit"
    dd_src = inspect.getsource(ja._debounced_dispatch)
    assert "handle_spoken_request" not in dd_src, "legacy fallback not removed from _debounced_dispatch"
    assert "_speak_gap_filler" not in dd_src, "gap_filler still scheduled in _debounced_dispatch"
