"""Phase 4 tests for review/api.py changes (D-01 relay room creation, D-05/D-07 cleanup)."""
import asyncio
from unittest.mock import AsyncMock, Mock, patch

import pytest


@pytest.mark.asyncio
async def test_start_bot_for_session_creates_relay_room():
    """D-01: _start_bot_for_session must call _create_recall_relay_room after the publisher room."""
    from confluence_logic.review import api as review_api
    from confluence_logic.review.api import StartBotRequest

    order = []

    async def _fake_create_livekit_room(session_id, bot_id):
        order.append(("publisher", session_id, bot_id))

    async def _fake_create_relay_room(session_id):
        order.append(("relay", session_id))

    async def _noop_inprocess(session_id):
        order.append(("inprocess", session_id))
        return None

    async def _noop_teardown(session_id):
        order.append(("teardown", session_id))

    # Patch each function at its origin in the jarvis_agentic module so the
    # deferred imports inside _start_bot_for_session resolve to the fakes.
    with patch("confluence_logic.jarvis_agentic.create_bot", return_value="bot-abc"), \
         patch("confluence_logic.jarvis_agentic._create_livekit_room", side_effect=_fake_create_livekit_room), \
         patch("confluence_logic.jarvis_agentic._create_recall_relay_room", side_effect=_fake_create_relay_room), \
         patch("confluence_logic.jarvis_agentic._teardown_livekit_room", side_effect=_noop_teardown), \
         patch("confluence_logic.jarvis_agentic._start_in_process_agent_session", side_effect=_noop_inprocess, create=True), \
         patch.object(review_api, "_persist_history_snapshot", lambda *a, **k: None):
        body = StartBotRequest(meeting_url="https://meet.google.com/abc-defg-hij", session_id="sess-r")
        result = await review_api._start_bot_for_session(body, session_id="sess-r")

    assert result["status"] == "in_meeting", result
    assert result["session_id"] == "sess-r"
    # Publisher must be created BEFORE relay (relay reuses room name but is a second participant).
    kinds = [step[0] for step in order]
    assert "publisher" in kinds and "relay" in kinds
    assert kinds.index("publisher") < kinds.index("relay"), (
        f"Publisher room must be created before relay room; order={order}"
    )
    # Relay must be invoked with the same session_id.
    relay_step = next(step for step in order if step[0] == "relay")
    assert relay_step[1] == "sess-r"


@pytest.mark.asyncio
async def test_start_bot_for_session_fails_clean_when_relay_room_errors():
    """D-01: if relay room fails, the publisher room is torn down and status=error returned."""
    from confluence_logic.review import api as review_api
    from confluence_logic.review.api import StartBotRequest

    teardown_called = []

    async def _fake_relay_fail(session_id):
        raise RuntimeError("LIVEKIT_API_KEY missing")

    async def _fake_teardown(session_id):
        teardown_called.append(session_id)

    with patch("confluence_logic.jarvis_agentic.create_bot", return_value="bot-zzz"), \
         patch("confluence_logic.jarvis_agentic._create_livekit_room", AsyncMock()), \
         patch("confluence_logic.jarvis_agentic._create_recall_relay_room", side_effect=_fake_relay_fail), \
         patch("confluence_logic.jarvis_agentic._teardown_livekit_room", side_effect=_fake_teardown), \
         patch("confluence_logic.jarvis_agentic._start_in_process_agent_session", AsyncMock(), create=True), \
         patch.object(review_api, "_persist_history_snapshot", lambda *a, **k: None):
        body = StartBotRequest(meeting_url="https://meet.google.com/abc-defg-hij", session_id="sess-fail")
        result = await review_api._start_bot_for_session(body, session_id="sess-fail")

    assert result["status"] == "error"
    assert "relay" in result["error"].lower()
    assert teardown_called == ["sess-fail"], teardown_called
