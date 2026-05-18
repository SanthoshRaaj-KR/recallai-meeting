"""Phase 4/6 tests for review/api.py changes.

Phase 4 relay room creation tests (D-01) are removed in Wave 3 (Plan 006-004) because
_create_recall_relay_room was deleted from jarvis_agentic.py and review/api.py.
The tests below verify the Wave 3 state: relay room is NOT called and the bot start
path connects directly without the relay creation step.
"""
import inspect
from unittest.mock import AsyncMock, patch

import pytest


@pytest.mark.asyncio
async def test_start_bot_for_session_does_not_call_relay_room():
    """Wave 3 (Plan 006-004): _start_bot_for_session must NOT call _create_recall_relay_room.

    The relay participant has been removed. Bot start goes directly from publisher room
    creation to bot creation without a relay hop.
    """
    from confluence_logic.review import api as review_api
    from confluence_logic.review.api import StartBotRequest

    order = []

    async def _fake_create_livekit_room(session_id, bot_id):
        order.append(("publisher", session_id, bot_id))

    with patch("confluence_logic.jarvis_agentic.create_bot", return_value="bot-abc"), \
         patch("confluence_logic.jarvis_agentic._create_livekit_room", side_effect=_fake_create_livekit_room), \
         patch.object(review_api, "_persist_history_snapshot", lambda *a, **k: None):
        body = StartBotRequest(meeting_url="https://meet.google.com/abc-defg-hij", session_id="sess-r")
        result = await review_api._start_bot_for_session(body, session_id="sess-r")

    assert result["status"] == "in_meeting", result
    assert result["session_id"] == "sess-r"
    # Publisher room must be created; relay must NOT appear in call order.
    kinds = [step[0] for step in order]
    assert "publisher" in kinds
    assert "relay" not in kinds, (
        "_start_bot_for_session must not create a relay room in Wave 3 (Plan 006-004)"
    )


def test_start_bot_for_session_does_not_reference_in_process_agent():
    """D-07 / Pitfall 6: _start_in_process_agent_session must be fully gone from review/api.py."""
    from confluence_logic.review import api as review_api
    src = inspect.getsource(review_api._start_bot_for_session)
    assert "_start_in_process_agent_session" not in src, (
        "review/api.py must not reference _start_in_process_agent_session (D-07 / Pitfall 6)"
    )
    # Wave 3 (Plan 006-004): relay room call is also fully removed.
    assert "_create_recall_relay_room" not in src, (
        "review/api.py must not call _create_recall_relay_room after Wave 3 (Plan 006-004)"
    )


def test_review_api_module_imports_clean():
    """Pitfall 6: review/api.py must import without ImportError after D-07 deletions."""
    import importlib
    # Force re-import to catch any stale cached state
    from confluence_logic.review import api as review_api
    importlib.reload(review_api)
    assert hasattr(review_api, "_start_bot_for_session")
    # Confirm the function exists and is callable (signature unchanged)
    sig = inspect.signature(review_api._start_bot_for_session)
    assert "body" in sig.parameters
    assert "session_id" in sig.parameters
