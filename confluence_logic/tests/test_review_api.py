from unittest.mock import Mock, patch

from confluence_logic.review import api


def test_latest_recall_status_code_reads_status_changes():
    payload = {
        "status": "in_call",
        "status_changes": [
            {"code": "joining_call"},
            {"code": "in_call"},
            {"code": "call_ended"},
        ],
    }

    assert api._latest_recall_status_code(payload) == "call_ended"


def test_refresh_session_status_marks_call_ended():
    state = {
        "bot_id": "bot-123",
        "is_active": True,
        "session_status": "in_meeting",
        "last_recall_status_checked_at": 0.0,
    }

    with patch.object(api, "_fetch_recall_bot_payload", return_value={"status_changes": [{"code": "call_ended"}]}), \
         patch.object(api, "_RECALL_STATUS_CACHE_SECONDS", 0.0):
        api._refresh_session_status_from_recall(state)

    assert state["is_active"] is False
    assert state["session_status"] == "ended"
    assert state["recall_status_code"] == "call_ended"
    assert state["ended_at"]


def test_refresh_session_status_marks_missing_bot_as_ended():
    state = {
        "bot_id": "bot-123",
        "is_active": True,
        "session_status": "in_meeting",
        "last_recall_status_checked_at": 0.0,
    }
    response = Mock(status_code=404)
    error = api.requests.HTTPError(response=response)

    with patch.object(api, "_fetch_recall_bot_payload", side_effect=error), \
         patch.object(api, "_RECALL_STATUS_CACHE_SECONDS", 0.0):
        api._refresh_session_status_from_recall(state)

    assert state["is_active"] is False
    assert state["session_status"] == "ended"
    assert state["end_reason"] == "Recall no longer returns this bot session."
