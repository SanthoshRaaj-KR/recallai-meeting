import base64
import gzip
import json
from unittest.mock import Mock, patch
from unittest.mock import AsyncMock

import pytest

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


def test_compress_transcript_round_trips_full_log():
    transcript = [
        {"participant": "Asha", "text": "We should launch next week.", "timestamp": 1.0},
        {"participant": "Jarvis", "text": "I noted the launch timing.", "timestamp": 2.0, "source": "jarvis"},
    ]

    payload = api._compress_transcript(transcript)

    assert payload["transcript_codec"] == "json+gzip+base64"
    assert payload["transcript_entry_count"] == 2
    compressed = base64.b64decode(payload["transcript_compressed"])
    restored = json.loads(gzip.decompress(compressed).decode("utf-8"))
    assert restored == transcript


def test_decompress_transcript_returns_empty_for_missing_payload():
    assert api._decompress_transcript({"session_id": "missing"}) == []


def test_persist_history_snapshot_stores_compressed_transcript():
    state = {
        "session_id": "session-1",
        "meeting_url": "https://meet.google.com/abc-defg-hij",
        "session_status": "ended",
        "transcript_log": [
            {"participant": "Asha", "text": "Please follow up.", "timestamp": 1.0},
        ],
        "change_count": 0,
    }

    with patch.object(api.supabase_store, "upsert_history") as upsert:
        api._persist_history_snapshot(state, {"id": "user-1"}, {"summary": "Done", "stats": {}})

    row = upsert.call_args.args[0]
    assert row["transcript_codec"] == "json+gzip+base64"
    assert row["transcript_entry_count"] == 1
    assert row["transcript_compressed"]


def test_persist_history_snapshot_embeds_pending_changes_in_summary_json():
    state = {
        "session_id": "session-1",
        "session_status": "ended",
        "transcript_log": [],
        "pending_changes": [
            {"id": 1, "status": "pending", "page_title": "Roadmap"},
        ],
        "change_count": 1,
    }

    with patch.object(api.supabase_store, "upsert_history") as upsert:
        api._persist_history_snapshot(state, {"id": "user-1"}, {"summary": "Done", "stats": {}})

    row = upsert.call_args.args[0]
    assert row["summary_json"]["pending_changes"][0]["page_title"] == "Roadmap"


def test_hydrate_state_from_history_item_restores_transcript_and_changes():
    transcript = [{"participant": "Asha", "text": "We chose Friday.", "timestamp": 1.0}]
    history_item = {
        "session_id": "session-restore",
        "meeting_url": "https://meet.google.com/abc-defg-hij",
        "status": "ended",
        "started_at": "2026-05-10T10:00:00+00:00",
        "summary_json": {
            "summary": "The team chose Friday.",
            "pending_changes": [{"id": 4, "status": "pending", "page_title": "Launch"}],
        },
        **api._compress_transcript(transcript),
    }
    state = {
        "session_id": "session-restore",
        "session_status": "idle",
        "transcript_log": [],
        "pending_changes": [],
    }

    api._hydrate_state_from_history_item(state, history_item, {"id": "user-1"})

    assert state["session_status"] == "ended"
    assert state["meeting_url"] == "https://meet.google.com/abc-defg-hij"
    assert state["auth_user_id"] == "user-1"
    assert state["transcript_log"] == transcript
    assert state["pending_changes"][0]["page_title"] == "Launch"
    assert state["change_count"] == 1


def test_stored_summary_response_uses_history_summary_after_restart():
    transcript = [{"participant": "Asha", "text": "We chose Friday.", "timestamp": 1.0}]
    history_item = {
        "session_id": "session-restore",
        "title": "Meeting - restored",
        "started_at": "2026-05-10T10:00:00+00:00",
        "summary_json": {
            "summary": "Stored summary",
            "key_topics": ["Launch"],
            "decisions": ["Launch Friday"],
            "action_items": [],
            "participants": ["Asha"],
            "mom": [],
        },
        **api._compress_transcript(transcript),
    }

    response = api._stored_summary_response(history_item, {"session_id": "session-restore"})

    assert response["summary"] == "Stored summary"
    assert response["session_id"] == "session-restore"
    assert response["stats"]["transcript_entries"] == 1
    assert response["transcript_highlights"][0]["text"] == "We chose Friday."


def test_stored_summary_response_ignores_prior_generation_failure():
    history_item = {
        "session_id": "session-restore",
        "summary_json": {
            "summary": "AI review generation is unavailable right now. The transcript was captured, but the post-meeting summary could not be generated.",
        },
        **api._compress_transcript([{"participant": "Asha", "text": "Retry this.", "timestamp": 1.0}]),
    }

    assert api._stored_summary_response(history_item, {"session_id": "session-restore"}) is None


@pytest.mark.asyncio
async def test_generate_review_insights_uses_parallel_specialists():
    transcript = [
        {"participant": "Asha", "text": "We decided to launch next week.", "timestamp": 1.0},
        {"participant": "Ben", "text": "I will prepare the rollout checklist.", "timestamp": 2.0},
    ]
    calls = []

    async def fake_agent(name, *_args, **_kwargs):
        calls.append(name)
        if name == "Executive summary":
            return {"summary": "Detailed summary"}
        if name == "Topics and decisions":
            return {"key_topics": ["Launch"], "decisions": ["Launch next week"]}
        if name == "Action items":
            return {"action_items": [{"description": "Prepare rollout checklist", "owner": "Ben", "due": None}]}
        if name == "Minutes of meeting":
            return {"mom": [{"topic": "Launch plan", "summary": "The team aligned on next week's launch."}]}
        return {}

    with patch.object(api, "_run_review_agent", side_effect=fake_agent):
        insights = await api._generate_review_insights(transcript, [], [])

    assert set(calls) == {"Executive summary", "Topics and decisions", "Action items", "Minutes of meeting"}
    assert insights["summary"] == "Detailed summary"
    assert insights["key_topics"] == ["Launch"]
    assert insights["decisions"] == ["Launch next week"]
    assert insights["action_items"][0]["owner"] == "Ben"
    assert insights["mom"][0]["topic"] == "Launch plan"


@pytest.mark.asyncio
async def test_chat_with_meeting_uses_session_context():
    from confluence_logic import jarvis_agentic

    session_id = "chat-session-review-api"
    state = jarvis_agentic.get_meeting_session_state(session_id)
    state["session_id"] = session_id
    state["transcript_log"] = [
        {"participant": "Asha", "text": "We chose the Friday launch.", "timestamp": 1.0},
    ]

    async def fake_answer(context, messages):
        assert context["transcript"][0]["text"] == "We chose the Friday launch."
        assert messages[0].content == "When are we launching?"
        return "The team chose the Friday launch."

    with patch.object(api, "_answer_meeting_chat", side_effect=fake_answer):
        response = await api.chat_with_meeting(
            session_id,
            api.MeetingChatRequest(messages=[
                api.MeetingChatMessage(role="user", content="When are we launching?"),
            ]),
            authorization=None,
        )

    assert response["answer"] == "The team chose the Friday launch."
    assert response["context"]["transcript_entries"] == 1


def test_replace_agent_generated_changes_preserves_manual_queue_items():
    state = {
        "session_id": "session-1",
        "pending_changes": [
            {
                "id": 3,
                "change_type": "edit",
                "page_id": "manual-page",
                "page_title": "Manual page",
                "status": "pending",
            },
            {
                "id": 4,
                "change_type": "edit",
                "page_id": "old-agent-page",
                "page_title": "Old generated page",
                "status": "pending",
                "source": "meeting_proposal_agent",
            },
        ],
    }

    generated = api._replace_agent_generated_changes(
        state,
        [
            {
                "change_type": "edit",
                "page_id": "new-page",
                "page_title": "New page",
                "section_heading": "Decisions",
                "before_content": None,
                "after_content": "Add the launch decision.",
                "rationale": "The meeting reached a launch decision.",
            }
        ],
        "session-1",
        "focus on decisions",
    )

    assert len(generated) == 1
    assert generated[0]["id"] == 4
    assert generated[0]["source"] == "meeting_proposal_agent"
    assert generated[0]["generation_query"] == "focus on decisions"
    assert state["pending_changes"][0]["page_id"] == "manual-page"
    assert state["pending_changes"][1]["page_id"] == "new-page"
    assert state["change_count"] == 2


@pytest.mark.asyncio
async def test_execute_changes_uses_existing_editor_agent_logic():
    editor = Mock()
    editor.handle_prepared_query = AsyncMock(return_value="Commit successful.")
    state = {
        "session_id": "session-1",
        "transcript_log": [],
        "pending_changes": [
            {
                "id": 7,
                "change_type": "edit",
                "page_id": "page-1",
                "page_title": "Launch Plan",
                "section_heading": "Decisions",
                "before_content": "Old launch date",
                "after_content": "Launch moved to Friday.",
                "status": "pending",
            }
        ],
    }

    with patch.object(api, "_get_editor_agent", return_value=editor):
        response = await api._execute_changes_for_state(state, [7])

    assert response["results"] == [{"id": 7, "success": True}]
    assert state["pending_changes"][0]["status"] == "executed"
    prepared_query = editor.handle_prepared_query.await_args.args[0]
    assert "page-1" in prepared_query
    assert "SAFETY RULES:" in prepared_query
    assert "Decisions" in prepared_query
    assert editor.handle_prepared_query.await_args.kwargs["original_query"] == "Approve Confluence change 7"


@pytest.mark.asyncio
async def test_meeting_chat_allows_independent_assessment():
    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[0].message.content = "I would challenge the plan and add a rollback gate."
    mock_client = Mock()
    mock_client.chat.completions.create.return_value = mock_response

    context = {
        "summary": {"summary": "The team agreed to launch on Friday.", "mom": []},
        "transcript": [{"participant": "Asha", "text": "Let's launch Friday.", "timestamp": 1.0}],
    }

    with patch.object(api, "_get_openai_client", return_value=mock_client):
        answer = await api._answer_meeting_chat(
            context,
            [api.MeetingChatMessage(role="user", content="What do you think about the launch plan?")],
        )

    assert answer == "I would challenge the plan and add a rollback gate."
    messages = mock_client.chat.completions.create.call_args.kwargs["messages"]
    system_text = "\n".join(message["content"] for message in messages if message["role"] == "system")
    assert "combine the meeting context with your general knowledge" in system_text
    assert "You may disagree with the plan" in system_text


@pytest.mark.asyncio
async def test_start_bot_uses_explicit_session_state():
    from confluence_logic import jarvis_agentic

    session_id = "test-session-review-api"

    async def _fake_create_room(sid, bid):
        return (Mock(), Mock())

    with patch.object(jarvis_agentic, "create_bot", return_value="bot-for-session"), \
         patch.object(jarvis_agentic, "_create_livekit_room", new=_fake_create_room):
        response = await api._start_bot_for_session(
            api.StartBotRequest(meeting_url="https://meet.google.com/abc-defg-hij"),
            session_id=session_id,
        )

    state = jarvis_agentic.get_meeting_session_state(session_id)

    assert response["status"] == "in_meeting"
    assert response["session_id"] == session_id
    assert state["bot_id"] == "bot-for-session"
    assert state["meeting_url"] == "https://meet.google.com/abc-defg-hij"
    assert jarvis_agentic.get_session_id_for_bot("bot-for-session") == session_id
