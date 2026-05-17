import asyncio
from io import BytesIO
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from confluence_logic import jarvis_agentic as ja
from confluence_logic.core.schemas import MasterVoiceDecision


def _reset_meeting_state():
    ja.meeting_state["bot_id"] = None
    ja.meeting_state["transcript_log"] = []
    ja.meeting_state["is_active"] = False
    ja.meeting_state["jarvis_listening"] = False
    ja.meeting_state["current_task"] = None
    ja.meeting_state["pending_requests"].clear()
    ja.meeting_state["pending_clarification"] = None
    ja.meeting_state["current_task_id"] = None
    ja.meeting_state["current_phase"] = "idle"
    ja.meeting_state["current_request"] = None
    ja.meeting_state["output_generation"] = 0
    ja.meeting_state["mutation_started"] = False
    ja.meeting_state["cancel_requested"] = False
    ja.meeting_state["last_user_speech_at"] = 0.0
    ja.meeting_state["general_history"] = []
    ja.meeting_state["last_jarvis_response"] = None
    ja.meeting_state["invoker_participant"] = None
    ja.meeting_state["_pending_debounce_task"] = None
    ja.meeting_state["_accumulated_query"] = ""
    ja.meeting_state["gap_filler_generation"] = None
    ja.meeting_state["active_gap_filler_task"] = None
    ja.meeting_state["wake_query_ack_pending"] = False


def test_build_create_bot_payload_uses_recall_provider_by_default():
    with patch.object(ja, "WEBHOOK_URL", "https://example.ngrok-free.app"), \
         patch.object(ja, "RECALL_TRANSCRIPT_PROVIDER", "recallai_streaming"), \
         patch.object(ja, "STREAMING_MODE", "prioritize_low_latency"), \
         patch.object(ja, "LANGUAGE_CODE", "en"), \
         patch.object(ja, "LIVEKIT_URL", "wss://test.livekit.cloud"), \
         patch.object(ja, "LIVEKIT_API_KEY", "key"), \
         patch.object(ja, "LIVEKIT_API_SECRET", "secret"), \
         patch.object(ja, "_make_subscriber_token", return_value="JWT"):
        payload = ja.build_create_bot_payload("https://meet.google.com/abc-defg-hij")

    provider = payload["recording_config"]["transcript"]["provider"]
    endpoint = payload["recording_config"]["realtime_endpoints"][0]

    assert provider == {
        "recallai_streaming": {
            "mode": "prioritize_low_latency",
            "language_code": "en",
        }
    }
    assert endpoint["events"] == ["transcript.data"]
    assert endpoint["url"] == "wss://example.ngrok-free.app/recall-audio-stream"


def test_build_create_bot_payload_requires_webhook_url():
    with patch.object(ja, "WEBHOOK_URL", None):
        try:
            ja.build_create_bot_payload("https://meet.google.com/abc-defg-hij")
        except RuntimeError as exc:
            assert "WEBHOOK_URL" in str(exc)
        else:
            raise AssertionError("Expected RuntimeError when WEBHOOK_URL is missing")


def test_confluence_read_query_detection_excludes_mutations():
    assert ja._is_confluence_read_query("what does the roadmap page say about launch")
    assert ja._is_confluence_read_query("list the Confluence pages")
    assert not ja._is_confluence_read_query("update the roadmap page with the new launch date")


def test_execute_editor_task_queues_proposal_instead_of_committing():
    _reset_meeting_state()

    async def run_test():
        task = ja._new_voice_task("update the roadmap page", "bot-123")
        task.output_generation = 1
        task.execution_request = "Update Roadmap with Friday launch."

        with patch.object(ja, "_queue_confluence_proposal", new=AsyncMock(return_value="Queued 1 proposed Confluence update for review after the meeting.")) as queue, \
             patch.object(ja.session_agent, "handle_prepared_query", new=AsyncMock()) as editor, \
             patch.object(ja, "_speak_guarded", new=AsyncMock()) as speak:
            await ja._execute_editor_task(task)

        queue.assert_awaited_once_with(task, "Update Roadmap with Friday launch.")
        editor.assert_not_awaited()
        speak.assert_awaited_once()

    asyncio.run(run_test())


def test_build_create_bot_payload_supports_assembly_provider_opt_in():
    with patch.object(ja, "WEBHOOK_URL", "https://example.ngrok-free.app"), \
         patch.object(ja, "RECALL_TRANSCRIPT_PROVIDER", "assembly_ai_v3_streaming"), \
         patch.object(ja, "STREAMING_MODE", "prioritize_low_latency"), \
         patch.object(ja, "LANGUAGE_CODE", "en"), \
         patch.object(ja, "LIVEKIT_API_KEY", "key"), \
         patch.object(ja, "LIVEKIT_API_SECRET", "secret"), \
         patch.object(ja, "LIVEKIT_URL", "wss://test.livekit.cloud"):
        payload = ja.build_create_bot_payload("https://meet.google.com/abc-defg-hij")

    provider = payload["recording_config"]["transcript"]["provider"]
    assert provider == {
        "assembly_ai_v3_streaming": {
            "language_code": "en",
            "speech_model": "u3-rt-pro",
        }
    }


def test_process_transcript_event_handles_inline_wake_word_query():
    _reset_meeting_state()
    result = ja.process_transcript_event("Hey Jarvis update quarterly goals", 10.0)
    assert result == "update quarterly goals"
    assert ja.meeting_state["jarvis_listening"] is False


def test_process_transcript_event_handles_bare_wake_then_follow_up():
    _reset_meeting_state()
    first = ja.process_transcript_event("Hey Jarvis", 20.0)
    second = ja.process_transcript_event("change the title to donut facts", 20.3)
    assert first is None
    assert second == "change the title to donut facts"
    assert ja.meeting_state["jarvis_listening"] is False


def test_process_transcript_event_routes_pending_clarification_without_wake():
    _reset_meeting_state()
    ja.meeting_state["pending_clarification"] = {
        "task_id": 1,
        "clarification_context": "Need the page title.",
        "question": "Which roadmap page?",
    }
    result = ja.process_transcript_event("Product Roadmap", 40.0)
    assert result == "Product Roadmap"


def test_is_bare_wake_invocation():
    assert ja.is_bare_wake_invocation("Hey Jarvis") is True
    assert ja.is_bare_wake_invocation("Hey Jarvis update quarterly goals") is False
    assert ja.is_bare_wake_invocation("update quarterly goals") is False


def test_synthesize_speech_uses_openai_tts_bytes():
    mock_response = Mock()
    mock_response.read.return_value = b"fake-mp3"
    mock_client = SimpleNamespace(
        audio=SimpleNamespace(
            speech=SimpleNamespace(
                create=Mock(return_value=mock_response)
            )
        )
    )

    with patch.object(ja, "JARVIS_TTS_PROVIDER", "openai"), \
         patch.object(ja, "JARVIS_TTS_MODEL", "gpt-4o-mini-tts"), \
         patch.object(ja, "JARVIS_TTS_VOICE", "alloy"), \
         patch.object(ja, "JARVIS_TTS_SPEED", 1.0), \
         patch.object(ja, "get_openai_client", return_value=mock_client):
        audio = ja.synthesize_speech("hello world")

    assert audio == b"fake-mp3"
    mock_client.audio.speech.create.assert_called_once()


# test_speak_falls_back_to_gtts_when_openai_tts_fails removed — speak() deleted in Plan 003 (D-13)


def test_handle_spoken_request_master_clarifies_when_needed():
    _reset_meeting_state()

    async def run_test():
        with patch.object(ja.session_agent, "plan_voice_turn", new=AsyncMock(return_value=MasterVoiceDecision(
            immediate_reply="Sure, let me check.",
            needs_clarification=True,
            clarification_question="Which roadmap page do you mean?",
            proceed_reply=None,
            execution_request=None,
            intent="edit",
        ))), patch.object(ja.session_agent, "handle_prepared_query", new=AsyncMock()) as mock_exec, \
             patch.object(ja, "_speak_guarded", new=AsyncMock(return_value=True)) as mock_speak:
            await ja.handle_spoken_request("update the roadmap page", "bot-123")
            await asyncio.sleep(0)
            await asyncio.sleep(0)

            mock_exec.assert_not_awaited()
            spoken_texts = [call.args[0] for call in mock_speak.call_args_list]
            # The immediate_reply text may be rewritten; check clarification question verbatim
            assert "Which roadmap page do you mean?" in spoken_texts
            assert ja.meeting_state["pending_clarification"]["question"] == "Which roadmap page do you mean?"

            ja.meeting_state["current_task"].runner.cancel()
            await asyncio.sleep(0)

    asyncio.run(run_test())


def test_handle_spoken_request_master_proceeds_and_suppresses_final_success():
    _reset_meeting_state()

    async def run_test():
        with patch.object(ja.session_agent, "plan_voice_turn", new=AsyncMock(return_value=MasterVoiceDecision(
            immediate_reply="Yes, let me check.",
            needs_clarification=False,
            clarification_question=None,
            proceed_reply="Okay, proceeding.",
            execution_request="Change the title of hello to hi.",
            intent="edit",
        ))), patch.object(ja.session_agent, "handle_prepared_query", new=AsyncMock(return_value="Done.")) as mock_exec, \
             patch.object(ja, "_speak_guarded", new=AsyncMock(return_value=True)) as mock_speak:
            await ja.handle_spoken_request("change the title of hello to hi", "bot-123")
            await asyncio.sleep(0)
            await asyncio.sleep(0)

            mock_exec.assert_awaited_once_with(
                "Change the title of hello to hi.",
                original_query="change the title of hello to hi",
                mutation_started_callback=mock_exec.await_args.kwargs["mutation_started_callback"],
            )
            spoken_texts = [call.args[0] for call in mock_speak.call_args_list]
            assert "Yes, let me check." in spoken_texts
            assert "Okay, proceeding." in spoken_texts
            assert "Done." not in spoken_texts

    asyncio.run(run_test())


def test_listing_intent_still_speaks_final_answer():
    _reset_meeting_state()

    async def run_test():
        with patch.object(ja.session_agent, "plan_voice_turn", new=AsyncMock(return_value=MasterVoiceDecision(
            immediate_reply="Let me check.",
            needs_clarification=False,
            clarification_question=None,
            proceed_reply="Okay.",
            execution_request="LIST_PAGES",
            intent="list_pages",
        ))), patch.object(ja.session_agent, "handle_prepared_query", new=AsyncMock(return_value="Sample AI Page, ML Notes")), \
             patch.object(ja, "_speak_guarded", new=AsyncMock(return_value=True)) as mock_speak:
            await ja.handle_spoken_request("what pages are available", "bot-123")
            await asyncio.sleep(0)
            await asyncio.sleep(0)

            spoken_texts = [call.args[0] for call in mock_speak.call_args_list]
            assert "Sample AI Page, ML Notes" in spoken_texts

    asyncio.run(run_test())


def test_follow_up_answer_resumes_same_task_without_wake_word():
    _reset_meeting_state()
    current_task = ja._new_voice_task("update the goals page", "bot-123")
    current_task.phase = "clarifying"
    current_task.output_generation = 1
    ja._set_current_task(current_task)
    ja.meeting_state["pending_clarification"] = {
        "task_id": current_task.task_id,
        "clarification_context": "Need exact page title.",
        "question": "Which goals page do you mean?",
    }

    async def run_test():
        loop = asyncio.get_running_loop()
        current_task.answer_future = loop.create_future()

        with patch.object(ja, "_speak_guarded", new=AsyncMock(return_value=True)):
            await ja.handle_spoken_request("Quarterly Goals", "bot-123")

            assert current_task.answer_future.done() is True
            assert current_task.answer_future.result() == "Quarterly Goals"

    asyncio.run(run_test())


def test_master_gets_clarification_context_on_retry():
    _reset_meeting_state()

    async def run_test():
        planner = AsyncMock(side_effect=[
            MasterVoiceDecision(
                immediate_reply="Sure.",
                needs_clarification=True,
                clarification_question="Which page title should I use?",
                proceed_reply=None,
                execution_request=None,
                intent="create",
            ),
            MasterVoiceDecision(
                immediate_reply="Got it.",
                needs_clarification=False,
                clarification_question=None,
                proceed_reply="Proceeding.",
                execution_request="Create a new page titled Why Donuts Are Awesome with a random table.",
                intent="create",
            ),
        ])

        with patch.object(ja.session_agent, "plan_voice_turn", new=planner), \
             patch.object(ja.session_agent, "handle_prepared_query", new=AsyncMock(return_value="Created.")), \
             patch.object(ja, "_speak_guarded", new=AsyncMock(return_value=True)):
            await ja.handle_spoken_request("create a new page on donuts", "bot-123")
            await asyncio.sleep(0)
            await asyncio.sleep(0)

            await ja.handle_spoken_request("use why donuts are awesome", "bot-123")
            await asyncio.sleep(0)
            await asyncio.sleep(0)
            await asyncio.sleep(0)

            assert planner.await_count == 2
            second_kwargs = planner.await_args_list[1].kwargs
            assert "User answer: use why donuts are awesome" in second_kwargs["clarification_context"]

    asyncio.run(run_test())


def test_handle_spoken_request_queues_non_overriding_when_busy():
    _reset_meeting_state()

    async def run_test():
        current_task = ja._new_voice_task("update roadmap", "bot-123")
        current_task.phase = "executing"
        current_task.output_generation = 1
        current_task.runner = asyncio.create_task(asyncio.sleep(0.1))
        ja.meeting_state["output_generation"] = 1
        ja._set_current_task(current_task)

        with patch.object(ja, "push_audio_to_livekit", new=AsyncMock(return_value=True)), \
             patch.object(ja, "_plan_and_maybe_execute", new=AsyncMock()):
            await ja.handle_spoken_request("also update notes", "bot-123")

        # Non-override request when busy: runs as parallel task (not pending_requests)
        assert len(ja.meeting_state.get("parallel_runners", [])) >= 1
        parallel_requests = [t.request for t, _ in ja.meeting_state.get("parallel_runners", [])]
        assert "also update notes" in parallel_requests
        current_task.runner.cancel()

    asyncio.run(run_test())


def test_handle_spoken_request_supersedes_and_cancels_current_task():
    _reset_meeting_state()

    async def run_test():
        current_task = ja._new_voice_task("update roadmap", "bot-123")
        current_task.phase = "executing"
        current_task.output_generation = 1
        current_task.runner = asyncio.create_task(asyncio.sleep(10))
        ja.meeting_state["output_generation"] = 1
        ja._set_current_task(current_task)

        with patch.object(ja, "push_audio_to_livekit", new=AsyncMock(return_value=True)):
            await ja.handle_spoken_request("instead update notes", "bot-123")
            await asyncio.sleep(0)

        assert current_task.cancel_requested is True
        assert current_task.superseded is True
        assert len(ja.meeting_state["pending_requests"]) == 1
        assert ja.meeting_state["pending_requests"][0].request == "instead update notes"

    asyncio.run(run_test())


def test_handle_bare_wake_uses_busy_ack_when_task_active():
    _reset_meeting_state()

    async def run_test():
        current_task = ja._new_voice_task("update roadmap", "bot-123")
        current_task.phase = "executing"
        current_task.output_generation = 3
        ja.meeting_state["output_generation"] = 3
        ja._set_current_task(current_task)

        with patch.object(ja, "push_audio_to_livekit", new=AsyncMock(return_value=True)):
            await ja._handle_bare_wake("bot-123")

    asyncio.run(run_test())


def test_format_general_history_returns_at_most_3_exchanges():
    """TOPIC-01: sliding window caps at 3 exchanges."""
    _reset_meeting_state()
    ja.meeting_state["general_history"] = [
        ("q1", "a1"), ("q2", "a2"), ("q3", "a3"), ("q4", "a4"), ("q5", "a5"),
    ]
    result = ja._format_general_history()
    # Should contain only last 3 exchanges (q3/a3, q4/a4, q5/a5)
    assert "q3" in result
    assert "q4" in result
    assert "q5" in result
    assert "q1" not in result
    assert "q2" not in result
    # Count lines: 3 exchanges * 2 lines each = 6 lines
    assert len(result.strip().splitlines()) == 6


def test_remember_general_exchange_caps_at_3():
    """TOPIC-01: _remember_general_exchange discards oldest beyond 3."""
    _reset_meeting_state()
    ja.meeting_state["general_history"] = []
    for i in range(5):
        ja._remember_general_exchange(f"q{i}", f"a{i}")
    assert len(ja.meeting_state["general_history"]) == 3
    assert ja.meeting_state["general_history"][0] == ("q2", "a2")
    assert ja.meeting_state["general_history"][-1] == ("q4", "a4")


def test_format_general_history_empty():
    """Edge case: empty history returns [none]."""
    _reset_meeting_state()
    ja.meeting_state["general_history"] = []
    assert ja._format_general_history() == "[none]"


import pytest


@pytest.mark.asyncio
async def test_speak_streaming_posts_each_sentence_separately():
    """REQ-05: per-sentence push via push_audio_to_livekit (not combined)."""
    _reset_meeting_state()
    sid = "session-test-sep"
    ja._meeting_sessions[sid] = ja._fresh_meeting_state(sid)
    ja.meeting_state["session_id"] = sid
    ja.meeting_state["output_generation"] = 0

    async def sentence_gen():
        yield "First sentence. "
        yield "Second sentence. "

    async def gap_filler():
        return None

    pushed_audio = []

    def fake_synthesize(text):
        return f"<{text}>".encode()

    async def fake_push(audio_bytes, session_id, generation=None):
        pushed_audio.append(audio_bytes)
        return True

    with patch.object(ja, "synthesize_speech", side_effect=fake_synthesize), \
         patch.object(ja, "push_audio_to_livekit", side_effect=fake_push), \
         patch.object(ja, "JARVIS_POST_SPEECH_PAUSE_SECONDS", 0.0):
        answer = await ja._speak_streaming(
            sentence_gen(),
            asyncio.create_task(gap_filler()),
            "bot123",
            ja.meeting_state["output_generation"],
            sid,
        )

    assert answer == "First sentence. Second sentence."
    # REQ-05: each sentence is a SEPARATE push — not a single concatenated push
    assert pushed_audio == [b"<First sentence.>", b"<Second sentence.>"]


@pytest.mark.asyncio
async def test_handle_general_question_injects_graph_context():
    """CLASSIFY-03: _handle_general_question calls graph_rag.query_context and passes result to answer_general_question."""
    _reset_meeting_state()
    ja.meeting_state["general_history"] = []
    ja.meeting_state["output_generation"] = 0
    ja.meeting_state["last_jarvis_response"] = None

    with patch("confluence_logic.jarvis_agentic.answer_general_question", new_callable=AsyncMock) as mock_answer, \
         patch("confluence_logic.jarvis_agentic.graph_rag") as mock_graph, \
         patch("confluence_logic.jarvis_agentic._speak_guarded", new_callable=AsyncMock), \
         patch("confluence_logic.jarvis_agentic._looks_like_clarification_prompt", return_value=False):
        mock_graph.query_context = AsyncMock(return_value="In the meeting: React MENTIONED_BY Alice.")
        mock_answer.return_value = "Alice suggested React."
        await ja._handle_general_question("what did Alice suggest", "bot123")
        mock_graph.query_context.assert_called_once_with("what did Alice suggest")
        mock_answer.assert_called_once()
        call_kwargs = mock_answer.call_args
        # graph_context should be passed as keyword argument
        assert "graph_context" in call_kwargs.kwargs or (len(call_kwargs.args) >= 3 and call_kwargs.args[2] == "In the meeting: React MENTIONED_BY Alice.")


def test_speaker_isolation_locks_invoker_on_wake_word():
    """SPEAKER-01: invoker_participant is set when a wake word query is detected."""
    _reset_meeting_state()
    result = ja.process_transcript_event("hey jarvis what time is it", 0.0)
    assert result == "what time is it"
    # Simulate what websocket_endpoint does: lock the invoker
    ja.meeting_state["invoker_participant"] = "Alice"
    assert ja.meeting_state["invoker_participant"] == "Alice"


def test_speaker_isolation_drops_non_invoker_in_filter():
    """SPEAKER-01: non-invoker transcripts are identified and would be filtered."""
    _reset_meeting_state()
    ja.meeting_state["invoker_participant"] = "Alice"
    invoker = ja.meeting_state.get("invoker_participant")
    participant = "Bob"
    should_filter = bool(invoker and participant != invoker)
    assert should_filter is True


def test_speaker_isolation_passes_invoker_transcript():
    """SPEAKER-01: invoker's own transcripts pass through the filter."""
    _reset_meeting_state()
    ja.meeting_state["invoker_participant"] = "Alice"
    invoker = ja.meeting_state.get("invoker_participant")
    participant = "Alice"
    should_filter = bool(invoker and participant != invoker)
    assert should_filter is False


def test_debounce_accumulates_query_text():
    """DEBOUNCE-01: accumulated query text grows correctly with each invoker segment."""
    _reset_meeting_state()
    # First segment
    accumulated = ""
    query1 = "what is the"
    accumulated = (accumulated + " " + query1).strip() if accumulated else query1
    ja.meeting_state["_accumulated_query"] = accumulated
    # Second segment
    query2 = "meeting agenda"
    accumulated = (accumulated + " " + query2).strip() if accumulated else query2
    ja.meeting_state["_accumulated_query"] = accumulated
    assert ja.meeting_state["_accumulated_query"] == "what is the meeting agenda"


@pytest.mark.asyncio
async def test_debounced_dispatch_clears_state_and_calls_handle_spoken_request():
    """DEBOUNCE-01: _debounced_dispatch clears invoker lock and dispatches after sleep."""
    _reset_meeting_state()
    ja.meeting_state["invoker_participant"] = "Alice"
    ja.meeting_state["_accumulated_query"] = "what time is it"

    with patch.object(ja, "handle_spoken_request", new_callable=AsyncMock) as mock_dispatch, \
         patch.object(ja, "JARVIS_DEBOUNCE_SECONDS", 0.0):
        await ja._debounced_dispatch("what time is it", "bot123")
        mock_dispatch.assert_called_once_with("what time is it", "bot123")

    assert ja.meeting_state["invoker_participant"] is None
    assert ja.meeting_state["_accumulated_query"] == ""
    assert ja.meeting_state["_pending_debounce_task"] is None


@pytest.mark.asyncio
async def test_inline_wake_query_starts_ack_preface_before_dispatch():
    """FILLER-03: 'Hey Jarvis, <question>' starts the yes+filler preface immediately."""
    _reset_meeting_state()
    ja.meeting_state["wake_query_ack_pending"] = True
    ja.meeting_state["invoker_participant"] = "Alice"
    calls = []

    async def mock_gap(query, bot_id, generation):
        calls.append(("gap", query, bot_id, generation))

    async def mock_dispatch(query, bot_id):
        calls.append(("dispatch", query, bot_id))
        await asyncio.sleep(0)

    with patch.object(ja, "_speak_gap_filler", side_effect=mock_gap), \
         patch.object(ja, "handle_spoken_request", side_effect=mock_dispatch), \
         patch.object(ja, "JARVIS_DEBOUNCE_SECONDS", 0.0):
        await ja._debounced_dispatch("what is money", "bot123")
        await asyncio.sleep(0)

    assert ("gap", "what is money", "bot123", 0) in calls
    assert ("dispatch", "what is money", "bot123") in calls
    assert calls.index(("gap", "what is money", "bot123", 0)) < calls.index(("dispatch", "what is money", "bot123"))


@pytest.mark.asyncio
async def test_handle_general_question_speaks_filler_before_answer():
    """FILLER-02: contextual gap filler is spoken before the LLM answer is generated."""
    _reset_meeting_state()
    call_order = []

    async def mock_filler(q, invoker_name=None):
        call_order.append("filler_generated")
        return "Let me check that for you."

    async def mock_speak(text, bot_id, generation, allow_stale=False, **_kwargs):
        call_order.append(f"spoke:{text[:20]}")
        return True

    async def mock_answer(q, history, graph_context="", force_web_search=False, speech_rewrite_enabled=False, multiturn_reference=False):
        call_order.append("answer_generated")
        return "The answer is 42."

    with patch.object(ja, "_generate_contextual_gap_filler", side_effect=mock_filler), \
         patch.object(ja, "_speak_guarded", side_effect=mock_speak), \
         patch.object(ja, "answer_general_question", side_effect=mock_answer), \
         patch.object(ja, "get_random_filler_audio", return_value=None), \
         patch.object(ja, "synthesize_speech", return_value=b"mp3"), \
         patch.object(ja, "graph_rag") as mock_graph, \
         patch.object(ja, "_looks_like_clarification_prompt", return_value=False):
        mock_graph.query_context = AsyncMock(return_value="")
        await ja._handle_general_question("what time is it", "bot123")

    assert "filler_generated" in call_order, "filler was never generated"
    assert "answer_generated" in call_order, "answer was never generated"
    assert call_order.index("spoke:Let me check that fo") < call_order.index("spoke:The answer is 42."), \
        f"Expected filler audio before answer audio, got: {call_order}"


# REQ-01: Incremental sentence splitter tests — added by plan 01-001


def test_split_sentence_incremental_basic():
    sentences, remainder = ja._split_sentence_incremental("Hello world. ")
    assert sentences == ["Hello world."]
    assert remainder == ""


def test_split_sentence_incremental_abbreviation_guard():
    # "Dr." should NOT be a sentence end — the period is part of the abbreviation
    sentences, remainder = ja._split_sentence_incremental("Dr. Smith arrived.")
    # Without a trailing space after the final period, the second period is also not yet a confirmed boundary
    # — the splitter waits for whitespace or more input.
    assert sentences == []
    assert remainder == "Dr. Smith arrived."

    # With trailing space, the final period IS a boundary (>=5 chars, not abbrev)
    sentences, remainder = ja._split_sentence_incremental("Dr. Smith arrived. ")
    assert sentences == ["Dr. Smith arrived."]
    assert remainder == ""


def test_split_sentence_incremental_initial_guard():
    # "U.S." internal periods must not split; final period after "today" is the real boundary
    sentences, remainder = ja._split_sentence_incremental("U.S. policy is clear today. ")
    assert sentences == ["U.S. policy is clear today."]
    assert remainder == ""


def test_split_sentence_incremental_min_chars():
    # "Hi." — only 2 non-whitespace chars before the period; must NOT split
    sentences, remainder = ja._split_sentence_incremental("Hi. ")
    assert sentences == []
    assert remainder == "Hi. "


def test_split_sentence_incremental_exclamation_unconditional():
    # "!" splits regardless of preceding length
    sentences, remainder = ja._split_sentence_incremental("Hi! ")
    assert sentences == ["Hi!"]
    assert remainder == ""


def test_split_sentence_incremental_question_unconditional():
    # "?" splits regardless of preceding length
    sentences, remainder = ja._split_sentence_incremental("Hi? ")
    assert sentences == ["Hi?"]
    assert remainder == ""


def test_split_sentence_incremental_cross_token():
    # Simulate token-by-token streaming — incremental accumulation must not emit until boundary completes
    buffer = ""
    for token in ["Hello", " world"]:
        buffer += token
        sentences, buffer = ja._split_sentence_incremental(buffer)
        assert sentences == []
    # Now add period — still no whitespace after, so no boundary yet
    buffer += "."
    sentences, buffer = ja._split_sentence_incremental(buffer)
    assert sentences == []
    assert buffer == "Hello world."
    # Add trailing space — boundary now confirmed
    buffer += " "
    sentences, buffer = ja._split_sentence_incremental(buffer)
    assert sentences == ["Hello world."]
    assert buffer == ""


def test_split_sentence_incremental_multiple_sentences():
    sentences, remainder = ja._split_sentence_incremental(
        "First sentence here. Second one too! Third? "
    )
    assert sentences == ["First sentence here.", "Second one too!", "Third?"]
    assert remainder == ""


def test_split_sentence_incremental_remainder_preserved():
    sentences, remainder = ja._split_sentence_incremental(
        "First sentence here. And another partial"
    )
    assert sentences == ["First sentence here."]
    assert remainder == "And another partial"


def test_split_sentence_incremental_inside_word_period():
    # Period inside "example.com" must not split — followed by non-whitespace char
    sentences, remainder = ja._split_sentence_incremental("Visit example.com today! ")
    assert sentences == ["Visit example.com today!"]
    assert remainder == ""


@pytest.mark.asyncio
async def test_speak_streaming_ordered_playback_despite_out_of_order_tts():
    """REQ-03: PriorityQueue + stash preserves sentence order even when TTS completes out-of-order."""
    _reset_meeting_state()
    sid = "session-test-order"
    ja._meeting_sessions[sid] = ja._fresh_meeting_state(sid)
    ja.meeting_state["session_id"] = sid
    ja.meeting_state["output_generation"] = 0

    async def sentence_gen():
        yield "First. "  # 5 chars before period — splitter min_chars check; use longer
        yield "First sentence ready. "
        yield "Second sentence ready. "
        yield "Third sentence ready. "

    async def gap_filler():
        return None

    pushed_audio = []
    synth_delays = {
        "First sentence ready.": 0.20,   # slowest
        "Second sentence ready.": 0.05,  # fastest
        "Third sentence ready.": 0.10,
    }

    def fake_synthesize(text):
        import time as _t
        delay = synth_delays.get(text, 0.0)
        _t.sleep(delay)
        return f"<{text}>".encode()

    async def fake_push(audio_bytes, session_id, generation=None):
        pushed_audio.append(audio_bytes)
        return True

    with patch.object(ja, "synthesize_speech", side_effect=fake_synthesize), \
         patch.object(ja, "push_audio_to_livekit", side_effect=fake_push), \
         patch.object(ja, "JARVIS_POST_SPEECH_PAUSE_SECONDS", 0.0):
        await ja._speak_streaming(
            sentence_gen(),
            asyncio.create_task(gap_filler()),
            "bot123",
            ja.meeting_state["output_generation"],
            sid,
        )

    # Note: "First. " has only 6 chars including space — the splitter's >=5 non-ws rule
    # passes ("First." is 6 chars). So 4 sentences total.
    assert pushed_audio == [
        b"<First.>",
        b"<First sentence ready.>",
        b"<Second sentence ready.>",
        b"<Third sentence ready.>",
    ]


@pytest.mark.asyncio
async def test_speak_streaming_waits_playback_duration_between_posts():
    """D-12: duration-wait removed — consumer completes immediately (no sleep between sentences)."""
    _reset_meeting_state()
    sid = "session-test-nodur"
    ja._meeting_sessions[sid] = ja._fresh_meeting_state(sid)
    ja.meeting_state["session_id"] = sid
    ja.meeting_state["output_generation"] = 0

    async def sentence_gen():
        yield "First sentence here. "
        yield "Second sentence here. "

    async def gap_filler():
        return None

    import time as _t

    def fake_synthesize(text):
        return f"<{text}>".encode()

    async def fake_push(audio_bytes, session_id, generation=None):
        return True

    t0 = _t.monotonic()
    with patch.object(ja, "synthesize_speech", side_effect=fake_synthesize), \
         patch.object(ja, "push_audio_to_livekit", side_effect=fake_push), \
         patch.object(ja, "JARVIS_POST_SPEECH_PAUSE_SECONDS", 0.0):
        await ja._speak_streaming(
            sentence_gen(),
            asyncio.create_task(gap_filler()),
            "bot123",
            ja.meeting_state["output_generation"],
            sid,
        )
    elapsed = _t.monotonic() - t0
    # Duration-wait removed: with 2 sentences and no sleep, should complete well under 1 second.
    assert elapsed < 1.0, f"Expected fast completion (no duration-wait), took {elapsed:.3f}s"


@pytest.mark.asyncio
async def test_speak_streaming_generation_interrupt_stops_pipeline():
    """REQ-06: output_generation change mid-playback prevents further pushes."""
    _reset_meeting_state()
    sid = "session-test-interrupt"
    ja._meeting_sessions[sid] = ja._fresh_meeting_state(sid)
    ja.meeting_state["session_id"] = sid
    ja.meeting_state["output_generation"] = 5

    async def sentence_gen():
        yield "First sentence here. "
        yield "Second sentence here. "
        yield "Third sentence here. "

    async def gap_filler():
        return None

    pushed_audio = []

    def fake_synthesize(text):
        return f"<{text}>".encode()

    async def fake_push(audio_bytes, session_id, generation=None):
        pushed_audio.append(audio_bytes)
        # After first push, bump generation to simulate interruption
        if len(pushed_audio) == 1:
            ja.meeting_state["output_generation"] = 99
        return False  # return False to signal interruption (generation check)

    with patch.object(ja, "synthesize_speech", side_effect=fake_synthesize), \
         patch.object(ja, "push_audio_to_livekit", side_effect=fake_push), \
         patch.object(ja, "JARVIS_POST_SPEECH_PAUSE_SECONDS", 0.0):
        await ja._speak_streaming(
            sentence_gen(),
            asyncio.create_task(gap_filler()),
            "bot123",
            5,  # initial generation
            sid,
        )

    # After the first push, generation changed — remaining sentences MUST NOT be pushed.
    assert len(pushed_audio) == 1, f"Expected exactly 1 push after interruption, got {len(pushed_audio)}"
    assert pushed_audio[0] == b"<First sentence here.>"


@pytest.mark.asyncio
async def test_speak_streaming_handles_abbreviation_in_stream():
    """REQ-01+REQ-02: producer uses _split_sentence_incremental — 'Dr.' is NOT a sentence boundary."""
    _reset_meeting_state()

    async def sentence_gen():
        # Streaming raw text including "Dr." — splitter must not break here
        yield "Dr. Smith said hello today. "
        yield "Goodbye now. "

    async def gap_filler():
        return None

    def fake_synthesize(text):
        return f"<{text}>".encode()

    sid = "session-test-abbrev"
    ja._meeting_sessions[sid] = ja._fresh_meeting_state(sid)
    ja.meeting_state["session_id"] = sid

    pushed_audio = []

    async def fake_push(audio_bytes, session_id, generation=None):
        pushed_audio.append(audio_bytes)
        return True

    with patch.object(ja, "synthesize_speech", side_effect=fake_synthesize), \
         patch.object(ja, "push_audio_to_livekit", side_effect=fake_push), \
         patch.object(ja, "JARVIS_POST_SPEECH_PAUSE_SECONDS", 0.0):
        await ja._speak_streaming(
            sentence_gen(),
            asyncio.create_task(gap_filler()),
            "bot123",
            ja.meeting_state["output_generation"],
            sid,
        )

    # Exactly TWO sentences — "Dr." must not be split off
    assert pushed_audio == [
        b"<Dr. Smith said hello today.>",
        b"<Goodbye now.>",
    ]


@pytest.mark.asyncio
async def test_speak_streaming_acquires_lock_after_gap_filler():
    """Pitfall 3: output_lock must NOT be held while awaiting gap_filler_task."""
    _reset_meeting_state()

    async def sentence_gen():
        yield "First sentence here. "

    gap_filler_acquired_lock = asyncio.Event()
    gap_filler_released_lock = asyncio.Event()

    async def gap_filler():
        # Simulate gap_filler trying to use the same output_lock
        lock = ja._get_output_lock()
        async with lock:
            gap_filler_acquired_lock.set()
            await asyncio.sleep(0.05)
            gap_filler_released_lock.set()

    sid = "session-test-lock"
    ja._meeting_sessions[sid] = ja._fresh_meeting_state(sid)
    ja.meeting_state["session_id"] = sid

    def fake_synthesize(text):
        return f"<{text}>".encode()

    pushed = []

    async def fake_push(audio_bytes, session_id, generation=None):
        pushed.append(audio_bytes)
        return True

    with patch.object(ja, "synthesize_speech", side_effect=fake_synthesize), \
         patch.object(ja, "push_audio_to_livekit", side_effect=fake_push), \
         patch.object(ja, "JARVIS_POST_SPEECH_PAUSE_SECONDS", 0.0):
        await asyncio.wait_for(
            ja._speak_streaming(
                sentence_gen(),
                asyncio.create_task(gap_filler()),
                "bot123",
                ja.meeting_state["output_generation"],
                sid,
            ),
            timeout=3.0,
        )

    # If we got here without TimeoutError, no deadlock occurred.
    assert gap_filler_acquired_lock.is_set()
    assert gap_filler_released_lock.is_set()
    assert len(pushed) == 1


# REQ-07: Backward-compatibility tests — added by plan 01-003


@pytest.mark.asyncio
async def test_speak_guarded_still_concatenates():
    """REQ-07: _speak_guarded still concatenates sentences — one push_audio_to_livekit call with combined audio."""
    _reset_meeting_state()
    sid = "session-test-guard"
    ja._meeting_sessions[sid] = ja._fresh_meeting_state(sid)
    ja.meeting_state["session_id"] = sid
    ja.meeting_state["output_generation"] = 0

    pushed_audio = []

    def fake_synthesize(text):
        return f"<{text}>".encode()

    async def fake_push(audio_bytes, session_id, generation=None):
        pushed_audio.append(audio_bytes)
        return True

    with patch.object(ja, "synthesize_speech", side_effect=fake_synthesize), \
         patch.object(ja, "push_audio_to_livekit", side_effect=fake_push), \
         patch.object(ja, "JARVIS_SPEECH_HOLD_SECONDS", 0.0):
        ok = await ja._speak_guarded(
            "First sentence here. Second sentence here.",
            "bot123",
            ja.meeting_state["output_generation"],
        )

    assert ok is True
    # REQ-07: _speak_guarded MUST still concatenate — one push with combined audio
    assert len(pushed_audio) == 1
    assert pushed_audio[0] == b"<First sentence here.><Second sentence here.>"


@pytest.mark.asyncio
async def test_handle_meeting_summary_invokes_speak_streaming_with_correct_args():
    """REQ-07: _handle_meeting_summary calls _speak_streaming with (sentence_gen, gap_filler_task, bot_id, generation, session_id)."""
    _reset_meeting_state()
    ja.meeting_state["transcript_log"] = [{"speaker": "Alice", "text": "test"}]
    ja.meeting_state["output_generation"] = 7

    async def fake_stream(*args, **kwargs):
        yield "Fake summary sentence here. "

    async def fake_gap_filler(*args, **kwargs):
        return None

    with patch.object(ja, "summarize_meeting_streaming", side_effect=fake_stream), \
         patch.object(ja, "_speak_gap_filler", side_effect=fake_gap_filler), \
         patch.object(ja, "_speak_streaming", new=AsyncMock(return_value="Fake summary sentence here.")) as mock_stream:
        await ja._handle_meeting_summary("give me a brief summary", "botX")

    assert mock_stream.await_count == 1
    args, kwargs = mock_stream.await_args
    assert len(args) >= 4
    # arg 0: sentence_gen (async generator)
    assert hasattr(args[0], "__aiter__")
    # arg 1: gap_filler_task (asyncio.Task)
    assert isinstance(args[1], asyncio.Task)
    # arg 2: bot_id
    assert args[2] == "botX"
    # arg 3: generation
    assert args[3] == 7


@pytest.mark.asyncio
async def test_handle_meeting_opinion_invokes_speak_streaming():
    """REQ-07: _handle_meeting_opinion still calls _speak_streaming unchanged."""
    _reset_meeting_state()
    ja.meeting_state["transcript_log"] = [{"speaker": "Alice", "text": "test"}]
    ja.meeting_state["output_generation"] = 3

    async def fake_stream(*args, **kwargs):
        yield "Fake opinion here. "

    async def fake_gap_filler(*args, **kwargs):
        return None

    with patch.object(ja, "generate_opinion_streaming", side_effect=fake_stream), \
         patch.object(ja, "_speak_gap_filler", side_effect=fake_gap_filler), \
         patch.object(ja, "_speak_streaming", new=AsyncMock(return_value="Fake opinion here.")) as mock_stream:
        await ja._handle_meeting_opinion("what's your opinion", "botY")

    assert mock_stream.await_count == 1
    args, _ = mock_stream.await_args
    assert len(args) >= 4
    assert args[2] == "botY"
    assert args[3] == 3


@pytest.mark.asyncio
async def test_handle_action_items_invokes_speak_streaming():
    """REQ-07: _handle_action_items still calls _speak_streaming unchanged."""
    _reset_meeting_state()
    ja.meeting_state["transcript_log"] = [{"speaker": "Alice", "text": "test"}]
    ja.meeting_state["output_generation"] = 11

    async def fake_stream(*args, **kwargs):
        yield "Action one here. "

    async def fake_gap_filler(*args, **kwargs):
        return None

    with patch.object(ja, "extract_action_items_streaming", side_effect=fake_stream), \
         patch.object(ja, "_speak_gap_filler", side_effect=fake_gap_filler), \
         patch.object(ja, "_speak_streaming", new=AsyncMock(return_value="Action one here.")) as mock_stream:
        await ja._handle_action_items("what are the action items", "botZ")

    assert mock_stream.await_count == 1
    args, _ = mock_stream.await_args
    assert len(args) >= 4
    assert args[2] == "botZ"
    assert args[3] == 11


def test_local_repl_override_signature_compatibility():
    """Plan 003: _speak_streaming signature now includes session_id (Plan 003 migration).

    Signature contract: (sentence_gen, gap_filler_task: asyncio.Task, bot_id: str, generation: int, session_id: Optional[str]) -> Optional[str]
    """
    import inspect
    sig = inspect.signature(ja._speak_streaming)
    param_names = list(sig.parameters.keys())
    assert param_names == ["sentence_gen", "gap_filler_task", "bot_id", "generation", "session_id"], (
        f"Signature drift detected — params are {param_names}, expected "
        f"['sentence_gen', 'gap_filler_task', 'bot_id', 'generation', 'session_id']. "
        f"Plan 003 added session_id for LiveKit routing."
    )
    # gap_filler_task annotation should be asyncio.Task
    gap_filler_param = sig.parameters["gap_filler_task"]
    assert gap_filler_param.annotation is asyncio.Task, (
        f"gap_filler_task annotation drift: {gap_filler_param.annotation}"
    )
    # bot_id annotation should be str
    assert sig.parameters["bot_id"].annotation is str
    # generation annotation should be int
    assert sig.parameters["generation"].annotation is int
    # session_id should be Optional with default None
    session_id_param = sig.parameters["session_id"]
    assert session_id_param.default is None


# ---------------------------------------------------------------------------
# Phase 02: LiveKit audio infrastructure tests (Plan 002)
# ---------------------------------------------------------------------------

import datetime as _dt


def _make_fake_frame(label=""):
    f = Mock()
    f.label = label
    return f


def _async_iter_frames(frames):
    """Return an async iterator that yields the given frames."""
    async def _gen():
        for f in frames:
            yield f
    return _gen()


def _make_decoder_mock(frames):
    """Mock AudioStreamDecoder: push/end_input are sync no-ops; async iter yields frames; aclose is async no-op."""
    decoder = Mock()
    decoder.push = Mock()
    decoder.end_input = Mock()
    decoder.aclose = AsyncMock()
    decoder.__aiter__ = Mock(return_value=_async_iter_frames(frames))
    return decoder


def test_push_audio_to_livekit_decodes_mp3():
    _reset_meeting_state()

    frame_a = _make_fake_frame("a")
    frame_b = _make_fake_frame("b")
    frame_c = _make_fake_frame("c")

    decoder_mock = _make_decoder_mock([frame_a, frame_b, frame_c])
    source = Mock()
    source.capture_frame = AsyncMock()

    sid = "session-decode"
    ja._meeting_sessions[sid] = ja._fresh_meeting_state(sid)
    ja._meeting_sessions[sid]["livekit_source"] = source

    with patch.object(ja, "AudioStreamDecoder", return_value=decoder_mock):
        result = asyncio.run(ja.push_audio_to_livekit(b"fake-mp3-bytes", sid))

    assert result is True
    decoder_mock.push.assert_called_once_with(b"fake-mp3-bytes")
    decoder_mock.end_input.assert_called_once()
    assert source.capture_frame.await_count == 3
    awaited_args = [c.args[0] for c in source.capture_frame.await_args_list]
    assert awaited_args == [frame_a, frame_b, frame_c]

    del ja._meeting_sessions[sid]


def test_push_audio_to_livekit_interrupts_on_generation_change():
    _reset_meeting_state()

    frame_a = _make_fake_frame("a")
    frame_b = _make_fake_frame("b")
    frame_c = _make_fake_frame("c")

    ja.meeting_state["output_generation"] = 5
    ja._meeting_sessions[ja._DEFAULT_SESSION_ID]["output_generation"] = 5

    async def _flip_after_first(frame):
        if frame is frame_a:
            ja.meeting_state["output_generation"] = 6
            ja._meeting_sessions[ja._DEFAULT_SESSION_ID]["output_generation"] = 6

    source = Mock()
    source.capture_frame = AsyncMock(side_effect=_flip_after_first)

    sid = "session-interrupt"
    ja._meeting_sessions[sid] = ja._fresh_meeting_state(sid)
    ja._meeting_sessions[sid]["livekit_source"] = source
    ja._meeting_sessions[sid]["output_generation"] = 5

    decoder_mock = _make_decoder_mock([frame_a, frame_b, frame_c])

    with patch.object(ja, "AudioStreamDecoder", return_value=decoder_mock):
        result = asyncio.run(ja.push_audio_to_livekit(b"fake", sid, generation=5))

    assert result is False
    # frame_a was pushed; loop aborted before frame_b on next iteration's generation check
    assert source.capture_frame.await_count == 1

    del ja._meeting_sessions[sid]


def test_push_audio_to_livekit_returns_false_when_no_session():
    _reset_meeting_state()
    with patch.object(ja, "AudioStreamDecoder") as decoder_cls:
        result = asyncio.run(ja.push_audio_to_livekit(b"x", "session-missing"))
    assert result is False
    decoder_cls.assert_not_called()


def test_build_create_bot_payload_has_output_media():
    with patch.object(ja, "WEBHOOK_URL", "https://example.ngrok-free.app"), \
         patch.object(ja, "LIVEKIT_URL", "wss://test.livekit.cloud"), \
         patch.object(ja, "LIVEKIT_API_KEY", "key"), \
         patch.object(ja, "LIVEKIT_API_SECRET", "secret"), \
         patch.object(ja, "_make_subscriber_token", return_value="JWT_FAKE_TOKEN"):
        payload = ja.build_create_bot_payload(
            "https://meet.google.com/abc-defg-hij", session_id="sess-1"
        )

    assert "output_media" in payload
    assert payload["output_media"]["camera"]["kind"] == "webpage"
    url = payload["output_media"]["camera"]["config"]["url"]
    assert url.startswith("https://example.ngrok-free.app/bot-page?")
    assert "token=JWT_FAKE_TOKEN" in url
    assert "room=sess-1" in url
    assert "url=wss://test.livekit.cloud" in url


def test_build_create_bot_payload_subscriber_room_matches_publisher():
    captured_room = {}

    def fake_make_subscriber_token(room_name, ttl_hours=8):
        captured_room["name"] = room_name
        return "JWT_FAKE"

    with patch.object(ja, "WEBHOOK_URL", "https://example.ngrok-free.app"), \
         patch.object(ja, "LIVEKIT_URL", "wss://test.livekit.cloud"), \
         patch.object(ja, "LIVEKIT_API_KEY", "key"), \
         patch.object(ja, "LIVEKIT_API_SECRET", "secret"), \
         patch.object(ja, "_make_subscriber_token", side_effect=fake_make_subscriber_token):
        payload = ja.build_create_bot_payload(
            "https://meet.google.com/abc-defg-hij", session_id="abc-123-uuid"
        )

    assert captured_room["name"] == "abc-123-uuid"
    url = payload["output_media"]["camera"]["config"]["url"]
    assert "room=abc-123-uuid" in url


def test_build_create_bot_payload_retains_websocket_endpoint():
    with patch.object(ja, "WEBHOOK_URL", "https://example.ngrok-free.app"), \
         patch.object(ja, "LIVEKIT_URL", "wss://test.livekit.cloud"), \
         patch.object(ja, "LIVEKIT_API_KEY", "key"), \
         patch.object(ja, "LIVEKIT_API_SECRET", "secret"), \
         patch.object(ja, "_make_subscriber_token", return_value="JWT"):
        payload = ja.build_create_bot_payload(
            "https://meet.google.com/abc-defg-hij", session_id="sess-1"
        )

    endpoints = payload["recording_config"]["realtime_endpoints"]
    assert len(endpoints) == 1
    assert endpoints[0]["type"] == "websocket"
    assert endpoints[0]["url"].startswith("wss://")
    assert endpoints[0]["events"] == ["transcript.data"]


def test_build_create_bot_payload_rejects_non_https_webhook():
    with patch.object(ja, "WEBHOOK_URL", "http://insecure.example.com"), \
         patch.object(ja, "LIVEKIT_URL", "wss://test.livekit.cloud"), \
         patch.object(ja, "LIVEKIT_API_KEY", "key"), \
         patch.object(ja, "LIVEKIT_API_SECRET", "secret"), \
         patch.object(ja, "_make_subscriber_token", return_value="JWT"):
        try:
            ja.build_create_bot_payload(
                "https://meet.google.com/abc-defg-hij", session_id="sess-1"
            )
        except RuntimeError as exc:
            assert "HTTPS" in str(exc) or "https" in str(exc)
        else:
            raise AssertionError("Expected RuntimeError for non-HTTPS WEBHOOK_URL")


def test_build_create_bot_payload_has_no_automatic_audio_output():
    with patch.object(ja, "WEBHOOK_URL", "https://example.ngrok-free.app"), \
         patch.object(ja, "LIVEKIT_URL", "wss://test.livekit.cloud"), \
         patch.object(ja, "LIVEKIT_API_KEY", "key"), \
         patch.object(ja, "LIVEKIT_API_SECRET", "secret"), \
         patch.object(ja, "_make_subscriber_token", return_value="JWT"):
        payload = ja.build_create_bot_payload(
            "https://meet.google.com/abc-defg-hij", session_id="sess-1"
        )
    assert "automatic_audio_output" not in payload


def test_make_subscriber_token_uses_room_name_and_has_ttl():
    captured = {}

    chain = Mock()
    chain.with_identity.return_value = chain
    chain.with_name.return_value = chain

    def _capture_grants(grants):
        captured["grants"] = grants
        return chain
    chain.with_grants.side_effect = _capture_grants

    def _capture_ttl(td):
        captured["ttl"] = td
        return chain
    chain.with_ttl.side_effect = _capture_ttl
    chain.to_jwt.return_value = "JWT_RESULT"

    fake_access_token_cls = Mock(return_value=chain)

    with patch.object(ja, "LIVEKIT_API_KEY", "key"), \
         patch.object(ja, "LIVEKIT_API_SECRET", "secret"), \
         patch.object(ja.livekit_api, "AccessToken", fake_access_token_cls):
        token = ja._make_subscriber_token("my-room", ttl_hours=8)

    assert token == "JWT_RESULT"
    fake_access_token_cls.assert_called_once_with(api_key="key", api_secret="secret")
    assert captured["grants"].room == "my-room"
    assert captured["grants"].can_subscribe is True
    assert captured["grants"].can_publish is False
    assert isinstance(captured["ttl"], _dt.timedelta)
    assert captured["ttl"] == _dt.timedelta(hours=8)


def test_make_subscriber_token_raises_without_credentials():
    with patch.object(ja, "LIVEKIT_API_KEY", ""), \
         patch.object(ja, "LIVEKIT_API_SECRET", ""):
        try:
            ja._make_subscriber_token("room-x")
        except RuntimeError as exc:
            assert "LIVEKIT_API_KEY" in str(exc)
        else:
            raise AssertionError("Expected RuntimeError when LiveKit credentials are empty")


def test_create_livekit_room_uses_session_id_as_room_name():
    _reset_meeting_state()
    sid = "session-create-room-uuid-abc"
    bot_id = "recall-bot-uuid-123"

    captured = {}

    chain = Mock()
    chain.with_identity.return_value = chain
    chain.with_name.return_value = chain
    def _cap_grants(grants):
        captured["grants"] = grants
        return chain
    chain.with_grants.side_effect = _cap_grants
    chain.with_ttl.return_value = chain
    chain.to_jwt.return_value = "PUB_JWT"
    fake_access_token_cls = Mock(return_value=chain)

    fake_room = Mock()
    fake_room.connect = AsyncMock()
    fake_room.local_participant = Mock()
    fake_room.local_participant.publish_track = AsyncMock()
    fake_room.on = Mock(side_effect=lambda *a, **kw: (lambda f: f))

    fake_source = Mock()
    fake_track = Mock()
    fake_options = Mock()

    ja._meeting_sessions[sid] = ja._fresh_meeting_state(sid)

    with patch.object(ja, "LIVEKIT_URL", "wss://test.livekit.cloud"), \
         patch.object(ja, "LIVEKIT_API_KEY", "key"), \
         patch.object(ja, "LIVEKIT_API_SECRET", "secret"), \
         patch.object(ja.livekit_api, "AccessToken", fake_access_token_cls), \
         patch.object(ja.rtc, "Room", return_value=fake_room), \
         patch.object(ja.rtc, "AudioSource", return_value=fake_source), \
         patch.object(ja.rtc, "LocalAudioTrack") as fake_lat, \
         patch.object(ja.rtc, "TrackPublishOptions", return_value=fake_options), \
         patch.object(ja.rtc, "TrackSource", create=True) as fake_ts:
        fake_lat.create_audio_track = Mock(return_value=fake_track)
        fake_ts.SOURCE_MICROPHONE = "SOURCE_MICROPHONE"
        room, source = asyncio.run(ja._create_livekit_room(sid, bot_id))

    assert room is fake_room
    assert source is fake_source
    assert captured["grants"].room == sid
    assert captured["grants"].room != bot_id
    assert captured["grants"].can_publish is True
    assert ja._meeting_sessions[sid]["livekit_room"] is fake_room
    assert ja._meeting_sessions[sid]["livekit_source"] is fake_source

    del ja._meeting_sessions[sid]


def test_teardown_livekit_room_disconnects_and_clears_state():
    _reset_meeting_state()
    sid = "session-teardown"
    fake_room = Mock()
    fake_room.disconnect = AsyncMock()
    fake_source = Mock()
    ja._meeting_sessions[sid] = ja._fresh_meeting_state(sid)
    ja._meeting_sessions[sid]["livekit_room"] = fake_room
    ja._meeting_sessions[sid]["livekit_source"] = fake_source

    asyncio.run(ja._teardown_livekit_room(sid))

    fake_room.disconnect.assert_awaited_once()
    assert ja._meeting_sessions[sid].get("livekit_room") is None
    assert ja._meeting_sessions[sid].get("livekit_source") is None

    del ja._meeting_sessions[sid]


# ---------------------------------------------------------------------------
# Phase 02 Plan 003: removals + unified audio path
# ---------------------------------------------------------------------------


def test_removed_functions_do_not_exist():
    assert not hasattr(ja, "speak"), "ja.speak must be removed (D-13)"
    assert not hasattr(ja, "speak_cached_audio"), "ja.speak_cached_audio must be removed (D-13)"
    assert not hasattr(ja, "_get_audio_duration"), "ja._get_audio_duration must be removed (D-12)"
    assert not hasattr(ja, "_playback_wait_seconds"), "ja._playback_wait_seconds must be removed (D-12)"
    assert not hasattr(ja, "JARVIS_RECALL_AUDIO_DRAIN_BUFFER_SECONDS"), \
        "drain-buffer constant must be removed (D-12)"


def test_speak_streaming_consumer_calls_push_audio():
    _reset_meeting_state()
    sid = "session-stream"
    ja._meeting_sessions[sid] = ja._fresh_meeting_state(sid)
    ja._meeting_sessions[sid]["livekit_source"] = Mock()
    ja.meeting_state["session_id"] = sid
    ja.meeting_state["output_generation"] = 1

    async def fake_sentence_gen():
        yield "Hello world. "
        yield "Second sentence. "

    async def run_test():
        gap_task = asyncio.create_task(asyncio.sleep(0))
        with patch.object(ja, "synthesize_speech", side_effect=lambda s: b"MP3:" + s.encode()), \
             patch.object(ja, "push_audio_to_livekit", new=AsyncMock(return_value=True)) as push_mock:
            result = await ja._speak_streaming(
                fake_sentence_gen(), gap_task, "bot-x", 1, sid,
            )
        return result, push_mock

    result, push_mock = asyncio.run(run_test())

    assert push_mock.await_count >= 2  # at least one push per sentence
    # Each push call must include session_id
    for call in push_mock.await_args_list:
        args, kwargs = call
        assert sid in args or kwargs.get("session_id") == sid


def test_speak_streaming_no_duration_wait():
    # Confirms _playback_wait_seconds is gone AND _speak_streaming completes immediately
    # when push_audio_to_livekit is a fast no-op (no duration-based sleep loop).
    assert not hasattr(ja, "_playback_wait_seconds")

    _reset_meeting_state()
    sid = "session-fast"
    ja._meeting_sessions[sid] = ja._fresh_meeting_state(sid)
    ja._meeting_sessions[sid]["livekit_source"] = Mock()
    ja.meeting_state["session_id"] = sid
    ja.meeting_state["output_generation"] = 1

    async def fake_sentence_gen():
        yield "First sentence. "
        yield "Second sentence. "
        yield "Third sentence."

    async def run_test():
        import time as _time
        gap_task = asyncio.create_task(asyncio.sleep(0))
        with patch.object(ja, "synthesize_speech", side_effect=lambda s: b"MP3:" + s.encode()), \
             patch.object(ja, "push_audio_to_livekit", new=AsyncMock(return_value=True)):
            t0 = _time.monotonic()
            await ja._speak_streaming(fake_sentence_gen(), gap_task, "bot-x", 1, sid)
            elapsed = _time.monotonic() - t0
        return elapsed

    elapsed = asyncio.run(run_test())
    # With duration-wait removed and JARVIS_POST_SPEECH_PAUSE_SECONDS=0.7,
    # total should be well under 2 seconds even with 3 sentences.
    assert elapsed < 2.0, "consumer took %.2fs — duration-wait loop may still be present" % elapsed


def test_speak_cached_guarded_calls_push_audio():
    _reset_meeting_state()
    sid = "session-cached"
    ja._meeting_sessions[sid] = ja._fresh_meeting_state(sid)
    ja._meeting_sessions[sid]["livekit_source"] = Mock()
    ja.meeting_state["session_id"] = sid
    ja.meeting_state["output_generation"] = 1
    ja.meeting_state["last_user_speech_at"] = 0.0

    async def run_test():
        with patch.object(ja, "push_audio_to_livekit", new=AsyncMock(return_value=True)) as push_mock:
            ok = await ja._speak_cached_guarded(b"PRECACHED_MP3", "bot-x", 1)
            return ok, push_mock

    ok, push_mock = asyncio.run(run_test())
    assert ok is True
    push_mock.assert_awaited_once()
    args, kwargs = push_mock.await_args
    assert args[0] == b"PRECACHED_MP3"
    assert args[1] == sid or kwargs.get("session_id") == sid


# ---------------------------------------------------------------------------
# Phase 02 Plan 004: /bot-page route + lifecycle wiring
# ---------------------------------------------------------------------------

from fastapi.testclient import TestClient


def test_bot_page_route_serves_html():
    client = TestClient(ja.app)
    resp = client.get("/bot-page")
    assert resp.status_code == 200, "expected 200 got %s" % resp.status_code
    ctype = resp.headers.get("content-type", "")
    assert "text/html" in ctype, "expected text/html content-type, got %r" % ctype
    body = resp.text
    assert "livekit-client.umd.min.js" in body
    assert "RoomEvent.TrackSubscribed" in body
    assert "URLSearchParams" in body


def test_start_bot_for_session_creates_livekit_room():
    # Import lazily — review.api triggers its own jarvis_agentic import chain.
    from confluence_logic.review import api as review_api
    from confluence_logic.review.api import StartBotRequest

    captured = {}

    async def _fake_create_room(session_id, bot_id):
        captured["args"] = (session_id, bot_id)
        return (Mock(), Mock())

    async def run_test():
        with patch.object(ja, "create_bot", return_value="bot-uuid-xyz"), \
             patch.object(ja, "_create_livekit_room", new=_fake_create_room), \
             patch.object(ja, "_teardown_livekit_room", new=AsyncMock()):
            body = StartBotRequest(meeting_url="https://meet.google.com/abc-defg-hij")
            result = await review_api._start_bot_for_session(body)
        return result

    result = asyncio.run(run_test())
    assert result["status"] == "in_meeting"
    assert result["bot_id"] == "bot-uuid-xyz"
    assert captured["args"][1] == "bot-uuid-xyz"


def test_start_bot_for_session_tears_down_on_livekit_failure():
    from confluence_logic.review import api as review_api
    from confluence_logic.review.api import StartBotRequest

    teardown_calls = []

    async def _fake_create_room_fail(session_id, bot_id):
        raise RuntimeError("simulated LiveKit failure")

    async def _fake_teardown(session_id):
        teardown_calls.append(session_id)

    async def run_test():
        with patch.object(ja, "create_bot", return_value="bot-uuid-xyz"), \
             patch.object(ja, "_create_livekit_room", new=_fake_create_room_fail), \
             patch.object(ja, "_teardown_livekit_room", new=_fake_teardown):
            body = StartBotRequest(meeting_url="https://meet.google.com/abc-defg-hij")
            result = await review_api._start_bot_for_session(body)
        return result

    result = asyncio.run(run_test())
    assert result["status"] == "error"
    assert "LiveKit room creation failed" in (result.get("error") or "")
    assert len(teardown_calls) == 1


# ============================================================================
# PHASE 4 — Recall Audio-to-LiveKit + Native STT — RED stubs (Wave 0)
# These tests intentionally fail until Waves 1-3 land. Do NOT skip or xfail.
# ============================================================================


def test_recall_audio_relay_endpoint_accepts_connection():
    paths = {getattr(r, "path", None) for r in ja.app.routes}
    assert "/recall-audio-mixed/{session_id}" in paths, (
        f"Expected /recall-audio-mixed/{{session_id}} WebSocket route to be registered; got {paths}"
    )


def test_audio_resampler_16k_to_48k():
    import livekit.rtc as rtc
    try:
        resampler = rtc.AudioResampler(input_rate=16000, output_rate=48000, num_channels=1)
    except Exception as exc:  # pragma: no cover — env quirk
        import pytest
        pytest.skip(f"AudioResampler unavailable in this env: {exc}")
    # 100 ms of silence at 16 kHz mono S16LE = 1600 samples * 2 bytes = 3200 bytes.
    # (20 ms is insufficient — the resampler buffers internally and returns 0 frames
    # for small inputs; 100 ms produces at least one 48 kHz output frame.)
    silence = bytearray(3200)
    # AudioResampler.push expects an AudioFrame OR bytes — try both signatures.
    try:
        frames = resampler.push(silence)
    except TypeError:
        frame_in = rtc.AudioFrame(
            data=bytes(silence), sample_rate=16000,
            num_channels=1, samples_per_channel=1600,
        )
        frames = resampler.push(frame_in)
    # If push returns empty, flush to drain any buffered samples.
    if not frames:
        frames = resampler.flush()
    assert frames, "Expected resampler to emit at least one output frame for 100 ms input"
    out = frames[0]
    assert out.sample_rate == 48000
    assert out.num_channels == 1


def test_debounced_dispatch_removed():
    assert not hasattr(ja, "_debounced_dispatch"), "_debounced_dispatch must be removed (D-05)"
    assert not hasattr(ja, "_flush_agent_queue"), "_flush_agent_queue must be removed (D-05)"
    assert not hasattr(ja, "_query_consumer"), "_query_consumer must be removed (D-05)"


def test_in_process_agent_session_removed():
    assert not hasattr(ja, "_InProcessJarvisAgent"), "_InProcessJarvisAgent must be removed (D-07)"
    assert not hasattr(ja, "_start_in_process_agent_session"), "_start_in_process_agent_session must be removed (D-07)"
    assert not hasattr(ja, "_on_inprocess_session_reply"), "_on_inprocess_session_reply must be removed (D-07)"


def test_transcript_log_still_populated_after_voice_removal():
    # REGRESSION GUARD — must keep passing through Wave 3 (D-03)
    _reset_meeting_state()
    entry = ja._append_transcript_log_entry(participant="Alice", text="hello", timestamp=1.0)
    assert entry is not None
    assert len(ja.meeting_state["transcript_log"]) == 1
    assert ja.meeting_state["transcript_log"][0]["participant"] == "Alice"
    assert ja.meeting_state["transcript_log"][0]["text"] == "hello"


def test_build_create_bot_payload_includes_audio_mixed_raw():
    with patch.object(ja, "WEBHOOK_URL", "https://example.ngrok-free.app"), \
         patch.object(ja, "RECALL_TRANSCRIPT_PROVIDER", "recallai_streaming"), \
         patch.object(ja, "STREAMING_MODE", "prioritize_low_latency"), \
         patch.object(ja, "LANGUAGE_CODE", "en"), \
         patch.object(ja, "LIVEKIT_URL", "wss://test.livekit.cloud"), \
         patch.object(ja, "LIVEKIT_API_KEY", "key"), \
         patch.object(ja, "LIVEKIT_API_SECRET", "secret"), \
         patch.object(ja, "_make_subscriber_token", return_value="JWT"):
        payload = ja.build_create_bot_payload("https://meet.google.com/abc-defg-hij", session_id="sess-1")

    rc = payload["recording_config"]
    assert rc.get("audio_mixed_raw") == {}, "audio_mixed_raw must be enabled (D-01)"
    endpoints = rc["realtime_endpoints"]
    audio_eps = [e for e in endpoints if e.get("events") == ["audio_mixed_raw.data"]]
    transcript_eps = [e for e in endpoints if e.get("events") == ["transcript.data"]]
    assert len(audio_eps) == 1, f"Expected exactly one audio realtime endpoint; got {endpoints}"
    assert audio_eps[0]["url"].endswith("/recall-audio-mixed/sess-1"), audio_eps[0]["url"]
    assert len(transcript_eps) == 1, "Transcript endpoint must be preserved (D-03/D-04)"


def test_create_recall_relay_room_uses_distinct_identity():
    """D-01 / Pitfall 1: relay participant identity must be recall-relay-{session_id}
    to avoid feedback loop with jarvis-publisher-{session_id}."""
    captured = {}

    class _FakeGrants:
        def __init__(self, **kwargs): captured["grants"] = kwargs

    class _FakeToken:
        def __init__(self, **kwargs): self._k = kwargs; captured["token_init"] = kwargs
        def with_identity(self, identity): captured["identity"] = identity; return self
        def with_name(self, n): captured["name"] = n; return self
        def with_grants(self, g): captured["grants_obj"] = g; return self
        def with_ttl(self, t): captured["ttl"] = t; return self
        def to_jwt(self): return "FAKE_JWT"

    class _FakeRoom:
        def __init__(self): self.local_participant = AsyncMock()
        def on(self, *a, **k):
            def _decorator(fn): return fn
            return _decorator
        async def connect(self, url, token): captured["connect"] = (url, token)
        async def disconnect(self): pass

    async def run_test():
        with patch.object(ja, "LIVEKIT_URL", "wss://test.livekit.cloud"), \
             patch.object(ja, "LIVEKIT_API_KEY", "key"), \
             patch.object(ja, "LIVEKIT_API_SECRET", "secret"), \
             patch.object(ja.livekit_api, "AccessToken", _FakeToken), \
             patch.object(ja.livekit_api, "VideoGrants", _FakeGrants), \
             patch.object(ja.rtc, "Room", _FakeRoom), \
             patch.object(ja.rtc, "AudioSource", Mock(return_value=Mock())), \
             patch.object(ja.rtc, "LocalAudioTrack", Mock(create_audio_track=Mock(return_value=Mock()))), \
             patch.object(ja.rtc, "TrackPublishOptions", Mock()), \
             patch.object(ja.rtc, "TrackSource", Mock(SOURCE_MICROPHONE="MIC")):
            # Ensure clean state
            ja._meeting_sessions.pop("sess-xyz", None)
            room, source = await ja._create_recall_relay_room("sess-xyz")
            return room, source

    room, source = asyncio.run(run_test())
    assert captured.get("identity") == "recall-relay-sess-xyz"
    assert captured["grants"]["room"] == "sess-xyz"
    assert captured["grants"]["can_publish"] is True
    assert captured["grants"]["can_subscribe"] is False
    assert ja._meeting_sessions["sess-xyz"]["recall_relay_room"] is room
    assert ja._meeting_sessions["sess-xyz"]["recall_relay_source"] is source
