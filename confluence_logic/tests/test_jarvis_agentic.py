from io import BytesIO
from types import SimpleNamespace
from unittest.mock import Mock, patch

from confluence_logic import jarvis_agentic as ja


def _reset_meeting_state():
    ja.meeting_state["bot_id"] = None
    ja.meeting_state["transcript_log"] = []
    ja.meeting_state["is_active"] = False
    ja.meeting_state["jarvis_listening"] = False


def test_build_create_bot_payload_uses_recall_provider_by_default():
    with patch.object(ja, "WEBHOOK_URL", "https://example.ngrok-free.app"), \
         patch.object(ja, "RECALL_TRANSCRIPT_PROVIDER", "recallai_streaming"), \
         patch.object(ja, "STREAMING_MODE", "prioritize_low_latency"), \
         patch.object(ja, "LANGUAGE_CODE", "en"):
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


def test_build_create_bot_payload_supports_assembly_provider_opt_in():
    with patch.object(ja, "WEBHOOK_URL", "https://example.ngrok-free.app"), \
         patch.object(ja, "RECALL_TRANSCRIPT_PROVIDER", "assembly_ai_v3_streaming"), \
         patch.object(ja, "STREAMING_MODE", "prioritize_low_latency"), \
         patch.object(ja, "LANGUAGE_CODE", "en"):
        payload = ja.build_create_bot_payload("https://meet.google.com/abc-defg-hij")

    provider = payload["recording_config"]["transcript"]["provider"]
    endpoint = payload["recording_config"]["realtime_endpoints"][0]

    assert provider == {"assembly_ai_v3_streaming": {}}
    assert endpoint["events"] == ["transcript.data"]


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


def test_process_transcript_event_bare_wake_arms_listening():
    _reset_meeting_state()
    result = ja.process_transcript_event("Hey Jarvis", 30.0)
    assert result is None
    assert ja.meeting_state["jarvis_listening"] is True


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


@patch("confluence_logic.jarvis_agentic.requests.post")
@patch("confluence_logic.jarvis_agentic.gTTS")
def test_speak_falls_back_to_gtts_when_openai_tts_fails(mock_gtts, mock_post):
    mock_post.return_value.status_code = 200

    def write_to_fp(file_obj):
        assert isinstance(file_obj, BytesIO)
        file_obj.write(b"fallback-mp3")

    mock_gtts.return_value.write_to_fp.side_effect = write_to_fp

    with patch.object(ja, "JARVIS_TTS_PROVIDER", "openai"), \
         patch.object(ja, "synthesize_speech", side_effect=RuntimeError("tts failed")):
        result = ja.speak("hello", "bot-123")

    assert result is True
    mock_gtts.assert_called_once()
    mock_post.assert_called_once()
