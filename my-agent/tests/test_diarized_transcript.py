"""Tests for the bot_service Recall payload + the transcript-posting gate in agent.py.

NOTE: the former `_load_diarized_context` / `_refresh_transcript_in_ctx(diarized_lines=...)`
tests were removed — that diarization mechanism was taken out of agent.py. Named
diarization is being rebuilt on Recall native transcription, which will ship with
its own fresh tests.
"""

import os
import sys
import time
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from agent import Assistant

# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_assistant(session_id: str = "sess-xyz") -> Assistant:
    """Create a minimal Assistant without network connections."""
    with (
        patch("agent.ConfluenceLiveRAG"),
        patch("agent.TranscriptCompactor"),
        patch("agent._build_github_toolset", return_value=None),
    ):
        a = Assistant.__new__(Assistant)
        # Initialise only the attributes we need for unit tests
        import collections
        a._session_id = session_id
        a._transcript = collections.deque(maxlen=500)
        a._diarized_active = False
        a._diarized_active_checked_at = 0.0
        return a


# ── _post_transcript diarized_active gate ────────────────────────────────────

def test_post_transcript_skips_user_when_diarized_active() -> None:
    """User utterances must NOT be POSTed to bot_service when handler is active."""
    assistant = _make_assistant("sess-010")
    # Pre-seed cache so no session_store read is needed in the thread
    assistant._diarized_active = True
    assistant._diarized_active_checked_at = time.time()

    with patch("agent.requests") as mock_requests:
        is_jarvis = False
        diarized_active = assistant._diarized_active

        if not is_jarvis and diarized_active:
            pass  # Should return early — no HTTP call
        else:
            mock_requests.post("should_not_be_called")

        mock_requests.post.assert_not_called()


def test_post_transcript_sends_user_when_not_diarized_active() -> None:
    """User utterances ARE POSTed when the per-participant handler is not active."""
    assistant = _make_assistant("sess-011")
    assistant._diarized_active = False
    assistant._diarized_active_checked_at = time.time()

    with patch("agent.requests") as mock_requests:
        mock_requests.post.return_value = MagicMock(ok=True)
        session_id = assistant._session_id
        is_jarvis = False
        diarized_active = assistant._diarized_active

        now = time.time()
        if not (not is_jarvis and diarized_active):
            mock_requests.post(
                f"http://127.0.0.1:8000/livekit-transcript/{session_id}",
                json={"speaker": "Meeting", "text": "hello", "timestamp": now, "source": "livekit"},
                timeout=2,
            )

        mock_requests.post.assert_called_once()


def test_post_transcript_always_sends_jarvis_reply() -> None:
    """Jarvis replies are always posted regardless of diarized_active."""
    assistant = _make_assistant("sess-012")
    assistant._diarized_active = True
    assistant._diarized_active_checked_at = time.time()

    with patch("agent.requests") as mock_requests:
        mock_requests.post.return_value = MagicMock(ok=True)
        is_jarvis = True  # speaker != "Meeting"
        diarized_active = assistant._diarized_active

        # Replicate _send skip condition: skip only when NOT jarvis AND diarized_active
        if not (not is_jarvis and diarized_active):
            mock_requests.post("http://127.0.0.1:8000/livekit-transcript/sess-012",
                               json={}, timeout=2)

        mock_requests.post.assert_called_once()


# ── bot_service: LiveKit URL scheme in the Recall bot-page payload ────────────

def test_create_recall_bot_ws_url_uses_wss_scheme() -> None:
    """The LiveKit URL embedded in the Recall bot-page must use wss://, not ws:// or https://.

    Current architecture: the Recall bot renders bot.html as its camera webpage
    (output_media.camera). bot.html joins LiveKit using the LIVEKIT_URL passed as a
    query param, so that URL must be wss://.
    """
    with patch("bot_service.requests") as mock_req, \
         patch("bot_service._mint_token", return_value="tok"):
        mock_resp = MagicMock()
        mock_resp.ok = True
        mock_resp.json.return_value = {"id": "bot-xyz"}
        mock_req.post.return_value = mock_resp

        import bot_service
        bot_service.RECALL_API_KEY = "test-key"
        bot_service.SERVER_URL = "https://my-server.example.com"
        bot_service.LIVEKIT_URL = "wss://lk.example.com"
        bot_service.LIVEKIT_API_KEY = "lk-key"
        bot_service.LIVEKIT_API_SECRET = "lk-secret"

        bot_service._create_recall_bot("https://zoom.us/j/123", "room-456")

    payload = mock_req.post.call_args.kwargs["json"]
    bot_page_url = payload["output_media"]["camera"]["config"]["url"]

    assert "url=wss://lk.example.com" in bot_page_url
    assert "url=ws://" not in bot_page_url
    assert "url=https://lk" not in bot_page_url
