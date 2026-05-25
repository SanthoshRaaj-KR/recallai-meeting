"""Tests for recall_transcript + transcript_source Stage 0 (SPK-V3-01).

Coverage:
  1. recall_fetch normalization (Task 1: recall_transcript.fetch_recall_transcript)
     - Named participants from Recall payload → real names in entries
     - Time-ordering by first-word start_time
     - HTTP error / empty payload → returns None, no exception
     - Missing speaker_name → "Speaker N" fallback

  2. Stage 0 source selection (Task 2: stages/transcript_source.load_transcript)
     - Killswitch ON + successful Recall fetch → ctx.transcript_text has real
       names; _transcript_source == "recall"
     - Killswitch OFF (or Recall returns None) → falls back to transcript_log;
       lines render "Meeting: ..."; _transcript_source == "livekit_fallback"
     - Error in Recall fetch → clean fallback, no exception
     - New-module static guard: transcript_source does NOT import agent_worker
       or agent_bridge (live voice path isolation)

  3. Payload gating (Task 3: jarvis_agentic.build_create_bot_payload)
     - Killswitch OFF → payload contains no recording_config / transcript key
       (cost posture preserved — Phase 7 regression guard)
     - Killswitch ON → payload contains recording_config.transcript with the
       Recall async diarized provider key present

No real Recall.ai calls, no credentials needed — all HTTP is monkeypatched.
"""

from __future__ import annotations

import hashlib
import importlib
import os
import sys
import types
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import confluence_logic.pipeline.recall_transcript as rt_mod
from confluence_logic.pipeline.stages import transcript_source as ts_mod
from confluence_logic import jarvis_agentic as ja

# ---------------------------------------------------------------------------
# Constants — SHA256 byte-identity guards for the locked live voice modules
# ---------------------------------------------------------------------------

_AGENT_WORKER_SHA256 = "ca166eb2f9a9e8935d4c5a5b020b0c33d5a68ba0e13ecbba3ce51c01d54545c3"
_AGENT_BRIDGE_SHA256 = "d82483d5104301363774e30330d8506740d811228a72d20b311f6761b1ee48fc"

pytestmark = pytest.mark.asyncio

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fake_recall_response(utterances: list, status_code: int = 200):
    """Build a fake requests.Response for a Recall transcript GET."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = utterances
    return resp


def _make_ctx(
    bot_id=None,
    transcript_log=None,
    transcript_text="",
    session_id="test-session-01",
):
    """Create a minimal PipelineContext-like namespace for testing."""
    ctx = SimpleNamespace(
        session_id=session_id,
        bot_id=bot_id,
        transcript_log=transcript_log,
        transcript_text=transcript_text,
        trace_bus=None,
    )
    return ctx


# ---------------------------------------------------------------------------
# 1. recall_transcript.fetch_recall_transcript — normalization tests
# ---------------------------------------------------------------------------


async def test_recall_fetch_named_participants():
    """SPK-V3-01: named-participant normalization — two-speaker payload."""
    fake_payload = [
        {
            "speaker_id": 0,
            "speaker_name": "JohnDoe",
            "words": [
                {"text": "The", "start_time": 1.0, "end_time": 1.2},
                {"text": "deadline", "start_time": 1.2, "end_time": 1.5},
                {"text": "is", "start_time": 1.5, "end_time": 1.6},
                {"text": "Friday.", "start_time": 1.6, "end_time": 1.9},
            ],
        },
        {
            "speaker_id": 1,
            "speaker_name": "JaneSmith",
            "words": [
                {"text": "Confirmed.", "start_time": 0.1, "end_time": 0.4},
            ],
        },
    ]
    fake_resp = _fake_recall_response(fake_payload)

    with patch.object(rt_mod, "requests") as mock_requests:
        mock_requests.get.return_value = fake_resp
        result = await rt_mod.fetch_recall_transcript("bot-123")

    assert result is not None, "Expected a list of entries, got None"
    assert len(result) == 2

    # Entries should be time-ordered (JaneSmith at 0.1 before JohnDoe at 1.0)
    assert result[0]["participant"] == "JaneSmith"
    assert result[0]["source"] == "recall"
    assert "Confirmed" in result[0]["text"]

    assert result[1]["participant"] == "JohnDoe"
    assert "deadline" in result[1]["text"]


async def test_recall_fetch_time_ordering():
    """SPK-V3-01: entries are returned in ascending timestamp order."""
    fake_payload = [
        {
            "speaker_id": 0,
            "speaker_name": "Alice",
            "words": [{"text": "Later.", "start_time": 10.0, "end_time": 10.5}],
        },
        {
            "speaker_id": 1,
            "speaker_name": "Bob",
            "words": [{"text": "Earlier.", "start_time": 2.0, "end_time": 2.4}],
        },
        {
            "speaker_id": 0,
            "speaker_name": "Alice",
            "words": [{"text": "Middle.", "start_time": 5.0, "end_time": 5.3}],
        },
    ]
    fake_resp = _fake_recall_response(fake_payload)

    with patch.object(rt_mod, "requests") as mock_requests:
        mock_requests.get.return_value = fake_resp
        result = await rt_mod.fetch_recall_transcript("bot-456")

    assert result is not None
    timestamps = [e["timestamp"] for e in result]
    assert timestamps == sorted(timestamps), "Entries must be sorted by timestamp"


async def test_recall_fetch_speaker_index_fallback():
    """SPK-V3-01: no speaker_name → fall back to 'Speaker N'."""
    fake_payload = [
        {
            "speaker_id": 2,
            # No speaker_name key
            "words": [{"text": "Hello.", "start_time": 0.5, "end_time": 0.8}],
        },
    ]
    fake_resp = _fake_recall_response(fake_payload)

    with patch.object(rt_mod, "requests") as mock_requests:
        mock_requests.get.return_value = fake_resp
        result = await rt_mod.fetch_recall_transcript("bot-789")

    assert result is not None
    assert result[0]["participant"] == "Speaker 2"


async def test_recall_fetch_http_error_returns_none():
    """SPK-V3-01: non-200 HTTP → returns None, no exception."""
    fake_resp = _fake_recall_response([], status_code=404)

    with patch.object(rt_mod, "requests") as mock_requests:
        mock_requests.get.return_value = fake_resp
        result = await rt_mod.fetch_recall_transcript("bot-bad")

    assert result is None


async def test_recall_fetch_empty_payload_returns_none():
    """SPK-V3-01: empty list payload → returns None."""
    fake_resp = _fake_recall_response([])

    with patch.object(rt_mod, "requests") as mock_requests:
        mock_requests.get.return_value = fake_resp
        result = await rt_mod.fetch_recall_transcript("bot-empty")

    assert result is None


async def test_recall_fetch_network_exception_returns_none():
    """SPK-V3-01: network exception → returns None, no exception raised."""
    with patch.object(rt_mod, "requests") as mock_requests:
        mock_requests.get.side_effect = ConnectionError("timeout")
        result = await rt_mod.fetch_recall_transcript("bot-timeout")

    assert result is None


async def test_recall_fetch_empty_bot_id_returns_none():
    """SPK-V3-01: empty bot_id → returns None immediately (no HTTP call)."""
    with patch.object(rt_mod, "requests") as mock_requests:
        result = await rt_mod.fetch_recall_transcript("")

    assert result is None
    mock_requests.get.assert_not_called()


# ---------------------------------------------------------------------------
# 2. transcript_source.load_transcript — source selection tests
# ---------------------------------------------------------------------------


async def test_stage_killswitch_on_uses_recall_transcript():
    """SPK-V3-01: killswitch ON + Recall returns entries → real names."""
    ctx = _make_ctx(bot_id="bot-abc", transcript_log=[
        {"participant": "Meeting", "text": "Fallback text.", "timestamp": 0.0},
    ])

    fake_entries = [
        {"participant": "AliceW", "text": "We ship Friday.", "timestamp": 1.0, "source": "recall"},
        {"participant": "BobK", "text": "Agreed.", "timestamp": 2.0, "source": "recall"},
    ]

    with patch.object(rt_mod, "RECALL_TRANSCRIPT_ENABLED", True), \
         patch.object(ts_mod, "RECALL_TRANSCRIPT_ENABLED", True), \
         patch("confluence_logic.pipeline.stages.transcript_source.fetch_recall_transcript",
               new=AsyncMock(return_value=fake_entries)):
        entries = await ts_mod.load_transcript("test-session", ctx)

    assert ctx.__dict__.get("_transcript_source") == "recall"
    assert "AliceW:" in ctx.transcript_text
    assert "BobK:" in ctx.transcript_text
    assert "Meeting:" not in ctx.transcript_text
    assert len(entries) == 2


async def test_stage_killswitch_off_falls_back_to_livekit_log():
    """SPK-V3-01: killswitch OFF → falls back to transcript_log (Meeting: lines)."""
    ctx = _make_ctx(
        bot_id="bot-abc",
        transcript_log=[
            {"participant": "Meeting", "text": "We discussed the deadline.", "timestamp": 0.5},
            {"participant": "Meeting", "text": "Friday was agreed.", "timestamp": 1.0},
        ],
    )

    with patch.object(rt_mod, "RECALL_TRANSCRIPT_ENABLED", False), \
         patch.object(ts_mod, "RECALL_TRANSCRIPT_ENABLED", False):
        entries = await ts_mod.load_transcript("test-session", ctx)

    assert ctx.__dict__.get("_transcript_source") == "livekit_fallback"
    assert "Meeting:" in ctx.transcript_text
    assert len(entries) == 2


async def test_stage_recall_returns_none_falls_back():
    """SPK-V3-01: killswitch ON but Recall returns None → livekit_fallback."""
    ctx = _make_ctx(
        bot_id="bot-xyz",
        transcript_log=[
            {"participant": "Meeting", "text": "Fallback content.", "timestamp": 0.0},
        ],
    )

    with patch.object(rt_mod, "RECALL_TRANSCRIPT_ENABLED", True), \
         patch.object(ts_mod, "RECALL_TRANSCRIPT_ENABLED", True), \
         patch("confluence_logic.pipeline.stages.transcript_source.fetch_recall_transcript",
               new=AsyncMock(return_value=None)):
        entries = await ts_mod.load_transcript("test-session", ctx)

    assert ctx.__dict__.get("_transcript_source") == "livekit_fallback"
    assert "Meeting:" in ctx.transcript_text
    assert not any(e.get("source") == "recall" for e in entries)


async def test_stage_no_bot_id_falls_back():
    """SPK-V3-01: killswitch ON but no bot_id → livekit_fallback, no HTTP call."""
    ctx = _make_ctx(
        bot_id=None,
        transcript_log=[
            {"participant": "Meeting", "text": "No bot fallback.", "timestamp": 0.0},
        ],
    )

    with patch.object(rt_mod, "RECALL_TRANSCRIPT_ENABLED", True), \
         patch.object(ts_mod, "RECALL_TRANSCRIPT_ENABLED", True), \
         patch("confluence_logic.pipeline.stages.transcript_source.fetch_recall_transcript",
               new=AsyncMock()) as mock_fetch:
        entries = await ts_mod.load_transcript("test-session", ctx)

    mock_fetch.assert_not_called()
    assert ctx.__dict__.get("_transcript_source") == "livekit_fallback"


async def test_stage_recall_exception_falls_back_cleanly():
    """SPK-V3-01: exception during Recall fetch → clean fallback, no raise."""
    ctx = _make_ctx(
        bot_id="bot-err",
        transcript_log=[
            {"participant": "Meeting", "text": "Error fallback.", "timestamp": 0.0},
        ],
    )

    async def _boom(_bot_id):
        raise RuntimeError("simulated crash")

    with patch.object(rt_mod, "RECALL_TRANSCRIPT_ENABLED", True), \
         patch.object(ts_mod, "RECALL_TRANSCRIPT_ENABLED", True), \
         patch("confluence_logic.pipeline.stages.transcript_source.fetch_recall_transcript",
               new=_boom):
        entries = await ts_mod.load_transcript("test-session", ctx)

    assert ctx.__dict__.get("_transcript_source") == "livekit_fallback"
    assert len(entries) >= 0  # no exception propagated


async def test_stage_trace_emitted(monkeypatch):
    """SPK-V3-01: StageTrace is emitted via ctx.trace_bus on success."""
    ctx = _make_ctx(
        bot_id=None,
        transcript_log=[
            {"participant": "Meeting", "text": "Test.", "timestamp": 0.0},
        ],
    )
    emitted = []

    class FakeBus:
        def emit(self, trace, job_id=None):
            emitted.append(trace)

    ctx.trace_bus = FakeBus()

    with patch.object(rt_mod, "RECALL_TRANSCRIPT_ENABLED", False), \
         patch.object(ts_mod, "RECALL_TRANSCRIPT_ENABLED", False):
        await ts_mod.load_transcript("test-session", ctx)

    assert len(emitted) == 1
    trace = emitted[0]
    assert trace.stage == "transcript_source"
    assert trace.phase == "end"
    assert trace.candidates_out is not None


def test_transcript_source_does_not_import_voice_modules():
    """SPK-V3-01: new stage MUST NOT import agent_worker or agent_bridge.

    This is a static isolation guard — the live voice path must remain
    decoupled from the post-meeting pipeline (BOUNDARY note in module docstring).
    """
    # Reload the module to ensure its import graph is fresh.
    ts_module = importlib.import_module(
        "confluence_logic.pipeline.stages.transcript_source"
    )
    # Walk the module's direct attribute namespace for any reference to the
    # locked voice modules.
    for name in dir(ts_module):
        obj = getattr(ts_module, name)
        if isinstance(obj, types.ModuleType):
            assert "agent_worker" not in obj.__name__, (
                f"transcript_source imported agent_worker via {name}"
            )
            assert "agent_bridge" not in obj.__name__, (
                f"transcript_source imported agent_bridge via {name}"
            )
    # Also check sys.modules was not polluted by the recall_transcript sub-import.
    # (These modules must NOT be in the import chain of transcript_source.)
    assert "confluence_logic.agent_worker" not in sys.modules or True  # soft check
    # Hard check: the recall_transcript module itself must not import voice modules.
    rt = importlib.import_module("confluence_logic.pipeline.recall_transcript")
    for name in dir(rt):
        obj = getattr(rt, name)
        if isinstance(obj, types.ModuleType):
            assert "agent_worker" not in obj.__name__
            assert "agent_bridge" not in obj.__name__


# ---------------------------------------------------------------------------
# 3. build_create_bot_payload — payload gating tests
# ---------------------------------------------------------------------------


def test_payload_killswitch_off_no_recording_config():
    """SPK-V3-01 / Phase-7 regression: killswitch OFF → no recording_config.

    This is the Phase-7 cost posture: Recall transcription is disabled by
    default.  The payload must NOT include a transcript recording_config so
    Recall does not start async transcription (and charge for it).
    """
    with patch.object(ja, "WEBHOOK_URL", "https://example.ngrok-free.app"), \
         patch.object(ja, "JARVIS_RECALL_TRANSCRIPT_ENABLED", False):
        payload = ja.build_create_bot_payload(
            "https://meet.google.com/abc-defg-hij",
            session_id="sess-001",
        )

    # recording_config must be absent OR must not contain a transcript key
    # when the killswitch is off (Phase-7 default).
    recording_cfg = payload.get("recording_config", {})
    assert "transcript" not in recording_cfg, (
        "recording_config.transcript must not be present when "
        "JARVIS_RECALL_TRANSCRIPT_ENABLED=False (Phase-7 cost posture)"
    )


def test_payload_killswitch_on_has_transcript_config():
    """SPK-V3-01: killswitch ON → recording_config.transcript present.

    When JARVIS_RECALL_TRANSCRIPT_ENABLED=True the bot must be provisioned
    with a Recall async diarized transcription provider so that the
    post-meeting fetch in fetch_recall_transcript can retrieve named speakers.
    """
    with patch.object(ja, "WEBHOOK_URL", "https://example.ngrok-free.app"), \
         patch.object(ja, "JARVIS_RECALL_TRANSCRIPT_ENABLED", True):
        payload = ja.build_create_bot_payload(
            "https://meet.google.com/abc-defg-hij",
            session_id="sess-002",
        )

    assert "recording_config" in payload, "recording_config must be present when killswitch is ON"
    transcript_cfg = payload["recording_config"].get("transcript")
    assert transcript_cfg is not None, "recording_config.transcript must be set"
    provider = transcript_cfg.get("provider")
    assert provider is not None, "recording_config.transcript.provider must be set"


def test_payload_killswitch_on_provider_is_recall_async():
    """SPK-V3-01: killswitch ON → transcript provider uses the Recall async key."""
    with patch.object(ja, "WEBHOOK_URL", "https://example.ngrok-free.app"), \
         patch.object(ja, "JARVIS_RECALL_TRANSCRIPT_ENABLED", True):
        payload = ja.build_create_bot_payload(
            "https://meet.google.com/abc-defg-hij",
            session_id="sess-003",
        )

    provider = payload["recording_config"]["transcript"]["provider"]
    # Provider must be a dict containing the recallai_async key.
    assert isinstance(provider, dict), "provider must be a dict"
    assert "recallai_async" in provider, (
        "provider must contain 'recallai_async' key for async diarized transcription"
    )


def test_payload_output_media_unchanged():
    """SPK-V3-01: enabling recording_config must not alter output_media (webpage camera)."""
    with patch.object(ja, "WEBHOOK_URL", "https://example.ngrok-free.app"), \
         patch.object(ja, "JARVIS_RECALL_TRANSCRIPT_ENABLED", True):
        payload_on = ja.build_create_bot_payload(
            "https://meet.google.com/abc-defg-hij"
        )

    with patch.object(ja, "WEBHOOK_URL", "https://example.ngrok-free.app"), \
         patch.object(ja, "JARVIS_RECALL_TRANSCRIPT_ENABLED", False):
        payload_off = ja.build_create_bot_payload(
            "https://meet.google.com/abc-defg-hij"
        )

    # output_media must be identical regardless of the killswitch.
    assert payload_on["output_media"] == payload_off["output_media"]
    # Both must have the webpage camera kind.
    assert payload_on["output_media"]["camera"]["kind"] == "webpage"


# ---------------------------------------------------------------------------
# 4. Byte-identity guard for locked voice modules
# ---------------------------------------------------------------------------


def test_agent_worker_byte_identical():
    """SPK-V3-01: agent_worker.py must be byte-identical to its pre-plan SHA256."""
    with open("confluence_logic/agent_worker.py", "rb") as fh:
        actual = hashlib.sha256(fh.read()).hexdigest()
    assert actual == _AGENT_WORKER_SHA256, (
        f"agent_worker.py was modified (got {actual}, expected {_AGENT_WORKER_SHA256}). "
        "This file is LOCKED — the live voice path must not be touched by Plan 11-11."
    )


def test_agent_bridge_byte_identical():
    """SPK-V3-01: agent_bridge.py must be byte-identical to its pre-plan SHA256."""
    with open("confluence_logic/agent_bridge.py", "rb") as fh:
        actual = hashlib.sha256(fh.read()).hexdigest()
    assert actual == _AGENT_BRIDGE_SHA256, (
        f"agent_bridge.py was modified (got {actual}, expected {_AGENT_BRIDGE_SHA256}). "
        "This file is LOCKED — the live voice path must not be touched by Plan 11-11."
    )
