"""Unit tests for the edge-tts LiveKit TTS adapter (no network).

Exercises EdgeChunkedStream._run in isolation with a mocked edge_tts.Communicate
and a mocked AudioEmitter, so we verify the adapter's contract (decode-by-mime,
push audio chunks, skip non-audio events, flush) without hitting Microsoft's
endpoint or constructing the heavy ChunkedStream base (which auto-starts tasks).
"""

import asyncio
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import edge_tts_plugin  # noqa: E402


def test_edge_tts_capabilities_and_format():
    tts = edge_tts_plugin.EdgeTTS(voice="en-US-AriaNeural")
    assert tts.sample_rate == 24000
    assert tts.num_channels == 1
    assert tts.capabilities.streaming is False
    assert tts.provider == "edge-tts"
    assert tts.model == "en-US-AriaNeural"


def test_run_pushes_audio_chunks_and_flushes():
    tts = edge_tts_plugin.EdgeTTS(voice="en-US-AriaNeural")

    # Build the stream WITHOUT the heavy base __init__ (which auto-starts network
    # tasks); we only want to test _run's logic.
    stream = object.__new__(edge_tts_plugin.EdgeChunkedStream)
    stream._tts = tts
    stream._input_text = "hello world"

    emitter = MagicMock()

    class FakeCommunicate:
        def __init__(self, text, voice, **kwargs):
            self.text = text
            self.voice = voice

        async def stream(self):
            yield {"type": "audio", "data": b"\x00\x01"}
            yield {"type": "WordBoundary", "offset": 0, "text": "hello"}
            yield {"type": "audio", "data": b"\x02\x03"}

    with patch.object(edge_tts_plugin.edge_tts, "Communicate", FakeCommunicate):
        asyncio.run(stream._run(emitter))

    # Decodes by mime type (MP3) — no local model, AudioEmitter handles PCM.
    emitter.initialize.assert_called_once()
    init_kwargs = emitter.initialize.call_args.kwargs
    assert init_kwargs["mime_type"] == "audio/mp3"
    assert init_kwargs["sample_rate"] == 24000
    assert init_kwargs["num_channels"] == 1

    # Only the two audio events are pushed; WordBoundary is skipped.
    assert emitter.push.call_count == 2
    emitter.push.assert_any_call(b"\x00\x01")
    emitter.push.assert_any_call(b"\x02\x03")
    emitter.flush.assert_called_once()


def test_run_passes_voice_to_communicate():
    tts = edge_tts_plugin.EdgeTTS(voice="en-GB-RyanNeural")
    stream = object.__new__(edge_tts_plugin.EdgeChunkedStream)
    stream._tts = tts
    stream._input_text = "test"

    seen = {}

    class FakeCommunicate:
        def __init__(self, text, voice, **kwargs):
            seen["text"] = text
            seen["voice"] = voice

        async def stream(self):
            yield {"type": "audio", "data": b"\x10\x11"}

    with patch.object(edge_tts_plugin.edge_tts, "Communicate", FakeCommunicate):
        asyncio.run(stream._run(MagicMock()))

    assert seen == {"text": "test", "voice": "en-GB-RyanNeural"}
