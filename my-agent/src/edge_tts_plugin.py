"""edge-tts TTS plugin for LiveKit Agents (free Microsoft Edge voices).

Replaces the LiveKit Cloud ``inference.TTS`` path when self-hosting LiveKit.
No model runs locally: edge-tts streams MP3 from Microsoft's public endpoint and
the LiveKit ``AudioEmitter`` decodes it to PCM frames for the agent session.

Voice is configurable via JARVIS_EDGE_TTS_VOICE (default: en-US-AriaNeural).
List voices with: ``uv run edge-tts --list-voices``.
"""

from __future__ import annotations

import os

import edge_tts
from livekit.agents.tts import TTS, AudioEmitter, ChunkedStream, TTSCapabilities
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, APIConnectOptions
from livekit.agents.utils import shortuuid

DEFAULT_VOICE = os.getenv("JARVIS_EDGE_TTS_VOICE", "en-US-AriaNeural")
# edge-tts streams MP3; Microsoft neural voices are 24 kHz mono.
_SAMPLE_RATE = 24000
_NUM_CHANNELS = 1


class EdgeTTS(TTS):
    """Non-streaming TTS backed by Microsoft Edge ``edge-tts`` (free)."""

    def __init__(self, *, voice: str = DEFAULT_VOICE, rate: str | None = None) -> None:
        super().__init__(
            capabilities=TTSCapabilities(streaming=False),
            sample_rate=_SAMPLE_RATE,
            num_channels=_NUM_CHANNELS,
        )
        self._voice = voice
        self._rate = rate

    @property
    def model(self) -> str:
        return self._voice

    @property
    def provider(self) -> str:
        return "edge-tts"

    def synthesize(
        self,
        text: str,
        *,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> EdgeChunkedStream:
        return EdgeChunkedStream(tts=self, input_text=text, conn_options=conn_options)


class EdgeChunkedStream(ChunkedStream):
    """Synthesizes the full input via edge-tts, pushing MP3 chunks to the emitter."""

    def __init__(
        self, *, tts: EdgeTTS, input_text: str, conn_options: APIConnectOptions
    ) -> None:
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)

    async def _run(self, output_emitter: AudioEmitter) -> None:
        tts: EdgeTTS = self._tts  # type: ignore[assignment]
        output_emitter.initialize(
            request_id=shortuuid(),
            sample_rate=tts.sample_rate,
            num_channels=tts.num_channels,
            mime_type="audio/mp3",
            stream=False,
        )

        kwargs: dict[str, str] = {}
        if tts._rate:
            kwargs["rate"] = tts._rate
        communicate = edge_tts.Communicate(self._input_text, tts._voice, **kwargs)

        async for chunk in communicate.stream():
            if chunk["type"] == "audio" and chunk.get("data"):
                output_emitter.push(chunk["data"])

        output_emitter.flush()
