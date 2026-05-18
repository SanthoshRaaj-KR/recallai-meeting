"""
LiveKit Native Agent Worker for Jarvis.

Run with:
    python -m confluence_logic.agent_worker dev      # hot reload
    python -m confluence_logic.agent_worker start    # production

FULL PIPELINE (what actually happens, in order):
  1. Recall headless Chrome loads bot.html via output_media API
  2. bot.html pubRoom joins LiveKit as "recall-browser-{session_id}"
     (pub_token MUST grant this exact identity — see IDENTITY CONTRACT below)
  3. pubRoom publishes getUserMedia() audio = mixed meeting audio
  4. AgentSession STT (Deepgram Nova-3) subscribes to that participant's audio
  5. VAD (Silero) detects speech segments, gates STT
  6. Deepgram streams partial + final transcripts in real-time
  7. on_user_turn_completed fires — wake word gate:
       no wake word  → clear message content → llm_node returns None → total silence
       bare "Jarvis" → clear message content + await session.say(ack mp3) → silence
       "Jarvis, X"   → rewrite message content to X only → llm_node dispatches to LLM
  8. llm_node → gpt-4o-mini streams tokens (preemptive_generation=True speeds this up)
  9. tts_node → yields pre-recorded ack MP3 frames immediately (local, ~0ms)
              → then Cartesia Sonic-3 streams TTS frames in parallel
 10. AgentSession publishes TTS audio as agent's own participant track
 11. bot.html subscriber room TrackSubscribed fires for agent's identity
 12. <audio> element plays → Recall bot's Chrome plays audio → meeting hears Jarvis

LATENCY BUDGET (what you should see in logs):
  VAD end-of-speech:            30–80 ms
  Deepgram final transcript:    100–200 ms
  on_user_turn_completed:       ~1 ms  (regex only, no I/O)
  LLM TTFT (gpt-4o-mini):       200–400 ms
  Ack MP3 first frame:          ~0 ms  (local disk)
  Cartesia TTFA:                40–90 ms
  ── Total perceived latency ── ~400–800 ms after user stops speaking

IDENTITY CONTRACT (tokens your backend must mint):
  pub_token  → identity MUST be: "recall-browser-{session_id}"
               This is the participant the AgentSession STT subscribes to.
               If this is wrong, the agent is completely deaf.

  token      → identity can be anything (e.g. "recall-listener-{session_id}")
               This room only plays back audio — it just needs to be in the room.

  Agent TTS  → published under the agent's own LiveKit participant identity.
               In bot.html, update the TrackSubscribed filter to match this identity.
               The safest approach: filter for NOT startsWith("recall-") instead of
               startsWith("jarvis-publisher-"), since the agent identity is auto-assigned.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import AsyncIterable

from dotenv import load_dotenv
from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    JobProcess,
    ModelSettings,
    TurnHandlingOptions,
    cli,
    inference,
    llm,
)
from livekit.agents.utils.codecs import AudioStreamDecoder
from livekit.agents.voice import room_io
from livekit.plugins import silero

from confluence_logic.agent_bridge import JARVIS_TOOLS
from confluence_logic.audio_cache import (
    get_random_quick_ack_audio,
    get_random_query_ack_audio,
    load_audio_cache,
)

_MODULE_DIR = Path(__file__).resolve().parent
load_dotenv(_MODULE_DIR.parent / ".env")
load_dotenv(_MODULE_DIR / ".env", override=True)

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)

# ── Config ────────────────────────────────────────────────────────────────────
JARVIS_AGENT_WORKER_NAME = os.getenv("JARVIS_AGENT_WORKER_NAME", "jarvis-agent").strip()
JARVIS_LK_TTS_PROVIDER   = os.getenv("JARVIS_LK_TTS_PROVIDER", "cartesia").strip().lower()
JARVIS_LK_TTS_VOICE      = os.getenv("JARVIS_LK_TTS_VOICE", "9626c31c-bec5-4cca-baa8-f8ba9e84c8bc").strip()
JARVIS_LK_LLM            = os.getenv("JARVIS_LK_LLM", "openai/gpt-4o-mini").strip()

# ── Wake word regex ───────────────────────────────────────────────────────────
_WAKE_ALIASES = r"(?:jarvis|jarvas|jervis|jarvus|jarves|jarvi|jarv)"
_CUSTOM_WAKE  = os.getenv("JARVIS_WAKE_ALIASES", "").strip()
if _CUSTOM_WAKE:
    _WAKE_ALIASES = rf"(?:jarvis|jarvas|jervis|jarvus|jarves|jarvi|jarv|{_CUSTOM_WAKE})"

_WAKE_PATTERN = re.compile(
    rf"(?:hey|yo|ok|hi|okay)[,\s]+{_WAKE_ALIASES}[,.\s!?]*\s*(.*)",
    re.IGNORECASE | re.DOTALL,
)


def _extract_query(text: str) -> str | None:
    """
    Returns:
      None  → no wake word detected  (silence — don't touch LLM)
      ""    → bare wake word only    (play ack, no LLM)
      "..."  → query after wake word  (dispatch to LLM, strip wake word)
    """
    m = _WAKE_PATTERN.search(text.strip())
    return m.group(1).strip() if m else None


# ── Audio helper ──────────────────────────────────────────────────────────────
async def _ack_audio_frames(ack_bytes: bytes) -> AsyncIterable:
    """Decode pre-recorded MP3 bytes into AudioFrames for session.say()."""
    decoder = AudioStreamDecoder(format="mp3", sample_rate=48000, num_channels=1)
    decoder.push(ack_bytes)
    decoder.end_input()
    async for frame in decoder:
        yield frame
    await decoder.aclose()


# ── Agent ─────────────────────────────────────────────────────────────────────
class JarvisAgent(Agent):
    """
    Wake-word gated voice agent for Jarvis.

    Three-stage pipeline:
      on_user_turn_completed → rewrites new_message.content (wake gate)
      llm_node               → returns None on empty content (total silence)
      tts_node               → ack MP3 first, then Cartesia TTS stream
    """

    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "You are Jarvis, an AI meeting assistant. You help teams update Confluence "
                "documentation based on what is discussed in meetings. Keep responses concise "
                "— you are speaking aloud in a live meeting. Do not use markdown, asterisks, "
                "bullet points, or emojis. One or two sentences unless more detail is essential."
            ),
            tools=JARVIS_TOOLS,
        )
        # FIFO queue: llm_node puts True, tts_node gets it.
        # One signal per LLM turn — no shared-bool race condition.
        self._ack_q: asyncio.Queue[bool] = asyncio.Queue()
        self._turn_t0: float = 0.0   # perf_counter at start of turn for latency logs

    async def on_enter(self) -> None:
        logger.info("✅ JarvisAgent entered room — listening for 'Hey Jarvis'")

    # ── Wake word gate ────────────────────────────────────────────────────────
    async def on_user_turn_completed(
        self,
        turn_ctx: llm.ChatContext,
        new_message: llm.ChatMessage,
    ) -> None:
        """
        Called immediately after STT finalises a transcript, before llm_node.

        Key facts from LiveKit source:
          - Returning early from this function does NOT suppress the LLM.
          - The ONLY way to suppress the LLM is to clear new_message.content
            so llm_node sees empty text and returns None.
          - new_message.content = [] is safe to mutate here.
        """
        self._turn_t0 = time.perf_counter()
        raw = new_message.text_content or ""

        query = _extract_query(raw)

        # ── Case 1: no wake word — regular meeting speech ─────────────────
        if query is None:
            logger.info("🔇 SILENT   (no wake word): %.80r", raw)
            new_message.content = []
            return

        # ── Case 2: bare wake word — ready prompt, no LLM ───────────────
        if not query:
            logger.info("👂 BARE WAKE — saying ready prompt")
            new_message.content = []
            await self.session.say("Yes, how can I help you?", add_to_chat_ctx=False)
            return

        # ── Case 3: full query — strip wake word, send to LLM ────────────
        logger.info(
            "🎯 WAKE QUERY (Δ=%.0fms since turn start): %.100r",
            (time.perf_counter() - self._turn_t0) * 1000,
            query,
        )
        # Rewrite to query-only — LLM never sees "hey jarvis"
        new_message.content = [query]
        # Signal ack here (after wake word confirmed) — not in llm_node, which
        # fires speculatively before this gate with preemptive_generation=True.
        self._ack_q.put_nowait(True)

    # ── LLM node ──────────────────────────────────────────────────────────────
    def llm_node(
        self,
        chat_ctx: llm.ChatContext,
        tools: list[llm.Tool],
        model_settings: ModelSettings,
    ):
        """
        Returning None → framework skips LLM + TTS entirely (complete silence).
        Returning a stream → framework calls tts_node with that text stream.

        We check the last user message content that on_user_turn_completed set.
        """
        last_user = next(
            (m for m in reversed(chat_ctx.messages()) if m.role == "user"), None
        )

        if last_user is None:
            # Shouldn't happen — be safe and delegate to default
            return Agent.default.llm_node(self, chat_ctx, tools, model_settings)

        text = (last_user.text_content or "").strip()

        if not text:
            # Content cleared by on_user_turn_completed → complete silence
            logger.debug("🚫 LLM suppressed (empty content)")
            return None

        logger.info(
            "🧠 LLM dispatching (Δ=%.0fms since wake): '%.80s'",
            (time.perf_counter() - self._turn_t0) * 1000,
            text,
        )
        return Agent.default.llm_node(self, chat_ctx, tools, model_settings)

    # ── TTS node ──────────────────────────────────────────────────────────────
    async def tts_node(
        self,
        text: AsyncIterable[str],
        model_settings: ModelSettings,
    ):
        """
        Fastest possible response path:
          1. Immediately yield pre-recorded ack MP3 frames (local disk, ~0ms latency)
             → user hears something while Cartesia warms up
          2. Stream Cartesia Sonic-3 TTS frames as they arrive (40–90ms TTFA)

        Only called by the framework when llm_node returned a non-None stream.
        The _ack_q has one entry when on_user_turn_completed confirmed a wake query.
        """
        try:
            should_ack = self._ack_q.get_nowait()
        except asyncio.QueueEmpty:
            should_ack = False
            logger.warning("⚠️  TTS ack queue empty — skipping ack")

        # Step 1: instant local ack (excludes "busy"/"yes" — those imply the wrong state)
        if should_ack:
            ack = get_random_query_ack_audio()
            if ack:
                name, data = ack
                logger.info(
                    "🔊 TTS ack '%s' (Δ=%.0fms since wake)",
                    name,
                    (time.perf_counter() - self._turn_t0) * 1000,
                )
                async for frame in _ack_audio_frames(data):
                    yield frame

        # Step 2: stream Cartesia TTS
        frame_count = 0
        async for frame in Agent.default.tts_node(self, text, model_settings):
            if frame_count == 0:
                logger.info(
                    "🗣️  TTS first Cartesia frame (Δ=%.0fms since wake)",
                    (time.perf_counter() - self._turn_t0) * 1000,
                )
            frame_count += 1
            yield frame

        logger.info(
            "✅ TTS done  frames=%d  total_Δ=%.0fms",
            frame_count,
            (time.perf_counter() - self._turn_t0) * 1000,
        )


# ── Worker ────────────────────────────────────────────────────────────────────
server = AgentServer()


def prewarm(proc: JobProcess) -> None:
    """Load VAD model and audio cache once at worker startup."""
    proc.userdata["vad"] = silero.VAD.load()
    load_audio_cache()
    logger.info("✅ Prewarm done — VAD + audio cache ready")


server.setup_fnc = prewarm


@server.rtc_session(agent_name=JARVIS_AGENT_WORKER_NAME)
async def entrypoint(ctx: JobContext) -> None:
    # ── 1. JOIN THE ROOM ──────────────────────────────────────────────────────
    # CRITICAL BUG FIX: ctx.connect() was missing in the old code.
    # Without this the agent process registers with LiveKit but never actually
    # joins the room, so it never subscribes to any audio tracks → completely deaf.
    await ctx.connect()

    # ── 2. Parse session_id from job metadata ─────────────────────────────────
    session_id = ""
    try:
        meta = json.loads(ctx.job.metadata or "{}")
        session_id = (meta.get("session_id") or "").strip()
    except (ValueError, TypeError) as exc:
        logger.warning("Bad job metadata (%s) — fallback to first participant", exc)

    logger.info(
        "🚀 AgentSession starting  agent=%s  llm=%s  tts=%s/sonic-3  session_id=%s",
        JARVIS_AGENT_WORKER_NAME, JARVIS_LK_LLM, JARVIS_LK_TTS_PROVIDER,
        session_id or "<none — first participant>",
    )

    # ── 3. Build AgentSession with fastest possible settings ──────────────────
    session = AgentSession(
        # Deepgram Nova-3 via LiveKit Inference
        # LiveKit Inference co-locates STT with your agent → lowest RTT
        # Has a Mumbai node → extra-low latency for Indian region users
        stt=inference.STT(model="deepgram/nova-3", language="multi"),

        # gpt-4o-mini: fastest OpenAI TTFT at conversational response lengths
        llm=inference.LLM(JARVIS_LK_LLM),

        # Cartesia Sonic-3: 40–90ms time-to-first-audio, best-in-class for latency
        tts=inference.TTS(
            f"{JARVIS_LK_TTS_PROVIDER}/sonic-3",
            voice=JARVIS_LK_TTS_VOICE,
        ),

        vad=ctx.proc.userdata["vad"],

        turn_handling=TurnHandlingOptions(
            endpointing={
                # Start processing 300ms after silence — aggressive but correct
                # for meeting context where people speak in complete sentences
                "min_delay": 0.3,
                # Never wait more than 1.5s — prevents hanging on trailing silence
                "max_delay": 1.5,
            },
            interruption={
                # Adaptive interruption model (livekit-agents >= 1.5):
                # distinguishes real interruptions from coughs, "mm-hmm", etc.
                "resume_false_interruption": True,
                "false_interruption_timeout": 1.2,
            },
        ),

        # SPEED: start LLM before end-of-turn is fully confirmed.
        # With wake-word gating via message content rewrite, this is safe:
        #   - no-wake speech: on_user_turn_completed clears content → speculative LLM discarded
        #   - wake speech: content set correctly → speculative output is used directly
        # Old code had preemptive_generation=False which added 200-400ms of unnecessary latency.
        preemptive_generation=True,

        tts_text_transforms=["filter_emoji", "filter_markdown"],
    )

    # ── 4. Start session linked to the correct audio participant ──────────────
    if session_id:
        # IDENTITY CONTRACT:
        # pub_token in bot.html MUST grant identity = "recall-browser-{session_id}"
        # That is the participant the AgentSession subscribes to for STT audio.
        # If pub_token grants ANY other identity, the agent hears nothing.
        logger.info("🎧 STT linked to participant: recall-browser-%s", session_id)
        await session.start(
            agent=JarvisAgent(),
            room=ctx.room,
            room_options=room_io.RoomOptions(
                participant_identity=f"recall-browser-{session_id}",
            ),
        )
    else:
        logger.warning("⚠️  No session_id — linking to first participant (dev mode only)")
        await session.start(agent=JarvisAgent(), room=ctx.room)


if __name__ == "__main__":
    cli.run_app(server)