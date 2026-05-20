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
  8. llm_node → gpt-5.4-nano streams tokens (preemptive_generation=True speeds this up)
  9. tts_node → yields pre-recorded ack MP3 frames immediately (local, ~0ms)
              → then Cartesia Sonic-3 streams TTS frames in parallel
 10. AgentSession publishes TTS audio as agent's own participant track
 11. bot.html subscriber room TrackSubscribed fires for agent's identity
 12. <audio> element plays → Recall bot's Chrome plays audio → meeting hears Jarvis

LATENCY BUDGET (what you should see in logs):
  VAD end-of-speech:            30–80 ms
  Deepgram final transcript:    100–200 ms
  on_user_turn_completed:       ~1 ms  (regex only, no I/O)
  LLM TTFT (gpt-5.4-nano):      150–250 ms
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
import functools
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, AsyncIterable

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
from livekit.agents.voice import room_io
from livekit.plugins import assemblyai, silero
from livekit import rtc
from livekit.agents import stt as _lk_stt
from livekit.agents.utils.codecs.decoder import AudioStreamDecoder
from livekit.agents.llm import FunctionTool

from confluence_logic.audio_cache import get_wake_ack_audio, get_random_query_ack_audio, load_audio_cache
from confluence_logic.agent_bridge import JARVIS_TOOLS

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
JARVIS_LK_LLM            = os.getenv("JARVIS_LK_LLM", "openai/gpt-4.1-mini").strip()
# Base URL of the Jarvis FastAPI server — used to POST LiveKit transcripts for
# the transcript_log (replaces Recall BYOB transcript WebSocket, Phase 7).
JARVIS_API_BASE          = os.getenv("WEBHOOK_URL", "").rstrip("/")

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


# ── Transcript logging (Phase 7) ─────────────────────────────────────────────
async def _post_transcript(session_id: str, text: str) -> None:
    """POST a final STT transcript to the Jarvis server for transcript_log population.

    Fires for EVERY utterance (not just wake-word ones) so the post-meeting
    Confluence review pipeline has full meeting context. Non-fatal on failure.
    """
    if not JARVIS_API_BASE or not text.strip():
        return
    try:
        import requests as _req
        await asyncio.to_thread(
            _req.post,
            f"{JARVIS_API_BASE}/livekit-transcript/{session_id}",
            json={"text": text, "speaker": "Meeting"},
            timeout=3,
        )
    except Exception as exc:
        logger.debug("transcript post failed (session=%s): %s", session_id, exc)


# ── Tool-call ack helpers ─────────────────────────────────────────────────────

# Instant tools return in <100ms — an ack would still be playing when the answer starts.
_INSTANT_TOOL_NAMES: frozenset[str] = frozenset({"get_current_datetime"})

# Dedup: fire at most one ack per agent turn (keyed by speech_handle.id).
_tool_ack_fired: set[str] = set()


async def _safe_play_query_ack(session: AgentSession, mp3_bytes: bytes) -> None:
    """Play a query-ack clip, swallowing all errors (fire-and-forget)."""
    try:
        await session.say("", audio=_mp3_bytes_to_frames(mp3_bytes), add_to_chat_ctx=False)
    except Exception as exc:
        logger.debug("tool ack play error: %s", exc)


def _maybe_fire_tool_ack(ctx: Any) -> None:
    """Fire a random query-ack clip once per speech turn — non-fatal."""
    try:
        speech_id = ctx.speech_handle.id
        if speech_id in _tool_ack_fired:
            return
        _tool_ack_fired.add(speech_id)
        if len(_tool_ack_fired) > 500:
            _tool_ack_fired.clear()
        cached = get_random_query_ack_audio()
        if cached is None:
            return
        _, mp3_bytes = cached
        asyncio.create_task(_safe_play_query_ack(ctx.session, mp3_bytes))
        logger.info("🔔 TOOL ACK firing (%s...)", speech_id[:8])
    except Exception as exc:
        logger.debug("_maybe_fire_tool_ack error: %s", exc)


def _add_tool_call_ack(tool: Any) -> Any:
    """Return a FunctionTool that fires a query-ack clip before the original executes."""
    if not isinstance(tool, FunctionTool) or tool.info.name in _INSTANT_TOOL_NAMES:
        return tool

    original_func = tool._func

    @functools.wraps(original_func)
    async def _with_ack(*args: Any, **kwargs: Any) -> Any:
        for arg in (*args, *kwargs.values()):
            if hasattr(arg, "speech_handle") and hasattr(arg, "session"):
                _maybe_fire_tool_ack(arg)
                break
        return await original_func(*args, **kwargs)

    return FunctionTool(_with_ack, tool.info)


_JARVIS_TOOLS_WITH_ACK = [_add_tool_call_ack(t) for t in JARVIS_TOOLS]


# == PCM ack helpers (Phase 7 D-05) ============================================
async def _mp3_bytes_to_frames(mp3_bytes: bytes):
    """Decode MP3 bytes to a 48kHz mono rtc.AudioFrame async iterator.

    Used by _play_ack_frames to feed AgentSession.say(audio=...) directly,
    bypassing the Cartesia TTS engine for pre-cached acknowledgements.
    """
    decoder = AudioStreamDecoder(
        sample_rate=48000,
        num_channels=1,
        format="mp3",
    )
    decoder.push(mp3_bytes)
    decoder.end_input()
    async for frame in decoder:
        yield frame


async def _play_ack_frames(session: AgentSession, key: str = "yes") -> None:
    """Phase 7 D-05: play a pre-cached ack clip with sub-5ms perceived latency.

    Pulls MP3 bytes from audio_cache, decodes to rtc.AudioFrame via AudioStreamDecoder,
    and feeds them into AgentSession.say(audio=...) - bypasses Cartesia TTS entirely
    (<5ms vs ~150ms round-trip).

    Falls back to live TTS session.say("Yes?", ...) on cache miss (e.g. yes.mp3 absent).
    """
    cached = get_wake_ack_audio()  # Returns ("yes", mp3_bytes) or None - key param is currently informational.
    if cached is None:
        logger.warning("_play_ack_frames: cache miss for %r - falling back to TTS", key)
        await session.say("Yes?", add_to_chat_ctx=False)
        return

    _, mp3_bytes = cached
    await session.say(
        "",  # empty text - `audio` overrides TTS routing
        audio=_mp3_bytes_to_frames(mp3_bytes),
        add_to_chat_ctx=False,
    )


# ── Agent ─────────────────────────────────────────────────────────────────────
class JarvisAgent(Agent):
    """
    Wake-word gated voice agent for Jarvis.

    Three-stage pipeline:
      on_user_turn_completed → rewrites new_message.content (wake gate)
      llm_node               → returns None on empty content (total silence)
      tts_node               → ack MP3 first, then Cartesia TTS stream
    """

    def __init__(self, session_id: str = "") -> None:
        super().__init__(
            instructions=(
                "You are Jarvis, an AI meeting assistant. You help teams update Confluence "
                "documentation and answer questions during live meetings. Keep responses concise "
                "— you are speaking aloud. Do not use markdown, asterisks, bullet points, or emojis. "
                "One or two sentences unless more detail is truly needed.\n\n"
                "TOOL USAGE RULES:\n"
                "- For ANY question about current date, time, or day: call get_current_datetime.\n"
                "- For ANY factual question, general knowledge question, or anything you are not "
                "100% certain about: call answer_general_question_tool with the user's question.\n"
                "- For meeting summaries, opinions, or action items: use the meeting tools.\n"
                "- For Confluence page operations: use the confluence tools.\n"
                "- Never answer factual questions from memory alone — use answer_general_question_tool."
            ),
            tools=_JARVIS_TOOLS_WITH_ACK,
        )
        self._session_id: str = session_id
        self._turn_t0: float = 0.0
        self._listening_mode: bool = False
        self._listening_since: float = 0.0
        # Partial-transcript wake detection (fires ack before VAD silence)
        self._partial_wake_detected: bool = False
        self._partial_wake_t: float = 0.0

    _LISTENING_TIMEOUT_S: float = 8.0  # reset listening mode if silent this long

    async def on_enter(self) -> None:
        logger.info("✅ JarvisAgent entered room — listening for 'Hey Jarvis'")

    # ── Partial-transcript wake detection ──────────────────────────────────────
    async def stt_node(
        self,
        audio: AsyncIterable[rtc.AudioFrame],
        model_settings: ModelSettings,
    ) -> AsyncIterable[_lk_stt.SpeechEvent | str]:
        """Spy on INTERIM_TRANSCRIPT events to detect 'Hey Jarvis' before VAD fires.

        When the wake word appears in a partial, the 'Yes?' ack is played immediately —
        shaving 100-200ms vs waiting for the final transcript + VAD silence window.
        """
        async for event in Agent.default.stt_node(self, audio, model_settings):
            if (
                not self._partial_wake_detected
                and event.type == _lk_stt.SpeechEventType.INTERIM_TRANSCRIPT
            ):
                text = event.alternatives[0].text if event.alternatives else ""
                if _WAKE_PATTERN.search(text):
                    self._partial_wake_detected = True
                    self._partial_wake_t = time.perf_counter()
                    logger.info("⚡ PARTIAL WAKE detected in interim: %.60r", text)
                    try:
                        asyncio.create_task(_play_ack_frames(self.session, "yes"))
                    except Exception as exc:
                        logger.debug("partial wake ack task failed: %s", exc)
            yield event

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
        # Snapshot and reset partial-wake state at the start of every turn.
        partial_detected = self._partial_wake_detected
        partial_t = self._partial_wake_t
        self._partial_wake_detected = False

        raw = new_message.text_content or ""

        if raw.strip() and self._session_id:
            asyncio.create_task(_post_transcript(self._session_id, raw))

        query = _extract_query(raw)

        # ── Case 1: listening mode — user said bare wake, now asking query ───
        if query is None and self._listening_mode:
            elapsed = time.perf_counter() - self._listening_since
            self._listening_mode = False
            if elapsed > self._LISTENING_TIMEOUT_S:
                logger.info(
                    "🔇 SILENT (listening mode timed out after %.1fs): %.80r",
                    elapsed, raw,
                )
                new_message.content = []
                return
            q = raw.strip()
            if not q:
                logger.info("🔇 SILENT (listening mode — empty transcript)")
                new_message.content = []
                return
            logger.info("👂 LISTENING QUERY (Δ=%.0fms): %.100r", elapsed * 1000, q)
            new_message.content = [q]
            return

        # ── Case 2: no wake word — regular meeting speech ─────────────────
        if query is None:
            logger.info("🔇 SILENT   (no wake word): %.80r", raw)
            new_message.content = []
            return

        # ── Case 3: bare wake word — ack + enter listening mode ──────────
        if not query:
            elapsed_partial_ms = (
                (time.perf_counter() - partial_t) * 1000 if partial_detected else None
            )
            logger.info(
                "👂 BARE WAKE — listening mode (ack %s)",
                f"pre-fired {elapsed_partial_ms:.0f}ms ago via partial" if elapsed_partial_ms is not None else "firing now",
            )
            self._listening_mode = True
            self._listening_since = time.perf_counter()
            new_message.content = []
            if not partial_detected:
                # Partial detection already played the ack; skip to avoid double-play.
                await _play_ack_frames(self.session, "yes")
            return

        # ── Case 4: full inline query — strip wake word, send to LLM ─────
        logger.info(
            "🎯 WAKE QUERY (Δ=%.0fms since turn start%s): %.100r",
            (time.perf_counter() - self._turn_t0) * 1000,
            f", partial Δ={((time.perf_counter() - partial_t) * 1000):.0f}ms" if partial_detected else "",
            query,
        )
        self._listening_mode = False
        new_message.content = [query]

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
        """Stream Cartesia TTS directly. No pre-recorded acks — simplest path for debugging."""

        async def _logged_text(stream: AsyncIterable[str]) -> AsyncIterable[str]:
            chunks: list[str] = []
            async for chunk in stream:
                chunks.append(chunk)
                yield chunk
            if chunks:
                logger.info("🤖 LLM: %s", "".join(chunks))

        frame_count = 0
        async for frame in Agent.default.tts_node(self, _logged_text(text), model_settings):
            if frame_count == 0:
                logger.info(
                    "🗣️  TTS first frame (Δ=%.0fms since wake)",
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
    proc.userdata["vad"] = silero.VAD.load()
    n_cached = load_audio_cache()
    logger.info("✅ Prewarm done — VAD ready, %d ack audio files cached", n_cached)


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
        "🚀 AgentSession starting  agent=%s  llm=%s  tts=%s/sonic-turbo  session_id=%s",
        JARVIS_AGENT_WORKER_NAME, JARVIS_LK_LLM, JARVIS_LK_TTS_PROVIDER,
        session_id or "<none — first participant>",
    )

    # ── 3. Build AgentSession with fastest possible settings ──────────────────
    session = AgentSession(
        # AssemblyAI Universal-3 Pro Streaming (plugin-direct, NOT via inference.STT — Inference does not expose keyterms_prompt).
        # keyterms_prompt locks "Jarvis" / "Hey Jarvis" recognition in noisy meeting audio.
        # language_detection=False eliminates the 30–80 ms multilingual overhead Deepgram added with its multilingual mode.
        stt=assemblyai.STT(
            model="u3-rt-pro",
            keyterms_prompt=["Jarvis", "Hey Jarvis"],
            language_detection=False,
        ),

        # gpt-5.4-nano: lowest TTFT at conversational response lengths (JARVIS_LK_LLM env override)
        llm=inference.LLM(JARVIS_LK_LLM),

        # Cartesia Sonic-Turbo: ~40ms time-to-first-audio (vs ~90ms for Sonic-3); voice UUID unchanged (cross-model compatible).
        tts=inference.TTS(
            f"{JARVIS_LK_TTS_PROVIDER}/sonic-turbo",
            voice=JARVIS_LK_TTS_VOICE,
        ),

        vad=ctx.proc.userdata["vad"],

        turn_handling=TurnHandlingOptions(
            endpointing={
                # Phase 7 D-04: 150ms floor (was 300ms). Aggressive but safe — premature
                # cuts on non-wake utterances are discarded by the wake-word regex anyway.
                "min_delay": 0.15,
                # Unchanged — prevents hanging on trailing silence
                "max_delay": 1.5,
            },
            interruption={
                # Adaptive interruption model (livekit-agents >= 1.5):
                # distinguishes real interruptions from coughs, "mm-hmm", etc.
                "resume_false_interruption": True,
                # Phase 7 D-04: 0.6s (was 1.2s) — halves resume-from-false-interruption latency.
                "false_interruption_timeout": 0.6,
            },
        ),

        # DISABLED: preemptive_generation=True caused audio glitching ("kirch kirch kirching").
        # Our wake-word rewrite in on_user_turn_completed ALWAYS changes the message content,
        # so the speculative LLM is ALWAYS discarded → framework cuts speculative TTS mid-frame
        # → audible glitch at the start of every response. The ack MP3 already covers the
        # latency gap, so preemptive_generation adds no benefit here.
        preemptive_generation=False,

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
            agent=JarvisAgent(session_id=session_id),
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