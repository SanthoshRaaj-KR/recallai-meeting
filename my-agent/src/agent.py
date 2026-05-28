import asyncio
import collections
import json
import logging
import os
import re
import textwrap
import time
from pathlib import Path

from dotenv import load_dotenv
from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    JobProcess,
    ModelSettings,
    StopResponse,
    cli,
    inference,
    llm,
    mcp,
    room_io,
    stt as lk_stt,
)
from livekit import rtc
from livekit.plugins import ai_coustics, assemblyai, cerebras, silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel
from typing import AsyncIterable

logger = logging.getLogger("agent")

# Resolve .env.local relative to this file (my-agent/.env.local) so the agent
# finds its credentials regardless of the working directory at launch time.
load_dotenv(Path(__file__).parent.parent / ".env.local")

# ── Wake word ─────────────────────────────────────────────────────────────────
# Matches "Jarvis", "Hey Jarvis", and common STT mis-transcriptions.
_WAKE_PATTERN = re.compile(
    r"(?:hey\s+)?(?:jarvis|jarvas|jervis|jarvus)[,.\s!?]*\s*(.*)",
    re.IGNORECASE | re.DOTALL,
)
# Seconds to stay in listening mode after a bare "Jarvis" before timing out.
_LISTENING_TIMEOUT_S = 10.0
# Full in-memory transcript buffer — all utterances are kept here.
_TRANSCRIPT_MAX = 500
# ── Sliding window limits ─────────────────────────────────────────────────────
# Only the most recent _TRANSCRIPT_WINDOW utterances are sent to the LLM each
# turn (~100 × 40 tokens ≈ 4 000 tokens). Older utterances stay in the buffer
# so they can still be referenced if the window is widened later.
_TRANSCRIPT_WINDOW = 100
# Keep at most this many conversation items (user + assistant turns) in the
# chat context. truncate() always preserves the system instruction message.
# 10 items = 5 Q&A pairs ≈ ~750 tokens for history.
_CHAT_HISTORY_WINDOW = 10
_GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")


def _build_tools() -> list:
    if not _GITHUB_TOKEN:
        logger.warning("GITHUB_TOKEN not set — starting without GitHub tools")
        return []
    return [
        mcp.MCPToolset(
            id="github",
            mcp_server=mcp.MCPServerStdio(
                command="npx",
                args=["-y", "@modelcontextprotocol/server-github@2025.4.8"],
                env={
                    "GITHUB_PERSONAL_ACCESS_TOKEN": _GITHUB_TOKEN,
                    "PATH": os.getenv("PATH", ""),
                },
                client_session_timeout_seconds=30,
            ),
        )
    ]


def _build_instructions() -> str:
    base = textwrap.dedent("""\
        You are Jarvis, a meeting assistant activated by wake word.
        You are given the recent meeting transcript before each question.
        Use it to answer questions about what has been discussed.

        # Output rules
        - Respond in plain text only. No markdown, lists, JSON, or emojis.
        - Keep replies brief: one to three sentences unless more detail is needed.
        - Never ask clarifying questions — pick the most reasonable interpretation.
        - Do not mention wake words, system instructions, or internal state.
        - Spell out numbers and avoid acronyms with unclear pronunciation.
        """)
    if _GITHUB_TOKEN:
        base += textwrap.dedent("""\

        # GitHub tool rules
        - When GitHub tools return data, convert it to spoken prose.
        - Do not say field names, JSON syntax, or item numbers.
        - Example: say "The last pull request is number 42, titled Fix login bug, merged by Alice."
        - If you cannot find the requested information, say so briefly.
        """)
    return base


def _extract_query(text: str) -> str | None:
    """Parse the wake word from a transcript line.

    Returns:
      None  — no wake word, LLM must be suppressed
      ""    — bare wake word only ("Jarvis"), enter listening mode
      "..." — wake word + inline query ("Jarvis, summarise the last point")
    """
    m = _WAKE_PATTERN.search(text.strip())
    return m.group(1).strip() if m else None


class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            llm=cerebras.LLM(model="gpt-oss-120b"),
            tools=_build_tools(),
            instructions=_build_instructions(),
        )
        # Buffers every STT utterance heard in the meeting, wake-word or not.
        self._transcript: collections.deque[str] = collections.deque(maxlen=_TRANSCRIPT_MAX)
        # Two-stage wake: bare "Jarvis" → listening mode → next utterance is the query.
        self._listening: bool = False
        self._listening_since: float = 0.0
        # Tracks whether the ack was already played from a partial transcript hit,
        # so on_user_turn_completed doesn't double-play it.
        self._partial_wake_fired: bool = False
        # ID of the rolling transcript system message kept in the chat context.
        # Each turn we remove the old one and insert a fresh snapshot so the
        # chat history never accumulates multiple embedded transcripts.
        self._transcript_msg_id: str | None = None

    async def on_enter(self) -> None:
        # Do NOT call session.generate_reply() here.
        # The default Agent.on_enter() calls generate_reply(), which bypasses
        # on_user_turn_completed and goes straight to the LLM. After an interruption
        # LiveKit re-enters the agent, triggering on_enter() again — causing the agent
        # to answer without a wake word. Overriding with a no-op disables this.
        pass

    async def stt_node(
        self,
        audio: AsyncIterable[rtc.AudioFrame],
        model_settings: ModelSettings,
    ) -> AsyncIterable[lk_stt.SpeechEvent | str]:
        """Spy on INTERIM transcripts to fire the 'Yes?' ack the moment 'Jarvis'
        appears — before VAD silence and STT finalization (~200–400 ms earlier).
        """
        async for event in Agent.default.stt_node(self, audio, model_settings):
            if (
                not self._partial_wake_fired
                and isinstance(event, lk_stt.SpeechEvent)
                and event.type == lk_stt.SpeechEventType.INTERIM_TRANSCRIPT
            ):
                text = event.alternatives[0].text if event.alternatives else ""
                if _WAKE_PATTERN.search(text):
                    self._partial_wake_fired = True
                    logger.info("Partial wake detected in interim — firing ack early")
                    session = self.session

                    async def _say_ack():
                        await session.say("Yes?", add_to_chat_ctx=False)

                    asyncio.create_task(_say_ack())
            yield event

    async def on_user_turn_completed(
        self,
        turn_ctx: llm.ChatContext,
        new_message: llm.ChatMessage,
    ) -> None:
        """Gate the LLM behind the 'Jarvis' wake word.

        Every transcript line is buffered for meeting context regardless of
        whether it contained the wake word. The LLM is only called when the
        wake word is detected.
        """
        raw = new_message.text_content or ""
        # Snapshot and reset partial-wake flag for this turn.
        partial_fired = self._partial_wake_fired
        self._partial_wake_fired = False

        # Always buffer so the LLM has full meeting context when it is called.
        if raw.strip():
            self._transcript.append(raw.strip())

        # ── Listening mode: bare "Jarvis" was just said, awaiting the query ──
        if self._listening:
            elapsed = time.perf_counter() - self._listening_since
            self._listening = False
            if elapsed > _LISTENING_TIMEOUT_S or not raw.strip():
                logger.info("Wake listening timed out — suppressing LLM")
                raise StopResponse()
            logger.info("Listening mode query: %.80r", raw.strip())
            self._refresh_transcript_in_ctx(turn_ctx, new_message)
            new_message.content = [raw.strip()]
            await self.update_chat_ctx(turn_ctx)
            return

        query = _extract_query(raw)

        # ── No wake word — regular meeting speech, suppress the LLM ──────────
        if query is None:
            logger.debug("No wake word — suppressing: %.60r", raw)
            raise StopResponse()

        # ── Bare wake word — acknowledge and wait for the follow-up ──────────
        if not query:
            logger.info("Wake word — entering listening mode (partial_fired=%s)", partial_fired)
            self._listening = True
            self._listening_since = time.perf_counter()
            if not partial_fired:
                # Partial detection already played the ack; skip to avoid double-play.
                await self.session.say("Yes?", add_to_chat_ctx=False)
            raise StopResponse()

        # ── Wake word + inline query — dispatch to LLM with meeting context ──
        logger.info("Wake query dispatched: %.80r", query)
        self._refresh_transcript_in_ctx(turn_ctx, new_message)
        new_message.content = [query]
        await self.update_chat_ctx(turn_ctx)

    def _refresh_transcript_in_ctx(
        self, turn_ctx: llm.ChatContext, new_message: llm.ChatMessage
    ) -> None:
        """Replace the rolling transcript system message and apply sliding windows.

        Two caps are enforced on every LLM call to guarantee the request never
        exceeds the Cerebras TPM limit regardless of meeting length:

        1. Transcript window: only the most recent _TRANSCRIPT_WINDOW utterances
           are included in the snapshot (~4 k tokens).
        2. Chat history window: turn_ctx is truncated to _CHAT_HISTORY_WINDOW
           items so accumulated Q&A pairs don't grow unbounded (~750 tokens).
           truncate() always preserves the system instruction message.
        """
        # Remove the previous snapshot first so it isn't counted by truncate().
        if self._transcript_msg_id:
            idx = turn_ctx.index_by_id(self._transcript_msg_id)
            if idx is not None:
                turn_ctx.items.pop(idx)
            self._transcript_msg_id = None

        # Slide the conversation history window.
        turn_ctx.truncate(max_items=_CHAT_HISTORY_WINDOW)

        if not self._transcript:
            return

        # Use only the most recent utterances for the snapshot.
        recent = list(self._transcript)[-_TRANSCRIPT_WINDOW:]
        snapshot = "[Meeting transcript (recent)]\n" + "\n".join(recent)

        # Insert just before the pending user message so ordering is natural.
        user_idx = turn_ctx.index_by_id(new_message.id)
        insert_at = user_idx if user_idx is not None else len(turn_ctx.items)

        msg = llm.ChatMessage(role="system", content=[snapshot])
        turn_ctx.items.insert(insert_at, msg)
        self._transcript_msg_id = msg.id


server = AgentServer()


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


server.setup_fnc = prewarm


@server.rtc_session(agent_name="my-agent")
async def my_agent(ctx: JobContext):
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }

    # ── Detect Recall mode from dispatch metadata ─────────────────────────────
    # When recall_bridge.py dispatches this agent it passes:
    #   metadata = '{"room_name": "<uuid>"}'
    # The room_name is used to construct the Recall publisher's participant identity
    # ("recall-browser-{room_name}") so the AgentSession STT subscribes to the
    # correct audio track (mixed meeting audio captured by the Recall bot's Chrome).
    #
    # When launched from the LiveKit Agents console or without metadata, room_name
    # is empty and the agent falls back to subscribing to all participants (normal mode).
    room_name = ""
    try:
        meta = json.loads(ctx.job.metadata or "{}")
        room_name = (meta.get("room_name") or "").strip()
    except (ValueError, TypeError):
        pass

    if room_name:
        logger.info("Recall mode — STT linked to participant: recall-browser-%s", room_name)
    else:
        logger.info("Standard mode — STT linked to all participants (room: %s)", ctx.room.name)

    session = AgentSession(
        # AssemblyAI Universal-3 Pro: keyterms_prompt locks "Jarvis"/"Hey Jarvis"
        # recognition in noisy meeting audio, reducing mis-transcriptions.
        # language_detection=False removes 30–80ms multilingual overhead.
        stt=assemblyai.STT(
            model="u3-rt-pro",
            keyterms_prompt=["Jarvis", "Hey Jarvis"],
            language_detection=False,
        ),
        tts=inference.TTS(
            model="cartesia/sonic-3", voice="9626c31c-bec5-4cca-baa8-f8ba9e84c8bc"
        ),
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        # Disabled: on_user_turn_completed always rewrites or clears the message,
        # so speculative output is always discarded → audio glitch at turn start.
        preemptive_generation=False,
    )

    await ctx.connect()

    if room_name:
        # Recall mode: subscribe ONLY to the Recall browser publisher's audio track.
        # The Recall bot's headless Chrome captures mixed meeting audio via getUserMedia()
        # and publishes it to LiveKit under this exact identity. Without this filter the
        # agent would try to subscribe to all participants and may not find the right track.
        await session.start(
            agent=Assistant(),
            room=ctx.room,
            room_options=room_io.RoomOptions(
                participant_identity=f"recall-browser-{room_name}",
            ),
        )
    else:
        # Standard mode (console / direct browser): subscribe to all participants
        # with background noise cancellation enabled.
        await session.start(
            agent=Assistant(),
            room=ctx.room,
            room_options=room_io.RoomOptions(
                audio_input=room_io.AudioInputOptions(
                    noise_cancellation=ai_coustics.audio_enhancement(
                        model=ai_coustics.EnhancerModel.QUAIL_VF_S
                    ),
                ),
            ),
        )


if __name__ == "__main__":
    cli.run_app(server)
