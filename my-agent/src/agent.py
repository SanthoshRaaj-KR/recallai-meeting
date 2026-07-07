import asyncio
import collections
import json
import logging
import os
import re
import requests
import textwrap
import time
from concurrent.futures import ThreadPoolExecutor
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
from livekit.plugins import cerebras, deepgram, silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel
from typing import AsyncIterable

try:
    from .memory_compaction import TranscriptCompactor
    from .confluence_rag import ConfluenceLiveRAG
    from . import session_store
except ImportError:  # Allows `python src/agent.py ...` from my-agent.
    from memory_compaction import TranscriptCompactor
    from confluence_rag import ConfluenceLiveRAG
    import session_store

logger = logging.getLogger("agent")

# Resolve .env.local relative to this file (my-agent/.env.local) so the agent
# finds its credentials regardless of the working directory at launch time.
load_dotenv(Path(__file__).parent.parent / ".env.local")

_MEMORY_HEADER_RE = re.compile(
    r"^\[Compacted meeting memory\].*?Key retained context:\s*",
    re.IGNORECASE | re.DOTALL,
)


def _extract_topic_hint(memory: str, max_chars: int = 150) -> str:
    """Strip compacted-memory headers and return a short topic string for query enrichment."""
    text = _MEMORY_HEADER_RE.sub("", (memory or "")).strip()
    text = re.sub(r"\s+", " ", text)
    return text[:max_chars].strip()


# ── Wake word ─────────────────────────────────────────────────────────────────
# Matches "Jarvis", "Hey Jarvis", and common STT mis-transcriptions.
_WAKE_PATTERN = re.compile(
    r"(?:hey\s+)?(?:jarvis|jarvas|jervis|jarvus)[,.\s!?]*\s*(.*)",
    re.IGNORECASE | re.DOTALL,
)
# ── Audio primer ──────────────────────────────────────────────────────────────
# Silent audio frames injected into the first TTS output to prime the WebRTC
# connection and jitter buffer before the greeting is heard.
# 24 kHz, 16-bit PCM — matches deepgram/aura-2 output.  10 ms chunks × N.
_SILENCE_PRIMER_SAMPLE_RATE = 24_000
_SILENCE_PRIMER_CHUNK_SAMPLES = _SILENCE_PRIMER_SAMPLE_RATE // 100  # 10 ms
_SILENCE_PRIMER_FRAMES = max(1, int(os.getenv("JARVIS_SILENCE_PRIMER_MS", "500")) // 10)

# Full in-memory transcript buffer — all utterances are kept here.
_TRANSCRIPT_MAX = 500
# ── Sliding window limits ─────────────────────────────────────────────────────
# Only the most recent _TRANSCRIPT_WINDOW utterances are sent to the LLM each
# turn (~100 × 40 tokens ≈ 4 000 tokens). Older utterances stay in the buffer
# so they can still be referenced if the window is widened later.
_TRANSCRIPT_WINDOW = 100
# Keep the recent/raw transcript budget independent from compacted memory. We
# approximate tokens by chars here to avoid adding a tokenizer dependency to the
# live agent path.
_CHARS_PER_TOKEN_APPROX = 4
_RECENT_TRANSCRIPT_TOKENS = int(os.getenv("JARVIS_RECENT_TRANSCRIPT_TOKENS", "4000"))
_RECENT_TRANSCRIPT_CHARS = int(
    os.getenv(
        "JARVIS_RECENT_TRANSCRIPT_CHARS",
        str(_RECENT_TRANSCRIPT_TOKENS * _CHARS_PER_TOKEN_APPROX),
    )
)
_COMPACTED_MEMORY_TOKENS = int(os.getenv("JARVIS_COMPACTED_MEMORY_TOKENS", "6000"))
_COMPACTED_MEMORY_CHARS = int(
    os.getenv(
        "JARVIS_COMPACTED_MEMORY_CHARS",
        str(_COMPACTED_MEMORY_TOKENS * _CHARS_PER_TOKEN_APPROX),
    )
)
# Keep at most this many conversation items (user + assistant turns) in the
# chat context. truncate() always preserves the system instruction message.
# 10 items = 5 Q&A pairs ≈ ~750 tokens for history.
_CHAT_HISTORY_WINDOW = 1
_GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")
# Optional default repo (owner/repo) used when the user doesn't name one.
_GITHUB_DEFAULT_REPO = os.getenv("GITHUB_DEFAULT_REPO", "")
_BRIDGE_INTERNAL_URL = os.getenv("BRIDGE_INTERNAL_URL", "http://127.0.0.1:8000").rstrip("/")
_OPENING_GREETING_DELAY_S = float(os.getenv("JARVIS_OPENING_GREETING_DELAY_SECONDS", "0.5"))

# Silent backend prompt sent to the LLM pipeline after Jarvis joins and tools are
# ready.  Meeting participants never hear this text — only Jarvis's generated reply
# plays in the meeting.  Keep in sync with src/warmup_prompt.wav.
_WARMUP_PROMPT_TEXT = os.getenv(
    "JARVIS_WARMUP_PROMPT",
    "Hello Jarvis, please introduce yourself to the meeting.",
)


def _trim_text_to_char_budget(text: str, max_chars: int) -> str:
    value = (text or "").strip()
    if len(value) <= max_chars:
        return value
    return value[-max_chars:].lstrip()


def _tail_lines_to_char_budget(lines: list[str], max_chars: int) -> list[str]:
    kept: collections.deque[str] = collections.deque()
    total = 0
    for line in reversed(lines):
        clean = line.strip()
        if not clean:
            continue
        line_cost = len(clean) + 1
        if kept and total + line_cost > max_chars:
            break
        if not kept and line_cost > max_chars:
            kept.appendleft(clean[-max_chars:].lstrip())
            break
        kept.appendleft(clean)
        total += line_cost
    return list(kept)


def _build_github_toolset() -> mcp.MCPToolset | None:
    """Build the GitHub MCPToolset, or return None if no token is configured.

    The toolset is created eagerly at startup but its MCP server subprocess is
    not launched until Assistant.on_enter() calls toolset.setup() in the
    background — so the npx/Node cold-start happens immediately when the
    session opens, not on the user's first GitHub question.
    """
    if not _GITHUB_TOKEN:
        logger.warning("GITHUB_TOKEN not set — starting without GitHub tools")
        return None
    # On Windows, Python subprocesses cannot resolve bare "npx" — use "npx.cmd".
    _npx = "npx.cmd" if os.name == "nt" else "npx"
    return mcp.MCPToolset(
        id="github",
        mcp_server=mcp.MCPServerStdio(
            command=_npx,
            args=["-y", "@modelcontextprotocol/server-github@2025.4.8"],
            env={
                **os.environ,
                "GITHUB_PERSONAL_ACCESS_TOKEN": _GITHUB_TOKEN,
            },
            client_session_timeout_seconds=60,
        ),
    )


def _build_instructions() -> str:
    base = textwrap.dedent("""\
        You are Jarvis, a voice meeting assistant. Your answers are spoken aloud during a live meeting.

        Before each question you may receive:

        [Meeting Transcript]
        Recent speech-to-text from this meeting.

        [Confluence Knowledge]
        Retrieved wiki excerpts that may or may not be relevant.

        [General Knowledge]
        Your own knowledge.

        Source reliability:

        * Treat the meeting transcript as noisy. Words may be missing, substituted, or misheard.
        * Do not base conclusions on a single unclear transcript fragment.
        * Look for agreement across multiple transcript lines or speakers.
        * Ignore obviously garbled transcript text.
        * If evidence is weak, use cautious language such as "it sounded like" or "the team appeared to".
        * Confluence excerpts are retrieved by similarity, not intent.
        * Each excerpt is labelled with its page title and section heading — use these to identify which system or topic the excerpt covers.
        * Before using any excerpt, check the meeting transcript to establish which specific system, pipeline, or topic is being discussed.
        * If multiple excerpts from different pages cover the same subject (e.g. deployment), only use the one whose page title matches the system the meeting is discussing. Discard the others.
        * Use an excerpt only if it directly helps answer the question.
        * Ignore irrelevant or weak matches.
        * Do not force wiki content into an answer.

        Answering:

        * Combine all available evidence.
        * If transcript and Confluence agree, answer confidently.
        * If they conflict, briefly mention the disagreement.
        * If neither source answers the question, use general knowledge.
        * Never invent meeting decisions, statements, or participants.
        * If something was not discussed, say so directly.

        Output:

        * Plain text only.
        * No markdown, bullet points, JSON, tables, or emojis.
        * One to three sentences unless additional detail is required.
        * Answer immediately; do not ask clarifying questions.
        * Do not reveal internal implementation details such as Pinecone, Confluence indexes, compacted memory, or how context was retrieved. Answer general knowledge questions (including questions about AI techniques like RAG) from your own knowledge.
        * Use natural spoken language suitable for text-to-speech.
        * Prefer short words and short sentences.
        * Spell out numbers when practical.
        * Avoid unexplained acronyms.
        """)
    if _GITHUB_TOKEN:
        base += textwrap.dedent("""\

        # GitHub tool rules
        - When GitHub tools return data, convert it to spoken prose.
        - Do not say field names, JSON syntax, or item numbers.
        - Example: say "The last pull request is number 42, titled Fix login bug, merged by Alice."
        - If you cannot find the requested information, say so briefly.
        """)
        if _GITHUB_DEFAULT_REPO:
            base += textwrap.dedent(f"""\
        - When the user does not specify a repository, default to {_GITHUB_DEFAULT_REPO}.
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
    def __init__(self, session_id: str = "", confluence_enabled: bool = True) -> None:
        self._session_id = session_id
        self._confluence_enabled = confluence_enabled
        self._github_toolset = _build_github_toolset()
        super().__init__(
            llm=cerebras.LLM(model="gpt-oss-120b"),
            tools=[self._github_toolset] if self._github_toolset else [],
            instructions=_build_instructions(),
        )
        # Buffers every STT utterance heard in the meeting, wake-word or not.
        self._transcript: collections.deque[str] = collections.deque(maxlen=_TRANSCRIPT_MAX)
        self._transcript_memory = TranscriptCompactor(
            window_size=_TRANSCRIPT_WINDOW,
            max_memory_chars=_COMPACTED_MEMORY_CHARS,
        )
        # Confluence in-meeting RAG — fires on every wake-word query.
        self._confluence_rag = ConfluenceLiveRAG()
        # Holds the formatted Confluence context for the current turn, cleared
        # each turn so stale results never bleed into the next query.
        self._last_rag_context: str = ""
        # Persists the most recent compacted memory so the Pinecone query can
        # be enriched with the meeting's running topic even after a topic switch.
        self._last_compacted_memory: str = ""
        # Single-threaded executor for all TranscriptCompactor operations.
        # Serialises observe_utterance and memory_text calls coming from both
        # the event loop and tts_node, eliminating the race condition on the
        # compactor's internal lists and keeping blocking LLM compaction calls
        # off the event loop.
        self._compactor_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="compactor")
        # Ensures the opening greeting fires exactly once, even though LiveKit
        # can re-enter on_enter() after an interruption.
        self._greeted: bool = False
        # Set to True when the partial-wake ack ("Yes?") fires from stt_node so
        # on_user_turn_completed doesn't double-play it.
        self._partial_wake_fired: bool = False
        # ID of the rolling transcript system message kept in the chat context.
        # Each turn we remove the old one and insert a fresh snapshot so the
        # chat history never accumulates multiple embedded transcripts.
        self._transcript_msg_id: str | None = None
        self._opening_greeting_task: asyncio.Task | None = None
        # Flipped to True just before the opening TTS call so the first audio
        # output is prefixed with silence to prime the WebRTC jitter buffer.
        self._should_prime_audio: bool = False

    async def on_enter(self) -> None:
        # Do NOT call session.generate_reply() here directly.
        # The default Agent.on_enter() calls generate_reply(), which bypasses
        # on_user_turn_completed and goes straight to the LLM. After an interruption
        # LiveKit re-enters the agent, triggering on_enter() again — causing the agent
        # to answer without a wake word. Overriding with a no-op disables this.

        # One-time warmup + intro sequence:
        # 1. Warm up GitHub MCP and Confluence RAG in parallel.
        # 2. Once tools are ready, send the pre-determined silent prompt
        #    ("Hello Jarvis, please introduce yourself") through the LLM pipeline
        #    so Jarvis's opening words are LLM-generated, not pre-canned.
        if not self._greeted:
            self._greeted = True
            self._opening_greeting_task = asyncio.create_task(
                self._warmup_and_intro(),
                name="opening_greeting",
            )

    async def _warmup_and_intro(self) -> None:
        """Warm up tools then send the silent intro prompt through the LLM pipeline.

        Tools (GitHub MCP, Confluence RAG) warm up in parallel first so they are
        available by the time the LLM generates the intro.  The warmup prompt text
        mirrors the pre-determined WAV file at src/warmup_prompt.wav but is sent
        directly to the LLM — meeting participants hear only Jarvis's reply.
        """
        async def _warmup_mcp() -> None:
            if not self._github_toolset:
                return
            try:
                await self._github_toolset.setup()
                logger.info("GitHub MCP toolset ready")
            except Exception as exc:
                logger.warning("GitHub MCP prewarm failed: %s", exc)

        async def _warmup_rag() -> None:
            if not self._confluence_rag.enabled:
                return
            try:
                await asyncio.to_thread(self._confluence_rag.warmup)
            except Exception as exc:
                logger.warning("Confluence RAG prewarm failed: %s", exc)

        # Step 1: warm up all tools in parallel.
        await asyncio.gather(_warmup_mcp(), _warmup_rag())

        # Step 2: send the silent backend prompt to the LLM pipeline.
        # generate_reply(user_input=...) bypasses the wake-word gate and goes
        # straight to the LLM, warming up the model so the first real query
        # from a meeting participant is served without cold-start latency.
        if _OPENING_GREETING_DELAY_S > 0:
            await asyncio.sleep(_OPENING_GREETING_DELAY_S)
        self._should_prime_audio = True
        try:
            logger.info("Sending warmup intro prompt to LLM pipeline")
            handle = self.session.generate_reply(
                user_input=_WARMUP_PROMPT_TEXT,
                allow_interruptions=False,
            )
            await handle.wait_for_playout()
        except Exception as exc:
            logger.warning("LLM intro warmup failed: %s — falling back to canned greeting", exc)
            try:
                handle = self.session.say(
                    "Hello everyone, Jarvis here. Just say Hey Jarvis whenever you need me.",
                    add_to_chat_ctx=False,
                    allow_interruptions=False,
                )
                await handle.wait_for_playout()
            except Exception as exc2:
                logger.warning("Fallback greeting also failed: %s", exc2)

    async def stt_node(
        self,
        audio: AsyncIterable[rtc.AudioFrame],
        model_settings: ModelSettings,
    ) -> AsyncIterable[lk_stt.SpeechEvent | str]:
        """Spy on INTERIM transcripts to fire "Yes?" the moment "Jarvis" appears.

        Fires ~200–400 ms before VAD silence + STT finalization, giving Jarvis
        an instant audio acknowledgment while the LLM call is still pending.
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

                    async def _say_ack() -> None:
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

        # Snapshot and reset so stt_node's partial ack doesn't double-fire.
        partial_fired = self._partial_wake_fired
        self._partial_wake_fired = False

        # Always buffer so the LLM has full meeting context when it is called.
        if raw.strip():
            self._transcript.append(raw.strip())
            asyncio.create_task(
                self._run_in_compactor(self._transcript_memory.observe_utterance, raw.strip())
            )
            self._post_transcript(raw.strip())

        query = _extract_query(raw)

        # ── No wake word — suppress the LLM silently ─────────────────────────
        if query is None:
            logger.debug("No wake word — suppressing: %.60r", raw)
            raise StopResponse()

        # ── Bare wake word only — ack already played by stt_node; no query ───
        if not query:
            logger.info("Bare wake word — ack fired=%s, no query to dispatch", partial_fired)
            if not partial_fired:
                # stt_node didn't catch it in interim — play the ack now
                await self.session.say("Yes?", add_to_chat_ctx=False)
            raise StopResponse()

        # ── Wake word + inline query — dispatch to LLM with meeting context ──
        logger.info("Wake query dispatched: %.80r", query)

        self._last_rag_context = ""
        if not self._confluence_enabled:
            logger.info("[Pinecone] skipped — Confluence disabled for this session")
        elif not self._confluence_rag.enabled:
            logger.info("[Pinecone] skipped — PINECONE_API_KEY not set in .env.local")
        else:
            try:
                topic_hint = _extract_topic_hint(self._last_compacted_memory)
                enriched_query = self._confluence_rag.build_search_query(
                    query, list(self._transcript), topic_hint=topic_hint
                )
                logger.info("[Pinecone] searching — enriched query: %.120r", enriched_query)
                hits = await asyncio.to_thread(self._confluence_rag.search, enriched_query)
                self._last_rag_context = self._confluence_rag.format_context(hits)
                logger.info(
                    "[Pinecone] %d chunk(s) will be injected into LLM context",
                    len(hits),
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("[Pinecone] lookup failed, continuing without Confluence context: %s", exc)

        # Fetch compacted memory off the event loop — memory_text() may trigger
        # a blocking LLM compaction call via force_compact().
        compacted_memory = _trim_text_to_char_budget(
            await self._run_in_compactor(self._transcript_memory.memory_text),
            _COMPACTED_MEMORY_CHARS,
        )
        if compacted_memory:
            self._last_compacted_memory = compacted_memory
        self._refresh_transcript_in_ctx(turn_ctx, new_message, compacted_memory)
        self._last_rag_context = ""  # consumed — clear so it cannot bleed into a subsequent turn
        new_message.content = [query]
        await self.update_chat_ctx(turn_ctx)

    def _post_transcript(self, text: str, speaker: str = "Meeting") -> None:
        """Send final STT turns (or Jarvis replies) to recall_bridge for post-meeting review.

        agent.py and recall_bridge.py usually run in separate processes, so the
        review pipeline cannot read this in-memory transcript directly.
        """
        if not self._session_id or not text:
            return

        def _send() -> None:
            try:
                requests.post(
                    f"{_BRIDGE_INTERNAL_URL}/livekit-transcript/{self._session_id}",
                    json={
                        "speaker": speaker,
                        "text": text,
                        "timestamp": time.time(),
                        "source": "livekit",
                    },
                    timeout=2,
                )
            except Exception as exc:  # noqa: BLE001
                logger.debug("Could not post transcript to bridge: %s", exc)

        asyncio.create_task(asyncio.to_thread(_send))

    async def tts_node(
        self,
        text: AsyncIterable[str],
        model_settings: ModelSettings,
    ) -> AsyncIterable[rtc.AudioFrame]:
        """Spy on text sent to TTS so Jarvis's spoken replies are added to the transcript."""
        if self._should_prime_audio:
            self._should_prime_audio = False
            _silence = rtc.AudioFrame(
                data=bytes(_SILENCE_PRIMER_CHUNK_SAMPLES * 2),
                sample_rate=_SILENCE_PRIMER_SAMPLE_RATE,
                num_channels=1,
                samples_per_channel=_SILENCE_PRIMER_CHUNK_SAMPLES,
            )
            for _ in range(_SILENCE_PRIMER_FRAMES):
                yield _silence

        collected: list[str] = []

        async def _spy(source: AsyncIterable[str]) -> AsyncIterable[str]:
            async for chunk in source:
                collected.append(chunk)
                yield chunk

        async for frame in Agent.default.tts_node(self, _spy(text), model_settings):
            yield frame

        reply = "".join(collected).strip()
        if reply:
            labelled = f"Jarvis: {reply}"
            self._transcript.append(labelled)
            asyncio.create_task(
                self._run_in_compactor(self._transcript_memory.observe_utterance, labelled)
            )
            self._post_transcript(reply, speaker="Jarvis")

    async def _run_in_compactor(self, fn, *args):
        """Run a TranscriptCompactor call in the dedicated single-threaded executor.

        Using a single-threaded executor (max_workers=1) for all compactor
        operations serialises observe_utterance and memory_text calls, eliminating
        the race condition between the event-loop path (user utterances) and the
        tts_node background path (Jarvis replies). Blocking OpenAI compaction
        calls run off the event loop so they never stall audio I/O.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._compactor_executor, fn, *args)

    def _refresh_transcript_in_ctx(
        self, turn_ctx: llm.ChatContext, new_message: llm.ChatMessage, compacted_memory: str = ""
    ) -> None:
        """Replace the rolling transcript system message and apply memory windows.

        Separate caps are enforced on every LLM call to keep the request bounded
        regardless of meeting length:

        1. Recent transcript: up to _TRANSCRIPT_WINDOW latest utterances, then
           trimmed to _RECENT_TRANSCRIPT_CHARS. This budget is reserved for raw
           transcript and is not consumed by compacted memory.
        2. Compacted memory: older utterances are rewritten into one bounded
           memory block, capped separately by _COMPACTED_MEMORY_CHARS.
        3. Chat history window: turn_ctx is truncated to _CHAT_HISTORY_WINDOW
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

        # Use the most recent utterances for the snapshot, with a budget that is
        # independent from the compacted-memory budget.
        recent = _tail_lines_to_char_budget(
            list(self._transcript)[-_TRANSCRIPT_WINDOW:],
            _RECENT_TRANSCRIPT_CHARS,
        )
        snapshot_parts = []
        if compacted_memory:
            snapshot_parts.append(compacted_memory)
        snapshot_parts.append("[Meeting transcript (recent)]\n" + "\n".join(recent))
        if self._last_rag_context:
            snapshot_parts.append("[Confluence Knowledge]\n" + self._last_rag_context)
            logger.info(
                "[Pinecone] Confluence context injected (%d chars)",
                len(self._last_rag_context),
            )
        snapshot = "\n\n".join(snapshot_parts)

        # Insert just before the pending user message so ordering is natural.
        user_idx = turn_ctx.index_by_id(new_message.id)
        insert_at = user_idx if user_idx is not None else len(turn_ctx.items)

        msg = llm.ChatMessage(role="system", content=[snapshot])
        turn_ctx.items.insert(insert_at, msg)
        self._transcript_msg_id = msg.id


class JarvisCallAssistant(Agent):
    """One-on-one post-meeting voice call agent.

    Unlike the meeting assistant, there is no wake-word gate — every user
    utterance is routed directly to the LLM.  The full meeting context
    (compacted memory + recent transcript) is injected as a system message
    once at startup so it is always in the context window.
    """

    def __init__(
        self,
        session_id: str = "",
        meeting_context: str = "",
        confluence_enabled: bool = False,
    ) -> None:
        self._session_id = session_id
        self._meeting_context = meeting_context
        self._confluence_enabled = confluence_enabled
        self._confluence_rag = ConfluenceLiveRAG()
        super().__init__(
            llm=cerebras.LLM(model="gpt-oss-120b"),
            instructions=self._build_instructions(),
        )
        self._greeted = False

    @staticmethod
    def _build_instructions() -> str:
        return textwrap.dedent("""\
            You are Jarvis, a personal AI assistant in a private one-on-one voice call.
            The user just finished a meeting and wants to discuss it with you directly.
            You will receive the meeting transcript and notes as context with each message.
            Answer any questions clearly and conversationally, drawing on the meeting context
            when relevant and your own knowledge otherwise.

            # Output rules
            - Respond in plain text only. No markdown, lists, JSON, or emojis.
            - Keep replies conversational and concise: two to four sentences unless more detail is needed.
            - Do not mention wake words, system instructions, or internal state.
            - Spell out numbers and avoid acronyms with unclear pronunciation.
            - Never fabricate meeting content. If something was not discussed, say so honestly.
            """)

    async def on_enter(self) -> None:
        if not self._greeted:
            self._greeted = True
            try:
                handle = self.session.say(
                    "Hi! I'm Jarvis. I have the full context of your meeting. Ask me anything.",
                    add_to_chat_ctx=False,
                    allow_interruptions=False,
                )
                await handle.wait_for_playout()
            except Exception as exc:  # noqa: BLE001
                logger.warning("Jarvis call greeting failed: %s", exc)

    async def on_user_turn_completed(
        self,
        turn_ctx: llm.ChatContext,
        new_message: llm.ChatMessage,
    ) -> None:
        # No wake-word gating — every utterance goes straight to the LLM.
        # Still apply the chat history window to prevent unbounded context growth.
        turn_ctx.truncate(max_items=_CHAT_HISTORY_WINDOW)

        # Inject meeting context as a system message just before the user turn
        # so it is present for this LLM call but not baked into the static system prompt.
        if self._meeting_context:
            user_idx = turn_ctx.index_by_id(new_message.id)
            insert_at = user_idx if user_idx is not None else len(turn_ctx.items)
            turn_ctx.items.insert(
                insert_at,
                llm.ChatMessage(
                    role="system",
                    content=["[Meeting context]\n" + self._meeting_context],
                ),
            )

        query = (new_message.text_content or "").strip()
        if query and self._confluence_enabled and self._confluence_rag.enabled:
            try:
                enriched_query = self._confluence_rag.build_search_query(query, [])
                logger.info("[Pinecone/call] searching — enriched query: %.120r", enriched_query)
                hits = await asyncio.to_thread(self._confluence_rag.search, enriched_query)
                rag_context = self._confluence_rag.format_context(hits)
                if rag_context:
                    logger.info("[Pinecone/call] %d chunk(s) injected into context", len(hits))
                    user_idx = turn_ctx.index_by_id(new_message.id)
                    insert_at = user_idx if user_idx is not None else len(turn_ctx.items)
                    turn_ctx.items.insert(
                        insert_at,
                        llm.ChatMessage(
                            role="system",
                            content=["[Confluence Knowledge]\n" + rag_context],
                        ),
                    )
            except Exception as exc:  # noqa: BLE001
                logger.warning("[Pinecone/call] lookup failed, continuing without context: %s", exc)
        elif self._confluence_enabled and not self._confluence_rag.enabled:
            logger.info("[Pinecone/call] skipped — PINECONE_API_KEY not set")

        await self.update_chat_ctx(turn_ctx)


def _load_meeting_context(session_id: str) -> str:
    """Load meeting transcript and compacted memory from session_store."""
    try:
        sess = session_store.get(session_id)
        if not sess:
            return ""
        parts: list[str] = []
        mem_text = (sess.get("transcript_memory_text") or "").strip()
        if mem_text:
            parts.append(f"[Meeting memory (compacted)]\n{mem_text}")
        transcript = session_store.get_transcript_turns(session_id)
        if transcript:
            recent = transcript[-150:]
            lines: list[str] = []
            for entry in recent:
                if isinstance(entry, dict):
                    speaker = entry.get("participant") or entry.get("speaker") or "Meeting"
                    text = entry.get("text") or ""
                    lines.append(f"{speaker}: {text}")
                elif isinstance(entry, str):
                    lines.append(entry)
            if lines:
                parts.append("[Meeting transcript (recent)]\n" + "\n".join(lines))
        return "\n\n".join(parts)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not load meeting context for jarvis call: %s", exc)
        return ""


server = AgentServer()


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()
    # MultilingualModel requires get_job_context().inference_executor and cannot
    # be instantiated in prewarm (no job context exists yet). It is created
    # inside the job entrypoint instead.


server.setup_fnc = prewarm


@server.rtc_session(agent_name="my-agent")
async def my_agent(ctx: JobContext):
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }

    # MultilingualModel needs a live JobContext (for inference_executor), so it
    # must be created here rather than in prewarm.
    turn_detector = MultilingualModel()

    # ── Parse dispatch metadata ───────────────────────────────────────────────
    mode = ""
    session_id = ""
    room_name = ""
    confluence_enabled = False
    try:
        meta = json.loads(ctx.job.metadata or "{}")
        mode = (meta.get("mode") or "").strip()
        session_id = (meta.get("session_id") or "").strip()
        # Legacy Recall bridge passes room_name directly without a mode field.
        room_name = (meta.get("room_name") or "").strip()
        confluence_enabled = bool(meta.get("confluence_enabled", False))
    except (ValueError, TypeError):
        pass

    # ── Mode: one-on-one Jarvis call (post-meeting voice Q&A) ────────────────
    if mode == "jarvis_call":
        logger.info(
            "Jarvis call mode — session_id=%s room=%s confluence=%s",
            session_id, ctx.room.name, confluence_enabled,
        )
        meeting_context = _load_meeting_context(session_id) if session_id else ""

        session = AgentSession(
            stt=deepgram.STT(
                model="nova-3",
                language="en",
            ),
            tts=inference.TTS(model="deepgram/aura-2", voice="athena", language="en"),
            turn_detection=turn_detector,
            vad=ctx.proc.userdata["vad"],
        )
        await ctx.connect()
        await session.start(
            agent=JarvisCallAssistant(
                session_id=session_id,
                meeting_context=meeting_context,
                confluence_enabled=confluence_enabled,
            ),
            room=ctx.room,
        )
        return

    # ── Mode: meeting assistant (wake-word activated, Recall bot audio) ───────
    # When recall_bridge.py dispatches this agent it passes:
    #   metadata = '{"room_name": "<uuid>"}'
    # The room_name is used to construct the Recall publisher's participant identity
    # ("recall-browser-{room_name}") so the AgentSession STT subscribes to the
    # correct audio track (mixed meeting audio captured by the Recall bot's Chrome).
    #
    # When launched from the LiveKit Agents console or without metadata, room_name
    # is empty and the agent falls back to subscribing to all participants (normal mode).
    if room_name:
        logger.info("Recall mode — STT linked to participant: recall-browser-%s", room_name)
    else:
        logger.info("Standard mode — STT linked to all participants (room: %s)", ctx.room.name)

    session = AgentSession(
        # Deepgram Nova-3: keyterm boosts "Jarvis"/"Hey Jarvis" recognition
        # in noisy meeting audio. language="en" removes multilingual overhead.
        stt=deepgram.STT(
            model="nova-3",
            language="en",
            keyterm=["Jarvis", "Hey Jarvis"],
        ),
        tts=inference.TTS(model="deepgram/aura-2", voice="athena", language="en"),
        turn_detection=turn_detector,
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
            agent=Assistant(session_id=room_name, confluence_enabled=confluence_enabled),
            room=ctx.room,
            room_options=room_io.RoomOptions(
                participant_identity=f"recall-browser-{room_name}",
            ),
        )
    else:
        # Standard mode (console / direct browser): subscribe to all participants.
        # (LiveKit Cloud-only ai_coustics noise cancellation removed for self-hosting.)
        await session.start(
            agent=Assistant(confluence_enabled=confluence_enabled),
            room=ctx.room,
        )


if __name__ == "__main__":
    # Observability: worker Prometheus metrics + optional OTLP tracing.
    # No-op unless deps installed and OTEL_*/METRICS_* env set — see deploy/observability/.
    try:
        try:
            from .observability import setup_worker_observability
        except ImportError:
            from observability import setup_worker_observability
        setup_worker_observability("agent-worker")
    except Exception as _obs_exc:  # never let observability break the worker
        logger.warning("observability setup skipped: %s", _obs_exc)

    cli.run_app(server)
