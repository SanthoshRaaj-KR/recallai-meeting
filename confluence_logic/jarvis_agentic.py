"""
Jarvis Meeting Assistant (Agentic Version)
===========================================
A wake-word activated AI assistant that joins meetings via Recall.ai.
This version uses the OpenAI Agents SDK for Confluence actions, AssemblyAI
transcription through Recall, and hosted TTS.

Usage:
  uvicorn confluence_logic.jarvis_agentic:app --host 0.0.0.0 --port 8000
"""

import asyncio
import base64
import logging
import os
import random
import re
import sys
import time
from collections import deque
from collections.abc import MutableMapping
from contextvars import ContextVar
from dataclasses import dataclass
from io import BytesIO
from itertools import count
from threading import Thread
from typing import Any, Deque, Iterator, Optional
from uuid import uuid4

import requests
import uvicorn
from dotenv import load_dotenv
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from gtts import gTTS
from openai import OpenAI

from .agents.editor_agent import EditorAgent
from .classifier import classify_intent
from .audio_cache import get_random_ack_audio, get_random_filler_audio
from .general_responder import answer_general_question
from .meeting_responder import (
    summarize_meeting, generate_opinion, extract_action_items, summarize_speaker,
    summarize_meeting_streaming, generate_opinion_streaming,
    extract_action_items_streaming, summarize_speaker_streaming,
)
from confluence_logic import graph_rag
from confluence_logic import confluence_page_graph
from confluence_logic.agents.proposed_changes_agent import ProposedChangesAgent

load_dotenv()

RECALL_API_KEY = os.getenv("RECALL_API_KEY")
RECALL_API_REGION = os.getenv("RECALL_API_REGION", "ap-northeast-1")
RECALL_BASE_URL = f"https://{RECALL_API_REGION}.recall.ai/api/v1"
WEBHOOK_URL = os.getenv("WEBHOOK_URL")
MEETING_URL = os.getenv("MEETING_URL")
BOT_NAME = os.getenv("BOT_NAME", "Jarvis")
LANGUAGE_CODE = os.getenv("LANGUAGE_CODE", "en")
STREAMING_MODE = os.getenv("STREAMING_MODE", "prioritize_low_latency")
APP_HOST = os.getenv("APP_HOST", "0.0.0.0")
APP_PORT = int(os.getenv("APP_PORT", "8000"))
JARVIS_AGENT_MODEL = os.getenv("JARVIS_AGENT_MODEL", "gpt-5-mini")
RECALL_TRANSCRIPT_PROVIDER = os.getenv("RECALL_TRANSCRIPT_PROVIDER", "recallai_streaming").strip()
ASSEMBLY_API = (os.getenv("ASSEMBLY_API") or "").strip()

JARVIS_TTS_PROVIDER = os.getenv("JARVIS_TTS_PROVIDER", "edge_tts").strip().lower()
JARVIS_TTS_MODEL = os.getenv("JARVIS_TTS_MODEL", "tts-1").strip()
JARVIS_TTS_VOICE = os.getenv("JARVIS_TTS_VOICE", "echo").strip()
JARVIS_TTS_SPEED = float(os.getenv("JARVIS_TTS_SPEED", "1.0"))
JARVIS_WAKE_ACK = os.getenv("JARVIS_WAKE_ACK", "Yes?").strip()
JARVIS_BUSY_ACK = os.getenv("JARVIS_BUSY_ACK", "I'm already on it. Give me a moment.").strip()
JARVIS_SPEECH_HOLD_SECONDS = float(os.getenv("JARVIS_SPEECH_HOLD_SECONDS", "0.8"))
JARVIS_INTER_SENTENCE_GAP_SECONDS = float(os.getenv("JARVIS_INTER_SENTENCE_GAP_SECONDS", "0.18"))
JARVIS_RECALL_AUDIO_DRAIN_BUFFER_SECONDS = float(os.getenv("JARVIS_RECALL_AUDIO_DRAIN_BUFFER_SECONDS", "0.35"))
JARVIS_GENERAL_CLARIFICATION_TIMEOUT = float(os.getenv("JARVIS_GENERAL_CLARIFICATION_TIMEOUT", "15.0"))
JARVIS_LISTENING_TIMEOUT = float(os.getenv("JARVIS_LISTENING_TIMEOUT", "10.0"))
JARVIS_DEBOUNCE_SECONDS = float(os.getenv("JARVIS_DEBOUNCE_SECONDS", "1.0"))
JARVIS_SPEECH_REWRITE_ENABLED = os.getenv("JARVIS_SPEECH_REWRITE_ENABLED", "false").strip().lower() == "true"
JARVIS_MICRO_ACK_ENABLED = os.getenv("JARVIS_MICRO_ACK_ENABLED", "true").strip().lower() == "true"
JARVIS_MICRO_ACK_TEXT = os.getenv("JARVIS_MICRO_ACK_TEXT", "Mhm.").strip()
JARVIS_YIELD_PHRASE = os.getenv("JARVIS_YIELD_PHRASE", "Of course \u2014 ").strip()
JARVIS_INTERRUPT_RECOVERY_ENABLED = os.getenv("JARVIS_INTERRUPT_RECOVERY_ENABLED", "true").strip().lower() == "true"
JARVIS_POST_SPEECH_PAUSE_SECONDS = float(os.getenv("JARVIS_POST_SPEECH_PAUSE_SECONDS", "0.7"))
JARVIS_DELETE_CONFIRM_ENABLED = os.getenv("JARVIS_DELETE_CONFIRM_ENABLED", "true").strip().lower() == "true"
JARVIS_DELETE_CONFIRM_TIMEOUT = float(os.getenv("JARVIS_DELETE_CONFIRM_TIMEOUT", "10.0"))
JARVIS_GARBLED_RECOVERY_ENABLED = os.getenv("JARVIS_GARBLED_RECOVERY_ENABLED", "true").strip().lower() == "true"
JARVIS_CONFIDENCE_SIGNAL_ENABLED = os.getenv("JARVIS_CONFIDENCE_SIGNAL_ENABLED", "true").strip().lower() == "true"

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Bot is started on-demand via POST /bot/start from the review UI.
    # Do not auto-join on startup — MEETING_URL in .env is ignored here.
    logger.info("Jarvis server ready — waiting for /bot/start from the UI")
    stop_graph_refresh = asyncio.Event()
    graph_refresh_task = asyncio.create_task(
        confluence_page_graph.refresh_known_user_graphs_forever(stop_graph_refresh)
    )
    try:
        yield
    finally:
        stop_graph_refresh.set()
        graph_refresh_task.cancel()
        try:
            await graph_refresh_task
        except asyncio.CancelledError:
            pass


app = FastAPI(lifespan=lifespan)

# Mount the review API router (GET /review/summary and related endpoints).
from .review.api import router as _review_router  # noqa: E402
app.include_router(_review_router)

session_agent = EditorAgent(model=JARVIS_AGENT_MODEL)
logger.info("Jarvis meeting agent using model: %s", JARVIS_AGENT_MODEL)
logger.info("Jarvis transcript provider: %s", RECALL_TRANSCRIPT_PROVIDER)
logger.info("Jarvis TTS provider: %s", JARVIS_TTS_PROVIDER)
if RECALL_TRANSCRIPT_PROVIDER == "assembly_ai_v3_streaming" and ASSEMBLY_API:
    logger.warning(
        "ASSEMBLY_API is set locally, but Recall BYOB transcription still requires the AssemblyAI key "
        "to be configured in the Recall transcription credentials dashboard."
    )


_CONFLUENCE_MUTATION_PATTERN = re.compile(
    r"\b(?:create|edit|update|delete|remove|rename|add|append|write|change|make|draft|save|put|move|replace)\b",
    re.IGNORECASE,
)

_CONFLUENCE_READ_PATTERN = re.compile(
    r"\b(?:what|which|who|when|where|why|how|summarize|summary|explain|tell|show|list|find|search|read|does|do|is|are)\b",
    re.IGNORECASE,
)

_DYNAMIC_ACK_FALLBACKS = {
    "queued": "Noted, I'll handle that shortly.",
    "switching": "On it, switching now.",
    "error": "Sorry, I hit a snag there.",
}

_INSTANT_ACKS = [
    "On it.",
    "Sure thing.",
    "Give me a sec.",
    "Got it.",
    "One moment.",
    "Right away.",
    "Working on it.",
    "Let me check.",
]

JARVIS_FILLER_PHRASES = [
    "Sure, let me retrieve that information for you now.",
    "Of course — pulling that together right away.",
    "Certainly, let me look into that for you.",
    "Let me bring up the relevant details — one moment.",
    "Understood — accessing that right away.",
    "Give me just a moment to work through this.",
    "Allow me a brief moment — I am on it.",
    "Bear with me for just a second while I process that.",
    "One moment — let me work through the details.",
    "Let me gather what you need — this will be brief.",
    "Let me review the discussion and come right back.",
    "I will scan through the conversation — just a moment.",
    "Let me go through the relevant context for you.",
    "Reviewing that now — I will be with you shortly.",
    "Noted — let me take care of that for you now.",
    "Right, let me get to that immediately.",
    "I will have that ready for you in just a moment.",
    "Sure — let me circle back on this right away.",
    "Let me verify that and report back shortly.",
    "I am checking on that now — please hold a moment.",
]


@dataclass
class VoiceTask:
    task_id: int
    request: str
    bot_id: str
    created_at: float
    phase: str = "queued"
    intent: str = "edit"
    cancel_requested: bool = False
    superseded: bool = False
    from_queue: bool = False
    mutation_started: bool = False
    clarification_context: str = ""
    question: str = ""
    latest_user_reply: str = ""
    output_generation: int = 0
    execution_request: str = ""
    pre_planned_decision: object = None
    answer_future: Optional[asyncio.Future] = None
    runner: Optional[asyncio.Task] = None

def _fresh_meeting_state(session_id: Optional[str] = None) -> dict:
    return {
        "session_id": session_id or "default",
        "bot_id": None,
        "meeting_url": None,
        "transcript_log": [],
        "is_active": False,
        "session_status": "idle",
        "started_at": None,
        "ended_at": None,
        "end_reason": None,
        "recall_status_code": None,
        "last_recall_status_checked_at": 0.0,
        "jarvis_listening": False,
        "jarvis_listening_at": 0.0,
        "current_task": None,
        "pending_requests": deque(),
        "pending_clarification": None,
        "pending_general_clarification": None,  # {"question": str, "original_query": str, "bot_id": str, "expires_at": float}
        "pending_summary_clarification": None,  # {"bot_id": str, "expires_at": float}
        "current_task_id": None,
        "current_phase": "idle",
        "current_request": None,
        "output_generation": 0,
        "mutation_started": False,
        "cancel_requested": False,
        "last_user_speech_at": 0.0,
        "parallel_runners": [],
        "general_history": [],
        "last_jarvis_response": None,
        "gap_filler_generation": None,
        "active_gap_filler_task": None,
        "wake_query_ack_pending": False,
        "invoker_participant": None,       # D-04: set when wake word is detected; D-03: cleared after dispatch
        "_pending_debounce_task": None,    # D-09: cancellable asyncio.Task for debounce window
        "_accumulated_query": "",          # D-07: space-joined query text from invoker segments
    }


_DEFAULT_SESSION_ID = "default"
_meeting_sessions: dict[str, dict] = {_DEFAULT_SESSION_ID: _fresh_meeting_state(_DEFAULT_SESSION_ID)}
_bot_session_ids: dict[str, str] = {}
_current_meeting_state: ContextVar[dict] = ContextVar(
    "current_meeting_state",
    default=_meeting_sessions[_DEFAULT_SESSION_ID],
)


class MeetingStateProxy(MutableMapping):
    """Context-aware mapping so existing code can remain session-scoped."""

    def _state(self) -> dict:
        return _current_meeting_state.get()

    def __getitem__(self, key: str) -> Any:
        return self._state()[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self._state()[key] = value

    def __delitem__(self, key: str) -> None:
        del self._state()[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._state())

    def __len__(self) -> int:
        return len(self._state())


meeting_state: MutableMapping[str, Any] = MeetingStateProxy()


def create_meeting_session() -> str:
    session_id = uuid4().hex
    _meeting_sessions[session_id] = _fresh_meeting_state(session_id)
    return session_id


def get_meeting_session_state(session_id: Optional[str] = None) -> dict:
    resolved = session_id or _DEFAULT_SESSION_ID
    if resolved not in _meeting_sessions:
        _meeting_sessions[resolved] = _fresh_meeting_state(resolved)
    return _meeting_sessions[resolved]


def _append_transcript_log_entry(
    participant: str,
    text: str,
    timestamp: Optional[float] = None,
    source: str = "transcript",
) -> Optional[dict]:
    entry_text = (text or "").strip()
    if not entry_text:
        return None

    log = meeting_state["transcript_log"]
    entry = {
        "participant": participant or "Unknown",
        "text": entry_text,
        "timestamp": timestamp or time.time(),
        "source": source,
    }
    log.append(entry)
    if len(log) > 500:
        meeting_state["transcript_log"] = log[-500:]
    return entry


def _record_jarvis_transcript(text: str) -> None:
    entry = _append_transcript_log_entry(BOT_NAME or "Jarvis", text, source="jarvis")
    if entry:
        logger.info("Transcript %s: %s", entry["participant"], entry["text"])


def bind_bot_to_session(bot_id: str, session_id: str) -> None:
    if bot_id:
        _bot_session_ids[bot_id] = session_id


def get_session_id_for_bot(bot_id: str) -> Optional[str]:
    return _bot_session_ids.get(bot_id)


def set_current_meeting_session(session_id: Optional[str] = None):
    return _current_meeting_state.set(get_meeting_session_state(session_id))


def reset_current_meeting_session(token) -> None:
    _current_meeting_state.reset(token)

_openai_client: Optional[OpenAI] = None
# WAKEALIAS-01: expanded wake word aliases with phonetic near-misses
_WAKE_ALIASES = r"(?:jarvis|jarvas|jervis|jarvus|jarves|jarvi|jarv)"
_CUSTOM_WAKE_ALIASES = os.getenv("JARVIS_WAKE_ALIASES", "").strip()
if _CUSTOM_WAKE_ALIASES:
    # User can add pipe-separated custom aliases, e.g., "jarvy|javis|jarviss"
    _WAKE_ALIASES = rf"(?:jarvis|jarvas|jervis|jarvus|jarves|jarvi|jarv|{_CUSTOM_WAKE_ALIASES})"
_WAKE_PATTERN = re.compile(
    rf"(?:(?:hey|yo|ok|hi)\s+)?{_WAKE_ALIASES}[,.\s!?]*\s*(.*)",
    re.IGNORECASE,
)
_OVERRIDE_PATTERN = re.compile(r"\b(stop|cancel|instead|forget that|never mind|nevermind|wait|change that|changed my mind|changed mind|don't do|dont do|undo)\b", re.IGNORECASE)
_ADDITIVE_PATTERN = re.compile(r"\b(also|after that|then|next)\b", re.IGNORECASE)
_UNAMBIGUOUS_VERBS = frozenset({
    "create", "list", "delete", "add", "update", "edit", "rename", "remove", "make", "show", "write",
})
_REFERENTIAL_TERMS = (" it ", " that ", " this ", " same ", "the one", "the page")

_MAX_GENERAL_HISTORY = 3  # turns

# CONFIDENCE-01: markers used to detect low-confidence answers (future enhancement)
_LOW_CONFIDENCE_MARKERS = (
    "i'm not sure", "i don't know", "i'm not certain",
    "i can't confirm", "i don't have", "i'm unable to verify",
    "it's unclear", "i couldn't find", "that's hard to say",
    "i'm not confident", "i may be wrong",
)


def _add_confidence_signal(answer: str) -> str:
    """Prepend a hedging phrase if the answer contains low-confidence markers.

    This makes Jarvis sound more trustworthy by signaling uncertainty explicitly
    rather than presenting uncertain information as fact.
    """
    if not JARVIS_CONFIDENCE_SIGNAL_ENABLED:
        return answer
    normalized = answer.lower()
    if any(marker in normalized for marker in _LOW_CONFIDENCE_MARKERS):
        # Already has hedging language — no need to double-hedge
        return answer
    # Additional confidence signal logic can be added here in future
    return answer


def _build_meeting_context_for_edit() -> str:
    """Return the last 10 transcript entries formatted as 'Speaker: text' lines, capped at 2000 chars."""
    transcript_log: list = meeting_state.get("transcript_log", [])
    if not transcript_log:
        return ""
    recent = transcript_log[-10:]
    lines = [f"{entry.get('participant', 'Unknown')}: {entry.get('text', '')}" for entry in recent]
    context = "\n".join(lines)
    if len(context) > 2000:
        context = context[-2000:]
    return context


def _current_confluence_graph_user_id() -> str:
    auth_user_id = meeting_state.get("auth_user_id")
    if auth_user_id:
        return f"supabase:{auth_user_id}"
    return f"session:{meeting_state.get('session_id') or 'default'}"


def _is_confluence_read_query(text: str) -> bool:
    normalized = (text or "").strip()
    if not normalized:
        return False
    if _CONFLUENCE_MUTATION_PATTERN.search(normalized):
        return False
    return bool(_CONFLUENCE_READ_PATTERN.search(normalized))


async def _queue_confluence_proposal(task: VoiceTask, prepared_request: str) -> str:
    from confluence_logic.review import api as review_api  # noqa: PLC0415

    state = meeting_state
    transcript_log = list(state.get("transcript_log") or [])
    transcript_text = review_api._format_transcript(transcript_log, review_api.JARVIS_REVIEW_MAX_INPUT_CHARS)
    meeting_context = _build_meeting_context_for_edit()
    summary = {
        "title": state.get("meeting_url") or "Live meeting",
        "summary": meeting_context,
        "key_topics": [],
        "decisions": [],
        "action_items": [],
        "mom": [],
        "participants": [],
    }
    proposal_query = prepared_request.strip() or task.request
    agent = ProposedChangesAgent(
        model=review_api.JARVIS_REVIEW_MODEL,
        client_factory=get_openai_client,
        max_tokens=review_api.JARVIS_PROPOSE_CHANGES_MAX_TOKENS,
    )
    proposals = await agent.propose(
        transcript_text=transcript_text or meeting_context,
        summary=summary,
        query=proposal_query,
        graph_user_id=_current_confluence_graph_user_id(),
    )
    generated = review_api._append_agent_generated_changes(
        state,
        proposals,
        state.get("session_id"),
        proposal_query,
        source="in_meeting_proposal_agent",
    )
    if generated:
        return f"Queued {len(generated)} proposed Confluence update{'s' if len(generated) != 1 else ''} for review after the meeting."
    return "I could not find a concrete Confluence change to queue from that request."


async def _answer_confluence_question(query: str) -> str:
    graph_user_id = _current_confluence_graph_user_id()
    try:
        await asyncio.wait_for(confluence_page_graph.ensure_user_confluence_graph(graph_user_id), timeout=0.7)
    except (asyncio.TimeoutError, Exception):
        asyncio.create_task(confluence_page_graph.ensure_user_confluence_graph(graph_user_id))

    if re.search(r"\b(?:list|show|what).*(?:pages|documents|docs)\b", query, re.IGNORECASE):
        pages = await confluence_page_graph.list_user_confluence_pages(graph_user_id, limit=10)
        if pages:
            titles = ", ".join(page.get("title") or "Untitled" for page in pages[:10])
            return f"I found these Confluence pages in the graph: {titles}."

    try:
        contexts = await asyncio.wait_for(
            confluence_page_graph.query_user_confluence_graph(graph_user_id, query, limit=6),
            timeout=0.8,
        )
    except (asyncio.TimeoutError, Exception):
        contexts = []

    if not contexts:
        return "I could not find a relevant Confluence page for that yet. The workspace graph may still be refreshing."

    context_text = json.dumps(
        [
            {
                "page_title": item.get("title"),
                "heading": item.get("heading"),
                "content": item.get("relevant_content"),
            }
            for item in contexts
        ],
        ensure_ascii=False,
    )
    response = await asyncio.to_thread(
        lambda: get_openai_client().chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You answer questions about Confluence pages using only the retrieved page/section context. "
                        "Keep the answer concise and useful for spoken delivery. If context is insufficient, say so."
                    ),
                },
                {"role": "user", "content": f"Question: {query}\n\nRetrieved Confluence context:\n{context_text}"},
            ],
            max_tokens=220,
            temperature=0.2,
        )
    )
    return (response.choices[0].message.content or "").strip() or "I could not answer that from the Confluence graph."


async def _handle_confluence_question(query: str, bot_id: str) -> None:
    generation = meeting_state["output_generation"]
    try:
        answer_task = asyncio.create_task(_answer_confluence_question(query))
        gap_filler_task = asyncio.create_task(_speak_gap_filler(query, bot_id, generation))
        answer = await answer_task
        await gap_filler_task
        await _speak_guarded(answer, bot_id, generation, allow_stale=True)
        if answer:
            meeting_state["last_jarvis_response"] = {"intent": "confluence_question", "query": query, "answer": answer}
    except Exception as exc:
        logger.error("Confluence question handling failed: %s", exc)
        await _speak_guarded("I hit an issue reading the Confluence graph.", bot_id, generation, allow_stale=True)


async def _confirm_delete_gate(task: VoiceTask) -> bool:
    """Ask the user to confirm a delete operation. Returns True if confirmed, False if denied/timeout."""
    if not JARVIS_DELETE_CONFIRM_ENABLED:
        return True
    generation = task.output_generation
    await _speak_guarded("Are you sure you want to delete that? Say yes to confirm.", task.bot_id, generation)
    task.answer_future = asyncio.get_running_loop().create_future()
    meeting_state["pending_clarification"] = {
        "task_id": task.task_id,
        "clarification_context": "Delete confirmation pending",
        "question": "Are you sure you want to delete that?",
    }
    try:
        answer = await asyncio.wait_for(task.answer_future, timeout=JARVIS_DELETE_CONFIRM_TIMEOUT)
    except asyncio.TimeoutError:
        await _speak_guarded("No confirmation received. Cancelling the delete.", task.bot_id, generation)
        meeting_state["pending_clarification"] = None
        return False
    finally:
        task.answer_future = None
    meeting_state["pending_clarification"] = None
    normalized = (answer or "").strip().lower()
    if normalized in ("yes", "yeah", "yep", "sure", "confirm", "do it", "go ahead", "yes please"):
        return True
    await _speak_guarded("Got it, I won't delete that.", task.bot_id, generation)
    return False


def _remember_general_exchange(question: str, answer: str) -> None:
    """Store a general Q&A exchange in meeting_state for context continuity."""
    # Build a new list to avoid mutating the shared reference mid-read
    history = list(meeting_state["general_history"])
    history.append((question.strip(), answer.strip()))
    if len(history) > _MAX_GENERAL_HISTORY:
        history = history[-_MAX_GENERAL_HISTORY:]
    meeting_state["general_history"] = history


def _format_general_history() -> str:
    """Format the general Q&A history as 'User/Assistant' lines."""
    history: list = meeting_state["general_history"]
    if not history:
        return "[none]"
    lines = []
    for user_text, assistant_text in history[-_MAX_GENERAL_HISTORY:]:
        lines.append(f"User: {user_text}")
        lines.append(f"Assistant: {assistant_text}")
    return "\n".join(lines)


def _is_followup(query: str) -> bool:
    """Return True if the query looks like a follow-up to a prior Jarvis answer.

    Uses word-boundary matching for short ambiguous words like 'it' and 'that'
    to avoid false positives (e.g., 'iterate' should not match 'it').
    """
    lowered = query.lower()
    # Word-boundary check for short, ambiguous words
    if re.search(r"\bit\b", lowered) or re.search(r"\bthat\b", lowered):
        return True
    # Simple substring check is safe for longer, unambiguous words
    follow_keywords = ("simpler", "explain", "elaborate", "again", "more", "rephrase", "clarify")
    return any(kw in lowered for kw in follow_keywords)


def _is_garbled_query(text: str) -> bool:
    """Detect likely garbled/unintelligible transcription noise.

    Heuristics:
    - Very short (1-2 chars) after stripping
    - Mostly non-alphabetic characters
    - No recognizable English words (all tokens < 2 chars)
    - Excessive repetition of single characters
    """
    if not JARVIS_GARBLED_RECOVERY_ENABLED:
        return False
    stripped = text.strip()
    if not stripped:
        return True
    # Too short to be meaningful (single char or two chars)
    if len(stripped) <= 2:
        return True
    # Ratio of alphabetic characters is very low
    alpha_count = sum(1 for c in stripped if c.isalpha())
    if len(stripped) > 3 and alpha_count / len(stripped) < 0.4:
        return True
    # All "words" are single characters (e.g., "a b c d")
    words = stripped.split()
    if len(words) >= 3 and all(len(w) <= 1 for w in words):
        return True
    # Excessive character repetition (e.g., "aaaaaa", "uh uh uh uh")
    if len(set(stripped.lower().replace(" ", ""))) <= 2 and len(stripped) > 4:
        return True
    return False


def _is_unambiguous_request(text: str) -> bool:
    """Return True for requests clear enough to skip the planning LLM call."""
    normalized = (text or "").strip().lower()
    words = normalized.split()
    if len(words) < 5:
        return False
    if words[0] not in _UNAMBIGUOUS_VERBS:
        return False
    padded = f" {normalized} "
    return not any(term in padded for term in _REFERENTIAL_TERMS)


_STATUS_PATTERN = re.compile(
    r"\b(what are you|what're you|whatcha|status|working on|doing right now"
    r"|what.*doing|are you busy|how.*going|update me|progress|what.*task"
    r"|are you idle|anything pending|what.*queue)",
    re.IGNORECASE,
)
_task_counter = count(1)
_state_locks: dict[str, tuple[Any, asyncio.Lock]] = {}
_output_locks: dict[str, tuple[Any, asyncio.Lock]] = {}


def get_openai_client() -> OpenAI:
    global _openai_client
    if _openai_client is None:
        _openai_client = OpenAI()
    return _openai_client


def _get_state_lock() -> asyncio.Lock:
    loop = asyncio.get_running_loop()
    session_id = meeting_state.get("session_id") or _DEFAULT_SESSION_ID
    lock_entry = _state_locks.get(session_id)
    if lock_entry is None or lock_entry[0] is not loop:
        lock_entry = (loop, asyncio.Lock())
        _state_locks[session_id] = lock_entry
    return lock_entry[1]


def _get_output_lock() -> asyncio.Lock:
    loop = asyncio.get_running_loop()
    session_id = meeting_state.get("session_id") or _DEFAULT_SESSION_ID
    lock_entry = _output_locks.get(session_id)
    if lock_entry is None or lock_entry[0] is not loop:
        lock_entry = (loop, asyncio.Lock())
        _output_locks[session_id] = lock_entry
    return lock_entry[1]


def _set_current_task(task: Optional[VoiceTask]) -> None:
    meeting_state["current_task"] = task
    meeting_state["current_task_id"] = task.task_id if task else None
    meeting_state["current_phase"] = task.phase if task else "idle"
    meeting_state["current_request"] = task.request if task else None
    meeting_state["mutation_started"] = task.mutation_started if task else False
    meeting_state["cancel_requested"] = task.cancel_requested if task else False


def _set_task_phase(task: VoiceTask, phase: str) -> None:
    task.phase = phase
    current = meeting_state.get("current_task")
    if current is not None and current.task_id == task.task_id:
        meeting_state["current_phase"] = phase


def _update_task_request(task: VoiceTask, request: str) -> None:
    task.request = request.strip()
    current = meeting_state.get("current_task")
    if current is not None and current.task_id == task.task_id:
        meeting_state["current_request"] = task.request


def _next_output_generation() -> int:
    meeting_state["output_generation"] += 1
    return meeting_state["output_generation"]


def _new_voice_task(request: str, bot_id: str) -> VoiceTask:
    return VoiceTask(
        task_id=next(_task_counter),
        request=request.strip(),
        bot_id=bot_id,
        created_at=time.time(),
    )


async def _generate_dynamic_ack(context: str, new_request: str = "", current_request: str = "") -> str:
    """Generate a short, natural acknowledgement via a fast LLM call instead of hardcoded text."""
    prompts = {
        "queued": (
            f"You're currently working on: '{current_request}'. "
            f"The user just asked: '{new_request}'. "
            "Generate a very short (4-8 words) natural voice acknowledgement that you'll handle "
            "their new request after you finish the current one. Be warm and natural."
        ),
        "switching": (
            f"You were working on: '{current_request}'. "
            f"The user wants you to switch to: '{new_request}'. "
            "Generate a very short (3-6 words) natural voice acknowledgement that you're switching tasks."
        ),
        "error": (
            f"You hit a technical error while working on: '{new_request}'. "
            "Generate a very short (5-10 words) natural voice apology. Be straightforward."
        ),
    }
    system_prompt = (
        "You are Jarvis, a professional voice assistant in a meeting. "
        "Respond with ONLY the short acknowledgement text. No quotes, no extra text."
    )
    user_prompt = prompts.get(context, f"Generate a very short natural acknowledgement for: {context}")

    try:
        response = await asyncio.to_thread(
            lambda: get_openai_client().chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=30,
                temperature=0.7,
            )
        )
        text = (response.choices[0].message.content or "").strip().strip('"\'')
        return text or _DYNAMIC_ACK_FALLBACKS.get(context, "Got it.")
    except Exception as e:
        logger.warning("Dynamic ack generation failed: %s", e)
        return _DYNAMIC_ACK_FALLBACKS.get(context, "Got it.")


def _looks_like_clarification_prompt(text: str) -> bool:
    normalized = (text or "").strip().lower()
    if not normalized or not normalized.endswith("?"):
        return False

    markers = (
        "which page",
        "what page",
        "do you mean",
        "which one",
        "what should i change",
        "what would you like me to change",
        "what should i update",
        "which title",
        "which section",
        "could you confirm",
        "can you confirm",
        "please confirm",
        "what page name",
    )
    return any(marker in normalized for marker in markers)


def _is_override_request(text: str) -> bool:
    return bool(_OVERRIDE_PATTERN.search((text or "").lower()))


def _is_additive_request(text: str) -> bool:
    return bool(_ADDITIVE_PATTERN.search((text or "").lower()))


def build_transcript_provider_config() -> dict:
    if RECALL_TRANSCRIPT_PROVIDER == "recallai_streaming":
        return {
            "recallai_streaming": {
                "mode": STREAMING_MODE,
                "language_code": LANGUAGE_CODE,
            }
        }

    if RECALL_TRANSCRIPT_PROVIDER in ("assembly_ai_v3", "assembly_ai_v3_streaming"):
        return {
            "assembly_ai_v3_streaming": {
                "language_code": LANGUAGE_CODE,
                "speech_model": os.getenv("ASSEMBLY_SPEECH_MODEL", "u3-rt-pro"),
            }
        }

    return {
        RECALL_TRANSCRIPT_PROVIDER: {
            "mode": STREAMING_MODE,
            "language_code": LANGUAGE_CODE,
        }
    }


def build_create_bot_payload(meeting_url: str, session_id: Optional[str] = None) -> dict:
    stream_path = f"/recall-audio-stream/{session_id}" if session_id else "/recall-audio-stream"
    ws_url = WEBHOOK_URL.replace("https://", "wss://") + stream_path
    return {
        "meeting_url": meeting_url,
        "bot_name": BOT_NAME,
        "metadata": {
            "session_id": session_id or _DEFAULT_SESSION_ID,
        },
        "recording_config": {
            "transcript": {
                "provider": build_transcript_provider_config()
            },
            "realtime_endpoints": [
                {
                    "type": "websocket",
                    "url": ws_url,
                    "events": ["transcript.data"],
                }
            ],
        },
    }


def create_bot(meeting_url: str, session_id: Optional[str] = None) -> Optional[str]:
    payload = build_create_bot_payload(meeting_url, session_id=session_id)
    try:
        response = requests.post(
            f"{RECALL_BASE_URL}/bot/",
            headers={"Authorization": f"Token {RECALL_API_KEY}", "Content-Type": "application/json"},
            json=payload,
            timeout=10,
        )
        if response.status_code in [200, 201]:
            bot_id = response.json()["id"]
            if session_id:
                bind_bot_to_session(bot_id, session_id)
            logger.info("Bot created: %s", bot_id)
            return bot_id
        logger.error("Bot creation failed: %s", response.text)
        return None
    except Exception as e:
        logger.error("Bot creation exception: %s", e)
        return None


def synthesize_speech(text: str) -> bytes:
    if JARVIS_TTS_PROVIDER == "gtts":
        buffer = BytesIO()
        gTTS(text, lang=LANGUAGE_CODE).write_to_fp(buffer)
        return buffer.getvalue()

    if JARVIS_TTS_PROVIDER == "edge_tts":
        try:
            import edge_tts as _edge_tts
            # OpenAI-only voice names are invalid for edge_tts; use a good default instead
            _OPENAI_ONLY_VOICES = {"echo", "alloy", "fable", "onyx", "nova", "shimmer"}
            voice = (
                JARVIS_TTS_VOICE
                if JARVIS_TTS_VOICE and JARVIS_TTS_VOICE not in _OPENAI_ONLY_VOICES
                else "en-US-GuyNeural"
            )
            communicate = _edge_tts.Communicate(text, voice)
            chunks = []
            async def _collect():
                async for chunk in communicate.stream():
                    if chunk["type"] == "audio":
                        chunks.append(chunk["data"])
            asyncio.run(_collect())
            return b"".join(chunks)
        except ImportError:
            logger.warning("edge_tts not installed, falling back to OpenAI TTS. Run: pip install edge-tts")

    with get_openai_client().audio.speech.with_streaming_response.create(
        model=JARVIS_TTS_MODEL,
        voice=JARVIS_TTS_VOICE,
        input=text,
        response_format="mp3",
        speed=JARVIS_TTS_SPEED,
    ) as response:
        return response.read()


def speak(text: str, bot_id: str) -> bool:
    try:
        audio_bytes = synthesize_speech(text)
        audio_b64 = base64.b64encode(audio_bytes).decode()
        resp = requests.post(
            f"{RECALL_BASE_URL}/bot/{bot_id}/output_audio/",
            headers={"Authorization": f"Token {RECALL_API_KEY}", "Content-Type": "application/json"},
            json={"kind": "mp3", "b64_data": audio_b64},
            timeout=10,
        )
        return resp.status_code == 200
    except Exception as e:
        logger.error("speak() error: %s", e)
        if JARVIS_TTS_PROVIDER != "gtts":
            try:
                buffer = BytesIO()
                gTTS(text, lang=LANGUAGE_CODE).write_to_fp(buffer)
                resp = requests.post(
                    f"{RECALL_BASE_URL}/bot/{bot_id}/output_audio/",
                    headers={"Authorization": f"Token {RECALL_API_KEY}", "Content-Type": "application/json"},
                    json={"kind": "mp3", "b64_data": base64.b64encode(buffer.getvalue()).decode()},
                    timeout=10,
                )
                return resp.status_code == 200
            except Exception as fallback_error:
                logger.error("gTTS fallback failed: %s", fallback_error)
        return False


def speak_cached_audio(audio_bytes: bytes, bot_id: str) -> bool:
    """Send pre-cached MP3 audio bytes directly to the Recall bot, skipping TTS synthesis."""
    try:
        audio_b64 = base64.b64encode(audio_bytes).decode()
        resp = requests.post(
            f"{RECALL_BASE_URL}/bot/{bot_id}/output_audio/",
            headers={"Authorization": f"Token {RECALL_API_KEY}", "Content-Type": "application/json"},
            json={"kind": "mp3", "b64_data": audio_b64},
            timeout=10,
        )
        return resp.status_code == 200
    except Exception as e:
        logger.error("speak_cached_audio() error: %s", e)
        return False


def _get_audio_duration(audio_bytes: bytes) -> float:
    """Compute exact MP3 playback duration by counting all MPEG Layer III frames.

    Algorithm:
    1. Skip ID3v2 tag to reach the first frame sync.
    2. Check for a Xing/Info VBR header in the first frame — if present, use
       stored frame_count × 1152 / sample_rate for exact VBR duration.
    3. Otherwise hop frame-by-frame (CBR: all frames same size ± 1 padding byte),
       count them, and return frame_count × 1152 / sample_rate.
    4. Final fallback: file_size / (bitrate / 8).

    This gives exact duration regardless of TTS provider, voice, or speed setting.
    """
    _BITRATES = [0, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320, 0]
    _SAMPLERATES = [44100, 48000, 32000, 0]
    SAMPLES_PER_FRAME = 1152  # MPEG-1 Layer III constant
    MIN_DURATION = 0.1

    data = audio_bytes
    if not data:
        return MIN_DURATION

    # Step 1: skip ID3v2 tag
    offset = 0
    if data[:3] == b"ID3" and len(data) >= 10:
        sz = (
            (data[6] & 0x7F) << 21
            | (data[7] & 0x7F) << 14
            | (data[8] & 0x7F) << 7
            | (data[9] & 0x7F)
        )
        offset = 10 + sz

    # Step 2: find first valid frame header
    first_bitrate = 0
    first_samplerate = 0
    first_pos = offset
    pos = offset
    while pos + 4 < len(data):
        b = data[pos:pos + 4]
        if b[0] == 0xFF and (b[1] & 0xE0) == 0xE0:
            layer = (b[1] >> 1) & 0x3
            bi = (b[2] >> 4) & 0xF
            si = (b[2] >> 2) & 0x3
            padding = (b[2] >> 1) & 0x1
            if layer == 1 and 0 < bi < 15 and si < 3:
                bitrate = _BITRATES[bi] * 1000
                samplerate = _SAMPLERATES[si]
                if bitrate > 0 and samplerate > 0:
                    first_bitrate = bitrate
                    first_samplerate = samplerate
                    first_pos = pos
                    # Step 3: check for Xing/Info VBR header
                    # Offset within frame: 4-byte header + 32-byte side info (stereo MPEG-1)
                    xing_off = pos + 36
                    if xing_off + 12 <= len(data):
                        tag = data[xing_off:xing_off + 4]
                        if tag in (b"Xing", b"Info"):
                            flags = int.from_bytes(data[xing_off + 4:xing_off + 8], "big")
                            if flags & 0x1:  # Frames field present
                                frame_count = int.from_bytes(data[xing_off + 8:xing_off + 12], "big")
                                if frame_count > 0:
                                    return max(MIN_DURATION, frame_count * SAMPLES_PER_FRAME / samplerate)
                    break
        pos += 1

    if not first_bitrate or not first_samplerate:
        # No valid frame found — fallback to 224 kbps assumption
        return max(MIN_DURATION, len(data) / 28000)

    # Step 4: CBR frame-hopping — count all frames
    frame_count = 0
    pos = first_pos
    while pos + 4 < len(data):
        b = data[pos:pos + 4]
        if b[0] == 0xFF and (b[1] & 0xE0) == 0xE0:
            layer = (b[1] >> 1) & 0x3
            bi = (b[2] >> 4) & 0xF
            si = (b[2] >> 2) & 0x3
            padding = (b[2] >> 1) & 0x1
            if layer == 1 and 0 < bi < 15 and si < 3:
                bitrate = _BITRATES[bi] * 1000
                samplerate = _SAMPLERATES[si]
                if bitrate > 0 and samplerate > 0:
                    frame_size = 144 * bitrate // samplerate + padding
                    frame_count += 1
                    pos += max(frame_size, 1)
                    continue
        pos += 1

    if frame_count > 0:
        return max(MIN_DURATION, frame_count * SAMPLES_PER_FRAME / first_samplerate)

    # Final fallback: size / bitrate
    return max(MIN_DURATION, (len(data) - offset) / (first_bitrate / 8))


async def _speak_cached_guarded(audio_bytes: bytes, bot_id: str, generation: int) -> bool:
    """Like _speak_guarded but sends pre-cached audio bytes instead of synthesizing."""
    output_lock = _get_output_lock()
    async with output_lock:
        if generation != meeting_state["output_generation"]:
            return False
        while True:
            remaining = meeting_state["last_user_speech_at"] + JARVIS_SPEECH_HOLD_SECONDS - time.time()
            if remaining <= 0:
                break
            await asyncio.sleep(min(remaining, 0.2))
            if generation != meeting_state["output_generation"]:
                return False
        ok = await asyncio.to_thread(speak_cached_audio, audio_bytes, bot_id)
        if ok:
            duration = _playback_wait_seconds(audio_bytes)
            elapsed = 0.0
            while elapsed < duration:
                await asyncio.sleep(0.1)
                elapsed += 0.1
                if generation != meeting_state["output_generation"]:
                    break
        return ok


def _format_pending_clarification(task: Optional[VoiceTask] = None) -> str:
    current_task = task or meeting_state.get("current_task")
    if current_task is None:
        return ""

    pending = meeting_state.get("pending_clarification")
    pending_for_task = None
    if pending and pending.get("task_id") == current_task.task_id:
        pending_for_task = pending

    lines = [
        f"Original request: {current_task.request}",
        f"Clarification context: {(pending_for_task or {}).get('clarification_context', current_task.clarification_context)}",
        f"Last question asked: {(pending_for_task or {}).get('question', current_task.question)}",
        f"Latest user reply: {current_task.latest_user_reply}",
    ]
    return "\n".join(line for line in lines if line.split(":", 1)[1].strip())


def _append_clarification_context(existing: str, addition: str) -> str:
    extra = (addition or "").strip()
    if not extra:
        return existing
    if not existing:
        return extra
    if extra in existing:
        return existing
    return f"{existing}\n{extra}"


def _record_clarification_answer(task: VoiceTask, answer: str) -> None:
    answer_text = (answer or "").strip()
    if not answer_text:
        return
    task.latest_user_reply = answer_text
    task.clarification_context = _append_clarification_context(
        task.clarification_context,
        f"User answer: {answer_text}",
    )


def _has_other_pending_work(task: VoiceTask) -> bool:
    current = meeting_state.get("current_task")
    if current is not None and current.task_id != task.task_id:
        return True

    if any(queued.task_id != task.task_id for queued in meeting_state["pending_requests"]):
        return True

    return False


def _build_request_reference(task: VoiceTask) -> str:
    request = (task.request or "").strip()
    normalized = request.lower()

    title_match = re.search(r"(?:title|rename|retitle).*?\bto\b\s+(.+)$", request, re.IGNORECASE)
    if title_match:
        new_title = title_match.group(1).strip().rstrip(".!?")
        if new_title:
            return f"the title change to {new_title}"
        return "the title change"

    if any(word in normalized for word in ("create", "new page")):
        topic_match = re.search(r"(?:about|on|called|titled)\s+(.+)$", request, re.IGNORECASE)
        if topic_match:
            topic = topic_match.group(1).strip().rstrip(".!?")
            if topic:
                return f"the page request about {topic}"
        return "the page creation request"

    if any(word in normalized for word in ("delete", "remove")):
        return "the removal request"

    if any(word in normalized for word in ("add", "update", "edit", "change", "replace", "rewrite")):
        words = request.split()
        trimmed = " ".join(words[:10]).strip().rstrip(",")
        if len(words) > 10:
            trimmed = f"{trimmed}"
        if trimmed:
            return f"the request to {trimmed.lower()}"
        return "the update request"

    return "this request"


def _split_into_sentences(text: str) -> list:
    """Split text into sentences for pipelined TTS delivery."""
    parts = re.split(r'(?<=[.!?])\s+', (text or "").strip())
    return [p.strip() for p in parts if p.strip()]


def _playback_wait_seconds(audio_bytes: bytes) -> float:
    """Return how long to hold the output lock after posting audio to Recall."""
    return _get_audio_duration(audio_bytes) + max(0.0, JARVIS_RECALL_AUDIO_DRAIN_BUFFER_SECONDS)


async def _speak_guarded(text: str, bot_id: str, generation: int, allow_stale: bool = False,
                         _preloaded_all: Optional[list] = None) -> bool:
    """Speak text as a single concatenated MP3 clip to eliminate inter-sentence overlap.

    All sentences are synthesised in parallel, their raw MP3 bytes are joined, and the
    result is sent to Recall.ai in one POST.  Because Recall.ai receives a single
    continuous audio stream there is no timing-based gap needed between sentences.

    _preloaded_all: list of pre-synthesized audio bytes for each sentence, in order.
    When provided, skips all TTS synthesis entirely.
    """
    output_lock = _get_output_lock()
    async with output_lock:
        if not allow_stale and generation != meeting_state["output_generation"]:
            return False
        while True:
            remaining = meeting_state["last_user_speech_at"] + JARVIS_SPEECH_HOLD_SECONDS - time.time()
            if remaining <= 0:
                break
            await asyncio.sleep(min(remaining, 0.2))
            if not allow_stale and generation != meeting_state["output_generation"]:
                return False

        sentences = _split_into_sentences(text)
        if not sentences:
            return False

        # Collect audio for all sentences
        if _preloaded_all and len(_preloaded_all) >= len(sentences):
            all_audio: list = list(_preloaded_all[:len(sentences)])
        else:
            # Synthesise all sentences in parallel, then gather in order
            tasks = [
                asyncio.create_task(asyncio.to_thread(synthesize_speech, s))
                for s in sentences
            ]
            if not allow_stale and generation != meeting_state["output_generation"]:
                for t in tasks:
                    t.cancel()
                return False
            all_audio = list(await asyncio.gather(*tasks))

        if not allow_stale and generation != meeting_state["output_generation"]:
            return False

        # Concatenate into one MP3 stream — single POST, zero timing gaps needed
        combined = b"".join(all_audio)
        ok = await asyncio.to_thread(speak_cached_audio, combined, bot_id)
        if not ok:
            return False

        _record_jarvis_transcript(" ".join(sentences))

        # Wait for the exact playback duration of the combined clip
        duration = _playback_wait_seconds(combined)
        elapsed = 0.0
        while elapsed < duration:
            await asyncio.sleep(0.1)
            elapsed += 0.1
            if generation != meeting_state["output_generation"]:
                return True

        return True


async def _speak_filler(bot_id: str, generation: int) -> None:
    """Speak a random filler phrase before a slow operation."""
    phrase = random.choice(JARVIS_FILLER_PHRASES)
    await _speak_guarded(phrase, bot_id, generation, allow_stale=True)


async def _speak_streaming(
    sentence_gen,
    gap_filler_task: asyncio.Task,
    bot_id: str,
    generation: int,
) -> Optional[str]:
    """Consume a streaming LLM response, then play it as one Recall audio clip.

    Recall's output_audio endpoint receives complete MP3 uploads, not a true
    realtime stream. Sending one upload per sentence can overlap in the meeting
    because local duration estimates and Recall playback buffering can drift.
    This keeps the low-latency LLM/TTS pipeline, but concatenates all sentence
    MP3s and sends a single output_audio request.
    """
    syn_queue: asyncio.Queue = asyncio.Queue()

    async def _produce():
        try:
            async for sentence in sentence_gen:
                if generation != meeting_state["output_generation"]:
                    break
                audio = await asyncio.to_thread(synthesize_speech, sentence)
                await syn_queue.put((sentence, audio))
        except Exception as exc:
            logger.error("Streaming TTS producer failed: %s", exc)
        finally:
            await syn_queue.put(None)

    producer_task = asyncio.create_task(_produce())

    full_sentences = []
    all_audio = []

    try:
        while True:
            if generation != meeting_state["output_generation"]:
                producer_task.cancel()
                return None

            try:
                item = await asyncio.wait_for(syn_queue.get(), timeout=0.5)
            except asyncio.TimeoutError:
                continue

            if item is None:
                break

            sentence_text, audio_bytes = item
            full_sentences.append(sentence_text)
            all_audio.append(audio_bytes)
    finally:
        try:
            await producer_task
        except asyncio.CancelledError:
            pass

    await gap_filler_task

    if not all_audio:
        return None

    output_lock = _get_output_lock()
    async with output_lock:
        if generation != meeting_state["output_generation"]:
            return None

        combined = b"".join(all_audio)
        ok = await asyncio.to_thread(speak_cached_audio, combined, bot_id)
        if not ok:
            return None

        spoken_answer = " ".join(full_sentences) or None
        if spoken_answer:
            _record_jarvis_transcript(spoken_answer)

        duration = _playback_wait_seconds(combined)
        elapsed = 0.0
        while elapsed < duration:
            await asyncio.sleep(0.1)
            elapsed += 0.1
            if generation != meeting_state["output_generation"]:
                return spoken_answer

    await asyncio.sleep(JARVIS_POST_SPEECH_PAUSE_SECONDS)
    return spoken_answer


async def _speak_gap_filler(query: str, bot_id: str, generation: int) -> None:
    """Bridge the gap between wake and answer with at most one filler per generation.

    Fast path: plays a random pre-generated MP3 from assets/audio/gap_filler_*.mp3
    (zero TTS latency — audio is already in memory).

    Fallback: calls the LLM to generate a contextual phrase, then speaks it via TTS.
    This fallback fires only when the audio cache hasn't been populated yet (e.g. the
    generate_wav_assets script hasn't been run).
    """
    play_wake_ack = False
    wait_for_existing: Optional[asyncio.Task] = None
    current_task = asyncio.current_task()
    async with _get_state_lock():
        if meeting_state.get("gap_filler_generation") == generation:
            existing = meeting_state.get("active_gap_filler_task")
            if existing is not None and existing is not current_task and not existing.done():
                wait_for_existing = existing
            else:
                logger.debug("Gap filler skipped — already played for generation %s", generation)
                return
        else:
            meeting_state["gap_filler_generation"] = generation
            meeting_state["active_gap_filler_task"] = current_task
            play_wake_ack = bool(meeting_state.get("wake_query_ack_pending"))
            meeting_state["wake_query_ack_pending"] = False

    if wait_for_existing is not None:
        try:
            await wait_for_existing
        except asyncio.CancelledError:
            pass
        return

    try:
        if play_wake_ack:
            cached_ack = get_random_ack_audio()
            if cached_ack and _get_audio_duration(cached_ack[1]) <= _MICRO_ACK_MAX_SECONDS:
                await _speak_cached_guarded(cached_ack[1], bot_id, generation)
            else:
                await _speak_guarded(JARVIS_WAKE_ACK, bot_id, generation, allow_stale=True)

        cached = get_random_filler_audio()
        if cached:
            _, audio_bytes = cached
            await _speak_cached_guarded(audio_bytes, bot_id, generation)
            return
        # Cache miss — fall back to LLM-generated contextual filler
        filler = await _generate_contextual_gap_filler(query, invoker_name=_get_clean_invoker_name())
        await _speak_guarded(filler, bot_id, generation, allow_stale=True)
    finally:
        async with _get_state_lock():
            if meeting_state.get("active_gap_filler_task") is current_task:
                meeting_state["active_gap_filler_task"] = None


_MICRO_ACK_MAX_SECONDS = 1.2  # cached clips longer than this are gap-fillers, not acks


async def _emit_micro_ack(bot_id: str) -> None:
    """Emit an ultra-short acknowledgment to fill dead air on wake detection.

    Guards:
    - Skips if the output_lock is already held (avoid overlap with a playing answer).
    - Skips if the best cached clip is a long gap-filler phrase (>1.2 s); those are
      reserved for handler gap-fill use only.  Without dedicated short ack files the
      TTS micro-ack text is used as fallback so users don't hear two long fillers.
    """
    if not JARVIS_MICRO_ACK_ENABLED:
        return
    output_lock = _get_output_lock()
    if output_lock.locked():
        logger.debug("Micro-ack skipped — output_lock held")
        return
    try:
        cached = get_random_ack_audio()
        if cached:
            _, ack_bytes = cached
            if _get_audio_duration(ack_bytes) > _MICRO_ACK_MAX_SECONDS:
                # All cached files are long gap-fillers; the handler will play one shortly.
                # Suppress the micro-ack so the user doesn't hear two filler phrases.
                logger.debug("Micro-ack skipped — cached audio too long (gap-filler only cache)")
                return
            await asyncio.to_thread(speak_cached_audio, ack_bytes, bot_id)
        else:
            await asyncio.to_thread(speak, JARVIS_MICRO_ACK_TEXT, bot_id)
        logger.debug("Micro-ack emitted for bot %s", bot_id)
    except Exception as e:
        logger.debug("Micro-ack failed (non-fatal): %s", e)


async def _handle_interruption(bot_id: str) -> None:
    """Cancel current TTS and emit a yield phrase when user speaks mid-speech."""
    if not JARVIS_INTERRUPT_RECOVERY_ENABLED:
        return
    try:
        # Bump generation to cancel any in-flight _speak_guarded waits
        _next_output_generation()
        # Speak yield phrase with new generation (allow_stale=True so it always plays)
        gen = meeting_state["output_generation"]
        await _speak_guarded(JARVIS_YIELD_PHRASE, bot_id, gen, allow_stale=True)
        logger.info("Interruption recovery: yield phrase emitted")
    except Exception as e:
        logger.debug("Interruption recovery failed (non-fatal): %s", e)


async def _debounced_dispatch(query: str, bot_id: str) -> None:
    """Wait JARVIS_DEBOUNCE_SECONDS then dispatch to handle_spoken_request and clear invoker lock.

    Per D-06: cancellable task. Per D-08: clears invoker_participant and _pending_debounce_task after firing.
    """
    await asyncio.sleep(JARVIS_DEBOUNCE_SECONDS)
    if meeting_state.get("wake_query_ack_pending"):
        generation = meeting_state["output_generation"]
        asyncio.create_task(_speak_gap_filler(query, bot_id, generation))
        await asyncio.sleep(0)
    meeting_state["invoker_participant"] = None
    meeting_state["_pending_debounce_task"] = None
    meeting_state["_accumulated_query"] = ""
    # Skip garbled queries early (before classification cost)
    if _is_garbled_query(query):
        logger.info("Garbled query detected in debounce, asking to repeat: %s", repr(query[:40]))
        generation = meeting_state["output_generation"]
        await _speak_guarded("Sorry, I didn't catch that. Could you say that again?", bot_id, generation, allow_stale=True)
        return
    await handle_spoken_request(query, bot_id)


def _get_clean_invoker_name() -> str:
    """Return the invoker's first name if it looks like a real human name, else empty string."""
    raw = meeting_state.get("invoker_participant") or ""
    raw = raw.strip()
    if not raw:
        return ""
    # Reject UUIDs, email-like strings, or single-char names
    if "@" in raw or len(raw) < 2 or re.match(r'^[0-9a-f-]{8,}$', raw, re.IGNORECASE):
        return ""
    # Use first name only (split on space, take first token)
    first = raw.split()[0]
    # Reject if it looks like a number or all-caps acronym > 3 chars
    if first.isdigit() or (first.isupper() and len(first) > 3):
        return ""
    return first


async def _generate_contextual_gap_filler(query: str, invoker_name: str = "") -> str:
    """Generate a short, contextual acknowledgment sentence for the given user query.
    Runs in parallel with the actual pipeline so there is no extra delay."""
    name_instruction = ""
    if invoker_name:
        name_instruction = (
            f" The person asking is named {invoker_name}. "
            f"Naturally include their name in your acknowledgment "
            f"(e.g., 'Sure {invoker_name}, let me check that.' or 'On it, {invoker_name}.')."
        )
    try:
        response = await asyncio.to_thread(
            lambda: get_openai_client().chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are Jarvis, a concise voice assistant in a meeting. "
                            "Given the user's request, respond with exactly ONE short sentence "
                            "(10 words or fewer) that acknowledges it and signals you are acting on it. "
                            "Be specific to what they asked. Do NOT answer the question itself. "
                            "Examples: 'Sure, let me pull up the meeting summary.', "
                            "'On it, fetching that for you.', 'Let me check that right now.'"
                        ) + name_instruction,
                    },
                    {"role": "user", "content": query},
                ],
                max_tokens=30,
                temperature=0.7,
            )
        )
        text = (response.choices[0].message.content or "").strip().strip('"\'')
        return text if text else random.choice(JARVIS_FILLER_PHRASES)
    except Exception:
        return random.choice(JARVIS_FILLER_PHRASES)


async def _rewrite_for_speech(text: str) -> str:
    """Condense a long LLM answer to 2-3 spoken sentences for verbal delivery.

    Short answers (<= 80 chars) are returned unchanged.
    On error, the original text is returned unchanged.
    """
    if not JARVIS_SPEECH_REWRITE_ENABLED or len(text) <= 80:
        return text
    try:
        response = await asyncio.to_thread(
            lambda: get_openai_client().chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a speech editor. Rewrite the following text for spoken delivery in a meeting.\n"
                            "Rules:\n"
                            "- Maximum 2-3 sentences\n"
                            "- End with a brief offer to elaborate (e.g., \"Want me to go into more detail?\" or \"I can elaborate if you'd like.\")\n"
                            "- Keep the core answer intact — do not lose factual content\n"
                            "- No markdown, no bullet points, no formatting\n"
                            "- Speak naturally as if talking aloud"
                        ),
                    },
                    {"role": "user", "content": text},
                ],
                max_tokens=120,
                temperature=0.5,
            )
        )
        result = (response.choices[0].message.content or "").strip()
        if not result:
            return text
        logger.info("Speech rewrite: %d -> %d chars", len(text), len(result))
        return result
    except Exception as e:
        logger.debug("Speech rewrite failed (non-fatal): %s", e)
        return text


async def _start_next_task_if_idle() -> None:
    state_lock = _get_state_lock()
    next_task: Optional[VoiceTask] = None
    async with state_lock:
        if meeting_state.get("current_task") is not None:
            return
        pending_requests: Deque[VoiceTask] = meeting_state["pending_requests"]
        while pending_requests and (pending_requests[0].cancel_requested or pending_requests[0].superseded):
            pending_requests.popleft()
        _set_current_task(None)
        if pending_requests:
            next_task = pending_requests.popleft()
            _set_current_task(next_task)
            next_task.runner = asyncio.create_task(_run_voice_task(next_task))

    if next_task is not None:
        return


async def _finish_voice_task(task: VoiceTask) -> None:
    state_lock = _get_state_lock()
    async with state_lock:
        pending = meeting_state.get("pending_clarification")
        current = meeting_state.get("current_task")
        if current is not None and current.task_id == task.task_id:
            _set_current_task(None)
            if pending and pending.get("task_id") == task.task_id:
                meeting_state["pending_clarification"] = None
    await _start_next_task_if_idle()


def _mark_mutation_started(task: VoiceTask, _action: str) -> None:
    task.mutation_started = True
    current = meeting_state.get("current_task")
    if current is not None and current.task_id == task.task_id:
        meeting_state["mutation_started"] = True


async def _handle_bare_wake(bot_id: str) -> None:
    state_lock = _get_state_lock()
    async with state_lock:
        current: Optional[VoiceTask] = meeting_state.get("current_task")
        generation = meeting_state["output_generation"]

    if current is not None:
        # Already busy — let the user know and clear listening state
        meeting_state["jarvis_listening"] = False
        await _speak_guarded(JARVIS_BUSY_ACK, bot_id, generation)
    else:
        # Speak "Yes" to signal readiness — user can now ask without repeating the wake word
        await _speak_guarded(JARVIS_WAKE_ACK, bot_id, generation)


def _should_speak_final_answer(task: VoiceTask, answer: str) -> bool:
    normalized = (answer or "").strip().lower()
    if not normalized:
        return False
    if task.intent == "list_pages":
        return True
    if normalized.startswith("the requested change did not complete"):
        return True
    if "encountered an issue" in normalized or "ran into an error" in normalized:
        return True
    if normalized.startswith("i couldn't find"):
        return True
    return False


async def _execute_editor_task(task: VoiceTask) -> Optional[str]:
    try:
        meeting_state["pending_clarification"] = None
        _set_task_phase(task, "executing")
        execution_request = task.execution_request or task.request
        answer = await _queue_confluence_proposal(task, execution_request)

        if task.cancel_requested or task.superseded:
            return None

        if _looks_like_clarification_prompt(answer):
            task.question = answer
            task.clarification_context = _append_clarification_context(
                task.clarification_context,
                f"Editor follow-up needed: {answer}",
            )
            return answer

        _set_task_phase(task, "speaking")
        await _speak_guarded(answer, task.bot_id, task.output_generation)
        return None
    except asyncio.CancelledError:
        logger.info("Editor task %s cancelled.", task.task_id)
        return None
    except Exception as e:
        logger.error("Editor task error: %s", e)
        if not task.cancel_requested and not task.superseded:
            error_ack = await _generate_dynamic_ack("error", task.request)
            await _speak_guarded(error_ack, task.bot_id, task.output_generation)
        return None


async def _run_voice_task(task: VoiceTask) -> None:
    first_turn = True
    try:
        while not task.cancel_requested:
            if first_turn:
                if not task.from_queue:
                    cached = get_random_ack_audio()
                    if cached:
                        ack_name, ack_bytes = cached
                        logger.info("Playing cached ack: %s", ack_name)
                        await _speak_cached_guarded(ack_bytes, task.bot_id, task.output_generation)
                    else:
                        # Fallback to live TTS if no cached audio available
                        ack = random.choice(_INSTANT_ACKS)
                        await _speak_guarded(ack, task.bot_id, task.output_generation)
                first_turn = False

            _set_task_phase(task, "planning")
            if task.pre_planned_decision is not None:
                decision = task.pre_planned_decision
                task.pre_planned_decision = None
            elif _is_unambiguous_request(task.request) and not task.clarification_context:
                # Fast-path: skip planning LLM call for clear, unambiguous requests
                from .core.schemas import MasterVoiceDecision
                # Detect delete intent for unambiguous requests
                _fast_intent = "edit"
                if any(w in task.request.lower().split()[:3] for w in ("delete", "remove")):
                    _fast_intent = "delete"
                decision = MasterVoiceDecision(
                    immediate_reply="",
                    needs_clarification=False,
                    execution_request=task.request,
                    intent=_fast_intent,
                    rationale="Fast-path: unambiguous request.",
                )
            else:
                decision = await session_agent.plan_voice_turn(
                    task.request,
                    clarification_context=_format_pending_clarification(task),
                    has_other_pending_work=_has_other_pending_work(task),
                    request_label=_build_request_reference(task),
                    is_queued_followup=task.from_queue,
                )
            if task.cancel_requested or task.superseded:
                return

            task.intent = decision.intent or "edit"

            if decision.needs_clarification:
                _set_task_phase(task, "clarifying")
                task.question = decision.clarification_question or "Which page should I work on?"
                task.clarification_context = _append_clarification_context(
                    task.clarification_context,
                    decision.rationale or "",
                )
                task.answer_future = asyncio.get_running_loop().create_future()
                meeting_state["pending_clarification"] = {
                    "task_id": task.task_id,
                    "clarification_context": task.clarification_context,
                    "question": task.question,
                }
                await _speak_guarded(task.question, task.bot_id, task.output_generation)
                try:
                    answer = await task.answer_future
                except asyncio.CancelledError:
                    return
                finally:
                    task.answer_future = None
                if task.cancel_requested:
                    return
                _record_clarification_answer(task, answer)
                await _speak_guarded("Got it.", task.bot_id, task.output_generation)
                task.question = ""
                continue


            task.execution_request = (decision.execution_request or task.request).strip()
            editor_follow_up = await _execute_editor_task(task)
            if task.cancel_requested or task.superseded:
                return
            if editor_follow_up:
                _set_task_phase(task, "clarifying")
                task.answer_future = asyncio.get_running_loop().create_future()
                meeting_state["pending_clarification"] = {
                    "task_id": task.task_id,
                    "clarification_context": task.clarification_context,
                    "question": task.question,
                }
                await _speak_guarded(task.question, task.bot_id, task.output_generation)
                try:
                    answer = await task.answer_future
                except asyncio.CancelledError:
                    return
                finally:
                    task.answer_future = None
                if task.cancel_requested:
                    return
                _record_clarification_answer(task, answer)
                await _speak_guarded("Got it.", task.bot_id, task.output_generation)
                task.question = ""
                task.execution_request = ""
                continue
            return
    except asyncio.CancelledError:
        logger.info("Voice task %s cancelled.", task.task_id)
        return
    except Exception as e:
        logger.error("Voice task error: %s", e)
        if not task.cancel_requested and not task.superseded:
            error_ack = await _generate_dynamic_ack("error", task.request)
            await _speak_guarded(error_ack, task.bot_id, task.output_generation)
    finally:
        await _finish_voice_task(task)


async def _plan_and_maybe_execute(task: VoiceTask) -> None:
    """Plan immediately. If confident, execute in parallel. If not, queue for serial."""
    try:
        ack = random.choice(_INSTANT_ACKS)
        await _speak_guarded(ack, task.bot_id, task.output_generation)

        decision = await session_agent.plan_voice_turn(
            task.request,
            is_queued_followup=True,
        )
        if task.cancel_requested or task.superseded:
            return

        if decision.needs_clarification:
            logger.info("Parallel task %s needs clarification — falling back to queue.", task.task_id)
            task.pre_planned_decision = decision
            task.from_queue = True
            async with _get_state_lock():
                meeting_state["pending_requests"].append(task)
            return

        logger.info("Parallel task %s is confident — executing in parallel.", task.task_id)
        task.intent = decision.intent or "edit"
        task.execution_request = (decision.execution_request or task.request).strip()
        await _execute_editor_task(task)
    except asyncio.CancelledError:
        logger.info("Parallel task %s cancelled.", task.task_id)
    except Exception as e:
        logger.error("Parallel task %s error: %s", task.task_id, e)
        if not task.cancel_requested and not task.superseded:
            error_ack = await _generate_dynamic_ack("error", task.request)
            await _speak_guarded(error_ack, task.bot_id, task.output_generation)
    finally:
        runners = meeting_state.get("parallel_runners", [])
        meeting_state["parallel_runners"] = [(t, r) for t, r in runners if t.task_id != task.task_id]


def _is_status_query(text: str) -> bool:
    return bool(_STATUS_PATTERN.search(text or ""))


async def _handle_status_query(query: str, bot_id: str) -> None:
    """Answer status/conversational queries instantly using gpt-4o-mini + meeting state."""
    current = meeting_state.get("current_task")
    pending = meeting_state.get("pending_requests", deque())

    state_lines = []
    if current:
        state_lines.append(f"Currently working on: {current.request} (phase: {current.phase})")
    else:
        state_lines.append("Currently idle — not working on anything.")

    queued = [t.request for t in pending if not t.cancel_requested]
    if queued:
        state_lines.append(f"Queued tasks: {', '.join(queued)}")

    parallel = [t.request for t, _ in meeting_state.get("parallel_runners", []) if not t.cancel_requested]
    if parallel:
        state_lines.append(f"Also running in parallel: {', '.join(parallel)}")

    recent_history = session_agent.get_recent_history_text()
    context = (
        f"User asked: {query}\n\n"
        f"Current state:\n" + "\n".join(state_lines) + "\n\n"
        f"Recent history:\n{recent_history}"
    )
    system = (
        "You are Jarvis, a voice assistant in a meeting. "
        "Answer the user's question about your current status briefly and naturally. "
        "Keep it to 1-2 short sentences. Be conversational."
    )

    try:
        response = await asyncio.to_thread(
            lambda: get_openai_client().chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": context},
                ],
                max_tokens=60,
                temperature=0.7,
            )
        )
        answer = (response.choices[0].message.content or "").strip().strip('"\'')
        if answer:
            generation = meeting_state["output_generation"]
            await _speak_guarded(answer, bot_id, generation, allow_stale=True)
    except Exception as e:
        logger.warning("Status query failed: %s", e)


async def _handle_general_question(query: str, bot_id: str, force_web_search: bool = False) -> None:
    """Answer a general (non-Confluence) question using the LLM and speak the response.
    If the answer is itself a clarifying question, set up a no-wake-word listening state."""
    generation = meeting_state["output_generation"]
    try:
        conversation_history = _format_general_history()
        # Cross-handler follow-up: prepend prior meeting response as context
        prior = meeting_state.get("last_jarvis_response")
        if prior and _is_followup(query):
            cross_context = f"User: {prior['query']}\nAssistant: {prior['answer']}"
            conversation_history = cross_context + "\n" + conversation_history if conversation_history and conversation_history.strip() != "[none]" else cross_context
        # Inject meeting context from graph (per D-01, D-02)
        # Cap at 0.5 s so a slow/unavailable graph never delays the answer.
        graph_context = ""
        try:
            graph_context = await asyncio.wait_for(graph_rag.query_context(query), timeout=0.5)
        except (asyncio.TimeoutError, Exception):
            pass  # Non-fatal — graph is optional
        # Determine multi-turn referencing
        use_multiturn_ref = bool(conversation_history and conversation_history.strip() != "[none]")
        # Run LLM and gap filler concurrently
        answer_task = asyncio.create_task(answer_general_question(
            query,
            conversation_history,
            graph_context=graph_context,
            force_web_search=force_web_search,
            speech_rewrite_enabled=JARVIS_SPEECH_REWRITE_ENABLED,
            multiturn_reference=use_multiturn_ref,
        ))
        gap_filler_task = asyncio.create_task(_speak_gap_filler(query, bot_id, generation))

        # When LLM answer arrives, immediately pre-synthesize sentence[0] while gap filler still plays
        answer = await answer_task
        if not answer:
            gap_filler_task.cancel()
            return

        answer = await _rewrite_for_speech(answer)
        # Confidence signaling (CONFIDENCE-01): web-search answers get attribution prefix
        if JARVIS_CONFIDENCE_SIGNAL_ENABLED and force_web_search and answer:
            if not answer.lower().startswith(("according to", "based on", "from what i found")):
                answer = "Based on what I found, " + answer[0].lower() + answer[1:]

        # Pre-synthesize ALL sentences in parallel while gap filler plays
        sentences = _split_into_sentences(answer)
        syn_tasks = [
            asyncio.create_task(asyncio.to_thread(synthesize_speech, s))
            for s in sentences
        ]
        await gap_filler_task  # ensure gap filler finishes before answer starts
        preloaded_all = list(await asyncio.gather(*syn_tasks)) if syn_tasks else None

        await _speak_guarded(answer, bot_id, generation, allow_stale=True, _preloaded_all=preloaded_all)
        await asyncio.sleep(JARVIS_POST_SPEECH_PAUSE_SECONDS)
        if answer:
            _remember_general_exchange(query, answer)

        # If the LLM's answer is a clarifying question, enter no-wake-word listening mode
        if _looks_like_clarification_prompt(answer):
            meeting_state["pending_general_clarification"] = {
                "question": answer,
                "original_query": query,
                "bot_id": bot_id,
                "conversation_history": conversation_history,
                "expires_at": time.time() + JARVIS_GENERAL_CLARIFICATION_TIMEOUT,
            }
            logger.info("General clarification mode activated for: %s (timeout: %.0fs)", query[:50], JARVIS_GENERAL_CLARIFICATION_TIMEOUT)
    except Exception as e:
        logger.error("General question handling failed: %s", e)


async def _handle_general_clarification_answer(answer_text: str, pending: dict) -> None:
    """Handle the user's answer to a general-question clarification, then clear the state."""
    bot_id = pending["bot_id"]
    original_query = pending["original_query"]
    conversation_history = pending.get("conversation_history", "")
    generation = meeting_state["output_generation"]

    # Clear the clarification state immediately so new transcripts go through normal flow
    state_lock = _get_state_lock()
    async with state_lock:
        meeting_state["pending_general_clarification"] = None

    try:
        # Build enriched context: original question + clarification exchange
        enriched_history = conversation_history
        if enriched_history and enriched_history.strip() != "[none]":
            enriched_history += f"\nUser: {original_query}\nAssistant: {pending['question']}\nUser: {answer_text}"
        else:
            enriched_history = f"User: {original_query}\nAssistant: {pending['question']}\nUser: {answer_text}"

        # Re-ask with full context
        final_answer = await answer_general_question(
            answer_text,
            enriched_history,
            speech_rewrite_enabled=JARVIS_SPEECH_REWRITE_ENABLED,
        )
        if final_answer:
            final_answer = await _rewrite_for_speech(final_answer)
            await _speak_guarded(final_answer, bot_id, generation, allow_stale=True)
            await asyncio.sleep(JARVIS_POST_SPEECH_PAUSE_SECONDS)
            _remember_general_exchange(answer_text, final_answer)
            # If the follow-up answer is itself a clarifying question, re-arm no-wake-word mode
            if _looks_like_clarification_prompt(final_answer):
                meeting_state["pending_general_clarification"] = {
                    "question": final_answer,
                    "original_query": answer_text,
                    "bot_id": bot_id,
                    "conversation_history": enriched_history,
                    "expires_at": time.time() + JARVIS_GENERAL_CLARIFICATION_TIMEOUT,
                }
                logger.info("General clarification re-armed after follow-up question.")
    except Exception as e:
        logger.error("General clarification resolution failed: %s", e)


JARVIS_SUMMARY_CLARIFICATION_TIMEOUT = float(os.getenv("JARVIS_SUMMARY_CLARIFICATION_TIMEOUT", "15.0"))


async def _handle_summary_clarification_answer(answer_text: str, pending: dict) -> None:
    """Handle user's brief/detailed response and generate the appropriate summary."""
    bot_id = pending["bot_id"]
    generation = meeting_state["output_generation"]
    meeting_state["pending_summary_clarification"] = None

    normalized = answer_text.strip().lower()
    if any(w in normalized for w in ("brief", "short", "quick", "concise")):
        detail_level = "brief"
    else:
        detail_level = "detailed"

    try:
        transcript_log = list(meeting_state["transcript_log"])
        # Start answer task first so it runs while the filler plays.
        summary_task = asyncio.create_task(summarize_meeting(transcript_log, detail_level=detail_level))
        await _speak_gap_filler(answer_text, bot_id, generation)
        answer = await summary_task
        answer = await _rewrite_for_speech(answer)
        await _speak_guarded(answer, bot_id, generation, allow_stale=True)
        await asyncio.sleep(JARVIS_POST_SPEECH_PAUSE_SECONDS)
        # If the summary response is itself a clarifying question, re-arm no-wake-word mode
        if _looks_like_clarification_prompt(answer):
            meeting_state["pending_summary_clarification"] = {
                "bot_id": bot_id,
                "expires_at": time.time() + JARVIS_SUMMARY_CLARIFICATION_TIMEOUT,
            }
            logger.info("Summary clarification re-armed after follow-up question.")
    except Exception as e:
        logger.error("Summary clarification resolution failed: %s", e)


async def _handle_meeting_summary(query: str, bot_id: str) -> None:
    """Stream the LLM summary sentence-by-sentence, playing under the output_lock."""
    generation = meeting_state["output_generation"]
    normalized_query = query.strip().lower()

    if any(w in normalized_query for w in ("detailed", "full", "long", "complete", "thorough")):
        detail_level = "detailed"
    else:
        detail_level = "brief"

    try:
        transcript_log = list(meeting_state["transcript_log"])
        sentence_gen = summarize_meeting_streaming(transcript_log, detail_level=detail_level)
        gap_filler_task = asyncio.create_task(_speak_gap_filler(query, bot_id, generation))
        answer = await _speak_streaming(sentence_gen, gap_filler_task, bot_id, generation)
        if answer:
            meeting_state["last_jarvis_response"] = {"intent": "meeting_summary", "query": query, "answer": answer}
    except Exception as e:
        logger.error("Meeting summary handling failed: %s", e)


async def _handle_meeting_opinion(query: str, bot_id: str) -> None:
    """Stream a first-person opinion grounded in the meeting transcript."""
    generation = meeting_state["output_generation"]
    try:
        transcript_log = list(meeting_state["transcript_log"])
        sentence_gen = generate_opinion_streaming(transcript_log, query=query)
        gap_filler_task = asyncio.create_task(_speak_gap_filler(query, bot_id, generation))
        answer = await _speak_streaming(sentence_gen, gap_filler_task, bot_id, generation)
        if answer:
            meeting_state["last_jarvis_response"] = {"intent": "meeting_opinion", "query": query, "answer": answer}
    except Exception as e:
        logger.error("Meeting opinion handling failed: %s", e)


def _normalize_split_verbs(text: str) -> str:
    """Collapse common STT word-split verbs before regex matching."""
    _SPLIT_VERB_PATTERNS = [
        (re.compile(r'\bcon\s+tribute\b', re.IGNORECASE), "contribute"),
        (re.compile(r'\bmen\s+tion\b', re.IGNORECASE), "mention"),
        (re.compile(r'\bsug\s+gest\b', re.IGNORECASE), "suggest"),
        (re.compile(r'\bpre\s+sent\b', re.IGNORECASE), "present"),
        (re.compile(r'\bdis\s+cuss\b', re.IGNORECASE), "discuss"),
        (re.compile(r'\bex\s+plain\b', re.IGNORECASE), "explain"),
        (re.compile(r'\bde\s+scribe\b', re.IGNORECASE), "describe"),
    ]
    for pattern, replacement in _SPLIT_VERB_PATTERNS:
        text = pattern.sub(replacement, text)
    return text


def _extract_speaker_name(query: str) -> str:
    """Extract the speaker name from a speaker query like 'what did Alice say'."""
    query = _normalize_split_verbs(query)
    patterns = [
        r"what (?:did|has|does) (\w+(?:\s+\w+)?) (?:say|said|mention|think|contribute|talk)",
        r"what (\w+(?:\s+\w+)?) (?:said|mentioned|talked|contributed)",
        r"summarize (?:what )?(\w+(?:\s+\w+)?) (?:said|mentioned)",
        r"tell me what (\w+(?:\s+\w+)?) (?:said|mentioned|talked)",
    ]
    for pattern in patterns:
        m = re.search(pattern, query, re.IGNORECASE)
        if m:
            name = m.group(1).strip()
            # Filter out common non-name words
            if name.lower() not in ("i", "we", "they", "he", "she", "you", "everyone", "somebody", "someone"):
                return name
    return ""


async def _handle_action_items(query: str, bot_id: str) -> None:
    """Stream action items extracted from the meeting transcript."""
    generation = meeting_state["output_generation"]
    try:
        transcript_log = list(meeting_state["transcript_log"])
        sentence_gen = extract_action_items_streaming(transcript_log)
        gap_filler_task = asyncio.create_task(_speak_gap_filler(query, bot_id, generation))
        answer = await _speak_streaming(sentence_gen, gap_filler_task, bot_id, generation)
        if answer:
            meeting_state["last_jarvis_response"] = {"intent": "action_items", "query": query, "answer": answer}
    except Exception as e:
        logger.error("Action items handling failed: %s", e)


def _extract_speaker_topic(query: str) -> str:
    """Extract an optional topic qualifier from a speaker query.

    E.g. "What did Rohan say about the database migration?" → "the database migration"
         "What did Deepa mention about churn?" → "churn"
         "What did Sneha say?" → ""
    """
    query = _normalize_split_verbs(query)
    m = re.search(
        r"(?:say|said|mention(?:ed)?|talk(?:ed)?|contribute[d]?) (?:about|regarding|on|regarding) (.+?)(?:\?|$)",
        query,
        re.IGNORECASE,
    )
    if m:
        return m.group(1).strip().rstrip("?").strip()
    return ""


async def _handle_speaker_query(query: str, bot_id: str) -> None:
    """Stream a summary of what a specific speaker said in the meeting."""
    generation = meeting_state["output_generation"]
    try:
        speaker_name = _extract_speaker_name(query)
        if not speaker_name:
            await _speak_guarded(
                "I'm not sure which participant you're asking about. Could you say their name again?",
                bot_id, generation, allow_stale=True,
            )
            return
        topic = _extract_speaker_topic(query)
        transcript_log = list(meeting_state["transcript_log"])
        sentence_gen = summarize_speaker_streaming(transcript_log, speaker_name, topic=topic)
        gap_filler_task = asyncio.create_task(_speak_gap_filler(query, bot_id, generation))
        answer = await _speak_streaming(sentence_gen, gap_filler_task, bot_id, generation)
        if answer:
            meeting_state["last_jarvis_response"] = {"intent": "speaker_query", "query": query, "answer": answer}
    except Exception as e:
        logger.error("Speaker query handling failed: %s", e)


async def handle_spoken_request(spoken_query: str, bot_id: str) -> None:
    # Intercept status/conversational queries — answer instantly, skip task queue
    if _is_status_query(spoken_query):
        await _handle_status_query(spoken_query, bot_id)
        return

    # Garbled query recovery (GARBLED-01)
    if _is_garbled_query(spoken_query):
        logger.info("Garbled query detected, asking to repeat: %s", repr(spoken_query[:40]))
        generation = meeting_state["output_generation"]
        await _speak_guarded("Sorry, I didn't catch that. Could you say that again?", bot_id, generation, allow_stale=True)
        return

    # Check for pending summary clarification (no wake word needed) — before intent classification
    pending_summary = meeting_state.get("pending_summary_clarification")
    if pending_summary:
        logger.info("Routing to summary clarification handler: %s", spoken_query[:60])
        asyncio.create_task(_handle_summary_clarification_answer(spoken_query, pending_summary))
        return

    # Classify intent: general questions get answered directly, not queued as tasks
    intent = await classify_intent(spoken_query)
    if intent == "general":
        logger.info("Classified as general question: %s", spoken_query[:60])
        asyncio.create_task(_handle_general_question(spoken_query, bot_id))
        return

    if intent == "web_search":
        logger.info("Classified as web search request: %s", spoken_query[:60])
        asyncio.create_task(_handle_general_question(spoken_query, bot_id, force_web_search=True))
        return

    # NEW: meeting transcript intents (per D-04 — routed before Confluence pipeline)
    if intent == "meeting_summary":
        logger.info("Classified as meeting summary request: %s", spoken_query[:60])
        asyncio.create_task(_handle_meeting_summary(spoken_query, bot_id))
        return

    if intent == "meeting_opinion":
        logger.info("Classified as meeting opinion request: %s", spoken_query[:60])
        asyncio.create_task(_handle_meeting_opinion(spoken_query, bot_id))
        return

    if intent == "action_items":
        logger.info("Classified as action items request: %s", spoken_query[:60])
        asyncio.create_task(_handle_action_items(spoken_query, bot_id))
        return

    if intent == "speaker_query":
        logger.info("Classified as speaker query: %s", spoken_query[:60])
        asyncio.create_task(_handle_speaker_query(spoken_query, bot_id))
        return

    if intent == "confluence" and _is_confluence_read_query(spoken_query):
        logger.info("Classified as Confluence read question: %s", spoken_query[:60])
        asyncio.create_task(_handle_confluence_question(spoken_query, bot_id))
        return

    # Check if this is an answer to a pending general clarification
    pending_general = meeting_state.get("pending_general_clarification")
    if pending_general and time.time() <= pending_general["expires_at"]:
        logger.info("Routing to general clarification handler: %s", spoken_query[:60])
        asyncio.create_task(_handle_general_clarification_answer(spoken_query, pending_general))
        return

    state_lock = _get_state_lock()
    ack_context: Optional[str] = None
    current_request_text: str = ""
    ack_generation: Optional[int] = None
    current_to_cancel: Optional[asyncio.Task] = None
    task_to_start: Optional[VoiceTask] = None
    should_cancel_now = False
    should_start_next = False

    async with state_lock:
        current: Optional[VoiceTask] = meeting_state.get("current_task")
        pending = meeting_state.get("pending_clarification")
        if pending and not _is_override_request(spoken_query):
            target_task = None
            if current is not None and current.task_id == pending.get("task_id"):
                target_task = current

            if target_task is not None and target_task.answer_future and not target_task.answer_future.done():
                try:
                    target_task.answer_future.set_result(spoken_query)
                except asyncio.InvalidStateError:
                    logger.warning("Clarification future already resolved/cancelled — ignoring answer.")
                return

        if current is None:
            new_task = _new_voice_task(spoken_query, bot_id)
            new_task.output_generation = _next_output_generation()
            _set_current_task(new_task)
            task_to_start = new_task
        elif _is_override_request(spoken_query):
            if current is not None:
                current.cancel_requested = True
                current.superseded = True
                current_to_cancel = current.runner
                should_cancel_now = not current.mutation_started and current.runner is not None
            meeting_state["cancel_requested"] = True
            if pending:
                meeting_state["pending_clarification"] = None
            pending_requests: Deque[VoiceTask] = meeting_state["pending_requests"]
            while pending_requests:
                queued = pending_requests.popleft()
                queued.cancel_requested = True
                queued.superseded = True
            for ptask, prunner in meeting_state.get("parallel_runners", []):
                ptask.cancel_requested = True
                ptask.superseded = True
                prunner.cancel()
            meeting_state["parallel_runners"] = []
            generation = _next_output_generation()
            replacement = _new_voice_task(spoken_query, bot_id)
            replacement.output_generation = generation
            if current is None:
                _set_current_task(replacement)
                task_to_start = replacement
            else:
                meeting_state["pending_requests"].appendleft(replacement)
            ack_context = "switching"
            current_request_text = current.request if current else ""
            ack_generation = generation
        else:
            parallel_task = _new_voice_task(spoken_query, bot_id)
            parallel_task.output_generation = _next_output_generation()
            runner = asyncio.create_task(_plan_and_maybe_execute(parallel_task))
            meeting_state["parallel_runners"].append((parallel_task, runner))

    if ack_context and ack_generation is not None:
        ack_text = await _generate_dynamic_ack(ack_context, spoken_query, current_request_text)
        await _speak_guarded(ack_text, bot_id, ack_generation)

    if should_cancel_now and current_to_cancel is not None:
        current_to_cancel.cancel()
        should_start_next = True

    if task_to_start is not None:
        task_to_start.runner = asyncio.create_task(_run_voice_task(task_to_start))

    if should_start_next:
        await _start_next_task_if_idle()


def extract_wake_and_query(text: str) -> Optional[str]:
    m = _WAKE_PATTERN.search(text)
    if m:
        return m.group(1).strip()
    return None


def is_bare_wake_invocation(text: str) -> bool:
    query = extract_wake_and_query(text)
    return query is not None and not query


def _extract_sentence(data_block: dict) -> str:
    words = data_block.get("words", []) or []
    if words:
        return " ".join(word.get("text", "") for word in words).strip()
    return (data_block.get("text") or "").strip()


def process_transcript_event(sentence: str, timestamp: float) -> Optional[str]:
    if not sentence:
        return None
    query = extract_wake_and_query(sentence)
    if query is not None:
        if query:
            meeting_state["jarvis_listening"] = False
            meeting_state["_query_from_wake_invocation"] = True
            return query
        meeting_state["jarvis_listening"] = True
        meeting_state["jarvis_listening_at"] = timestamp
        return None

    if meeting_state.get("pending_clarification"):
        meeting_state["jarvis_listening"] = False
        return sentence

    # Check for pending general question clarification (no wake word needed)
    pending_general = meeting_state.get("pending_general_clarification")
    if pending_general:
        if time.time() > pending_general["expires_at"]:
            # Timeout expired — clear state, require wake word again
            logger.info("General clarification timeout expired, clearing state.")
            meeting_state["pending_general_clarification"] = None
        else:
            # User is answering a general clarification — bypass wake word
            meeting_state["jarvis_listening"] = False
            return sentence

    # Check for pending summary clarification (no wake word needed)
    pending_summary = meeting_state.get("pending_summary_clarification")
    if pending_summary:
        if time.time() > pending_summary["expires_at"]:
            logger.info("Summary clarification timeout expired, clearing state.")
            meeting_state["pending_summary_clarification"] = None
        else:
            meeting_state["jarvis_listening"] = False
            return sentence

    if meeting_state["jarvis_listening"]:
        meeting_state["jarvis_listening"] = False
        meeting_state["_query_from_listening"] = True
        meeting_state["_query_from_wake_invocation"] = False
        return sentence or None
    return None


@app.websocket("/recall-audio-stream")
async def websocket_endpoint(websocket: WebSocket):
    await _websocket_endpoint_for_session(websocket, _DEFAULT_SESSION_ID)


@app.websocket("/recall-audio-stream/{session_id}")
async def websocket_endpoint_for_session(websocket: WebSocket, session_id: str):
    await _websocket_endpoint_for_session(websocket, session_id)


async def _websocket_endpoint_for_session(websocket: WebSocket, session_id: str):
    token = set_current_meeting_session(session_id)
    await websocket.accept()
    logger.info("WebSocket connected for session %s", session_id)

    try:
        while True:
            payload = await websocket.receive_json()
            event_name = payload.get("event")
            if event_name != "transcript.data":
                continue

            data_block = payload["data"]["data"]
            participant = data_block["participant"]["name"]
            sentence = _extract_sentence(data_block)

            if not sentence:
                continue

            if BOT_NAME.lower() in participant.lower():
                logger.debug("Skipping bot transcript event from command processing: %s", sentence[:40])
                continue

            # Only count speech from the active invoker toward the hold timer.
            # Background speakers should not block Jarvis from answering.
            invoker_at_arrival = meeting_state.get("invoker_participant")
            if not invoker_at_arrival or participant == invoker_at_arrival:
                meeting_state["last_user_speech_at"] = time.time()

            # Only dispatch commands on final segments — partial segments arrive mid-sentence
            if not data_block.get("is_final", True):
                logger.debug("Partial transcript (skipping dispatch): %s", sentence[:40])
                continue

            logger.info("Transcript %s: %s", participant, sentence)

            bot_id = meeting_state.get("bot_id")
            if not bot_id and os.path.exists("bot_id.txt"):
                with open("bot_id.txt", "r") as f:
                    bot_id = f.read().strip()
                    meeting_state["bot_id"] = bot_id
                    bind_bot_to_session(bot_id, session_id)

            if not bot_id:
                continue

            entry = _append_transcript_log_entry(
                participant=participant,
                text=sentence,
                timestamp=time.time(),
            )

            # Real-time graph ingestion (fire-and-forget, per D-13)
            try:
                if entry:
                    asyncio.create_task(graph_rag.ingest_transcript_entry(entry))
            except Exception:
                pass  # Non-fatal — graph is optional

            # Speaker isolation: skip non-invoker transcripts during active debounce window (D-02)
            invoker = meeting_state.get("invoker_participant")
            if invoker and participant != invoker:
                logger.debug(
                    "Ignoring transcript from %s — active invoker is %s", participant, invoker
                )
            else:
                query = process_transcript_event(sentence, time.time())
                if query:
                    # Lock invoker on wake word detection (D-01) and accumulate query text (D-07)
                    meeting_state["invoker_participant"] = participant
                    accumulated = meeting_state.get("_accumulated_query", "")
                    accumulated = (accumulated + " " + query).strip() if accumulated else query
                    meeting_state["_accumulated_query"] = accumulated
                    _from_listening = meeting_state.pop("_query_from_listening", False)
                    _from_wake_invocation = meeting_state.pop("_query_from_wake_invocation", False)
                    if _from_wake_invocation and not _from_listening:
                        meeting_state["wake_query_ack_pending"] = True
                    # INTERRUPT-01: if TTS is currently playing (output_lock held), emit yield phrase
                    output_lock = _get_output_lock()
                    if output_lock.locked():
                        asyncio.create_task(_handle_interruption(bot_id))
                    # Cancel existing debounce task and schedule new one (D-05, D-06)
                    pending = meeting_state.get("_pending_debounce_task")
                    if pending and not pending.done():
                        pending.cancel()
                    task = asyncio.create_task(_debounced_dispatch(accumulated, bot_id))
                    meeting_state["_pending_debounce_task"] = task
                elif meeting_state["jarvis_listening"] and is_bare_wake_invocation(sentence):
                    # Lock invoker for bare wake so only their follow-up is accepted (D-01)
                    meeting_state["invoker_participant"] = participant
                    asyncio.create_task(_handle_bare_wake(bot_id))
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected for session %s", session_id)
    except Exception as e:
        logger.error("WebSocket error: %s", e)
    finally:
        reset_current_meeting_session(token)


@app.get("/health")
async def health():
    return {
        "status": "healthy_agentic",
        "bot_id": meeting_state["bot_id"],
        "transcript_provider": RECALL_TRANSCRIPT_PROVIDER,
        "tts_provider": JARVIS_TTS_PROVIDER,
    }


def _start_server():
    uvicorn.run(app, host=APP_HOST, port=APP_PORT, log_level="warning")


def main():
    print("=" * 60)
    print(f"{BOT_NAME} AGENTIC Editor Bot")
    print("=" * 60)

    Thread(target=_start_server, daemon=True).start()
    time.sleep(2)

    meeting_url = MEETING_URL or input("\nMeeting URL (Google Meet / Zoom / Teams): ").strip()
    if not meeting_url:
        print("No meeting URL provided")
        sys.exit(1)

    print(f"Joining: {meeting_url}")
    bot_id = create_bot(meeting_url)
    if not bot_id:
        print("Failed to spawn bot")
        sys.exit(1)

    meeting_state["bot_id"] = bot_id
    meeting_state["is_active"] = True

    with open("bot_id.txt", "w") as f:
        f.write(bot_id)

    print(f"Bot joined! ID: {bot_id}")
    print(f'Listening for "Hey {BOT_NAME}"...')

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nshutting down.")


if __name__ == "__main__":
    main()
