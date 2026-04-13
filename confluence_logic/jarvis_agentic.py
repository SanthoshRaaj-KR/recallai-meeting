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
from dataclasses import dataclass
from io import BytesIO
from itertools import count
from threading import Thread
from typing import Deque, Optional

import requests
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from gtts import gTTS
from openai import OpenAI

from .agents.editor_agent import EditorAgent
from .classifier import classify_intent
from .audio_cache import get_random_ack_audio
from .general_responder import answer_general_question
from .meeting_responder import summarize_meeting, generate_opinion
from confluence_logic import graph_rag

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

JARVIS_TTS_PROVIDER = os.getenv("JARVIS_TTS_PROVIDER", "openai").strip().lower()
JARVIS_TTS_MODEL = os.getenv("JARVIS_TTS_MODEL", "gpt-4o-mini-tts").strip()
JARVIS_TTS_VOICE = os.getenv("JARVIS_TTS_VOICE", "echo").strip()
JARVIS_TTS_SPEED = float(os.getenv("JARVIS_TTS_SPEED", "1.0"))
JARVIS_WAKE_ACK = os.getenv("JARVIS_WAKE_ACK", "Yes?").strip()
JARVIS_BUSY_ACK = os.getenv("JARVIS_BUSY_ACK", "I'm already on it. Give me a moment.").strip()
JARVIS_SPEECH_HOLD_SECONDS = float(os.getenv("JARVIS_SPEECH_HOLD_SECONDS", "0.8"))
JARVIS_GENERAL_CLARIFICATION_TIMEOUT = float(os.getenv("JARVIS_GENERAL_CLARIFICATION_TIMEOUT", "15.0"))
JARVIS_DEBOUNCE_SECONDS = float(os.getenv("JARVIS_DEBOUNCE_SECONDS", "1.0"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI()
session_agent = EditorAgent(model=JARVIS_AGENT_MODEL)
logger.info("Jarvis meeting agent using model: %s", JARVIS_AGENT_MODEL)
logger.info("Jarvis transcript provider: %s", RECALL_TRANSCRIPT_PROVIDER)
logger.info("Jarvis TTS provider: %s", JARVIS_TTS_PROVIDER)
if RECALL_TRANSCRIPT_PROVIDER == "assembly_ai_v3_streaming" and ASSEMBLY_API:
    logger.warning(
        "ASSEMBLY_API is set locally, but Recall BYOB transcription still requires the AssemblyAI key "
        "to be configured in the Recall transcription credentials dashboard."
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
    "Sure! Give me a sec.",
    "On it!",
    "Just a moment.",
    "Let me check that for you.",
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

meeting_state = {
    "bot_id": None,
    "transcript_log": [],
    "is_active": False,
    "jarvis_listening": False,
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
    "invoker_participant": None,       # D-04: set when wake word is detected; D-03: cleared after dispatch
    "_pending_debounce_task": None,    # D-09: cancellable asyncio.Task for debounce window
    "_accumulated_query": "",          # D-07: space-joined query text from invoker segments
}

_openai_client: Optional[OpenAI] = None
_WAKE_PATTERN = re.compile(r"(?:hey\s+)?jarvis[,.]?\s*(.*)", re.IGNORECASE)
_OVERRIDE_PATTERN = re.compile(r"\b(stop|cancel|instead|forget that|never mind|nevermind|wait|change that|changed my mind|changed mind|don't do|dont do|undo)\b", re.IGNORECASE)
_ADDITIVE_PATTERN = re.compile(r"\b(also|after that|then|next)\b", re.IGNORECASE)
_UNAMBIGUOUS_VERBS = frozenset({
    "create", "list", "delete", "add", "update", "edit", "rename", "remove", "make", "show", "write",
})
_REFERENTIAL_TERMS = (" it ", " that ", " this ", " same ", "the one", "the page")

_MAX_GENERAL_HISTORY = 3  # turns


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
_state_lock: Optional[asyncio.Lock] = None
_state_lock_loop = None
_output_lock: Optional[asyncio.Lock] = None
_output_lock_loop = None


def get_openai_client() -> OpenAI:
    global _openai_client
    if _openai_client is None:
        _openai_client = OpenAI()
    return _openai_client


def _get_state_lock() -> asyncio.Lock:
    global _state_lock, _state_lock_loop
    loop = asyncio.get_running_loop()
    if _state_lock is None or _state_lock_loop is not loop:
        _state_lock = asyncio.Lock()
        _state_lock_loop = loop
    return _state_lock


def _get_output_lock() -> asyncio.Lock:
    global _output_lock, _output_lock_loop
    loop = asyncio.get_running_loop()
    if _output_lock is None or _output_lock_loop is not loop:
        _output_lock = asyncio.Lock()
        _output_lock_loop = loop
    return _output_lock


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


def build_create_bot_payload(meeting_url: str) -> dict:
    ws_url = WEBHOOK_URL.replace("https://", "wss://") + "/recall-audio-stream"
    return {
        "meeting_url": meeting_url,
        "bot_name": BOT_NAME,
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


def create_bot(meeting_url: str) -> Optional[str]:
    payload = build_create_bot_payload(meeting_url)
    try:
        response = requests.post(
            f"{RECALL_BASE_URL}/bot/",
            headers={"Authorization": f"Token {RECALL_API_KEY}", "Content-Type": "application/json"},
            json=payload,
            timeout=10,
        )
        if response.status_code in [200, 201]:
            bot_id = response.json()["id"]
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

    response = get_openai_client().audio.speech.create(
        model=JARVIS_TTS_MODEL,
        voice=JARVIS_TTS_VOICE,
        input=text,
        response_format="mp3",
        speed=JARVIS_TTS_SPEED,
    )

    if hasattr(response, "read"):
        return response.read()
    if hasattr(response, "content"):
        return response.content
    if hasattr(response, "iter_bytes"):
        return b"".join(response.iter_bytes())

    raise TypeError("Unexpected OpenAI TTS response type.")


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


def _estimate_cached_duration(audio_bytes: bytes) -> float:
    """Estimate MP3 playback duration from byte size (assumes ~32 kbps = 4000 bytes/sec, min 0.3s)."""
    return max(0.3, len(audio_bytes) / 4000)


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
            duration = _estimate_cached_duration(audio_bytes)
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


def _estimate_speech_duration(text: str) -> float:
    """Estimate playback duration in seconds from text word count (2.5 words/sec, min 0.5s)."""
    words = len((text or "").split())
    return max(0.5, words / 2.5)


async def _speak_guarded(text: str, bot_id: str, generation: int, allow_stale: bool = False) -> bool:
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
        ok = await asyncio.to_thread(speak, text, bot_id)
        if ok:
            duration = _estimate_speech_duration(text)
            elapsed = 0.0
            while elapsed < duration:
                await asyncio.sleep(0.1)
                elapsed += 0.1
                if generation != meeting_state["output_generation"]:
                    break
        return ok


async def _speak_filler(bot_id: str, generation: int) -> None:
    """Speak a random filler phrase before a slow operation."""
    phrase = random.choice(JARVIS_FILLER_PHRASES)
    await _speak_guarded(phrase, bot_id, generation, allow_stale=True)


async def _debounced_dispatch(query: str, bot_id: str) -> None:
    """Wait JARVIS_DEBOUNCE_SECONDS then dispatch to handle_spoken_request and clear invoker lock.

    Per D-06: cancellable task. Per D-08: clears invoker_participant and _pending_debounce_task after firing.
    """
    await asyncio.sleep(JARVIS_DEBOUNCE_SECONDS)
    meeting_state["invoker_participant"] = None
    meeting_state["_pending_debounce_task"] = None
    meeting_state["_accumulated_query"] = ""
    await handle_spoken_request(query, bot_id)


async def _generate_contextual_gap_filler(query: str) -> str:
    """Generate a short, contextual acknowledgment sentence for the given user query.
    Runs in parallel with the actual pipeline so there is no extra delay."""
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
                        ),
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
        # Already busy — let the user know
        await _speak_guarded(JARVIS_BUSY_ACK, bot_id, generation)
    # else: silently start listening — no "Yes?" acknowledgment


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
        if execution_request:
            answer = await session_agent.handle_prepared_query(
                execution_request,
                original_query=task.request,
                mutation_started_callback=lambda action: _mark_mutation_started(task, action),
            )
        else:
            answer = await session_agent.handle_voice_query(
                task.request,
                clarification_context=_format_pending_clarification(task),
                mutation_started_callback=lambda action: _mark_mutation_started(task, action),
            )

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
        if _should_speak_final_answer(task, answer):
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
                decision = MasterVoiceDecision(
                    immediate_reply="",
                    needs_clarification=False,
                    execution_request=task.request,
                    intent="edit",
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


async def _handle_general_question(query: str, bot_id: str) -> None:
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
        graph_context = ""
        try:
            graph_context = await graph_rag.query_context(query)
        except Exception:
            pass  # Non-fatal — graph is optional
        answer = await answer_general_question(query, conversation_history, graph_context=graph_context)
        if not answer:
            return

        await _speak_guarded(answer, bot_id, generation, allow_stale=True)
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
        final_answer = await answer_general_question(answer_text, enriched_history)
        if final_answer:
            await _speak_guarded(final_answer, bot_id, generation, allow_stale=True)
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
        # Fire gap filler and actual summary in parallel; gap filler plays first.
        gap_filler_task = asyncio.create_task(_generate_contextual_gap_filler(answer_text))
        summary_task = asyncio.create_task(summarize_meeting(transcript_log, detail_level=detail_level))

        gap_filler = await gap_filler_task
        await _speak_guarded(gap_filler, bot_id, generation, allow_stale=True)

        answer = await summary_task
        await _speak_guarded(answer, bot_id, generation, allow_stale=True)
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
    """Speak acknowledgment, optionally ask brief/detailed, then generate summary."""
    generation = meeting_state["output_generation"]
    normalized_query = query.strip().lower()

    # Detect detail level from original query
    if any(w in normalized_query for w in ("brief", "short", "quick", "concise")):
        detail_level = "brief"
        specified = True
    elif any(w in normalized_query for w in ("detailed", "full", "long", "complete", "thorough")):
        detail_level = "detailed"
        specified = True
    else:
        detail_level = None
        specified = False

    if specified:
        # Type already known — generate directly
        try:
            transcript_log = list(meeting_state["transcript_log"])
            # Fire gap filler and actual summary in parallel; gap filler plays first.
            gap_filler_task = asyncio.create_task(_generate_contextual_gap_filler(query))
            summary_task = asyncio.create_task(summarize_meeting(transcript_log, detail_level=detail_level))

            gap_filler = await gap_filler_task
            await _speak_guarded(gap_filler, bot_id, generation, allow_stale=True)

            answer = await summary_task
            meeting_state["last_jarvis_response"] = {
                "intent": "meeting_summary",
                "query": query,
                "answer": answer,
            }
            await _speak_guarded(answer, bot_id, generation, allow_stale=True)
        except Exception as e:
            logger.error("Meeting summary handling failed: %s", e)
    else:
        # Ask clarifying question and enter listen state
        await _speak_guarded("Sure!", bot_id, generation, allow_stale=True)
        await _speak_guarded("Do you want a detailed or a brief summary?", bot_id, generation, allow_stale=True)
        meeting_state["pending_summary_clarification"] = {
            "bot_id": bot_id,
            "expires_at": time.time() + JARVIS_SUMMARY_CLARIFICATION_TIMEOUT,
        }
        logger.info("Summary clarification mode activated (timeout: %.0fs)", JARVIS_SUMMARY_CLARIFICATION_TIMEOUT)


async def _handle_meeting_opinion(query: str, bot_id: str) -> None:
    """Generate and speak a first-person opinion grounded in the meeting transcript."""
    generation = meeting_state["output_generation"]
    try:
        transcript_log = list(meeting_state["transcript_log"])
        # Fire gap filler and actual opinion in parallel; gap filler plays first.
        gap_filler_task = asyncio.create_task(_generate_contextual_gap_filler(query))
        opinion_task = asyncio.create_task(generate_opinion(transcript_log, query=query))

        gap_filler = await gap_filler_task
        await _speak_guarded(gap_filler, bot_id, generation, allow_stale=True)

        answer = await opinion_task
        meeting_state["last_jarvis_response"] = {
            "intent": "meeting_opinion",
            "query": query,
            "answer": answer,
        }
        await _speak_guarded(answer, bot_id, generation, allow_stale=True)
    except Exception as e:
        logger.error("Meeting opinion handling failed: %s", e)


async def handle_spoken_request(spoken_query: str, bot_id: str) -> None:
    # Intercept status/conversational queries — answer instantly, skip task queue
    if _is_status_query(spoken_query):
        await _handle_status_query(spoken_query, bot_id)
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

    # NEW: meeting transcript intents (per D-04 — routed before Confluence pipeline)
    if intent == "meeting_summary":
        logger.info("Classified as meeting summary request: %s", spoken_query[:60])
        asyncio.create_task(_handle_meeting_summary(spoken_query, bot_id))
        return

    if intent == "meeting_opinion":
        logger.info("Classified as meeting opinion request: %s", spoken_query[:60])
        asyncio.create_task(_handle_meeting_opinion(spoken_query, bot_id))
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
            return query
        meeting_state["jarvis_listening"] = True
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
        return sentence or None
    return None


@app.websocket("/recall-audio-stream")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("WebSocket connected")

    try:
        while True:
            payload = await websocket.receive_json()
            event_name = payload.get("event")
            if event_name != "transcript.data":
                continue

            data_block = payload["data"]["data"]
            participant = data_block["participant"]["name"]
            sentence = _extract_sentence(data_block)

            if not sentence or BOT_NAME.lower() in participant.lower():
                continue

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

            if not bot_id:
                continue

            log = meeting_state["transcript_log"]
            log.append({
                "participant": participant,
                "text": sentence,
                "timestamp": time.time(),
            })
            # Keep memory bounded — drop oldest entries beyond limit
            if len(log) > 500:
                meeting_state["transcript_log"] = log[-500:]

            # Real-time graph ingestion (fire-and-forget, per D-13)
            try:
                asyncio.create_task(graph_rag.ingest_transcript_entry(log[-1]))
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
        logger.info("WebSocket disconnected")
    except Exception as e:
        logger.error("WebSocket error: %s", e)


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
