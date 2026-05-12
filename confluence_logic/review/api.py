"""
Review API router — FastAPI routes for meeting review and Confluence change management.

Mounted into the main jarvis_agentic app via app.include_router(router).
All routes are served at the root prefix (e.g. GET /review/summary, POST /bot/start).
"""
from __future__ import annotations

import asyncio
import base64
import gzip
import json
import logging
import os
import re
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional

from fastapi import APIRouter, Header, HTTPException
from openai import OpenAI
from pydantic import BaseModel
import requests

from . import supabase_store
from confluence_logic.agents.proposed_changes_agent import ProposedChangesAgent
from confluence_logic import confluence_page_graph
from confluence_logic.agents.fact_extraction_agent import (
    ExtractedFacts,
    _run_fact_extraction,
    _merged_rag_retrieval,
    JARVIS_FACT_INPUT_MAX_CHARS,
    JARVIS_PIPELINE_MAX_PAGES,
)
from confluence_logic.agents.drafter_agent import _run_drafter
from confluence_logic.agents.verifier_agent import _run_verifier

logger = logging.getLogger(__name__)

router = APIRouter()
_openai_client: Optional[OpenAI] = None

_RECALL_ENDED_CODES = {
    "call_ended",
    "done",
    "completed",
    "finished",
    "bot_left",
    "left_call",
    "removed_from_call",
    "kicked",
    "kicked_from_call",
    "host_ended_call",
    "meeting_ended",
}
_RECALL_ERROR_CODES = {
    "fatal",
    "error",
    "failed",
    "errored",
    "call_error",
    "joining_call_failed",
}
_RECALL_ACTIVE_CODES = {
    "joining_call",
    "in_call",
    "recording",
    "recording_permission_allowed",
}
_RECALL_STATUS_CACHE_SECONDS = float(os.getenv("RECALL_STATUS_CACHE_SECONDS", "4.0"))
JARVIS_REVIEW_MODEL = os.getenv("JARVIS_REVIEW_MODEL", "gpt-5.4-mini").strip()
JARVIS_REVIEW_MAX_INPUT_CHARS = int(os.getenv("JARVIS_REVIEW_MAX_INPUT_CHARS", "0"))
JARVIS_REVIEW_SUMMARY_MAX_TOKENS = int(os.getenv("JARVIS_REVIEW_SUMMARY_MAX_TOKENS", "1100"))
JARVIS_REVIEW_MOM_MAX_TOKENS = int(os.getenv("JARVIS_REVIEW_MOM_MAX_TOKENS", "900"))
JARVIS_REVIEW_TOPICS_MAX_TOKENS = int(os.getenv("JARVIS_REVIEW_TOPICS_MAX_TOKENS", "700"))
JARVIS_REVIEW_ACTION_ITEMS_MAX_TOKENS = int(os.getenv("JARVIS_REVIEW_ACTION_ITEMS_MAX_TOKENS", "700"))
JARVIS_PROPOSE_CHANGES_MAX_TOKENS = int(os.getenv("JARVIS_PROPOSE_CHANGES_MAX_TOKENS", "1400"))


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _utc_now_iso() -> str:
    return _utc_now().isoformat()


# ---------------------------------------------------------------------------
# Pydantic request/response models
# ---------------------------------------------------------------------------

class StartBotRequest(BaseModel):
    meeting_url: str
    session_id: Optional[str] = None


class ExecuteChangesRequest(BaseModel):
    ids: List[int]


class ProposeChangesRequest(BaseModel):
    query: Optional[str] = None


class PipelineStartRequest(BaseModel):
    session_id: str


class MeetingChatMessage(BaseModel):
    role: str
    content: str


class MeetingChatRequest(BaseModel):
    messages: List[MeetingChatMessage]


class ChangeItem(BaseModel):
    """Structured proposal for a single Confluence page change.

    Fields transcript_evidence, confidence, risk, and verifier_note have safe
    defaults so existing Supabase rows (which lack these columns) can be read
    in Phase 1 without validation errors. Phase 2 agents will populate them.
    """
    id: int
    change_type: str
    page_id: Optional[str] = None
    page_title: str
    section_heading: Optional[str] = None
    before_content: Optional[str] = None
    after_content: Optional[str] = None
    timestamp: str
    session_id: str
    status: str = "pending"
    source: Optional[str] = None
    rationale: Optional[str] = None
    generation_query: Optional[str] = None
    # --- Phase 2 verifier fields (safe defaults for Phase 1 backward-compat) ---
    transcript_evidence: List[str] = []
    confidence: Literal["high", "medium", "low"] = "low"
    risk: Literal["safe", "review", "risky"] = "safe"
    verifier_note: Optional[str] = None


def _bearer_token(authorization: Optional[str]) -> str:
    if not authorization:
        return ""
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer":
        return ""
    return token.strip()


def _auth_user_from_header(authorization: Optional[str]) -> Optional[Dict[str, Any]]:
    return supabase_store.user_from_bearer(_bearer_token(authorization))


def _history_user_or_401(authorization: Optional[str]) -> Dict[str, Any]:
    if not supabase_store.is_configured():
        raise HTTPException(status_code=503, detail="Supabase is not configured on the backend.")
    user = _auth_user_from_header(authorization)
    if not user:
        raise HTTPException(status_code=401, detail="Sign in with Google to view meeting history.")
    return user


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_meeting_state(session_id: Optional[str] = None) -> Dict[str, Any]:
    """Late-import session state to avoid circular imports at module load time."""
    try:
        from confluence_logic.jarvis_agentic import get_meeting_session_state, meeting_state  # noqa: PLC0415
        if session_id:
            return get_meeting_session_state(session_id)
        return meeting_state
    except Exception:
        return {}


def _get_local_nodes(_session_id: Optional[str] = None) -> Dict[str, Dict]:
    """Late-import graph_rag._local_nodes — populated during the meeting session."""
    try:
        from confluence_logic import graph_rag  # noqa: PLC0415
        return graph_rag._local_nodes
    except Exception:
        return {}


def _get_openai_client() -> OpenAI:
    global _openai_client
    if _openai_client is None:
        _openai_client = OpenAI()
    return _openai_client


def _openai_completion_options(model: str, max_tokens: int, temperature: float = 0.2) -> Dict[str, Any]:
    opts: Dict[str, Any] = {"model": model}
    if model.startswith(("gpt-5", "o1", "o3", "o4")):
        opts["max_completion_tokens"] = max_tokens
    else:
        opts["max_tokens"] = max_tokens
        opts["temperature"] = temperature
    return opts


def _extract_topics(local_nodes: Dict[str, Dict]) -> List[str]:
    """Return unique Topic names from the in-memory graph (title-cased, deduplicated)."""
    seen: set = set()
    topics: List[str] = []
    for meta in local_nodes.values():
        if meta.get("type") == "Topic":
            name = meta.get("name", "")
            key = name.lower()
            if name and key not in seen:
                seen.add(key)
                topics.append(name)
    return topics


def _extract_decisions(local_nodes: Dict[str, Dict]) -> List[str]:
    """Return unique Decision text values from the in-memory graph."""
    seen: set = set()
    decisions: List[str] = []
    for meta in local_nodes.values():
        if meta.get("type") == "Decision":
            text = meta.get("text") or meta.get("name", "")
            key = text.lower()[:80] if text else ""
            if text and key not in seen:
                seen.add(key)
                decisions.append(text)
    return decisions


def _extract_participants(local_nodes: Dict[str, Dict]) -> List[str]:
    """Return unique Person names from the in-memory graph."""
    seen: set = set()
    people: List[str] = []
    for meta in local_nodes.values():
        if meta.get("type") == "Person":
            name = meta.get("name", "")
            key = name.lower()
            if name and key not in seen:
                seen.add(key)
                people.append(name)
    return people


def _build_summary_text(transcript_log: List[Dict[str, Any]]) -> str:
    """Concatenate last 20 transcript entries as 'Speaker: text' lines."""
    if not transcript_log:
        return ""
    recent = transcript_log[-20:]
    lines = [f"{e.get('participant', 'Unknown')}: {e.get('text', '')}" for e in recent]
    return "\n".join(lines)


def _format_transcript(transcript_log: List[Dict[str, Any]], max_chars: Optional[int] = None) -> str:
    lines = [
        f"{entry.get('participant', 'Unknown')}: {entry.get('text', '')}"
        for entry in transcript_log
        if entry.get("text")
    ]
    text = "\n".join(lines)
    if max_chars is None or max_chars <= 0 or len(text) <= max_chars:
        return text
    # Guard: head_size must not exceed max_chars itself (fixes negative tail_size when max_chars < 2000)
    head_size = min(2000, max_chars // 2)
    tail_size = max_chars - head_size
    head = text[:head_size]
    tail = text[-tail_size:] if tail_size > 0 else ""
    return f"{head}\n[... middle transcript omitted ...]\n{tail}"


def _compress_transcript(transcript_log: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not transcript_log:
        return None
    raw = json.dumps(transcript_log, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    compressed = gzip.compress(raw, compresslevel=6)
    return {
        "transcript_compressed": base64.b64encode(compressed).decode("ascii"),
        "transcript_codec": "json+gzip+base64",
        "transcript_entry_count": len(transcript_log),
        "transcript_uncompressed_bytes": len(raw),
        "transcript_compressed_bytes": len(compressed),
    }


def _decompress_transcript(row: Dict[str, Any]) -> List[Dict[str, Any]]:
    if row.get("transcript_codec") != "json+gzip+base64" or not row.get("transcript_compressed"):
        return []
    try:
        compressed = base64.b64decode(row["transcript_compressed"])
        decoded = gzip.decompress(compressed).decode("utf-8")
        data = json.loads(decoded)
        return data if isinstance(data, list) else []
    except Exception as exc:
        logger.warning("Could not decode stored transcript for session %s: %s", row.get("session_id"), exc)
        return []


def _format_offset(timestamp: Any, started_at: Optional[str], first_timestamp: Optional[float]) -> str:
    try:
        ts = float(timestamp)
    except (TypeError, ValueError):
        return ""

    base: Optional[float] = first_timestamp
    if started_at:
        try:
            base = datetime.fromisoformat(started_at).timestamp()
        except Exception:
            base = first_timestamp
    if base is None:
        return ""

    seconds = max(0, int(ts - base))
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


def _build_transcript_highlights(
    transcript_log: List[Dict[str, Any]],
    started_at: Optional[str],
    limit: int = 8,
) -> List[Dict[str, str]]:
    if not transcript_log:
        return []

    first_timestamp = None
    for entry in transcript_log:
        try:
            first_timestamp = float(entry.get("timestamp"))
            break
        except (TypeError, ValueError):
            continue

    step = max(1, len(transcript_log) // limit)
    sampled = transcript_log[-limit:] if len(transcript_log) <= limit else transcript_log[::step][:limit]
    highlights = []
    for entry in sampled:
        text = (entry.get("text") or "").strip()
        if not text:
            continue
        highlights.append(
            {
                "time": _format_offset(entry.get("timestamp"), started_at, first_timestamp),
                "speaker": entry.get("participant") or "Unknown",
                "text": text,
            }
        )
    return highlights


def _fallback_review_insights(
    transcript_log: List[Dict[str, Any]],
    key_topics: List[str],
    decisions: List[str],
) -> Dict[str, Any]:
    transcript_text = _build_summary_text(transcript_log)
    summary = (
        "AI review generation is unavailable right now. The transcript was captured, but the post-meeting "
        "summary could not be generated. Check the OpenAI API key/model and retry."
        if transcript_text
        else "No transcript has been captured yet. Start or continue the meeting to populate this summary."
    )

    mom = []
    for i, entry in enumerate(transcript_log[-6:], start=1):
        text = (entry.get("text") or "").strip()
        if text:
            mom.append(
                {
                    "topic": f"Discussion point {i}",
                    "summary": f"{entry.get('participant', 'Unknown')}: {text}",
                }
            )

    action_items = []
    action_pattern = re.compile(
        r"\b(?:todo|to do|action item|follow up|own|take|prepare|send|share|create|draft|review|set up|configure|investigate|schedule)\b",
        re.IGNORECASE,
    )
    for entry in transcript_log:
        text = (entry.get("text") or "").strip()
        if text and action_pattern.search(text):
            action_items.append(
                {
                    "description": text,
                    "owner": entry.get("participant") or None,
                    "due": None,
                }
            )

    return {
        "summary": summary,
        "key_topics": key_topics,
        "decisions": decisions,
        "action_items": action_items[:8],
        "mom": mom,
    }


def _coerce_string_list(value: Any) -> List[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _coerce_object_list(value: Any, keys: List[str]) -> List[Dict[str, Any]]:
    if not isinstance(value, list):
        return []
    items: List[Dict[str, Any]] = []
    for raw in value:
        if isinstance(raw, dict):
            items.append({key: raw.get(key) for key in keys})
        elif isinstance(raw, str) and raw.strip():
            items.append({keys[0]: raw.strip(), **{key: None for key in keys[1:]}})
    return items


async def _run_review_agent(
    name: str,
    system_prompt: str,
    transcript_text: str,
    fallback: Dict[str, Any],
    max_tokens: int,
    hints: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    user_content = (
        f"Optional hints:\n{json.dumps(hints or {}, ensure_ascii=False)}\n\n"
        f"Full transcript:\n{transcript_text}"
    )
    try:
        response = await asyncio.to_thread(
            lambda: _get_openai_client().chat.completions.create(
                **_openai_completion_options(JARVIS_REVIEW_MODEL, max_tokens),
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
                response_format={"type": "json_object"},
            )
        )
        raw = response.choices[0].message.content or "{}"
        data = json.loads(raw)
        return data if isinstance(data, dict) else fallback
    except Exception as exc:
        logger.warning("%s review agent failed; using fallback: %s", name, exc)
        return fallback


async def _generate_review_insights(
    transcript_log: List[Dict[str, Any]],
    key_topics: List[str],
    decisions: List[str],
) -> Dict[str, Any]:
    fallback = _fallback_review_insights(transcript_log, key_topics, decisions)
    transcript_text = _format_transcript(transcript_log, JARVIS_REVIEW_MAX_INPUT_CHARS)
    if not transcript_text:
        return fallback

    hints = {
        "graph_key_topics": key_topics,
        "graph_decisions": decisions,
    }

    summary_prompt = (
        "You are the executive-summary specialist for a processed meeting UI. "
        "Return ONLY JSON shaped as {\"summary\": string}. "
        "Read the full transcript, including Jarvis turns, and write a detailed executive summary. "
        "Cover every important discussion, tradeoff, risk, decision, blocker, and follow-up supported by the transcript. "
        "Use 2-4 short paragraphs followed by concise bullets or a compact markdown table if that improves scanability. "
        "Do not invent facts."
    )
    topics_prompt = (
        "You are the topic and decision extraction specialist for a processed meeting UI. "
        "Return ONLY JSON shaped as {\"key_topics\": string[], \"decisions\": string[]}. "
        "Use the full transcript as source of truth; optional graph hints are only hints. "
        "Key topics should be concise labels. Decisions must be explicit or strongly evidenced by the discussion. "
        "Do not invent decisions."
    )
    action_prompt = (
        "You are the action-item extraction specialist for a processed meeting UI. "
        "Return ONLY JSON shaped as {\"action_items\": [{\"description\": string, \"owner\": string|null, \"due\": string|null}]}. "
        "Extract concrete follow-ups, ownership, due dates, blockers to unblock, and promised next steps. "
        "Only include items supported by the transcript; use null when owner or due date is not stated."
    )
    mom_prompt = (
        "You are the minutes-of-meeting specialist for a processed meeting UI. "
        "Return ONLY JSON shaped as {\"mom\": [{\"topic\": string, \"summary\": string}]}. "
        "Create clean chronological minutes that cover the meeting's important discussion sections. "
        "Each topic should be short; each summary should capture the substance, context, and outcome of that section."
    )

    summary_data, topic_data, action_data, mom_data = await asyncio.gather(
        _run_review_agent(
            "Executive summary",
            summary_prompt,
            transcript_text,
            {"summary": fallback["summary"]},
            JARVIS_REVIEW_SUMMARY_MAX_TOKENS,
            hints,
        ),
        _run_review_agent(
            "Topics and decisions",
            topics_prompt,
            transcript_text,
            {"key_topics": key_topics, "decisions": decisions},
            JARVIS_REVIEW_TOPICS_MAX_TOKENS,
            hints,
        ),
        _run_review_agent(
            "Action items",
            action_prompt,
            transcript_text,
            {"action_items": fallback["action_items"]},
            JARVIS_REVIEW_ACTION_ITEMS_MAX_TOKENS,
            hints,
        ),
        _run_review_agent(
            "Minutes of meeting",
            mom_prompt,
            transcript_text,
            {"mom": fallback["mom"]},
            JARVIS_REVIEW_MOM_MAX_TOKENS,
            hints,
        ),
    )

    generated_topics = _coerce_string_list(topic_data.get("key_topics"))
    generated_decisions = _coerce_string_list(topic_data.get("decisions"))
    return {
        "summary": str(summary_data.get("summary") or fallback["summary"]).strip(),
        "key_topics": generated_topics or key_topics,
        "decisions": generated_decisions or decisions,
        "action_items": (
            _coerce_object_list(action_data.get("action_items"), ["description", "owner", "due"])
            or fallback["action_items"]
        ),
        "mom": _coerce_object_list(mom_data.get("mom"), ["topic", "summary"]) or fallback["mom"],
    }


def _session_status(state: Dict[str, Any]) -> str:
    """Map meeting_state fields to the SessionStatus string the UI expects."""
    explicit = state.get("session_status")
    if explicit:
        return explicit
    if state.get("bot_id") and state.get("is_active"):
        return "in_meeting"
    if state.get("bot_id") and not state.get("is_active"):
        return "ended"
    return "idle"


def _normalize_recall_code(value: Any) -> str:
    return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")


def _latest_recall_status_code(payload: Dict[str, Any]) -> str:
    """Return the most useful Recall bot status code from a bot payload.

    Recall bot payloads have changed over time, so this checks both top-level
    fields and the latest status_changes entry.
    """
    direct_candidates = [
        payload.get("status"),
        payload.get("code"),
        payload.get("state"),
    ]
    for key in ("status_changes", "statusChanges"):
        changes = payload.get(key)
        if isinstance(changes, list) and changes:
            latest = changes[-1]
            if isinstance(latest, dict):
                direct_candidates.extend(
                    [
                        latest.get("code"),
                        latest.get("status"),
                        latest.get("state"),
                    ]
                )
            else:
                direct_candidates.append(latest)

    for candidate in reversed(direct_candidates):
        code = _normalize_recall_code(candidate)
        if code:
            return code
    return ""


def _recall_code_to_session_status(code: str, payload: Dict[str, Any]) -> Optional[str]:
    if code in _RECALL_ENDED_CODES:
        return "ended"
    if code in _RECALL_ERROR_CODES:
        return "error"
    if code in _RECALL_ACTIVE_CODES:
        return "in_meeting"

    # Defensive fallback for payload versions that expose terminal timestamps
    # instead of a compact status code.
    terminal_fields = (
        "ended_at",
        "end_time",
        "completed_at",
        "recording_completed_at",
        "left_call_at",
        "removed_at",
    )
    if any(payload.get(field) for field in terminal_fields):
        return "ended"
    return None


def _fetch_recall_bot_payload(bot_id: str) -> Dict[str, Any]:
    from confluence_logic.jarvis_agentic import RECALL_API_KEY, RECALL_BASE_URL  # noqa: PLC0415

    if not RECALL_API_KEY:
        raise RuntimeError("RECALL_API_KEY is not configured")

    response = requests.get(
        f"{RECALL_BASE_URL}/bot/{bot_id}/",
        headers={"Authorization": f"Token {RECALL_API_KEY}", "Accept": "application/json"},
        timeout=8,
    )
    response.raise_for_status()
    return response.json()


def _refresh_session_status_from_recall(state: Dict[str, Any]) -> None:
    bot_id = state.get("bot_id")
    if not bot_id or state.get("session_status") in {"ended", "error"}:
        return

    now = time.time()
    last_checked = float(state.get("last_recall_status_checked_at") or 0)
    if now - last_checked < _RECALL_STATUS_CACHE_SECONDS:
        return

    state["last_recall_status_checked_at"] = now
    try:
        payload = _fetch_recall_bot_payload(bot_id)
    except requests.HTTPError as exc:
        status_code = getattr(exc.response, "status_code", None)
        if status_code == 404:
            state["is_active"] = False
            state["session_status"] = "ended"
            state["ended_at"] = _utc_now_iso()
            state["end_reason"] = "Recall no longer returns this bot session."
            return
        logger.warning("Recall bot status lookup failed: %s", exc)
        return
    except Exception as exc:
        logger.warning("Recall bot status lookup failed: %s", exc)
        return

    code = _latest_recall_status_code(payload)
    mapped = _recall_code_to_session_status(code, payload)
    state["recall_status_code"] = code
    state["recall_status_payload"] = {
        "status": payload.get("status"),
        "latest_code": code,
    }
    if mapped == "ended":
        state["is_active"] = False
        state["session_status"] = "ended"
        state["ended_at"] = _utc_now_iso()
        state["end_reason"] = code or "Recall reported the bot left the meeting."
    elif mapped == "error":
        state["is_active"] = False
        state["session_status"] = "error"
        state["ended_at"] = _utc_now_iso()
        state["end_reason"] = code or "Recall reported a bot error."
    elif mapped == "in_meeting":
        state["is_active"] = True
        state["session_status"] = "in_meeting"


# ---------------------------------------------------------------------------
# POST /bot/start
# ---------------------------------------------------------------------------

async def _start_bot_for_session(
    body: StartBotRequest,
    session_id: Optional[str] = None,
    user: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Start the Recall.ai meeting bot for the given meeting URL.

    Calls create_bot() and stores bot_id + meeting_url in meeting_state.
    Returns SessionStatus shape: {status, bot_id, meeting_url, change_count}.
    """
    from confluence_logic.jarvis_agentic import (
        bind_bot_to_session,
        create_bot,
        create_meeting_session,
        get_meeting_session_state,
    )  # noqa: PLC0415

    meeting_url = body.meeting_url.strip()
    resolved_session_id = session_id or body.session_id or create_meeting_session()
    state = get_meeting_session_state(resolved_session_id)
    if not meeting_url:
        return {"status": "error", "session_id": resolved_session_id, "bot_id": None, "meeting_url": None, "change_count": 0,
                "error": "meeting_url is required"}

    bot_id = create_bot(meeting_url, session_id=resolved_session_id)
    if not bot_id:
        return {"status": "error", "session_id": resolved_session_id, "bot_id": None, "meeting_url": meeting_url,
                "change_count": 0, "error": "Failed to create bot — check RECALL_API_KEY and meeting URL"}

    state["session_id"] = resolved_session_id
    state["auth_user_id"] = user.get("id") if user else None
    state["bot_id"] = bot_id
    state["meeting_url"] = meeting_url
    state["is_active"] = True
    state["session_status"] = "in_meeting"
    state["started_at"] = _utc_now_iso()
    state["ended_at"] = None
    state["end_reason"] = None
    state["recall_status_code"] = None
    state["last_recall_status_checked_at"] = 0.0
    bind_bot_to_session(bot_id, resolved_session_id)
    _persist_history_snapshot(state, user)

    logger.info("Bot started: session=%s bot_id=%s meeting=%s", resolved_session_id, bot_id, meeting_url)
    return {
        "status": "in_meeting",
        "session_id": resolved_session_id,
        "bot_id": bot_id,
        "meeting_url": meeting_url,
        "change_count": 0,
    }


@router.post("/bot/start")
async def start_bot(body: StartBotRequest, authorization: Optional[str] = Header(default=None)) -> Dict[str, Any]:
    return await _start_bot_for_session(body, user=_auth_user_from_header(authorization))


@router.post("/sessions/{session_id}/bot/start")
async def start_bot_for_session(
    session_id: str,
    body: StartBotRequest,
    authorization: Optional[str] = Header(default=None),
) -> Dict[str, Any]:
    return await _start_bot_for_session(body, session_id=session_id, user=_auth_user_from_header(authorization))


# ---------------------------------------------------------------------------
# GET /bot/status
# ---------------------------------------------------------------------------

def _build_bot_status_response(state: Dict[str, Any]) -> Dict[str, Any]:
    _refresh_session_status_from_recall(state)
    status = _session_status(state)

    change_count: int = state.get("change_count", 0)

    return {
        "status": status,
        "session_id": state.get("session_id"),
        "bot_id": state.get("bot_id"),
        "meeting_url": state.get("meeting_url"),
        "change_count": change_count,
        "ended_at": state.get("ended_at"),
        "end_reason": state.get("end_reason"),
        "recall_status_code": state.get("recall_status_code"),
    }


def _history_title(state: Dict[str, Any]) -> str:
    meeting_url = state.get("meeting_url")
    if not meeting_url:
        return "Meeting Summary"
    try:
        from urllib.parse import urlparse  # noqa: PLC0415
        parsed = urlparse(meeting_url)
        return f"Meeting - {parsed.netloc}{parsed.path}"
    except Exception:
        return str(meeting_url)


def _meeting_chat_context(
    state: Dict[str, Any],
    history_item: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    summary_json = (history_item or {}).get("summary_json") or {}
    transcript_log = _decompress_transcript(history_item or {}) if history_item else []

    if not transcript_log:
        transcript_log = state.get("transcript_log") or []

    if not summary_json:
        summary_json = {
            "title": (history_item or {}).get("title") or _history_title(state),
            "summary": (history_item or {}).get("summary"),
            "key_topics": [],
            "decisions": [],
            "action_items": [],
            "mom": [],
            "participants": _extract_participants(_get_local_nodes(state.get("session_id"))),
        }

    return {
        "summary": summary_json,
        "transcript": transcript_log,
    }


def _hydrate_state_from_history_item(
    state: Dict[str, Any],
    history_item: Optional[Dict[str, Any]],
    user: Optional[Dict[str, Any]] = None,
) -> None:
    if not history_item:
        return

    summary_json = history_item.get("summary_json") or {}
    if user and user.get("id"):
        state["auth_user_id"] = user["id"]

    for source_key, state_key in (
        ("session_id", "session_id"),
        ("meeting_url", "meeting_url"),
        ("status", "session_status"),
        ("started_at", "started_at"),
        ("ended_at", "ended_at"),
        ("change_count", "change_count"),
    ):
        value = history_item.get(source_key)
        current = state.get(state_key)
        is_default_status = state_key == "session_status" and current == "idle"
        if value is not None and (not current or is_default_status):
            state[state_key] = value

    if not state.get("transcript_log"):
        transcript_log = _decompress_transcript(history_item)
        if transcript_log:
            state["transcript_log"] = transcript_log

    if isinstance(summary_json, dict):
        pending_changes = summary_json.get("pending_changes")
        if pending_changes and not state.get("pending_changes"):
            state["pending_changes"] = pending_changes
            state["change_count"] = len([change for change in pending_changes if change.get("status") == "pending"])


def _history_item_for_request(
    session_id: Optional[str],
    authorization: Optional[str],
) -> tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    user = _auth_user_from_header(authorization)
    if not user or not session_id:
        return None, user
    return supabase_store.get_history_item(user["id"], session_id), user


def _format_chat_context(context: Dict[str, Any]) -> str:
    summary = context.get("summary") or {}
    transcript = context.get("transcript") or []
    context_payload = {
        "title": summary.get("title"),
        "date": summary.get("date"),
        "executive_summary": summary.get("summary"),
        "key_topics": summary.get("key_topics") or [],
        "decisions": summary.get("decisions") or [],
        "action_items": summary.get("action_items") or [],
        "minutes_of_meeting": summary.get("mom") or [],
        "participants": summary.get("participants") or [],
    }
    return (
        f"Processed meeting context:\n{json.dumps(context_payload, ensure_ascii=False)}\n\n"
        f"Full meeting transcript:\n{_format_transcript(transcript, JARVIS_REVIEW_MAX_INPUT_CHARS)}"
    )


def _coerce_chat_messages(messages: List[MeetingChatMessage]) -> List[Dict[str, str]]:
    coerced: List[Dict[str, str]] = []
    for message in messages[-12:]:
        role = message.role if message.role in {"user", "assistant"} else "user"
        content = message.content.strip()
        if content:
            coerced.append({"role": role, "content": content})
    return coerced


def _next_change_id(pending: List[Dict[str, Any]]) -> int:
    existing_ids = [int(change.get("id") or 0) for change in pending if str(change.get("id") or "").isdigit()]
    return (max(existing_ids) if existing_ids else 0) + 1


def _confluence_graph_user_id(user: Optional[Dict[str, Any]], session_id: Optional[str]) -> str:
    if user and user.get("id"):
        return f"supabase:{user['id']}"
    return f"session:{session_id or 'default'}"


def _state_graph_user_id(state: Dict[str, Any], session_id: Optional[str] = None) -> str:
    auth_user_id = state.get("auth_user_id")
    if auth_user_id:
        return f"supabase:{auth_user_id}"
    return _confluence_graph_user_id(None, session_id or state.get("session_id"))


def _replace_agent_generated_changes(
    state: Dict[str, Any],
    proposals: List[Dict[str, Any]],
    session_id: Optional[str],
    query: str,
) -> List[Dict[str, Any]]:
    pending = [
        change for change in (state.get("pending_changes") or [])
        if change.get("source") != "meeting_proposal_agent"
    ]
    next_id = _next_change_id(pending)
    timestamp = _utc_now_iso()
    generated: List[Dict[str, Any]] = []

    for proposal in proposals:
        generated.append(
            {
                "id": next_id,
                "change_type": proposal.get("change_type") or "edit",
                "page_id": proposal.get("page_id"),
                "page_title": proposal.get("page_title") or "Confluence page",
                "section_heading": proposal.get("section_heading"),
                "before_content": proposal.get("before_content"),
                "after_content": proposal.get("after_content"),
                "timestamp": timestamp,
                "session_id": session_id or state.get("session_id") or "",
                "status": "pending",
                "source": "meeting_proposal_agent",
                "rationale": proposal.get("rationale"),
                "generation_query": query or None,
                "transcript_evidence": [],
                "confidence": "low",
                "risk": "safe",
                "verifier_note": None,
            }
        )
        next_id += 1

    state["pending_changes"] = pending + generated
    state["change_count"] = len([change for change in state["pending_changes"] if change.get("status") == "pending"])
    return generated


def _append_agent_generated_changes(
    state: Dict[str, Any],
    proposals: List[Dict[str, Any]],
    session_id: Optional[str],
    query: str,
    source: str = "meeting_proposal_agent",
) -> List[Dict[str, Any]]:
    pending = state.get("pending_changes") or []
    next_id = _next_change_id(pending)
    timestamp = _utc_now_iso()
    generated: List[Dict[str, Any]] = []

    for proposal in proposals:
        generated.append(
            {
                "id": next_id,
                "change_type": proposal.get("change_type") or "edit",
                "page_id": proposal.get("page_id"),
                "page_title": proposal.get("page_title") or "Confluence page",
                "section_heading": proposal.get("section_heading"),
                "before_content": proposal.get("before_content"),
                "after_content": proposal.get("after_content"),
                "timestamp": timestamp,
                "session_id": session_id or state.get("session_id") or "",
                "status": "pending",
                "source": source,
                "rationale": proposal.get("rationale"),
                "generation_query": query or None,
                "transcript_evidence": [],
                "confidence": "low",
                "risk": "safe",
                "verifier_note": None,
            }
        )
        next_id += 1

    state["pending_changes"] = pending + generated
    state["change_count"] = len([change for change in state["pending_changes"] if change.get("status") == "pending"])
    return generated


def _get_editor_agent():
    try:
        from confluence_logic.jarvis_agentic import session_agent  # noqa: PLC0415
        return session_agent
    except Exception:
        from confluence_logic.agents.editor_agent import EditorAgent  # noqa: PLC0415
        return EditorAgent(model=JARVIS_REVIEW_MODEL)


def _format_approved_change_request(change: Dict[str, Any]) -> str:
    change_type = str(change.get("change_type") or "edit").lower()
    page_title = change.get("page_title") or "Confluence page"
    page_id = change.get("page_id") or "NONE"
    heading = change.get("section_heading") or "UNKNOWN"
    before = change.get("before_content") or ""
    after = change.get("after_content") or ""
    rationale = change.get("rationale") or ""

    action = {
        "create": "create",
        "delete": "delete",
        "title": "title",
    }.get(change_type, "edit")

    if change_type == "create":
        instruction = (
            f"Create a new Confluence page titled '{page_title}' with the approved content below. "
            "Use the existing create page tool and do not edit an unrelated existing page."
        )
    elif change_type == "delete":
        instruction = (
            f"Delete the approved content from the existing Confluence page '{page_title}'. "
            "Use fetch, preview_delete, and commit_delete. Do not create a new page."
        )
    elif change_type == "title":
        instruction = (
            f"Rename the existing Confluence page '{page_title}' using the approved title/content below. "
            "Use update_page_title. Do not create a replacement page."
        )
    else:
        instruction = (
            f"Update the existing Confluence page '{page_title}' with the approved content below. "
            "Use fetch_live_page, preview_edit, and commit_document_edit. Do not create a new page."
        )

    return (
        "Resolver context:\n"
        f"ACTION: {action}\n"
        f"PAGE_TITLE: {page_title}\n"
        f"PAGE_ID: {page_id}\n"
        f"HEADING: {heading}\n"
        f"REFRAMED_REQUEST: {instruction}\n"
        f"RATIONALE: {rationale or 'Approved from meeting proposal queue.'}\n\n"
        "Approved Confluence change request:\n"
        f"{instruction}\n\n"
        f"Target page title: {page_title}\n"
        f"Target page id: {page_id}\n"
        f"Target section heading: {heading}\n"
        f"Existing content or anchor excerpt:\n{before or '[none supplied]'}\n\n"
        f"Approved new content:\n{after or '[none supplied]'}\n\n"
        "Apply only this approved change. If the target cannot be verified with existing Confluence tools, fail clearly."
    )


async def _execute_single_change(
    state: Dict[str, Any],
    change: Dict[str, Any],
) -> Dict[str, Any]:
    editor_agent = _get_editor_agent()
    context = _meeting_chat_context(state)
    meeting_context = _format_chat_context(context)
    prepared_query = _format_approved_change_request(change)
    change["status"] = "approved"

    graph_user_id = _confluence_graph_user_id(
        {"id": state.get("auth_user_id")} if state.get("auth_user_id") else None,
        state.get("session_id"),
    )
    graph_token = confluence_page_graph.set_current_graph_user_id(graph_user_id)
    try:
        try:
            answer = await editor_agent.handle_prepared_query(
                prepared_query,
                original_query=f"Approve Confluence change {change.get('id')}",
                meeting_context=meeting_context,
            )
        except Exception as exc:
            change["status"] = "failed"
            change["execution_error"] = str(exc)
            return {"id": change.get("id"), "success": False, "error": str(exc)}
    finally:
        confluence_page_graph.reset_current_graph_user_id(graph_token)

    normalized_answer = (answer or "").strip()
    if normalized_answer.lower().startswith(("error:", "the requested change did not complete", "i encountered an issue")):
        change["status"] = "failed"
        change["execution_error"] = normalized_answer
        return {"id": change.get("id"), "success": False, "error": normalized_answer}

    change["status"] = "executed"
    change["execution_result"] = normalized_answer
    return {"id": change.get("id"), "success": True}


async def _execute_changes_for_state(state: Dict[str, Any], ids: List[int]) -> Dict[str, Any]:
    pending: List[Dict[str, Any]] = state.get("pending_changes", [])
    results = []

    for change_id in ids:
        match = next((c for c in pending if c.get("id") == change_id), None)
        if not match:
            results.append({"id": change_id, "success": False, "error": "Change not found"})
            continue
        if match.get("status") not in {None, "pending", "approved", "failed"}:
            results.append({"id": change_id, "success": False, "error": f"Change is already {match.get('status')}"})
            continue
        results.append(await _execute_single_change(state, match))

    state["change_count"] = len([change for change in pending if change.get("status") == "pending"])
    return {"results": results}


async def _propose_changes_for_state(
    state: Dict[str, Any],
    body: ProposeChangesRequest,
    session_id: Optional[str] = None,
    authorization: Optional[str] = None,
) -> Dict[str, Any]:
    query = (body.query or "").strip()
    transcript_log: List[Dict[str, Any]] = state.get("transcript_log") or []
    summary = await _get_review_summary_for_state(state, session_id=session_id)
    transcript_text = _format_transcript(transcript_log, JARVIS_REVIEW_MAX_INPUT_CHARS)

    if not transcript_text and not summary.get("summary"):
        raise HTTPException(status_code=404, detail="No meeting context is available for this session.")

    user = _auth_user_from_header(authorization)
    if user:
        state["auth_user_id"] = user.get("id")
    agent = ProposedChangesAgent(
        model=JARVIS_REVIEW_MODEL,
        client_factory=_get_openai_client,
        max_tokens=JARVIS_PROPOSE_CHANGES_MAX_TOKENS,
    )
    proposals = await agent.propose(
        transcript_text=transcript_text,
        summary=summary,
        query=query,
        graph_user_id=_confluence_graph_user_id(user, session_id or state.get("session_id")) if user else _state_graph_user_id(state, session_id),
    )
    generated = _replace_agent_generated_changes(state, proposals, session_id, query)
    _persist_history_snapshot(state, user, summary)
    return {
        "changes": state.get("pending_changes", []),
        "generated_count": len(generated),
    }


async def _answer_meeting_chat(context: Dict[str, Any], messages: List[MeetingChatMessage]) -> str:
    chat_messages = _coerce_chat_messages(messages)
    if not chat_messages:
        raise HTTPException(status_code=400, detail="At least one chat message is required.")
    if not context.get("transcript") and not (context.get("summary") or {}).get("summary"):
        raise HTTPException(status_code=404, detail="No meeting context is available for this session.")

    system_prompt = (
        "You are a meeting-specific chat assistant with independent judgment. "
        "Use the provided meeting context to understand what happened: executive summary, minutes, topics, "
        "decisions, action items, participants, and transcript. The transcript may include Jarvis as a meeting participant. "
        "When the user asks what happened, who said something, what was decided, or what action items exist, answer strictly from this meeting context. "
        "When the user asks for your opinion, critique, strategy, risks, how to fix something, or what you think about a plan, "
        "combine the meeting context with your general knowledge and reasoning. You may disagree with the plan discussed in the meeting, "
        "propose a better plan, point out missing risks, and be creative. Clearly separate meeting facts from your own assessment when needed. "
        "Be respectful, practical, and specific. If something is not in the meeting context, say that clearly before giving external analysis."
    )
    response = await asyncio.to_thread(
        lambda: _get_openai_client().chat.completions.create(
            **_openai_completion_options(JARVIS_REVIEW_MODEL, 800),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "system", "content": _format_chat_context(context)},
                *chat_messages,
            ],
        )
    )
    return (response.choices[0].message.content or "").strip() or "I could not answer that from this meeting."


def _persist_history_snapshot(
    state: Dict[str, Any],
    user: Optional[Dict[str, Any]],
    summary: Optional[Dict[str, Any]] = None,
) -> None:
    if not user:
        return

    summary_payload = dict(summary or {})
    pending_changes = state.get("pending_changes") or []
    if pending_changes:
        summary_payload["pending_changes"] = pending_changes

    stats = summary_payload.get("stats") if summary_payload else None
    transcript_payload = _compress_transcript(state.get("transcript_log") or []) or {}
    supabase_store.upsert_history(
        {
            "user_id": user.get("id"),
            "session_id": state.get("session_id"),
            "title": summary_payload.get("title") or _history_title(state),
            "meeting_url": state.get("meeting_url"),
            "status": _session_status(state),
            "started_at": state.get("started_at"),
            "ended_at": state.get("ended_at"),
            "summary": summary_payload.get("summary"),
            "summary_json": summary_payload or None,
            "change_count": state.get("change_count", 0),
            "stats": stats,
            **transcript_payload,
        }
    )


@router.get("/bot/status")
async def get_bot_status(authorization: Optional[str] = Header(default=None)) -> Dict[str, Any]:
    """
    Return current bot/session status.

    Response shape: {status, bot_id, meeting_url, change_count}
    status values: "idle" | "in_meeting" | "ended" | "error"
    """
    state = _get_meeting_state()
    response = _build_bot_status_response(state)
    _persist_history_snapshot(state, _auth_user_from_header(authorization))
    return response


@router.get("/sessions/{session_id}/bot/status")
async def get_bot_status_for_session(
    session_id: str,
    authorization: Optional[str] = Header(default=None),
) -> Dict[str, Any]:
    state = _get_meeting_state(session_id)
    history_item, user = _history_item_for_request(session_id, authorization)
    _hydrate_state_from_history_item(state, history_item, user)
    response = _build_bot_status_response(state)
    _persist_history_snapshot(state, user)
    return response


# ---------------------------------------------------------------------------
# GET /review/changes
# ---------------------------------------------------------------------------

@router.get("/review/changes")
async def get_review_changes() -> List[Dict[str, Any]]:
    """
    Return pending Confluence changes for the current session.

    Returns an empty list when no changes are queued (change_queue not yet wired).
    """
    state = _get_meeting_state()
    # When the SQLite change_queue is wired, query by session_id here.
    pending: List[Dict[str, Any]] = state.get("pending_changes", [])
    return pending


@router.get("/sessions/{session_id}/review/changes")
async def get_review_changes_for_session(
    session_id: str,
    authorization: Optional[str] = Header(default=None),
) -> List[Dict[str, Any]]:
    state = _get_meeting_state(session_id)
    history_item, user = _history_item_for_request(session_id, authorization)
    _hydrate_state_from_history_item(state, history_item, user)
    pending: List[Dict[str, Any]] = state.get("pending_changes", [])
    return pending


@router.post("/review/changes/propose")
async def propose_review_changes(
    body: ProposeChangesRequest,
    authorization: Optional[str] = Header(default=None),
) -> Dict[str, Any]:
    state = _get_meeting_state()
    return await _propose_changes_for_state(state, body, authorization=authorization)


@router.post("/sessions/{session_id}/review/changes/propose")
async def propose_review_changes_for_session(
    session_id: str,
    body: ProposeChangesRequest,
    authorization: Optional[str] = Header(default=None),
) -> Dict[str, Any]:
    state = _get_meeting_state(session_id)
    history_item, user = _history_item_for_request(session_id, authorization)
    _hydrate_state_from_history_item(state, history_item, user)
    return await _propose_changes_for_state(state, body, session_id=session_id, authorization=authorization)


# ---------------------------------------------------------------------------
# POST /review/execute
# ---------------------------------------------------------------------------

@router.post("/review/execute")
async def execute_review_changes(body: ExecuteChangesRequest) -> Dict[str, Any]:
    """
    Mark the selected change IDs as approved and execute them.

    Returns per-change results: {results: [{id, success, error?}]}
    """
    state = _get_meeting_state()
    return await _execute_changes_for_state(state, body.ids)


@router.post("/sessions/{session_id}/review/execute")
async def execute_review_changes_for_session(session_id: str, body: ExecuteChangesRequest) -> Dict[str, Any]:
    state = _get_meeting_state(session_id)
    return await _execute_changes_for_state(state, body.ids)


# ---------------------------------------------------------------------------
# GET /history
# ---------------------------------------------------------------------------

@router.get("/history")
async def get_history(authorization: Optional[str] = Header(default=None)) -> List[Dict[str, Any]]:
    user = _history_user_or_401(authorization)
    return supabase_store.list_history(user["id"])


@router.get("/history/{session_id}")
async def get_history_item(session_id: str, authorization: Optional[str] = Header(default=None)) -> Dict[str, Any]:
    user = _history_user_or_401(authorization)
    item = supabase_store.get_history_item(user["id"], session_id)
    if not item:
        raise HTTPException(status_code=404, detail="Meeting history item not found.")
    return item


# ---------------------------------------------------------------------------
# GET /review/summary
# ---------------------------------------------------------------------------

async def _get_review_summary_for_state(state: Dict[str, Any], session_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Return a structured meeting summary for the results page.

    Response matches the TypeScript MeetingSummary type used by sync-sage-bot.
    """
    local_nodes = _get_local_nodes(session_id)

    meeting_url: Optional[str] = state.get("meeting_url") or os.getenv("MEETING_URL") or None
    started_at: Optional[str] = state.get("started_at") or None

    # Derive a human-readable title from the meeting URL
    title = "Meeting Summary"
    if meeting_url:
        # e.g. "Google Meet — meet.google.com/abc-def-ghi"
        from urllib.parse import urlparse  # noqa: PLC0415
        try:
            parsed = urlparse(meeting_url)
            title = f"Meeting — {parsed.netloc}{parsed.path}"
        except Exception:
            title = meeting_url

    # Format date from started_at timestamp or fall back to today
    date_str = ""
    if started_at:
        try:
            date_str = datetime.fromisoformat(started_at).strftime("%B %d, %Y %H:%M UTC")
        except Exception:
            date_str = started_at
    else:
        date_str = _utc_now().strftime("%B %d, %Y")

    transcript_log: List[Dict[str, Any]] = state.get("transcript_log") or []

    # Graph RAG-derived data
    key_topics = _extract_topics(local_nodes)
    decisions = _extract_decisions(local_nodes)
    participants = _extract_participants(local_nodes)

    # Fallback: derive participants from transcript if graph has none
    if not participants and transcript_log:
        seen: set = set()
        for entry in transcript_log:
            p = entry.get("participant", "")
            if p and p not in seen:
                seen.add(p)
                participants.append(p)

    insights = await _generate_review_insights(transcript_log, key_topics, decisions)
    transcript_highlights = _build_transcript_highlights(transcript_log, started_at)

    return {
        "title": title,
        "session_id": state.get("session_id"),
        "date": date_str,
        "summary": insights["summary"],
        "key_topics": insights["key_topics"],
        "action_items": insights["action_items"],
        "decisions": insights["decisions"],
        "participants": participants,
        "mom": insights["mom"],
        "transcript_highlights": transcript_highlights,
        "stats": {
            "transcript_entries": len(transcript_log),
            "topic_count": len(insights["key_topics"]),
            "decision_count": len(insights["decisions"]),
            "action_item_count": len(insights["action_items"]),
        },
    }


def _stored_summary_response(
    history_item: Optional[Dict[str, Any]],
    state: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    if not history_item:
        return None
    summary_json = history_item.get("summary_json")
    if not isinstance(summary_json, dict) or not summary_json.get("summary"):
        return None
    if str(summary_json.get("summary") or "").startswith("AI review generation is unavailable right now."):
        return None

    transcript_log = state.get("transcript_log") or _decompress_transcript(history_item)
    stored = {
        key: value
        for key, value in summary_json.items()
        if key in {
            "title",
            "session_id",
            "date",
            "summary",
            "key_topics",
            "action_items",
            "decisions",
            "participants",
            "mom",
            "transcript_highlights",
            "stats",
        }
    }
    stored.setdefault("title", history_item.get("title") or _history_title(state))
    stored["session_id"] = history_item.get("session_id") or state.get("session_id")
    stored.setdefault("date", history_item.get("started_at") or history_item.get("updated_at") or "")
    stored.setdefault("key_topics", [])
    stored.setdefault("action_items", [])
    stored.setdefault("decisions", [])
    stored.setdefault("participants", [])
    stored.setdefault("mom", [])
    stored["transcript_highlights"] = stored.get("transcript_highlights") or _build_transcript_highlights(
        transcript_log,
        history_item.get("started_at") or state.get("started_at"),
    )
    if not stored.get("stats"):
        stored["stats"] = {
            "transcript_entries": len(transcript_log),
            "topic_count": len(stored["key_topics"]),
            "decision_count": len(stored["decisions"]),
            "action_item_count": len(stored["action_items"]),
        }
    return stored


@router.get("/review/summary")
async def get_review_summary(authorization: Optional[str] = Header(default=None)) -> Dict[str, Any]:
    state = _get_meeting_state()
    summary = await _get_review_summary_for_state(state)
    _persist_history_snapshot(state, _auth_user_from_header(authorization), summary)
    return summary


@router.get("/sessions/{session_id}/review/summary")
async def get_review_summary_for_session(
    session_id: str,
    authorization: Optional[str] = Header(default=None),
) -> Dict[str, Any]:
    state = _get_meeting_state(session_id)
    history_item, user = _history_item_for_request(session_id, authorization)
    _hydrate_state_from_history_item(state, history_item, user)
    stored = _stored_summary_response(history_item, state)
    if stored:
        return stored
    summary = await _get_review_summary_for_state(state, session_id=session_id)
    _persist_history_snapshot(state, user, summary)
    return summary


@router.post("/sessions/{session_id}/review/chat")
async def chat_with_meeting(
    session_id: str,
    body: MeetingChatRequest,
    authorization: Optional[str] = Header(default=None),
) -> Dict[str, Any]:
    user = _auth_user_from_header(authorization)
    history_item = supabase_store.get_history_item(user["id"], session_id) if user else None
    state = _get_meeting_state(session_id)

    if user and not history_item and not state.get("transcript_log"):
        raise HTTPException(status_code=404, detail="Meeting history item not found.")

    context = _meeting_chat_context(state, history_item)
    answer = await _answer_meeting_chat(context, body.messages)
    return {
        "answer": answer,
        "session_id": session_id,
        "context": {
            "transcript_entries": len(context.get("transcript") or []),
            "has_summary": bool((context.get("summary") or {}).get("summary")),
        },
    }


# ---------------------------------------------------------------------------
# Multi-agent pipeline helpers and endpoint (PIPE-01 through PIPE-04)
# ---------------------------------------------------------------------------


async def _draft_verify_persist(
    page: Dict[str, Any],
    facts: ExtractedFacts,
    transcript_text: str,
    job_id: str,
    session_id: str,
    user_id: Optional[str],
) -> None:
    """Draft, verify, and immediately persist one page's proposal. Errors are isolated."""
    try:
        draft = await _run_drafter(page, facts, transcript_text)
        verified = await _run_verifier(
            draft,
            transcript_text,
            page.get("relevant_content") or "",
        )
        await asyncio.to_thread(
            supabase_store.upsert_proposal,
            {
                **verified,
                "job_id": job_id,
                "session_id": session_id,
                "user_id": user_id,
                "source": "pipeline",
                "status": "pending",
            },
        )
    except Exception as exc:
        logger.warning(
            "Draft-verify-persist failed for page %s: %s",
            page.get("page_id"),
            exc,
        )


async def _run_pipeline(
    session_id: str,
    job_id: str,
    user_id: Optional[str],
    graph_user_id: str,
) -> None:
    """Background pipeline orchestrator. Detached via asyncio.create_task.

    Exceptions are caught and recorded in pipeline_jobs; never propagated to the event loop.
    graph_user_id is passed explicitly — do NOT rely on ContextVar inheritance (Pitfall 4).
    """
    if not user_id:
        logger.error("Pipeline %s: user_id is required but was None — aborting", job_id)
        await asyncio.to_thread(
            supabase_store.update_pipeline_job,
            job_id, None, "failed", "user_id is required", _utc_now_iso(),
        )
        return
    try:
        # Stage 1: Fact Extraction
        await asyncio.to_thread(
            supabase_store.update_pipeline_job,
            job_id, "fact_extraction", "running", None, None,
        )
        state = _get_meeting_state(session_id)
        transcript_log = state.get("transcript_log") or []

        # D-12: fetch history_item for summary_json enrichment; also fallback transcript source
        history_item = await asyncio.to_thread(
            supabase_store.get_history_item, user_id, session_id
        ) or {}
        if not transcript_log:
            # D-12: in-memory state gone — decompress from Supabase
            transcript_log = _decompress_transcript(history_item)

        transcript_text = _format_transcript(transcript_log, JARVIS_FACT_INPUT_MAX_CHARS)

        # D-02: prepend stored summary_json metadata (decisions/action_items/key_topics)
        # above the transcript so FactExtractionAgent has full meeting context
        summary_json = history_item.get("summary_json") or {}
        metadata_parts: list = []
        if summary_json.get("decisions"):
            metadata_parts.append("Prior decisions: " + "; ".join(summary_json["decisions"][:10]))
        if summary_json.get("action_items"):
            items_str = "; ".join(
                (a.get("description") or str(a)) for a in summary_json["action_items"][:10]
            )
            metadata_parts.append("Action items: " + items_str)
        if summary_json.get("key_topics"):
            metadata_parts.append("Key topics: " + "; ".join(summary_json["key_topics"][:10]))
        if metadata_parts:
            metadata_prefix = "[Meeting Context]\n" + "\n".join(metadata_parts) + "\n\n[Transcript]\n"
            transcript_text = metadata_prefix + transcript_text

        facts = await _run_fact_extraction(transcript_text)

        # Stage 2: Merged RAG Retrieval
        await asyncio.to_thread(
            supabase_store.update_pipeline_job,
            job_id, "retrieval", None, None, None,
        )
        candidate_pages = await _merged_rag_retrieval(graph_user_id, facts.query_terms)

        # D-06: Zero-RAG fallback — generate create proposals from doc_worthy_updates
        if not candidate_pages and facts.doc_worthy_updates:
            candidate_pages = [
                {
                    "page_id": None,
                    "title": f"New page: {item[:60]}",
                    "relevant_content": "",
                    "score": 0.0,
                }
                for item in facts.doc_worthy_updates[:3]
            ]

        # D-09: Deduplicate by page_id; cap at JARVIS_PIPELINE_MAX_PAGES
        seen_ids: set = set()
        deduplicated: list = []
        for p in candidate_pages:
            pid = p.get("page_id")
            key = pid if pid else id(p)  # None page_ids (create proposals) each get own key
            if key not in seen_ids:
                seen_ids.add(key)
                deduplicated.append(p)
        candidate_pages = deduplicated[:JARVIS_PIPELINE_MAX_PAGES]

        # Stage 3: Parallel Draft + Verify + Persist
        await asyncio.to_thread(
            supabase_store.update_pipeline_job,
            job_id, "drafting", None, None, None,
        )
        tasks = [
            _draft_verify_persist(page, facts, transcript_text, job_id, session_id, user_id)
            for page in candidate_pages
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for r in results:
            if isinstance(r, Exception):
                logger.warning("Pipeline task failed (non-fatal): %s", r)

        # Stage 4: Complete
        await asyncio.to_thread(
            supabase_store.update_pipeline_job,
            job_id, "complete", "completed", None, _utc_now_iso(),
        )

    except Exception as exc:
        logger.error("Pipeline %s failed: %s", job_id, exc)
        try:
            await asyncio.to_thread(
                supabase_store.update_pipeline_job,
                job_id, None, "failed", str(exc), _utc_now_iso(),
            )
        except Exception:
            pass  # Supabase update failure is non-fatal


@router.post("/review/pipeline/start", status_code=202)
async def start_pipeline(
    body: PipelineStartRequest,
    authorization: Optional[str] = Header(default=None),
) -> Dict[str, Any]:
    """Start the multi-agent proposal pipeline for a session. Returns 202 immediately.

    The pipeline runs as a background asyncio task. Poll /review/pipeline/{job_id}
    or stream /review/pipeline/{job_id}/stream (Phase 3) for progress.
    """
    user = _auth_user_from_header(authorization)
    if not user:
        raise HTTPException(status_code=401, detail="Authentication required to start pipeline.")

    job_id = await asyncio.to_thread(
        supabase_store.create_pipeline_job,
        body.session_id,
        user["id"],
    )
    # job_id may be None if Supabase is not configured — pipeline still runs, just not tracked
    graph_user_id = _confluence_graph_user_id(user, body.session_id)
    asyncio.create_task(
        _run_pipeline(body.session_id, job_id or "", user["id"], graph_user_id)
    )
    return {"job_id": job_id, "status": "accepted"}
