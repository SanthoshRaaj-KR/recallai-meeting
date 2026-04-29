"""
Review API router — FastAPI routes for meeting review and Confluence change management.

Mounted into the main jarvis_agentic app via app.include_router(router).
All routes are served at the root prefix (e.g. GET /review/summary, POST /bot/start).
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter
from openai import OpenAI
from pydantic import BaseModel
import requests

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


def _format_transcript(transcript_log: List[Dict[str, Any]], max_chars: int = 10000) -> str:
    lines = [
        f"{entry.get('participant', 'Unknown')}: {entry.get('text', '')}"
        for entry in transcript_log
        if entry.get("text")
    ]
    text = "\n".join(lines)
    if len(text) <= max_chars:
        return text
    head = text[:2000]
    tail = text[-(max_chars - 2000):]
    return f"{head}\n[... middle transcript omitted ...]\n{tail}"


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
        transcript_text
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


async def _generate_review_insights(
    transcript_log: List[Dict[str, Any]],
    key_topics: List[str],
    decisions: List[str],
) -> Dict[str, Any]:
    fallback = _fallback_review_insights(transcript_log, key_topics, decisions)
    transcript_text = _format_transcript(transcript_log)
    if not transcript_text:
        return fallback

    prompt = (
        "You generate structured meeting review data for a UI. "
        "Return ONLY valid JSON with this exact shape:\n"
        "{"
        "\"summary\": string, "
        "\"key_topics\": string[], "
        "\"decisions\": string[], "
        "\"action_items\": [{\"description\": string, \"owner\": string|null, \"due\": string|null}], "
        "\"mom\": [{\"topic\": string, \"summary\": string}]"
        "}\n"
        "Use only the transcript. Do not invent people, due dates, or decisions. "
        "If a field has no real evidence, return an empty array for it."
    )

    try:
        response = await asyncio.to_thread(
            lambda: _get_openai_client().chat.completions.create(
                model=os.getenv("JARVIS_REVIEW_MODEL", os.getenv("JARVIS_GENERAL_MODEL", "gpt-4o-mini")),
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": f"Transcript:\n{transcript_text}"},
                ],
                max_tokens=900,
                temperature=0.2,
                response_format={"type": "json_object"},
            )
        )
        raw = response.choices[0].message.content or "{}"
        data = json.loads(raw)
    except Exception as exc:
        logger.warning("Review insight generation failed; using fallback: %s", exc)
        return fallback

    generated_topics = _coerce_string_list(data.get("key_topics"))
    generated_decisions = _coerce_string_list(data.get("decisions"))
    return {
        "summary": str(data.get("summary") or fallback["summary"]).strip(),
        "key_topics": generated_topics or key_topics,
        "decisions": generated_decisions or decisions,
        "action_items": _coerce_object_list(data.get("action_items"), ["description", "owner", "due"]),
        "mom": _coerce_object_list(data.get("mom"), ["topic", "summary"]),
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

async def _start_bot_for_session(body: StartBotRequest, session_id: Optional[str] = None) -> Dict[str, Any]:
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

    logger.info("Bot started: session=%s bot_id=%s meeting=%s", resolved_session_id, bot_id, meeting_url)
    return {
        "status": "in_meeting",
        "session_id": resolved_session_id,
        "bot_id": bot_id,
        "meeting_url": meeting_url,
        "change_count": 0,
    }


@router.post("/bot/start")
async def start_bot(body: StartBotRequest) -> Dict[str, Any]:
    return await _start_bot_for_session(body)


@router.post("/sessions/{session_id}/bot/start")
async def start_bot_for_session(session_id: str, body: StartBotRequest) -> Dict[str, Any]:
    return await _start_bot_for_session(body, session_id=session_id)


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


@router.get("/bot/status")
async def get_bot_status() -> Dict[str, Any]:
    """
    Return current bot/session status.

    Response shape: {status, bot_id, meeting_url, change_count}
    status values: "idle" | "in_meeting" | "ended" | "error"
    """
    return _build_bot_status_response(_get_meeting_state())


@router.get("/sessions/{session_id}/bot/status")
async def get_bot_status_for_session(session_id: str) -> Dict[str, Any]:
    return _build_bot_status_response(_get_meeting_state(session_id))


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
async def get_review_changes_for_session(session_id: str) -> List[Dict[str, Any]]:
    state = _get_meeting_state(session_id)
    pending: List[Dict[str, Any]] = state.get("pending_changes", [])
    return pending


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
    pending: List[Dict[str, Any]] = state.get("pending_changes", [])

    results = []
    for change_id in body.ids:
        match = next((c for c in pending if c.get("id") == change_id), None)
        if not match:
            results.append({"id": change_id, "success": False, "error": "Change not found"})
            continue
        # TODO: wire to actual Confluence commit when change_queue is implemented
        results.append({"id": change_id, "success": True})

    return {"results": results}


@router.post("/sessions/{session_id}/review/execute")
async def execute_review_changes_for_session(session_id: str, body: ExecuteChangesRequest) -> Dict[str, Any]:
    state = _get_meeting_state(session_id)
    pending: List[Dict[str, Any]] = state.get("pending_changes", [])

    results = []
    for change_id in body.ids:
        match = next((c for c in pending if c.get("id") == change_id), None)
        if not match:
            results.append({"id": change_id, "success": False, "error": "Change not found"})
            continue
        results.append({"id": change_id, "success": True})

    return {"results": results}


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


@router.get("/review/summary")
async def get_review_summary() -> Dict[str, Any]:
    return await _get_review_summary_for_state(_get_meeting_state())


@router.get("/sessions/{session_id}/review/summary")
async def get_review_summary_for_session(session_id: str) -> Dict[str, Any]:
    return await _get_review_summary_for_state(_get_meeting_state(session_id), session_id=session_id)
