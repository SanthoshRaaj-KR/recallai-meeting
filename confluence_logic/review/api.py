"""
Review API router — FastAPI routes for meeting review and Confluence change management.

Mounted into the main jarvis_agentic app via app.include_router(router).
All routes are served at the root prefix (e.g. GET /review/summary, POST /bot/start).
"""
from __future__ import annotations

import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter()


# ---------------------------------------------------------------------------
# Pydantic request/response models
# ---------------------------------------------------------------------------

class StartBotRequest(BaseModel):
    meeting_url: str


class ExecuteChangesRequest(BaseModel):
    ids: List[int]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_meeting_state() -> Dict[str, Any]:
    """Late-import meeting_state to avoid circular imports at module load time."""
    try:
        from confluence_logic.jarvis_agentic import meeting_state  # noqa: PLC0415
        return meeting_state
    except Exception:
        return {}


def _get_local_nodes() -> Dict[str, Dict]:
    """Late-import graph_rag._local_nodes — populated during the meeting session."""
    try:
        from confluence_logic import graph_rag  # noqa: PLC0415
        return graph_rag._local_nodes
    except Exception:
        return {}


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


# ---------------------------------------------------------------------------
# POST /bot/start
# ---------------------------------------------------------------------------

@router.post("/bot/start")
async def start_bot(body: StartBotRequest) -> Dict[str, Any]:
    """
    Start the Recall.ai meeting bot for the given meeting URL.

    Calls create_bot() and stores bot_id + meeting_url in meeting_state.
    Returns SessionStatus shape: {status, bot_id, meeting_url, change_count}.
    """
    from confluence_logic.jarvis_agentic import create_bot, meeting_state  # noqa: PLC0415

    meeting_url = body.meeting_url.strip()
    if not meeting_url:
        return {"status": "error", "bot_id": None, "meeting_url": None, "change_count": 0,
                "error": "meeting_url is required"}

    bot_id = create_bot(meeting_url)
    if not bot_id:
        return {"status": "error", "bot_id": None, "meeting_url": meeting_url,
                "change_count": 0, "error": "Failed to create bot — check RECALL_API_KEY and meeting URL"}

    # Update shared meeting state
    meeting_state["bot_id"] = bot_id
    meeting_state["meeting_url"] = meeting_url
    meeting_state["is_active"] = True
    meeting_state["session_status"] = "in_meeting"
    meeting_state["started_at"] = datetime.utcnow().isoformat()

    logger.info("Bot started via /bot/start: bot_id=%s meeting=%s", bot_id, meeting_url)
    return {
        "status": "in_meeting",
        "bot_id": bot_id,
        "meeting_url": meeting_url,
        "change_count": 0,
    }


# ---------------------------------------------------------------------------
# GET /bot/status
# ---------------------------------------------------------------------------

@router.get("/bot/status")
async def get_bot_status() -> Dict[str, Any]:
    """
    Return current bot/session status.

    Response shape: {status, bot_id, meeting_url, change_count}
    status values: "idle" | "in_meeting" | "ended" | "error"
    """
    state = _get_meeting_state()
    status = _session_status(state)

    # change_count: number of pending changes in the queue.
    # When the full change_queue is wired, read from DB here.
    change_count: int = state.get("change_count", 0)

    return {
        "status": status,
        "bot_id": state.get("bot_id"),
        "meeting_url": state.get("meeting_url"),
        "change_count": change_count,
    }


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


# ---------------------------------------------------------------------------
# GET /review/summary
# ---------------------------------------------------------------------------

@router.get("/review/summary")
async def get_review_summary() -> Dict[str, Any]:
    """
    Return a structured meeting summary for the results page.

    Response matches the TypeScript MeetingSummary type:
    {title, date, summary, key_topics, action_items, decisions, participants}
    """
    state = _get_meeting_state()
    local_nodes = _get_local_nodes()

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
        date_str = datetime.utcnow().strftime("%B %d, %Y")

    # Transcript-derived summary
    transcript_log: List[Dict[str, Any]] = state.get("transcript_log") or []
    summary_text = _build_summary_text(transcript_log)

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

    return {
        "title": title,
        "date": date_str,
        "summary": summary_text,
        "key_topics": key_topics,
        "action_items": [],
        "decisions": decisions,
        "participants": participants,
    }
