"""
Review API router — FastAPI routes for meeting review and Confluence change management.

Mounted into the main jarvis_agentic app via app.include_router(router).
All routes are served at the root prefix (e.g. GET /review/summary).
"""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

from fastapi import APIRouter

logger = logging.getLogger(__name__)

router = APIRouter()


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
    """
    Build a plain-text summary from the transcript log.

    Strategy: concatenate the last 20 transcript entries as 'Speaker: text' lines.
    This gives the results page a readable snapshot without needing an LLM call
    at review time (the meeting is already over and latency matters less here,
    but we keep it synchronous to avoid async complexity in the router).
    """
    if not transcript_log:
        return ""
    recent = transcript_log[-20:]
    lines = [f"{e.get('participant', 'Unknown')}: {e.get('text', '')}" for e in recent]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# GET /review/summary
# ---------------------------------------------------------------------------

@router.get("/review/summary")
async def get_review_summary() -> Dict[str, Any]:
    """
    Return a structured meeting summary for the results page.

    Response shape:
    {
      "session_id": str | null,
      "meeting_url": str | null,
      "started_at": str | null,
      "ended_at": str | null,
      "summary": str,
      "topics": [str, ...],
      "action_items": [{"item": str, "owner": str | null}, ...],
      "decisions": [str, ...],
      "participants": [str, ...]
    }

    Data sources:
    - session_id / meeting_url: meeting_state (runtime dict in jarvis_agentic)
    - topics / decisions / participants: graph_rag in-memory graph
    - summary: last 20 transcript entries concatenated (simple, synchronous)
    - action_items: empty list (extracted verbally during meeting via MeetingResponder;
      a future plan can wire structured extraction here)

    Returns sensible empty defaults — never raises 404.
    """
    state = _get_meeting_state()
    local_nodes = _get_local_nodes()

    # Session metadata
    session_id: Optional[str] = state.get("bot_id") or None
    meeting_url: Optional[str] = (
        state.get("meeting_url")
        or os.getenv("MEETING_URL")
        or None
    )
    started_at: Optional[str] = state.get("started_at") or None
    ended_at: Optional[str] = state.get("ended_at") or None

    # Transcript-derived data
    transcript_log: List[Dict[str, Any]] = state.get("transcript_log") or []
    summary_text = _build_summary_text(transcript_log)

    # Graph RAG-derived data
    topics = _extract_topics(local_nodes)
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
        "session_id": session_id,
        "meeting_url": meeting_url,
        "started_at": started_at,
        "ended_at": ended_at,
        "summary": summary_text,
        "topics": topics,
        "action_items": [],  # Structured action item extraction is a future plan
        "decisions": decisions,
        "participants": participants,
    }
