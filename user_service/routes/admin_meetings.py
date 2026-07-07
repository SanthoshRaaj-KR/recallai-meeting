"""Admin/CEO meeting-oversight routes.

Org-wide visibility and control over Jarvis bot meetings:

  GET  /admin/meetings/live            — meetings running right now (org-scoped)
  GET  /admin/meetings                 — meeting history (org-scoped, filterable)
  GET  /admin/meetings/{session_id}    — full detail incl. summary / MOM + transcript
  POST /admin/meetings/{session_id}/kick — remove the bot from a running meeting

All routes require ADMIN or CEO. A meeting is "owned" by the caller's org iff its
`team_id` belongs to a team in the caller's org — every query is scoped that way so
one org can never see or kick another org's meetings.

org-service is the *authorized front door*: the kick endpoint verifies role +
org-ownership, then proxies to bot-service (the executor) which tells Recall to
leave the call. The browser never calls bot-service's stop endpoint directly.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone

import requests
from fastapi import APIRouter, Depends, HTTPException

from ..database import select, select_one
from ..models import OrgRole
from ..rbac import require_admin_or_above

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/admin/meetings", tags=["admin-meetings"])

# bot-service (port 8000, VM-A) base URL — needed only for the kick proxy.
_BOT_SERVICE_URL = os.getenv("BOT_SERVICE_URL", "").rstrip("/")
# Optional shared secret; sent as a header so bot-service can require it later
# without breaking this caller. bot-service does not enforce it today.
_INTERNAL_SECRET = os.getenv("INTERNAL_API_SECRET", "")

_LIVE_STATUSES = ("joining", "in_meeting")


# ── Helpers ──────────────────────────────────────────────────────────────────

def _parse_dt(raw: str | None) -> datetime | None:
    if not raw:
        return None
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except Exception:
        return None


def _duration_mins(s: dict) -> float:
    start = _parse_dt(s.get("started_at"))
    end = _parse_dt(s.get("ended_at"))
    if start and end:
        return round(max(0.0, (end - start).total_seconds() / 60), 1)
    return 0.0


def _elapsed_seconds(s: dict) -> int:
    start = _parse_dt(s.get("started_at"))
    if not start:
        return 0
    return int(max(0.0, (datetime.now(timezone.utc) - start).total_seconds()))


def _org_team_map(org_id: str | None) -> dict[str, str]:
    """Return {team_id: team_name} for every team in the org."""
    if not org_id:
        return {}
    teams = select("org_teams", {"org_id": f"eq.{org_id}"})
    return {t["id"]: t["name"] for t in teams}


# Skinny projection for list/board views — never loads the transcript / summary /
# changes / diagnostics blobs. `title` comes from summary->>title server-side, and
# `change_count` is the generated column (3b), so counting is free.
_LIST_COLS = (
    "session_id,team_id,status,meeting_url,bot_id,started_at,ended_at,"
    "change_count,title:summary->>title"
)


def _title(s: dict) -> str:
    # Projected rows carry a top-level `title`; full rows carry summary.title.
    title = s.get("title")
    if title:
        return title
    summary = s.get("summary") or {}
    return summary.get("title") or f"Meeting {str(s.get('session_id', ''))[:8]}"


def _base_fields(s: dict, team_map: dict[str, str]) -> dict:
    tid = s.get("team_id")
    # Prefer the generated change_count column; fall back to counting a loaded blob.
    change_count = s.get("change_count")
    if change_count is None:
        change_count = len(s.get("changes") or [])
    return {
        "session_id": s.get("session_id"),
        "team_id": tid,
        "team_name": team_map.get(tid) if tid else None,
        "title": _title(s),
        "meeting_url": s.get("meeting_url"),
        "status": s.get("status"),
        "bot_id": s.get("bot_id"),
        "started_at": s.get("started_at"),
        "ended_at": s.get("ended_at"),
        "change_count": change_count,
    }


def _sessions_for_org(team_ids: list[str], extra: dict[str, str]) -> list[dict]:
    """Query jarvis_sessions scoped to the org's teams plus any extra filters."""
    if not team_ids:
        return []
    csv = ",".join(team_ids)
    filters = {"team_id": f"in.({csv})", **extra}
    return select("jarvis_sessions", filters, columns=_LIST_COLS)


# ── Routes ───────────────────────────────────────────────────────────────────

@router.get("/live")
def live_meetings(claims: dict = Depends(require_admin_or_above())):
    """All meetings currently running in the caller's org (newest first)."""
    team_map = _org_team_map(claims.get("org_id"))
    sessions = _sessions_for_org(
        list(team_map.keys()),
        {"status": f"in.({','.join(_LIVE_STATUSES)})", "order": "started_at.desc"},
    )
    out = []
    for s in sessions:
        item = _base_fields(s, team_map)
        item["elapsed_seconds"] = _elapsed_seconds(s)
        out.append(item)
    return out


@router.get("")
def list_meetings(
    team_id: str | None = None,
    status: str | None = None,
    limit: int = 100,
    claims: dict = Depends(require_admin_or_above()),
):
    """Org-wide meeting history (newest first). Optional team_id / status filters."""
    team_map = _org_team_map(claims.get("org_id"))
    team_ids = list(team_map.keys())

    if team_id:
        if team_id not in team_map:
            raise HTTPException(403, "That team is not in your organisation")
        team_ids = [team_id]

    extra: dict[str, str] = {"order": "started_at.desc", "limit": str(max(1, min(limit, 500)))}
    if status:
        extra["status"] = f"eq.{status}"

    sessions = _sessions_for_org(team_ids, extra)
    out = []
    for s in sessions:
        item = _base_fields(s, team_map)
        item["duration_mins"] = _duration_mins(s)
        out.append(item)
    return out


def _require_owned_session(session_id: str, claims: dict) -> tuple[dict, dict[str, str]]:
    """Load a session and assert it belongs to the caller's org. Returns (session, team_map)."""
    s = select_one("jarvis_sessions", {"session_id": f"eq.{session_id}"})
    if not s:
        raise HTTPException(404, "Meeting not found")
    team_map = _org_team_map(claims.get("org_id"))
    if s.get("team_id") not in team_map:
        raise HTTPException(403, "This meeting is not in your organisation")
    return s, team_map


@router.get("/{session_id}")
def meeting_detail(session_id: str, claims: dict = Depends(require_admin_or_above())):
    """Full meeting detail: summary / MOM, transcript, and proposed changes."""
    s, team_map = _require_owned_session(session_id, claims)
    item = _base_fields(s, team_map)
    item["duration_mins"] = _duration_mins(s)
    item["summary"] = s.get("summary") or {}
    item["transcript"] = s.get("transcript") or []
    item["changes"] = s.get("changes") or []
    return item


@router.post("/{session_id}/kick")
def kick_meeting(session_id: str, claims: dict = Depends(require_admin_or_above())):
    """Remove the Jarvis bot from a running meeting (org-ownership enforced)."""
    s, _ = _require_owned_session(session_id, claims)

    if s.get("status") in ("ended", "error"):
        raise HTTPException(409, "This meeting has already ended")

    if not _BOT_SERVICE_URL:
        raise HTTPException(
            503, "BOT_SERVICE_URL is not configured on org-service; cannot reach the bot."
        )

    headers = {}
    if _INTERNAL_SECRET:
        headers["X-Internal-Secret"] = _INTERNAL_SECRET

    try:
        resp = requests.post(
            f"{_BOT_SERVICE_URL}/sessions/{session_id}/bot/stop",
            headers=headers,
            timeout=15,
        )
    except Exception as exc:
        logger.warning("kick: bot-service unreachable for %s: %s", session_id, exc)
        raise HTTPException(502, "Could not reach the bot service to stop the meeting")

    if not resp.ok:
        logger.warning("kick: bot-service returned %s for %s", resp.status_code, session_id)
        raise HTTPException(502, f"Bot service rejected the stop ({resp.status_code})")

    logger.info(
        "kick: admin %s stopped meeting %s", claims.get("sub"), session_id
    )
    try:
        return resp.json()
    except Exception:
        return {"status": "ended", "session_id": session_id}
