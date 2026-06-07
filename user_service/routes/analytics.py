"""Bot usage analytics routes.

Provides usage stats at three scopes:
  GET /analytics/usage/me          — own usage (any role)
  GET /analytics/usage/teams/{id}  — per-team (team member, manager, or admin/CEO)
  GET /analytics/usage/org         — org-wide with per-team + per-user breakdown (CEO/ADMIN only)
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, HTTPException

from ..auth import get_current_user
from ..database import select, select_one
from ..models import OrgRole, TeamRole

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/analytics", tags=["analytics"])


# ── Helpers ────────────────────────────────────────────────────────────────────

def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _parse_dt(raw: str | None) -> datetime | None:
    if not raw:
        return None
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except Exception:
        return None


def _session_duration_mins(s: dict) -> float:
    start = _parse_dt(s.get("started_at"))
    end = _parse_dt(s.get("ended_at"))
    if start and end:
        return max(0.0, (end - start).total_seconds() / 60)
    return 0.0


def _compute_stats(sessions: list[dict]) -> dict:
    """Aggregate a list of jarvis_sessions rows into a usage-stats dict."""
    now = _now_utc()
    week_ago = now - timedelta(days=7)
    month_ago = now - timedelta(days=30)

    total_mins = 0.0
    this_week = 0
    this_month = 0
    durations: list[float] = []

    for s in sessions:
        dur = _session_duration_mins(s)
        total_mins += dur
        if dur > 0:
            durations.append(dur)
        start = _parse_dt(s.get("started_at"))
        if start:
            if start >= week_ago:
                this_week += 1
            if start >= month_ago:
                this_month += 1

    avg = round(sum(durations) / len(durations), 1) if durations else 0.0
    return {
        "total_sessions": len(sessions),
        "total_duration_mins": round(total_mins),
        "avg_duration_mins": avg,
        "sessions_this_week": this_week,
        "sessions_this_month": this_month,
    }


def _is_team_member(team_id: str, user_id: str) -> bool:
    return select_one("org_team_members", {
        "team_id": f"eq.{team_id}", "user_id": f"eq.{user_id}",
    }) is not None


# ── Routes ─────────────────────────────────────────────────────────────────────

@router.get("/usage/me")
def my_usage(claims: dict = Depends(get_current_user)):
    """Return bot usage stats for the current user's teams."""
    user_id = claims["sub"]
    memberships = select("org_team_members", {"user_id": f"eq.{user_id}"})
    team_ids = [m["team_id"] for m in memberships]

    if not team_ids:
        return {
            "total_sessions": 0,
            "total_duration_mins": 0,
            "avg_duration_mins": 0.0,
            "sessions_this_week": 0,
            "sessions_this_month": 0,
            "by_team": [],
        }

    all_sessions: list[dict] = []
    by_team: list[dict] = []

    for tid in team_ids:
        team = select_one("org_teams", {"id": f"eq.{tid}"})
        sessions = select("jarvis_sessions", {"team_id": f"eq.{tid}", "status": "eq.ended"})
        stats = _compute_stats(sessions)
        all_sessions.extend(sessions)
        by_team.append({
            "team_id": tid,
            "team_name": team["name"] if team else tid,
            **stats,
        })

    overall = _compute_stats(all_sessions)
    return {**overall, "by_team": by_team}


@router.get("/usage/teams/{team_id}")
def team_usage(team_id: str, claims: dict = Depends(get_current_user)):
    """Return bot usage stats for a specific team, including per-member breakdown."""
    team = select_one("org_teams", {"id": f"eq.{team_id}"})
    if not team:
        raise HTTPException(404, "Team not found")

    if claims["role"] not in (OrgRole.CEO, OrgRole.ADMIN) and not _is_team_member(team_id, claims["sub"]):
        raise HTTPException(403, "Not authorised to view this team's usage")

    sessions = select("jarvis_sessions", {"team_id": f"eq.{team_id}", "status": "eq.ended"})
    stats = _compute_stats(sessions)
    session_ids = {s["session_id"] for s in sessions}

    # Per-member breakdown via user_meeting_activity
    members = select("org_team_members", {"team_id": f"eq.{team_id}"})
    by_member: list[dict] = []

    for m in members:
        uid = m["user_id"]
        user_row = select_one("org_users", {"id": f"eq.{uid}"})
        activity = select("user_meeting_activity", {
            "user_id": f"eq.{uid}", "team_id": f"eq.{team_id}",
        })
        attended = [a for a in activity if a.get("session_id") in session_ids]
        total_mins = sum(float(a.get("duration_mins") or 0) for a in attended)
        by_member.append({
            "user_id": uid,
            "name": user_row["name"] if user_row else uid,
            "email": user_row.get("email", "") if user_row else "",
            "team_role": m["role"],
            "sessions_attended": len(attended),
            "total_duration_mins": round(total_mins),
        })

    return {
        "team_id": team_id,
        "team_name": team["name"],
        **stats,
        "by_member": sorted(by_member, key=lambda x: x["sessions_attended"], reverse=True),
    }


@router.get("/usage/org")
def org_usage(claims: dict = Depends(get_current_user)):
    """Org-wide usage: total + per-team + top users. CEO/ADMIN only."""
    if claims["role"] not in (OrgRole.CEO, OrgRole.ADMIN):
        raise HTTPException(403, "Only CEO or ADMIN can view org-wide usage")

    org_id = claims["org_id"]
    teams = select("org_teams", {"org_id": f"eq.{org_id}"})
    all_sessions: list[dict] = []
    by_team: list[dict] = []

    for team in teams:
        tid = team["id"]
        sessions = select("jarvis_sessions", {"team_id": f"eq.{tid}", "status": "eq.ended"})
        stats = _compute_stats(sessions)
        all_sessions.extend(sessions)
        by_team.append({
            "team_id": tid,
            "team_name": team["name"],
            **stats,
        })

    org_stats = _compute_stats(all_sessions)

    # Top users by meeting attendance across the org
    all_users = select("org_users", {"org_id": f"eq.{org_id}"})
    top_users: list[dict] = []

    for user in all_users:
        uid = user["id"]
        activity = select("user_meeting_activity", {"user_id": f"eq.{uid}"})
        total_mins = sum(float(a.get("duration_mins") or 0) for a in activity)
        if activity:
            top_users.append({
                "user_id": uid,
                "name": user["name"],
                "email": user["email"],
                "org_role": user["role"],
                "sessions_attended": len(activity),
                "total_duration_mins": round(total_mins),
            })

    top_users.sort(key=lambda x: x["sessions_attended"], reverse=True)

    return {
        "org_total": org_stats,
        "by_team": sorted(by_team, key=lambda x: x["total_sessions"], reverse=True),
        "top_users": top_users[:20],
    }
