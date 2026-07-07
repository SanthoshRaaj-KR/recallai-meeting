"""Bot usage analytics routes.

Provides usage stats at three scopes:
  GET /analytics/usage/me          — own usage (any role)
  GET /analytics/usage/teams/{id}  — per-team (team member, manager, or admin/CEO)
  GET /analytics/usage/org         — org-wide with per-team + per-user breakdown (CEO/ADMIN only)

Stats are cached for 60 seconds per key to avoid hammering Supabase on every
page load. Date-filtered queries fetch only the last 30 days of sessions for
weekly/monthly counts rather than scanning all history.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, HTTPException

from ..auth import get_current_user
from ..database import select, select_one
from ..models import OrgRole

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/analytics", tags=["analytics"])


# ── TTL cache ──────────────────────────────────────────────────────────────────

_cache: dict[str, tuple[float, dict]] = {}
_CACHE_TTL = 60.0  # seconds


def _cache_get(key: str) -> dict | None:
    entry = _cache.get(key)
    if entry and time.monotonic() - entry[0] < _CACHE_TTL:
        return entry[1]
    _cache.pop(key, None)
    return None


def _cache_set(key: str, data: dict) -> None:
    _cache[key] = (time.monotonic(), data)


def invalidate_team(team_id: str) -> None:
    """Call this when a meeting ends to bust the per-team cache entry."""
    _cache.pop(f"team:{team_id}", None)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _iso(dt: datetime) -> str:
    return dt.replace(microsecond=0).isoformat()


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


def _fetch_team_sessions(team_id: str, since: datetime | None = None) -> list[dict]:
    """Fetch ended sessions for a team, with optional date lower-bound.

    Projects only the three fields the stats need — never loads the
    transcript/summary/changes blobs (which duration math never touches).
    """
    filters: dict[str, str] = {"team_id": f"eq.{team_id}", "status": "eq.ended"}
    if since:
        filters["started_at"] = f"gte.{_iso(since)}"
    return select("jarvis_sessions", filters, columns="session_id,started_at,ended_at")


def _compute_stats(all_sessions: list[dict], recent: list[dict]) -> dict:
    """Aggregate stats from two separate session lists.

    all_sessions  — full history for totals (count + minutes)
    recent        — last 30 days for weekly/monthly counts (avoids full-scan)
    """
    now = _now_utc()
    week_ago = now - timedelta(days=7)
    month_ago = now - timedelta(days=30)

    total_mins = 0.0
    durations: list[float] = []
    for s in all_sessions:
        dur = _session_duration_mins(s)
        total_mins += dur
        if dur > 0:
            durations.append(dur)

    avg = round(sum(durations) / len(durations), 1) if durations else 0.0

    this_week = sum(
        1 for s in recent
        if (st := _parse_dt(s.get("started_at"))) and st >= week_ago
    )
    this_month = sum(
        1 for s in recent
        if (st := _parse_dt(s.get("started_at"))) and st >= month_ago
    )

    return {
        "total_sessions": len(all_sessions),
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
    cache_key = f"me:{user_id}"
    if cached := _cache_get(cache_key):
        return cached

    memberships = select("org_team_members", {"user_id": f"eq.{user_id}"})
    team_ids = [m["team_id"] for m in memberships]

    if not team_ids:
        result: dict = {
            "total_sessions": 0,
            "total_duration_mins": 0,
            "avg_duration_mins": 0.0,
            "sessions_this_week": 0,
            "sessions_this_month": 0,
            "by_team": [],
        }
        _cache_set(cache_key, result)
        return result

    month_ago = _now_utc() - timedelta(days=30)
    all_global: list[dict] = []
    all_recent: list[dict] = []
    by_team: list[dict] = []

    for tid in team_ids:
        team = select_one("org_teams", {"id": f"eq.{tid}"})
        all_s = _fetch_team_sessions(tid)
        recent_s = _fetch_team_sessions(tid, since=month_ago)
        stats = _compute_stats(all_s, recent_s)
        all_global.extend(all_s)
        all_recent.extend(recent_s)
        by_team.append({
            "team_id": tid,
            "team_name": team["name"] if team else tid,
            **stats,
        })

    overall = _compute_stats(all_global, all_recent)
    result = {**overall, "by_team": by_team}
    _cache_set(cache_key, result)
    return result


@router.get("/usage/teams/{team_id}")
def team_usage(team_id: str, claims: dict = Depends(get_current_user)):
    """Return bot usage stats for a specific team, including per-member breakdown."""
    team = select_one("org_teams", {"id": f"eq.{team_id}"})
    if not team:
        raise HTTPException(404, "Team not found")

    if claims["role"] not in (OrgRole.CEO, OrgRole.ADMIN) and not _is_team_member(team_id, claims["sub"]):
        raise HTTPException(403, "Not authorised to view this team's usage")

    cache_key = f"team:{team_id}"
    if cached := _cache_get(cache_key):
        return cached

    month_ago = _now_utc() - timedelta(days=30)
    all_s = _fetch_team_sessions(team_id)
    recent_s = _fetch_team_sessions(team_id, since=month_ago)
    stats = _compute_stats(all_s, recent_s)
    session_ids = {s["session_id"] for s in all_s}

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

    result = {
        "team_id": team_id,
        "team_name": team["name"],
        **stats,
        "by_member": sorted(by_member, key=lambda x: x["sessions_attended"], reverse=True),
    }
    _cache_set(cache_key, result)
    return result


@router.get("/usage/org")
def org_usage(claims: dict = Depends(get_current_user)):
    """Org-wide usage: total + per-team + top users. CEO/ADMIN only."""
    if claims["role"] not in (OrgRole.CEO, OrgRole.ADMIN):
        raise HTTPException(403, "Only CEO or ADMIN can view org-wide usage")

    org_id = claims["org_id"]
    cache_key = f"org:{org_id}"
    if cached := _cache_get(cache_key):
        return cached

    month_ago = _now_utc() - timedelta(days=30)
    teams = select("org_teams", {"org_id": f"eq.{org_id}"})
    all_global: list[dict] = []
    all_recent: list[dict] = []
    by_team: list[dict] = []

    for team in teams:
        tid = team["id"]
        all_s = _fetch_team_sessions(tid)
        recent_s = _fetch_team_sessions(tid, since=month_ago)
        stats = _compute_stats(all_s, recent_s)
        all_global.extend(all_s)
        all_recent.extend(recent_s)
        by_team.append({
            "team_id": tid,
            "team_name": team["name"],
            **stats,
        })

    org_stats = _compute_stats(all_global, all_recent)

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

    result = {
        "org_total": org_stats,
        "by_team": sorted(by_team, key=lambda x: x["total_sessions"], reverse=True),
        "top_users": top_users[:20],
    }
    _cache_set(cache_key, result)
    return result
