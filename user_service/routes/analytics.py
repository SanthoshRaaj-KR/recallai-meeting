"""Bot usage analytics routes.

Provides usage stats at three scopes:
  GET /analytics/usage/me          — own usage (any role)
  GET /analytics/usage/teams/{id}  — per-team (team member, manager, or admin/CEO)
  GET /analytics/usage/org         — org-wide with per-team + per-user breakdown (CEO/ADMIN only)

Aggregation runs in Postgres via RPCs (migrations/007_analytics_rpcs.sql) so each
endpoint makes 1-3 round-trips instead of looping per-team/per-user in Python.
Results are cached for 60 seconds per key.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, HTTPException

from ..auth import get_current_user
from ..database import rpc, select, select_one
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


def _stats_from_row(r: dict) -> dict:
    """Shape a team_usage_stats row (or a summed rollup) into the response stats.

    ``avg_duration_mins`` = total positive minutes / number of positive-duration
    sessions — matches the old Python _compute_stats semantics.
    """
    total_min = float(r.get("total_minutes") or 0)
    dur_count = int(r.get("dur_count") or 0)
    return {
        "total_sessions": int(r.get("total_sessions") or 0),
        "total_duration_mins": round(total_min),
        "avg_duration_mins": round(total_min / dur_count, 1) if dur_count else 0.0,
        "sessions_this_week": int(r.get("sessions_week") or 0),
        "sessions_this_month": int(r.get("sessions_month") or 0),
    }


def _shape_team_row(r: dict) -> dict:
    return {"team_id": r.get("team_id"), "team_name": r.get("team_name"), **_stats_from_row(r)}


def _rollup(rows: list[dict]) -> dict:
    """Aggregate per-team rows into org/overall totals (sums are additive; the
    average is recomputed from summed minutes and positive-duration counts)."""
    agg = {
        "total_sessions": sum(int(r.get("total_sessions") or 0) for r in rows),
        "total_minutes": sum(float(r.get("total_minutes") or 0) for r in rows),
        "dur_count": sum(int(r.get("dur_count") or 0) for r in rows),
        "sessions_week": sum(int(r.get("sessions_week") or 0) for r in rows),
        "sessions_month": sum(int(r.get("sessions_month") or 0) for r in rows),
    }
    return _stats_from_row(agg)


def _team_usage_rows(team_ids: list[str]) -> list[dict]:
    if not team_ids:
        return []
    now = _now_utc()
    return rpc("team_usage_stats", {
        "p_team_ids": team_ids,
        "p_week": _iso(now - timedelta(days=7)),
        "p_month": _iso(now - timedelta(days=30)),
    })


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

    memberships = select("org_team_members", {"user_id": f"eq.{user_id}"}, columns="team_id")
    team_ids = [m["team_id"] for m in memberships]

    rows = _team_usage_rows(team_ids)
    result = {**_rollup(rows), "by_team": [_shape_team_row(r) for r in rows]}
    _cache_set(cache_key, result)
    return result


@router.get("/usage/teams/{team_id}")
def team_usage(team_id: str, claims: dict = Depends(get_current_user)):
    """Return bot usage stats for a specific team, including per-member breakdown."""
    team = select_one("org_teams", {"id": f"eq.{team_id}"}, columns="id,name")
    if not team:
        raise HTTPException(404, "Team not found")

    if claims["role"] not in (OrgRole.CEO, OrgRole.ADMIN) and not _is_team_member(team_id, claims["sub"]):
        raise HTTPException(403, "Not authorised to view this team's usage")

    cache_key = f"team:{team_id}"
    if cached := _cache_get(cache_key):
        return cached

    rows = _team_usage_rows([team_id])
    stats = _stats_from_row(rows[0]) if rows else _stats_from_row({})

    members = rpc("team_member_usage", {"p_team": team_id})
    by_member = [
        {
            "user_id": m["user_id"],
            "name": m.get("name") or m["user_id"],
            "email": m.get("email") or "",
            "team_role": m.get("team_role"),
            "sessions_attended": int(m.get("sessions_attended") or 0),
            "total_duration_mins": round(float(m.get("total_minutes") or 0)),
        }
        for m in members
    ]
    by_member.sort(key=lambda x: x["sessions_attended"], reverse=True)

    result = {
        "team_id": team_id,
        "team_name": team["name"],
        **stats,
        "by_member": by_member,
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

    team_ids = [t["id"] for t in select("org_teams", {"org_id": f"eq.{org_id}"}, columns="id")]
    rows = _team_usage_rows(team_ids)
    by_team = [_shape_team_row(r) for r in rows]

    top = rpc("org_top_users", {"p_org": org_id, "p_limit": 20})
    top_users = [
        {
            "user_id": u["user_id"],
            "name": u.get("name"),
            "email": u.get("email"),
            "org_role": u.get("org_role"),
            "sessions_attended": int(u.get("sessions_attended") or 0),
            "total_duration_mins": round(float(u.get("total_minutes") or 0)),
        }
        for u in top
    ]

    result = {
        "org_total": _rollup(rows),
        "by_team": sorted(by_team, key=lambda x: x["total_sessions"], reverse=True),
        "top_users": top_users,
    }
    _cache_set(cache_key, result)
    return result
