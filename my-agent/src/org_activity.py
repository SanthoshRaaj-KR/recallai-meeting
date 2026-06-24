"""
Org meeting-activity recorder.

When a real bot meeting ends, this writes one `user_meeting_activity` row per
member of the session's team, so the org dashboard's per-member / per-team usage
reflects real meetings instead of seeded dummy data.

Why team members (not individual participants): a bot session carries a
`team_id` (set at /bot/start) but no per-attendee org identity — Recall gives
display names, not org emails. Crediting the team is the reliable signal we have
today; finer attendance would require mapping Recall participants → org_users.

Safe by construction: no-op if Supabase or team_id is absent, idempotent per
session (UNIQUE(user_id, session_id) + a pre-check), and never raises into the
caller (meeting-end paths must not break on analytics).
"""

import datetime
import logging
import os

import requests
from dotenv import load_dotenv
from pathlib import Path

load_dotenv(Path(__file__).parent.parent / ".env.local")

logger = logging.getLogger(__name__)

_SUPABASE_URL = os.getenv("SUPABASE_URL", "").rstrip("/")
_SUPABASE_KEY = (
    os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    or os.getenv("SUPABASE_ANON_KEY")
    or ""
)


def _configured() -> bool:
    return bool(_SUPABASE_URL and _SUPABASE_KEY)


def _headers(return_repr: bool = False, prefer: str | None = None) -> dict[str, str]:
    h = {
        "apikey": _SUPABASE_KEY,
        "Authorization": f"Bearer {_SUPABASE_KEY}",
        "Content-Type": "application/json",
    }
    if prefer:
        h["Prefer"] = prefer
    elif return_repr:
        h["Prefer"] = "return=representation"
    return h


def _url(table: str) -> str:
    return f"{_SUPABASE_URL}/rest/v1/{table}"


def _parse_dt(raw: str | None) -> datetime.datetime | None:
    if not raw:
        return None
    try:
        return datetime.datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except Exception:
        return None


def _duration_mins(session: dict) -> float:
    start = _parse_dt(session.get("started_at"))
    end = _parse_dt(session.get("ended_at"))
    if start and end:
        return round(max(0.0, (end - start).total_seconds() / 60), 1)
    return 0.0


def record_meeting_activity(session: dict) -> None:
    """Record per-member attendance for an ended meeting. Never raises."""
    try:
        if not _configured():
            return
        team_id = session.get("team_id")
        session_id = session.get("session_id")
        if not team_id or not session_id:
            return  # not team-scoped → nothing to attribute
        if str(session_id).startswith("seed-"):
            return  # defensive: never touch seeded sessions

        # Idempotency: skip if we already recorded attendance for this session.
        existing = requests.get(
            _url("user_meeting_activity"),
            headers=_headers(),
            params={"session_id": f"eq.{session_id}", "select": "id", "limit": "1"},
            timeout=5,
        )
        if existing.ok and existing.json():
            return

        members = requests.get(
            _url("org_team_members"),
            headers=_headers(),
            params={"team_id": f"eq.{team_id}", "select": "user_id"},
            timeout=5,
        )
        if not members.ok:
            logger.warning("org_activity: member lookup failed: %s", members.text[:120])
            return
        member_ids = [m["user_id"] for m in members.json() if m.get("user_id")]
        if not member_ids:
            return

        duration = _duration_mins(session)
        joined_at = session.get("started_at") or datetime.datetime.now(
            datetime.timezone.utc
        ).isoformat()
        rows = [
            {
                "user_id": uid,
                "session_id": session_id,
                "team_id": team_id,
                "joined_at": joined_at,
                "duration_mins": duration,
            }
            for uid in member_ids
        ]
        resp = requests.post(
            _url("user_meeting_activity"),
            headers=_headers(prefer="resolution=ignore-duplicates"),
            json=rows,
            timeout=8,
        )
        if resp.ok or resp.status_code == 409:
            logger.info(
                "org_activity: recorded %d attendees for session %s (team %s, %.1f min)",
                len(rows), session_id, team_id, duration,
            )
            # Bust the analytics cache so the next request sees fresh stats.
            try:
                from user_service.routes.analytics import invalidate_team  # type: ignore[import]
                invalidate_team(team_id)
            except Exception:
                pass  # cache invalidation is best-effort
        else:
            logger.warning("org_activity: insert failed (%s): %s", resp.status_code, resp.text[:120])
    except Exception as exc:  # analytics must never break meeting teardown
        logger.warning("org_activity.record_meeting_activity skipped: %s", exc)
