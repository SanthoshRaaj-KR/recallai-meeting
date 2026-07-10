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
import re

import requests
from dotenv import load_dotenv
from pathlib import Path

try:
    from . import name_match
except ImportError:
    import name_match

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


# ── Identity resolution: Recall display name → org_user (Phase 2) ───────────────

def _normalize_name(name: str | None) -> str:
    """Lowercase, strip, collapse whitespace, drop punctuation — so 'Alice B.' and
    'alice  b' compare equal. Device/label names ('iPhone') simply won't match."""
    if not name:
        return ""
    n = re.sub(r"[^a-z0-9 ]+", "", name.strip().lower())
    return re.sub(r"\s+", " ", n).strip()


def _resolve_participant(rec_name: str, candidates: list[dict]) -> tuple[str | None, str]:
    """Match a Recall display name to one org user via token-aware matching.

    `candidates` is [{user_id, name, aliases}]. Returns (user_id, confidence) with
    confidence 'high' / 'medium' / 'none' (see name_match.resolve). Ambiguous or
    unmatched names return (None, 'none') so the caller records a guest row.
    """
    return name_match.resolve(rec_name, candidates)


def _team_roster(team_id: str) -> tuple[list[dict], str | None]:
    """Return ([{user_id, name, aliases}] for the team's members, org_id)."""
    members = requests.get(
        _url("org_team_members"),
        headers=_headers(),
        params={"team_id": f"eq.{team_id}", "select": "user_id"},
        timeout=5,
    )
    if not members.ok:
        return [], None
    uids = [m["user_id"] for m in members.json() if m.get("user_id")]
    if not uids:
        return [], None
    users = requests.get(
        _url("org_users"),
        headers=_headers(),
        params={"id": f"in.({','.join(uids)})", "select": "id,name,org_id,display_name_aliases"},
        timeout=5,
    )
    if not users.ok:
        return [], None
    candidates: list[dict] = []
    org_id: str | None = None
    for u in users.json():
        org_id = org_id or u.get("org_id")
        candidates.append({
            "user_id": u["id"],
            "name": u.get("name") or "",
            "aliases": u.get("display_name_aliases") or [],
        })
    return candidates, org_id


def _participant_duration_mins(entry: dict, session: dict) -> float | None:
    """Per-person minutes from join/leave. Missing leave → stayed until meeting end;
    missing join → present from meeting start."""
    joined = _parse_dt(entry.get("joined_at")) or _parse_dt(session.get("started_at"))
    left = _parse_dt(entry.get("left_at")) or _parse_dt(session.get("ended_at"))
    if joined and left:
        return round(max(0.0, (left - joined).total_seconds() / 60), 1)
    return None


def record_participants(session: dict) -> None:
    """Write one meeting_participants row per REAL attendee (from the Phase-1
    presence map), resolving each Recall name to an org_user. Never raises."""
    try:
        if not _configured():
            return
        session_id = session.get("session_id")
        team_id = session.get("team_id")
        presence = session.get("participants") or {}
        if not session_id or not team_id or not presence:
            return
        if str(session_id).startswith("seed-"):
            return

        roster, org_id = _team_roster(team_id)

        rows = []
        for pid, entry in presence.items():
            rec_name = entry.get("name")
            uid, confidence = _resolve_participant(rec_name, roster)
            rows.append({
                "session_id": session_id,
                "team_id": team_id,
                "org_id": org_id,
                "recall_participant_id": str(pid),
                "recall_name": rec_name,
                "user_id": uid,
                "match_confidence": confidence,
                "is_guest": uid is None,
                "joined_at": entry.get("joined_at"),
                "left_at": entry.get("left_at"),
                "duration_mins": _participant_duration_mins(entry, session),
                "source": "presence",
            })
        if not rows:
            return

        resp = requests.post(
            _url("meeting_participants"),
            headers=_headers(prefer="resolution=merge-duplicates"),
            params={"on_conflict": "session_id,recall_participant_id"},
            json=rows,
            timeout=8,
        )
        if resp.ok or resp.status_code == 409:
            matched = sum(1 for r in rows if r["user_id"])
            logger.info(
                "org_activity: recorded %d participants (%d matched) for session %s",
                len(rows), matched, session_id,
            )
            try:
                from user_service.routes.analytics import invalidate_team  # type: ignore[import]
                invalidate_team(team_id)
            except Exception:
                pass
        else:
            logger.warning(
                "org_activity: participant insert failed (%s): %s",
                resp.status_code, resp.text[:120],
            )
    except Exception as exc:  # analytics must never break meeting teardown
        logger.warning("org_activity.record_participants skipped: %s", exc)


def _session_has_participants(session_id: str) -> bool:
    resp = requests.get(
        _url("meeting_participants"),
        headers=_headers(),
        params={"session_id": f"eq.{session_id}", "select": "id", "limit": "1"},
        timeout=5,
    )
    return bool(resp.ok and resp.json())


def backfill_participants_from_transcript(session: dict, speaker_names: list[str]) -> int:
    """Best-effort historical attendance: derive attendees from a past meeting's
    diarized speaker names (no join/leave available). Rows are flagged
    source='backfill' and credited the whole-meeting duration. Skips any session
    that already has real presence rows. Returns rows written. Never raises."""
    try:
        if not _configured():
            return 0
        session_id = session.get("session_id")
        team_id = session.get("team_id")
        if not session_id or not team_id or not speaker_names:
            return 0
        if str(session_id).startswith("seed-"):
            return 0
        if _session_has_participants(session_id):
            return 0  # real attendance already recorded — don't overwrite with a guess

        roster, org_id = _team_roster(team_id)
        duration = _duration_mins(session)
        started = session.get("started_at")
        ended = session.get("ended_at")

        # Distinct speaker names (case/space-insensitive), ignore the bot itself.
        seen: dict[str, str] = {}
        for raw in speaker_names:
            key = _normalize_name(raw)
            if not key or key in ("jarvis", "meeting", "meeting assistant"):
                continue
            seen.setdefault(key, raw)

        rows = []
        for key, raw in seen.items():
            uid, confidence = _resolve_participant(raw, roster)
            rows.append({
                "session_id": session_id,
                "team_id": team_id,
                "org_id": org_id,
                "recall_participant_id": f"backfill:{key}",
                "recall_name": raw,
                "user_id": uid,
                "match_confidence": confidence,
                "is_guest": uid is None,
                "joined_at": started,
                "left_at": ended,
                "duration_mins": duration or None,
                "source": "backfill",
            })
        if not rows:
            return 0
        resp = requests.post(
            _url("meeting_participants"),
            headers=_headers(prefer="resolution=merge-duplicates"),
            params={"on_conflict": "session_id,recall_participant_id"},
            json=rows,
            timeout=8,
        )
        if resp.ok or resp.status_code == 409:
            logger.info("backfill: %d participants for session %s", len(rows), session_id)
            return len(rows)
        logger.warning("backfill insert failed (%s): %s", resp.status_code, resp.text[:120])
        return 0
    except Exception as exc:
        logger.warning("org_activity.backfill_participants skipped: %s", exc)
        return 0


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
