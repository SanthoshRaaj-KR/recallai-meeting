"""Team bot assignment + team-scoped meeting session routes."""

import datetime

from fastapi import APIRouter, Depends, HTTPException, status

from ..auth import get_current_user
from ..database import select, select_one, insert, DBError
from ..models import BotCreate, BotOut, OrgRole, TeamRole
from ..rbac import require_admin_or_above, can_oversee_team

router = APIRouter(tags=["bots"])


def _is_team_member(team_id: str, user_id: str) -> bool:
    return select_one("org_team_members", {
        "team_id": f"eq.{team_id}", "user_id": f"eq.{user_id}",
    }) is not None


def _parse_dt(raw: str | None) -> datetime.datetime | None:
    if not raw:
        return None
    try:
        return datetime.datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except Exception:
        return None


def _duration_mins(s: dict) -> float:
    start = _parse_dt(s.get("started_at"))
    end = _parse_dt(s.get("ended_at"))
    if start and end:
        return round(max(0.0, (end - start).total_seconds() / 60), 1)
    return 0.0


# ── Bot assignment ─────────────────────────────────────────────────────────────

@router.get("/teams/{team_id}/bot", response_model=BotOut)
def get_team_bot(team_id: str, claims: dict = Depends(get_current_user)):
    if claims["role"] not in (OrgRole.CEO, OrgRole.ADMIN) and not _is_team_member(team_id, claims["sub"]):
        raise HTTPException(403, "Not authorised to view this team's bot")
    bot = select_one("org_team_bots", {"team_id": f"eq.{team_id}"})
    if not bot:
        raise HTTPException(404, "No bot assigned to this team")
    return BotOut(
        id=bot["id"], team_id=bot["team_id"], name=bot["name"],
        config=bot.get("config") or {}, created_at=bot["created_at"],
    )


@router.post("/teams/{team_id}/bot", response_model=BotOut, status_code=status.HTTP_201_CREATED)
def assign_team_bot(team_id: str, body: BotCreate, claims: dict = Depends(require_admin_or_above())):
    """CEO or ADMIN: assign (or re-assign) a bot to a team."""
    team = select_one("org_teams", {"id": f"eq.{team_id}"})
    if not team:
        raise HTTPException(404, "Team not found")
    existing = select_one("org_team_bots", {"team_id": f"eq.{team_id}"})
    if existing:
        raise HTTPException(409, "Team already has a bot assigned. Remove it first.")
    try:
        row = insert("org_team_bots", {"team_id": team_id, "name": body.name, "config": body.config})
    except DBError as e:
        raise HTTPException(500, str(e))
    return BotOut(
        id=row["id"], team_id=row["team_id"], name=row["name"],
        config=row.get("config") or {}, created_at=row["created_at"],
    )


# ── Team-scoped meetings ───────────────────────────────────────────────────────

@router.get("/teams/{team_id}/meetings")
def list_team_meetings(team_id: str, claims: dict = Depends(get_current_user)):
    """
    Return meeting sessions scoped to this team.

    CEO/ADMIN can see any team in their org. Team members can only see their own
    team's meetings. Cross-org access is denied for everyone.
    """
    team = select_one("org_teams", {"id": f"eq.{team_id}"}, columns="id,org_id")
    if not team:
        raise HTTPException(404, "Team not found")
    if team.get("org_id") != claims.get("org_id"):
        raise HTTPException(403, "This team is not in your organisation")
    if claims["role"] not in (OrgRole.CEO, OrgRole.ADMIN) and not _is_team_member(team_id, claims["sub"]):
        raise HTTPException(403, "Not authorised to view this team's meetings")

    sessions = select("jarvis_sessions", {"team_id": f"eq.{team_id}", "order": "updated_at.desc"})
    result = []
    for s in sessions:
        summary_obj = s.get("summary") or {}
        result.append({
            "session_id": s["session_id"],
            "title": summary_obj.get("title") or f"Meeting {s['session_id'][:8]}",
            "meeting_url": s.get("meeting_url"),
            "status": s.get("status"),
            "started_at": s.get("started_at"),
            "ended_at": s.get("ended_at"),
            "duration_mins": _duration_mins(s),
            "change_count": len(s.get("changes") or []),
            "team_id": team_id,
        })
    return result


@router.get("/teams/{team_id}/meetings/{session_id}/participants")
def team_meeting_participants(
    team_id: str, session_id: str, claims: dict = Depends(get_current_user),
):
    """Per-person in-call time for one of a team's meetings — the "active time of
    each person in the meeting" view. Team MANAGER (or ADMIN/CEO) only.

    In-call time is the real join→leave duration from meeting_participants; guests
    (Recall names not matched to an org user) are flagged so they read clearly.
    """
    team = select_one("org_teams", {"id": f"eq.{team_id}"}, columns="id,org_id")
    if not team:
        raise HTTPException(404, "Team not found")
    if team.get("org_id") != claims.get("org_id"):
        raise HTTPException(403, "This team is not in your organisation")
    if not can_oversee_team(claims, team_id):
        raise HTTPException(403, "Only a team manager, ADMIN, or CEO can view participant activity")

    s = select_one("jarvis_sessions", {"session_id": f"eq.{session_id}"}, columns="session_id,team_id")
    if not s:
        raise HTTPException(404, "Meeting not found")
    if s.get("team_id") != team_id:
        raise HTTPException(403, "This meeting does not belong to that team")

    rows = select("meeting_participants", {
        "session_id": f"eq.{session_id}", "order": "joined_at.asc",
    })

    # Resolve matched attendees to display names in one batched query.
    uids = [r["user_id"] for r in rows if r.get("user_id")]
    names: dict[str, str] = {}
    if uids:
        for u in select("org_users", {"id": f"in.({','.join(uids)})", "select": "id,name"}):
            names[u["id"]] = u["name"]

    return [
        {
            "id": r.get("id"),
            "recall_name": r.get("recall_name"),
            "name": names.get(r.get("user_id")) or r.get("recall_name"),
            "user_id": r.get("user_id"),
            "is_guest": r.get("is_guest"),
            "joined_at": r.get("joined_at"),
            "left_at": r.get("left_at"),
            "duration_mins": r.get("duration_mins"),
        }
        for r in rows
    ]


@router.post("/teams/{team_id}/meetings/{session_id}/kick")
def kick_team_meeting(
    team_id: str, session_id: str, claims: dict = Depends(get_current_user),
):
    """Remove the Jarvis bot from one of a team's live meetings. Team MANAGER (or
    ADMIN/CEO) only — the manager counterpart to the org-wide admin kick."""
    team = select_one("org_teams", {"id": f"eq.{team_id}"}, columns="id,org_id")
    if not team:
        raise HTTPException(404, "Team not found")
    if team.get("org_id") != claims.get("org_id"):
        raise HTTPException(403, "This team is not in your organisation")
    if not can_oversee_team(claims, team_id):
        raise HTTPException(403, "Only a team manager, ADMIN, or CEO can kick a meeting")

    s = select_one("jarvis_sessions", {"session_id": f"eq.{session_id}"}, columns="session_id,team_id,status")
    if not s:
        raise HTTPException(404, "Meeting not found")
    if s.get("team_id") != team_id:
        raise HTTPException(403, "This meeting does not belong to that team")

    from .admin_meetings import stop_meeting_bot
    return stop_meeting_bot(session_id, s, actor=claims.get("sub"))
