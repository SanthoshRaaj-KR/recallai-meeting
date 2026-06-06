"""Team bot assignment + team-scoped meeting session routes."""

from fastapi import APIRouter, Depends, HTTPException, status

from ..auth import get_current_user
from ..database import select, select_one, insert, DBError
from ..models import BotCreate, BotOut, OrgRole, TeamRole
from ..rbac import require_ceo

router = APIRouter(tags=["bots"])


def _is_team_member(team_id: str, user_id: str) -> bool:
    return select_one("org_team_members", {
        "team_id": f"eq.{team_id}", "user_id": f"eq.{user_id}",
    }) is not None


# ── Bot assignment ─────────────────────────────────────────────────────────────

@router.get("/teams/{team_id}/bot", response_model=BotOut)
def get_team_bot(team_id: str, claims: dict = Depends(get_current_user)):
    if claims["role"] != OrgRole.CEO and not _is_team_member(team_id, claims["sub"]):
        raise HTTPException(403, "Not authorised to view this team's bot")
    bot = select_one("org_team_bots", {"team_id": f"eq.{team_id}"})
    if not bot:
        raise HTTPException(404, "No bot assigned to this team")
    return BotOut(
        id=bot["id"], team_id=bot["team_id"], name=bot["name"],
        config=bot.get("config") or {}, created_at=bot["created_at"],
    )


@router.post("/teams/{team_id}/bot", response_model=BotOut, status_code=status.HTTP_201_CREATED)
def assign_team_bot(team_id: str, body: BotCreate, claims: dict = Depends(require_ceo())):
    """CEO only: assign (or re-assign) a bot to a team."""
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

    CEO can see all teams. Team members can only see their own team's meetings.
    Bot context is isolated: members cannot see sessions from other teams.
    """
    if claims["role"] != OrgRole.CEO and not _is_team_member(team_id, claims["sub"]):
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
            "change_count": len(s.get("changes") or []),
            "team_id": team_id,
        })
    return result
