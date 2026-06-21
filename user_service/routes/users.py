"""User CRUD routes."""

import logging
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, HTTPException

from ..auth import get_current_user
from ..database import select, select_one, update, DBError
from ..models import UserOut, UserUpdate, MeetingStats, OrgRole
from ..rbac import require_admin_or_above

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/users", tags=["users"])


def _to_user_out(row: dict) -> UserOut:
    return UserOut(
        id=row["id"],
        email=row["email"],
        name=row["name"],
        role=row["role"],
        org_id=row.get("org_id"),
        is_active=row.get("is_active", True),
        created_at=row["created_at"],
        job_title=row.get("job_title"),
        department=row.get("department"),
        phone=row.get("phone"),
        bio=row.get("bio"),
        avatar_url=row.get("avatar_url"),
    )


@router.get("", response_model=list[UserOut])
def list_users(claims: dict = Depends(require_admin_or_above())):
    """CEO or ADMIN: list all users in the organisation."""
    rows = select("org_users", {"org_id": f"eq.{claims['org_id']}"})
    return [_to_user_out(r) for r in rows]


@router.get("/me", response_model=UserOut)
def get_me(claims: dict = Depends(get_current_user)):
    row = select_one("org_users", {"id": f"eq.{claims['sub']}"})
    if not row:
        raise HTTPException(404, "User not found")
    return _to_user_out(row)


@router.get("/{user_id}", response_model=UserOut)
def get_user(user_id: str, claims: dict = Depends(get_current_user)):
    row = select_one("org_users", {"id": f"eq.{user_id}"})
    if not row:
        raise HTTPException(404, "User not found")
    if claims["role"] not in (OrgRole.CEO, OrgRole.ADMIN) and claims["sub"] != user_id:
        raise HTTPException(403, "Not authorised to view this user")
    return _to_user_out(row)


@router.patch("/{user_id}", response_model=UserOut)
def update_user(user_id: str, body: UserUpdate, claims: dict = Depends(get_current_user)):
    if claims["role"] not in (OrgRole.CEO, OrgRole.ADMIN) and claims["sub"] != user_id:
        raise HTTPException(403, "Not authorised to update this user")
    updates = body.model_dump(exclude_none=True)
    if "role" in updates:
        if claims["role"] not in (OrgRole.CEO, OrgRole.ADMIN):
            raise HTTPException(403, "Only CEO or ADMIN can change roles")
        if updates["role"] not in OrgRole.all:
            raise HTTPException(400, f"Invalid role. Choose from: {OrgRole.all}")
        if claims["role"] == OrgRole.ADMIN:
            if updates["role"] == OrgRole.CEO:
                raise HTTPException(403, "Only CEO can assign the CEO role")
            target = select_one("org_users", {"id": f"eq.{user_id}"})
            if target and target.get("role") == OrgRole.CEO:
                raise HTTPException(403, "Only CEO can modify the CEO's role")
    if not updates:
        raise HTTPException(400, "No fields to update")
    try:
        rows = update("org_users", {"id": f"eq.{user_id}"}, updates)
    except DBError as e:
        raise HTTPException(500, str(e))
    if not rows:
        raise HTTPException(404, "User not found")
    return _to_user_out(rows[0])


@router.get("/by-email/{email}", response_model=UserOut)
def get_user_by_email(email: str, claims: dict = Depends(get_current_user)):
    """Look up a user by email. CEO or manager can use this for the invite flow."""
    if claims["role"] not in OrgRole.managers_and_above:
        raise HTTPException(403, "Only managers, ADMIN, or CEO can look up users by email")
    row = select_one("org_users", {"email": f"eq.{email}"})
    if not row:
        raise HTTPException(404, f"No user found with email {email}")
    return _to_user_out(row)


@router.get("/{user_id}/reports", response_model=list[UserOut])
def get_direct_reports(user_id: str, claims: dict = Depends(get_current_user)):
    """Return the direct reports of a user (depth=1 in closure table)."""
    if claims["role"] not in (OrgRole.CEO, OrgRole.ADMIN) and claims["sub"] != user_id:
        raise HTTPException(403, "Not authorised")
    rows = select("org_reporting_hierarchy", {
        "ancestor_id": f"eq.{user_id}",
        "depth": "eq.1",
    })
    if not rows:
        return []
    descendant_ids = [r["descendant_id"] for r in rows]
    # Fetch user records for each descendant
    result = []
    for uid in descendant_ids:
        u = select_one("org_users", {"id": f"eq.{uid}"})
        if u:
            result.append(_to_user_out(u))
    return result


@router.get("/me/stats", response_model=MeetingStats)
def get_my_stats(claims: dict = Depends(get_current_user)):
    """Return meeting stats for the current user based on their team memberships."""
    user_id = claims["sub"]
    memberships = select("org_team_members", {"user_id": f"eq.{user_id}"})
    team_ids = [m["team_id"] for m in memberships]

    if not team_ids:
        return MeetingStats(total_meetings=0, total_minutes=0)

    now = datetime.now(timezone.utc)
    month_ago = now - timedelta(days=30)
    week_ago = now - timedelta(days=7)
    month_ago_iso = month_ago.replace(microsecond=0).isoformat()

    all_sessions: list[dict] = []
    recent_sessions: list[dict] = []
    for tid in team_ids:
        all_sessions.extend(select("jarvis_sessions", {"team_id": f"eq.{tid}", "status": "eq.ended"}))
        recent_sessions.extend(select("jarvis_sessions", {
            "team_id": f"eq.{tid}", "status": "eq.ended", "started_at": f"gte.{month_ago_iso}",
        }))

    total_minutes = 0.0
    durations: list[float] = []
    last_meeting_at: str | None = None

    for s in all_sessions:
        started_raw = s.get("started_at")
        ended_raw = s.get("ended_at")
        if started_raw and ended_raw:
            try:
                start = datetime.fromisoformat(started_raw.replace("Z", "+00:00"))
                end = datetime.fromisoformat(ended_raw.replace("Z", "+00:00"))
                dur = (end - start).total_seconds() / 60
                total_minutes += dur
                durations.append(dur)
            except Exception:
                pass
        if ended_raw and (last_meeting_at is None or ended_raw > last_meeting_at):
            last_meeting_at = ended_raw

    meetings_this_week = 0
    meetings_this_month = 0
    for s in recent_sessions:
        try:
            start = datetime.fromisoformat((s.get("started_at") or "").replace("Z", "+00:00"))
            if start >= week_ago:
                meetings_this_week += 1
            if start >= month_ago:
                meetings_this_month += 1
        except Exception:
            pass

    avg_mins = round(sum(durations) / len(durations), 1) if durations else 0.0

    return MeetingStats(
        total_meetings=len(all_sessions),
        total_minutes=round(total_minutes),
        last_meeting_at=last_meeting_at,
        meetings_this_week=meetings_this_week,
        meetings_this_month=meetings_this_month,
        avg_meeting_duration_mins=avg_mins,
    )


@router.get("/{user_id}/manager", response_model=UserOut)
def get_manager(user_id: str, claims: dict = Depends(get_current_user)):
    """Return the direct manager (depth=1 ancestor) of a user."""
    if claims["role"] not in (OrgRole.CEO, OrgRole.ADMIN) and claims["sub"] != user_id:
        raise HTTPException(403, "Not authorised")
    rows = select("org_reporting_hierarchy", {
        "descendant_id": f"eq.{user_id}",
        "depth": "eq.1",
    })
    if not rows:
        raise HTTPException(404, "No manager found")
    manager = select_one("org_users", {"id": f"eq.{rows[0]['ancestor_id']}"})
    if not manager:
        raise HTTPException(404, "Manager user record not found")
    return _to_user_out(manager)
