"""User CRUD routes."""

import logging
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, HTTPException

from ..auth import get_current_user
from ..database import rpc, select, select_one, update, DBError, find_by_text_ci
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
    row = find_by_text_ci("org_users", "email", email)
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
    """Return REAL meeting stats for the current user — meetings they actually
    attended and their true time-in-call, from meeting_participants (Phase 4).

    Previously this counted every one of the user's team meetings and credited the
    full duration whether or not they joined; now a no-show correctly shows zero.
    """
    user_id = claims["sub"]
    now = datetime.now(timezone.utc)
    week_iso = (now - timedelta(days=7)).replace(microsecond=0).isoformat()
    month_iso = (now - timedelta(days=30)).replace(microsecond=0).isoformat()

    try:
        rows = rpc("user_attendance_stats", {
            "p_user": user_id, "p_week": week_iso, "p_month": month_iso,
        })
    except DBError as exc:
        logger.warning("me/stats rpc failed for %s: %s", user_id, exc)
        return MeetingStats(total_meetings=0, total_minutes=0)

    r = rows[0] if rows else {}
    attended = int(r.get("meetings_attended") or 0)
    total_min = float(r.get("total_minutes") or 0)
    dur_count = int(r.get("dur_count") or 0)

    return MeetingStats(
        total_meetings=attended,
        total_minutes=round(total_min),
        last_meeting_at=r.get("last_attended"),
        meetings_this_week=int(r.get("attended_week") or 0),
        meetings_this_month=int(r.get("attended_month") or 0),
        avg_meeting_duration_mins=round(total_min / dur_count, 1) if dur_count else 0.0,
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
