"""User CRUD routes."""

from fastapi import APIRouter, Depends, HTTPException

from ..auth import get_current_user
from ..database import select, select_one, update, DBError
from ..models import UserOut, UserUpdate, OrgRole
from ..rbac import require_ceo

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
    )


@router.get("", response_model=list[UserOut])
def list_users(claims: dict = Depends(require_ceo())):
    """CEO only: list all users in the organisation."""
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
    # CEO can see anyone; others can only see themselves or same-team members.
    if claims["role"] != OrgRole.CEO and claims["sub"] != user_id:
        raise HTTPException(403, "Not authorised to view this user")
    return _to_user_out(row)


@router.patch("/{user_id}", response_model=UserOut)
def update_user(user_id: str, body: UserUpdate, claims: dict = Depends(get_current_user)):
    if claims["role"] != OrgRole.CEO and claims["sub"] != user_id:
        raise HTTPException(403, "Not authorised to update this user")
    updates = body.model_dump(exclude_none=True)
    if not updates:
        raise HTTPException(400, "No fields to update")
    try:
        rows = update("org_users", {"id": f"eq.{user_id}"}, updates)
    except DBError as e:
        raise HTTPException(500, str(e))
    if not rows:
        raise HTTPException(404, "User not found")
    return _to_user_out(rows[0])


@router.get("/{user_id}/reports", response_model=list[UserOut])
def get_direct_reports(user_id: str, claims: dict = Depends(get_current_user)):
    """Return the direct reports of a user (depth=1 in closure table)."""
    if claims["role"] != OrgRole.CEO and claims["sub"] != user_id:
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


@router.get("/{user_id}/manager", response_model=UserOut)
def get_manager(user_id: str, claims: dict = Depends(get_current_user)):
    """Return the direct manager (depth=1 ancestor) of a user."""
    if claims["role"] != OrgRole.CEO and claims["sub"] != user_id:
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
