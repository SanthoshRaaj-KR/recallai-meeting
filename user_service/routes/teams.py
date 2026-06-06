"""Team CRUD + member management routes."""

import datetime
from fastapi import APIRouter, Depends, HTTPException, status

from ..auth import get_current_user, verify_admin_secret
from ..database import select, select_one, insert, update, delete, DBError
from ..models import (
    TeamCreate, TeamOut, TeamUpdate,
    AddMemberRequest, RemoveMemberRequest, MemberOut, UserOut,
    OrgRole, TeamRole,
)
from ..rbac import require_ceo, require_manager_or_above

router = APIRouter(prefix="/teams", tags=["teams"])


def _is_team_manager(team_id: str, user_id: str) -> bool:
    row = select_one("org_team_members", {
        "team_id": f"eq.{team_id}",
        "user_id": f"eq.{user_id}",
        "role": f"eq.{TeamRole.MANAGER}",
    })
    return row is not None


def _enrich_team(row: dict) -> TeamOut:
    members = select("org_team_members", {"team_id": f"eq.{row['id']}"})
    bot = select_one("org_team_bots", {"team_id": f"eq.{row['id']}"})
    return TeamOut(
        id=row["id"],
        name=row["name"],
        org_id=row["org_id"],
        created_at=row["created_at"],
        member_count=len(members),
        bot_assigned=bot is not None,
    )


@router.get("", response_model=list[TeamOut])
def list_teams(claims: dict = Depends(get_current_user)):
    if claims["role"] == OrgRole.CEO:
        rows = select("org_teams", {"org_id": f"eq.{claims['org_id']}"})
    else:
        # Return only teams this user belongs to
        memberships = select("org_team_members", {"user_id": f"eq.{claims['sub']}"})
        team_ids = [m["team_id"] for m in memberships]
        rows = [select_one("org_teams", {"id": f"eq.{tid}"}) for tid in team_ids]
        rows = [r for r in rows if r]
    return [_enrich_team(r) for r in rows]


@router.post("", response_model=TeamOut, status_code=status.HTTP_201_CREATED)
def create_team(body: TeamCreate, claims: dict = Depends(require_ceo())):
    try:
        row = insert("org_teams", {"name": body.name, "org_id": body.org_id})
    except DBError as e:
        raise HTTPException(500, str(e))
    return _enrich_team(row)


@router.get("/{team_id}", response_model=TeamOut)
def get_team(team_id: str, claims: dict = Depends(get_current_user)):
    row = select_one("org_teams", {"id": f"eq.{team_id}"})
    if not row:
        raise HTTPException(404, "Team not found")
    if claims["role"] != OrgRole.CEO:
        member = select_one("org_team_members", {
            "team_id": f"eq.{team_id}", "user_id": f"eq.{claims['sub']}",
        })
        if not member:
            raise HTTPException(403, "Not a member of this team")
    return _enrich_team(row)


@router.patch("/{team_id}", response_model=TeamOut)
def update_team(team_id: str, body: TeamUpdate, claims: dict = Depends(get_current_user)):
    row = select_one("org_teams", {"id": f"eq.{team_id}"})
    if not row:
        raise HTTPException(404, "Team not found")
    if claims["role"] != OrgRole.CEO and not _is_team_manager(team_id, claims["sub"]):
        raise HTTPException(403, "Only team manager or CEO can update team")
    updates = body.model_dump(exclude_none=True)
    if not updates:
        raise HTTPException(400, "No fields to update")
    rows = update("org_teams", {"id": f"eq.{team_id}"}, updates)
    return _enrich_team(rows[0])


@router.delete("/{team_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_team(team_id: str, claims: dict = Depends(require_ceo())):
    delete("org_teams", {"id": f"eq.{team_id}"})


# ── Members ────────────────────────────────────────────────────────────────────

@router.get("/{team_id}/members", response_model=list[MemberOut])
def list_members(team_id: str, claims: dict = Depends(get_current_user)):
    row = select_one("org_teams", {"id": f"eq.{team_id}"})
    if not row:
        raise HTTPException(404, "Team not found")
    if claims["role"] != OrgRole.CEO:
        member = select_one("org_team_members", {
            "team_id": f"eq.{team_id}", "user_id": f"eq.{claims['sub']}",
        })
        if not member:
            raise HTTPException(403, "Not authorised to view this team")

    members = select("org_team_members", {"team_id": f"eq.{team_id}"})
    result = []
    for m in members:
        user_row = select_one("org_users", {"id": f"eq.{m['user_id']}"})
        user_out = None
        if user_row:
            user_out = UserOut(
                id=user_row["id"], email=user_row["email"], name=user_row["name"],
                role=user_row["role"], org_id=user_row.get("org_id"),
                is_active=user_row.get("is_active", True), created_at=user_row["created_at"],
            )
        result.append(MemberOut(
            user_id=m["user_id"], team_id=team_id,
            role=m["role"], joined_at=m["joined_at"], user=user_out,
        ))
    return result


@router.post("/{team_id}/members", response_model=MemberOut, status_code=status.HTTP_201_CREATED)
def add_member(team_id: str, body: AddMemberRequest, claims: dict = Depends(get_current_user)):
    """Add a user to a team. Requires admin_secret in body; only CEO or team manager may call."""
    verify_admin_secret(body.admin_secret)
    team = select_one("org_teams", {"id": f"eq.{team_id}"})
    if not team:
        raise HTTPException(404, "Team not found")
    if claims["role"] != OrgRole.CEO and not _is_team_manager(team_id, claims["sub"]):
        raise HTTPException(403, "Only team manager or CEO can add members")
    if body.role not in (TeamRole.MANAGER, TeamRole.MEMBER, TeamRole.ASSOCIATE):
        raise HTTPException(400, f"Invalid team role: {body.role}")

    # Prevent duplicate
    existing = select_one("org_team_members", {
        "team_id": f"eq.{team_id}", "user_id": f"eq.{body.user_id}",
    })
    if existing:
        raise HTTPException(409, "User is already a member of this team")

    try:
        row = insert("org_team_members", {
            "team_id": team_id, "user_id": body.user_id, "role": body.role,
        })
    except DBError as e:
        raise HTTPException(500, str(e))

    # Wire hierarchy: if the added user is MANAGER, link them to CEO in closure table.
    # For MEMBER/ASSOCIATE, find their team's manager and create reporting links.
    _wire_hierarchy(team_id, body.user_id, body.role)

    return MemberOut(
        user_id=body.user_id, team_id=team_id,
        role=body.role, joined_at=row["joined_at"],
    )


@router.delete("/{team_id}/members/{user_id}", status_code=status.HTTP_204_NO_CONTENT)
def remove_member(team_id: str, user_id: str, body: RemoveMemberRequest, claims: dict = Depends(get_current_user)):
    verify_admin_secret(body.admin_secret)
    team = select_one("org_teams", {"id": f"eq.{team_id}"})
    if not team:
        raise HTTPException(404, "Team not found")
    if claims["role"] != OrgRole.CEO and not _is_team_manager(team_id, claims["sub"]):
        raise HTTPException(403, "Only team manager or CEO can remove members")
    delete("org_team_members", {"team_id": f"eq.{team_id}", "user_id": f"eq.{user_id}"})
    # Remove from hierarchy
    delete("org_reporting_hierarchy", {"descendant_id": f"eq.{user_id}"})
    delete("org_reporting_hierarchy", {"ancestor_id": f"eq.{user_id}", "depth": "gt.0"})


def _wire_hierarchy(team_id: str, user_id: str, team_role: str) -> None:
    """Insert closure table rows so the new member appears in the right sub-tree."""
    try:
        # Always add self-loop
        insert("org_reporting_hierarchy", {
            "ancestor_id": user_id, "descendant_id": user_id, "depth": 0,
        })
    except DBError:
        pass  # already exists (re-added after removal)

    if team_role == TeamRole.MANAGER:
        # Find the CEO to link MANAGER → CEO at depth 1
        ceo = select_one("org_users", {"role": f"eq.{OrgRole.CEO}"})
        if ceo:
            try:
                insert("org_reporting_hierarchy", {
                    "ancestor_id": ceo["id"], "descendant_id": user_id, "depth": 1,
                })
            except DBError:
                pass
    else:
        # Find the team's manager to link MEMBER/ASSOCIATE → MANAGER at depth 1
        manager_row = select_one("org_team_members", {
            "team_id": f"eq.{team_id}", "role": f"eq.{TeamRole.MANAGER}",
        })
        if manager_row:
            manager_id = manager_row["user_id"]
            try:
                insert("org_reporting_hierarchy", {
                    "ancestor_id": manager_id, "descendant_id": user_id, "depth": 1,
                })
                # Also propagate the manager's ancestors (CEO) to this member at depth+1
                ancestors = select("org_reporting_hierarchy", {
                    "descendant_id": f"eq.{manager_id}", "depth": "gt.0",
                })
                for anc in ancestors:
                    try:
                        insert("org_reporting_hierarchy", {
                            "ancestor_id": anc["ancestor_id"],
                            "descendant_id": user_id,
                            "depth": anc["depth"] + 1,
                        })
                    except DBError:
                        pass
            except DBError:
                pass
