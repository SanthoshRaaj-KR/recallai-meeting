"""Invite routes: look up and accept team invitations."""
from __future__ import annotations

import datetime as dt

from fastapi import APIRouter, HTTPException

from ..auth import create_access_token, hash_password
from ..database import DBError, insert, select_one, update
from ..models import AcceptInviteRequest, OrgRole, TeamInviteOut

router = APIRouter(prefix="/invites", tags=["invites"])


def _load_and_validate(code: str) -> dict:
    invite = select_one("org_team_invitations", {"code": f"eq.{code}", "status": "eq.pending"})
    if not invite:
        raise HTTPException(404, "Invite not found or already used")
    expires = dt.datetime.fromisoformat(invite["expires_at"].replace("Z", "+00:00"))
    if dt.datetime.now(dt.timezone.utc) > expires:
        try:
            update("org_team_invitations", {"code": f"eq.{code}"}, {"status": "expired"})
        except Exception:
            pass
        raise HTTPException(410, "This invite has expired")
    return invite


@router.get("/{code}", response_model=TeamInviteOut)
def get_invite(code: str):
    invite = _load_and_validate(code)
    team = select_one("org_teams", {"id": f"eq.{invite['team_id']}"})
    existing = select_one("org_users", {"email": f"eq.{invite['email']}"})
    return TeamInviteOut(
        id=invite["id"],
        team_id=invite["team_id"],
        email=invite["email"],
        role=invite["role"],
        code=code,
        status=invite["status"],
        inviter_id=invite.get("inviter_id"),
        created_at=invite["created_at"],
        expires_at=invite["expires_at"],
        team_name=team["name"] if team else None,
        user_exists=existing is not None,
    )


@router.post("/{code}/accept")
def accept_invite(code: str, body: AcceptInviteRequest):
    invite = _load_and_validate(code)

    team = select_one("org_teams", {"id": f"eq.{invite['team_id']}"})
    if not team:
        raise HTTPException(404, "Team no longer exists")

    existing_user = select_one("org_users", {"email": f"eq.{invite['email']}"})

    if existing_user:
        user_id = existing_user["id"]
    else:
        if not body.name or not body.password:
            raise HTTPException(400, "name and password are required to create a new account")
        try:
            new_user = insert("org_users", {
                "email": invite["email"],
                "name": body.name,
                "password_hash": hash_password(body.password),
                "role": OrgRole.MEMBER,
                "org_id": team["org_id"],
                "is_active": True,
            })
            user_id = new_user["id"]
        except DBError as e:
            raise HTTPException(500, str(e))
        try:
            insert("org_reporting_hierarchy", {
                "ancestor_id": user_id, "descendant_id": user_id, "depth": 0,
            })
        except DBError:
            pass

    existing_membership = select_one(
        "org_team_members",
        {"team_id": f"eq.{invite['team_id']}", "user_id": f"eq.{user_id}"},
    )
    if not existing_membership:
        try:
            insert("org_team_members", {
                "team_id": invite["team_id"],
                "user_id": user_id,
                "role": invite["role"],
            })
        except DBError as e:
            raise HTTPException(500, str(e))

    try:
        update("org_team_invitations", {"code": f"eq.{code}"}, {"status": "accepted"})
    except DBError:
        pass

    user = select_one("org_users", {"id": f"eq.{user_id}"})
    access = create_access_token(user_id, user["role"], user.get("org_id"))
    return {
        "access_token": access,
        "team_name": team["name"],
        "role": invite["role"],
        "message": f"You've joined {team['name']} as {invite['role']}!",
    }
