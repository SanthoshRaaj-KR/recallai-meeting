"""Role-based access control helpers."""

from fastapi import Depends, HTTPException, status

from .auth import get_current_user
from .database import select_one
from .models import OrgRole, TeamRole


def is_team_manager(team_id: str, user_id: str) -> bool:
    """True if `user_id` holds the MANAGER team-role within `team_id`."""
    return select_one("org_team_members", {
        "team_id": f"eq.{team_id}",
        "user_id": f"eq.{user_id}",
        "role": f"eq.{TeamRole.MANAGER}",
    }) is not None


def can_manage_team(claims: dict, team_id: str) -> bool:
    """Effective team-write privilege: org ADMIN/CEO, or the MANAGER of *this* team.

    This is the single source of truth behind every team-scoped write (add/remove
    member, invite, edit) and the manager-only meeting participant view.
    """
    if claims.get("role") in OrgRole.admin_and_above:
        return True
    return is_team_manager(team_id, claims.get("sub"))


def require_roles(*roles: str):
    """FastAPI dependency: raises 403 if the caller's role is not in `roles`."""
    def _check(claims: dict = Depends(get_current_user)) -> dict:
        if claims.get("role") not in roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Requires role {' or '.join(roles)}, got {claims.get('role')}",
            )
        return claims
    return _check


def require_ceo():
    return require_roles(OrgRole.CEO)


def require_admin_or_above():
    return require_roles(OrgRole.CEO, OrgRole.ADMIN)


def require_manager_or_above():
    return require_roles(OrgRole.CEO, OrgRole.ADMIN, OrgRole.MANAGER)


def require_any_role():
    return require_roles(*OrgRole.all)
