"""Role-based access control helpers."""

from fastapi import Depends, HTTPException, status

from .auth import get_current_user
from .models import OrgRole


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


def require_manager_or_above():
    return require_roles(OrgRole.CEO, OrgRole.MANAGER)


def require_any_role():
    return require_roles(*OrgRole.all)
