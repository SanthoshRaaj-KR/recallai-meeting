"""Invite routes: look up and accept team invitations.

Both routes here are deliberately unauthenticated — an invitee is not in the org
yet — which makes them the one guessable entry point into an organisation. They
are protected by a long code (64 bits, see teams.invite_member), a one-hour
expiry, verified-identity binding on accept, and the per-IP throttle below.
"""
from __future__ import annotations

import datetime as dt
import logging
import os
import threading
import time

from fastapi import APIRouter, HTTPException, Request

from ..auth import create_access_token, verify_supabase_token
from ..database import DBError, delete, insert, select_one, update, find_by_text_ci
from ..models import AcceptInviteRequest, OrgRole, TeamInviteOut
from .teams import _wire_hierarchy

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/invites", tags=["invites"])

# ── Per-IP throttle on invalid codes ─────────────────────────────────────────
# Guessing a code is the only way to reach an org you weren't invited to, so
# WRONG guesses are what we count — a legitimate invitee, who has a real code,
# is never throttled no matter how often they reload.
#
# In-process, like the rest of this stack's ephemeral state: org-service runs
# single-replica (deploy/EXECUTION_PLAN.md). Multiple replicas would each hold
# their own counter and need a shared store.
_MAX_BAD_ATTEMPTS = int(os.getenv("INVITE_MAX_BAD_ATTEMPTS", "10"))
_WINDOW_S = int(os.getenv("INVITE_THROTTLE_WINDOW_SECONDS", "900"))  # 15 minutes

_bad_attempts: dict[str, list[float]] = {}
_throttle_lock = threading.Lock()


def _client_ip(request: Request) -> str:
    """Caller IP, honouring the proxy header Caddy sets in front of this service."""
    fwd = request.headers.get("x-forwarded-for", "")
    if fwd:
        return fwd.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def _check_throttle(request: Request) -> str:
    """Raise 429 if this IP has burned through its bad-guess budget."""
    ip = _client_ip(request)
    cutoff = time.monotonic() - _WINDOW_S
    with _throttle_lock:
        recent = [t for t in _bad_attempts.get(ip, []) if t > cutoff]
        if recent:
            _bad_attempts[ip] = recent
        else:
            _bad_attempts.pop(ip, None)
        if len(recent) >= _MAX_BAD_ATTEMPTS:
            logger.warning("[invite] throttling %s after %d bad codes", ip, len(recent))
            raise HTTPException(
                429, "Too many invalid invite codes. Try again in a few minutes."
            )
    return ip


def _record_bad_attempt(ip: str) -> None:
    with _throttle_lock:
        _bad_attempts.setdefault(ip, []).append(time.monotonic())


def _clear_attempts(ip: str) -> None:
    """A valid code proves this isn't a guessing client — reset its budget."""
    with _throttle_lock:
        _bad_attempts.pop(ip, None)


def _load_and_validate(code: str) -> dict:
    invite = select_one("org_team_invitations", {"code": f"eq.{code}"})
    if not invite:
        raise HTTPException(404, "Invite not found. Check the code and try again.")
    if invite["status"] == "accepted":
        raise HTTPException(410, "This invite has already been used.")
    if invite["status"] != "pending":
        raise HTTPException(404, "Invite not found. Check the code and try again.")
    expires = dt.datetime.fromisoformat(invite["expires_at"].replace("Z", "+00:00"))
    if dt.datetime.now(dt.timezone.utc) > expires:
        try:
            delete("org_team_invitations", {"code": f"eq.{code}"})
        except Exception:
            pass
        raise HTTPException(410, "This invite has expired. Ask the team admin to send a new one.")
    return invite


@router.get("/{code}", response_model=TeamInviteOut)
def get_invite(code: str, request: Request):
    ip = _check_throttle(request)
    try:
        invite = _load_and_validate(code)
    except HTTPException:
        _record_bad_attempt(ip)
        raise
    _clear_attempts(ip)
    team = select_one("org_teams", {"id": f"eq.{invite['team_id']}"})
    existing = find_by_text_ci("org_users", "email", invite["email"])
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
def accept_invite(code: str, body: AcceptInviteRequest, request: Request):
    ip = _check_throttle(request)
    try:
        invite = _load_and_validate(code)
    except HTTPException:
        _record_bad_attempt(ip)
        raise
    _clear_attempts(ip)

    team = select_one("org_teams", {"id": f"eq.{invite['team_id']}"})
    if not team:
        raise HTTPException(404, "Team no longer exists")

    # The invite code alone proves nothing about who is holding it. Require a
    # verified Google identity and bind it to the address the invite was sent to,
    # otherwise anyone with (or guessing) a code could create an account under
    # someone else's email and receive a valid session for it.
    identity = verify_supabase_token(body.supabase_token)
    if identity["email"].strip().lower() != (invite["email"] or "").strip().lower():
        raise HTTPException(
            403,
            f"This invite was sent to {invite['email']}. "
            f"You are signed in as {identity['email']} — sign in with the invited "
            f"account to accept.",
        )

    existing_user = find_by_text_ci("org_users", "email", invite["email"])

    if existing_user:
        user_id = existing_user["id"]
        # Attach an org-less account to the inviting team's org so org-scoped
        # views (and the user's JWT) reflect membership immediately on accept.
        if not existing_user.get("org_id"):
            try:
                update("org_users", {"id": f"eq.{user_id}"}, {"org_id": team["org_id"]})
            except DBError:
                pass
    else:
        # Prefer the name Google gives us; body.name is only a fallback.
        display_name = (identity.get("full_name") or body.name or "").strip()
        if not display_name:
            raise HTTPException(400, "name is required to create a new account")
        try:
            new_user = insert("org_users", {
                # Store the VERIFIED email, not the invited string, so casing
                # matches what google-exchange will look up later.
                "email": identity["email"],
                "name": display_name,
                # No password: invited users authenticate through Google.
                "password_hash": None,
                "role": OrgRole.MEMBER,
                "org_id": team["org_id"],
                "is_active": True,
                "supabase_user_id": identity.get("supabase_uid"),
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
        # Wire into the reporting hierarchy (under the team manager, or under the
        # CEO if joining as MANAGER) — mirrors a direct add_member so invited
        # members show up correctly in the org chart.
        _wire_hierarchy(invite["team_id"], user_id, invite["role"])

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
