"""Auth routes: register, login, google-exchange, refresh, logout."""

import logging

from fastapi import APIRouter, HTTPException, status, Depends

from ..auth import (
    hash_password, verify_password,
    create_access_token, decode_token,
    get_current_user, verify_supabase_token,
)
from ..database import select_one, insert, update, DBError, find_by_text_ci
from ..models import RegisterRequest, LoginRequest, GoogleExchangeRequest, TokenResponse, OrgRole

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/register", response_model=TokenResponse, status_code=status.HTTP_201_CREATED)
def register(body: RegisterRequest):
    if body.role not in OrgRole.all:
        raise HTTPException(400, f"Invalid role. Choose from: {OrgRole.all}")

    # Only one CEO allowed — prevents privilege escalation via self-registration
    if body.role == OrgRole.CEO:
        existing_ceo = select_one("org_users", {"role": f"eq.{OrgRole.CEO}"})
        if existing_ceo:
            raise HTTPException(409, "An organisation already has a CEO. Contact your admin to be invited.")

    existing = find_by_text_ci("org_users", "email", body.email)
    if existing:
        raise HTTPException(409, "Email already registered")

    org_id = body.org_id

    # CEO without an org_id auto-creates their organisation
    if body.role == OrgRole.CEO and not org_id:
        try:
            org = insert("organizations", {"name": f"{body.name}'s Organization"})
            org_id = org["id"]
        except DBError as e:
            raise HTTPException(500, f"Failed to create organisation: {e}")

    user_data = {
        "email": body.email,
        "name": body.name,
        "password_hash": hash_password(body.password),
        "role": body.role,
        "org_id": org_id,
        "is_active": True,
    }
    for field in ("job_title", "department", "phone", "bio", "avatar_url"):
        val = getattr(body, field, None)
        if val is not None:
            user_data[field] = val

    try:
        user = insert("org_users", user_data)
    except DBError as e:
        raise HTTPException(500, str(e))

    # CEO gets a self-loop in the hierarchy closure table
    if body.role == OrgRole.CEO:
        try:
            insert("org_reporting_hierarchy", {
                "ancestor_id": user["id"],
                "descendant_id": user["id"],
                "depth": 0,
            })
        except DBError:
            pass

    # Send welcome email (non-blocking — failure doesn't abort registration)
    try:
        from ..email import send_welcome_email
        send_welcome_email(body.email, body.name)
    except Exception as exc:
        logger.warning("[register] Welcome email failed for %s: %s", body.email, exc)

    # user["org_id"] may still be None if Supabase didn't return it; fall back to what we resolved above
    resolved_org_id = user.get("org_id") or org_id
    access = create_access_token(user["id"], user["role"], resolved_org_id)
    return TokenResponse(
        access_token=access,
        user_id=user["id"],
        role=user["role"],
        org_id=resolved_org_id,
    )


@router.post("/login", response_model=TokenResponse)
def login(body: LoginRequest):
    user = find_by_text_ci("org_users", "email", body.email)
    if not user or not verify_password(body.password, user["password_hash"]):
        raise HTTPException(401, "Invalid email or password")
    if not user.get("is_active"):
        raise HTTPException(403, "Account is inactive")

    access = create_access_token(user["id"], user["role"], user.get("org_id"))
    return TokenResponse(
        access_token=access,
        user_id=user["id"],
        role=user["role"],
        org_id=user.get("org_id"),
    )


@router.post("/refresh", response_model=TokenResponse)
def refresh(body: dict):
    token = body.get("refresh_token", "")
    claims = decode_token(token)
    if claims.get("type") != "refresh":
        raise HTTPException(401, "Expected refresh token")
    user = select_one("org_users", {"id": f"eq.{claims['sub']}"})
    if not user or not user.get("is_active"):
        raise HTTPException(401, "User not found or inactive")
    access = create_access_token(user["id"], user["role"], user.get("org_id"))
    return TokenResponse(
        access_token=access,
        user_id=user["id"],
        role=user["role"],
        org_id=user.get("org_id"),
    )


@router.post("/google-exchange", response_model=TokenResponse)
def google_exchange(body: GoogleExchangeRequest):
    """Exchange a Supabase Google access token for an org JWT.

    Called by the frontend after every Google sign-in. Looks up the user's
    email in org_users and returns an org-scoped JWT. Returns 404 with
    detail='not_in_org' if the user is not part of any organization.
    """
    identity = verify_supabase_token(body.supabase_token)
    email: str = identity["email"]
    supabase_uid: str | None = identity["supabase_uid"]

    # Case-insensitive match: Google/Supabase may return the email in a different
    # case than it was stored (invite/manual-add), which previously mis-reported an
    # existing member — even a manager — as "not_in_org" and stranded them at login.
    user = find_by_text_ci("org_users", "email", email)
    if not user:
        # uvicorn's access log records the 404 with neither the email nor a
        # reason, which made a real "wrong account" bug indistinguishable from
        # a not-yet-a-member browser retrying in the background — this line is
        # what makes that diagnosable from the logs.
        logger.info("[google-exchange] not_in_org for %s", email)
        raise HTTPException(404, "not_in_org")

    if not user.get("is_active"):
        logger.warning("[google-exchange] inactive account attempted sign-in: %s", email)
        raise HTTPException(403, "Account is inactive. Contact your organization admin.")

    # Link supabase_user_id on first exchange so future lookups can use it
    if supabase_uid and not user.get("supabase_user_id"):
        try:
            update("org_users", {"id": f"eq.{user['id']}"}, {"supabase_user_id": supabase_uid})
        except Exception:
            pass

    access = create_access_token(user["id"], user["role"], user.get("org_id"))
    return TokenResponse(
        access_token=access,
        user_id=user["id"],
        role=user["role"],
        org_id=user.get("org_id"),
    )


@router.post("/logout")
def logout(claims: dict = Depends(get_current_user)):
    return {"ok": True}
