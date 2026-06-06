"""Auth routes: register, login, refresh, logout."""

from fastapi import APIRouter, HTTPException, status, Depends

from ..auth import (
    hash_password, verify_password,
    create_access_token, decode_token,
    get_current_user,
)
from ..database import select_one, insert, DBError
from ..models import RegisterRequest, LoginRequest, TokenResponse, OrgRole

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/register", response_model=TokenResponse, status_code=status.HTTP_201_CREATED)
def register(body: RegisterRequest):
    if body.role not in OrgRole.all:
        raise HTTPException(400, f"Invalid role. Choose from: {OrgRole.all}")

    existing = select_one("org_users", {"email": f"eq.{body.email}"})
    if existing:
        raise HTTPException(409, "Email already registered")

    try:
        user = insert("org_users", {
            "email": body.email,
            "name": body.name,
            "password_hash": hash_password(body.password),
            "role": body.role,
            "org_id": body.org_id,
            "is_active": True,
        })
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

    access = create_access_token(user["id"], user["role"], user.get("org_id"))
    return TokenResponse(
        access_token=access,
        user_id=user["id"],
        role=user["role"],
        org_id=user.get("org_id"),
    )


@router.post("/login", response_model=TokenResponse)
def login(body: LoginRequest):
    user = select_one("org_users", {"email": f"eq.{body.email}"})
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


@router.post("/logout")
def logout(claims: dict = Depends(get_current_user)):
    # Stateless JWT: client just discards the token.
    return {"ok": True}
