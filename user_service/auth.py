"""JWT authentication helpers for the user service."""

import os
from datetime import datetime, timedelta, timezone

import bcrypt as _bcrypt
import requests as _requests
from jose import JWTError, jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

_SECRET = os.getenv("JWT_SECRET", "change-me-in-production-please-use-a-strong-random-secret")
_ADMIN_SECRET = os.getenv("ADMIN_SECRET_KEY", "")
_ALGORITHM = "HS256"
_ACCESS_TTL_MINUTES = int(os.getenv("JWT_ACCESS_TTL_MINUTES", "60"))
_REFRESH_TTL_DAYS = int(os.getenv("JWT_REFRESH_TTL_DAYS", "7"))

_bearer = HTTPBearer()


def hash_password(plain: str) -> str:
    return _bcrypt.hashpw(plain.encode(), _bcrypt.gensalt(rounds=12)).decode()


def verify_password(plain: str, hashed: str) -> bool:
    try:
        return _bcrypt.checkpw(plain.encode(), hashed.encode())
    except Exception:
        return False


def create_access_token(user_id: str, role: str, org_id: str | None) -> str:
    expire = datetime.now(timezone.utc) + timedelta(minutes=_ACCESS_TTL_MINUTES)
    return jwt.encode(
        {"sub": user_id, "role": role, "org_id": org_id, "exp": expire, "type": "access"},
        _SECRET, algorithm=_ALGORITHM,
    )


def create_refresh_token(user_id: str) -> str:
    expire = datetime.now(timezone.utc) + timedelta(days=_REFRESH_TTL_DAYS)
    return jwt.encode(
        {"sub": user_id, "exp": expire, "type": "refresh"},
        _SECRET, algorithm=_ALGORITHM,
    )


def decode_token(token: str) -> dict:
    try:
        return jwt.decode(token, _SECRET, algorithms=[_ALGORITHM])
    except JWTError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid or expired token: {exc}",
        )


def verify_supabase_token(supabase_token: str) -> dict:
    """Validate a Supabase access token and return its verified identity.

    Returns ``{"email": str, "supabase_uid": str | None, "full_name": str | None}``.
    The email is asserted by Supabase Auth (Google OAuth), so it is proof the
    caller controls that mailbox — which is what lets an invite be bound to the
    address it was sent to.

    Raises HTTPException on an unreachable/misconfigured Supabase or an invalid
    token.
    """
    supabase_url = os.getenv("SUPABASE_URL", "").rstrip("/")
    service_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")
    if not supabase_url or not service_key:
        raise HTTPException(500, "SUPABASE_URL / SUPABASE_SERVICE_ROLE_KEY not configured")

    try:
        resp = _requests.get(
            f"{supabase_url}/auth/v1/user",
            headers={"Authorization": f"Bearer {supabase_token}", "apikey": service_key},
            timeout=6,
        )
    except Exception as exc:
        raise HTTPException(503, f"Could not reach Supabase Auth: {exc}")

    if not resp.ok:
        raise HTTPException(401, "Invalid or expired Supabase token")

    data = resp.json()
    email = data.get("email")
    if not email:
        raise HTTPException(401, "Supabase token has no email claim")
    meta = data.get("user_metadata") or {}
    return {
        "email": email,
        "supabase_uid": data.get("id"),
        "full_name": meta.get("full_name") or meta.get("name"),
    }


def verify_admin_secret(secret: str) -> None:
    if not _ADMIN_SECRET:
        raise HTTPException(status_code=500, detail="ADMIN_SECRET_KEY is not configured on the server")
    if secret != _ADMIN_SECRET:
        raise HTTPException(status_code=403, detail="Invalid admin secret")


class CurrentUser:
    """Dependency: extracts and validates the JWT, returns claims dict."""

    def __call__(self, creds: HTTPAuthorizationCredentials = Depends(_bearer)) -> dict:
        claims = decode_token(creds.credentials)
        if claims.get("type") != "access":
            raise HTTPException(status_code=401, detail="Expected access token")
        return claims


get_current_user = CurrentUser()
