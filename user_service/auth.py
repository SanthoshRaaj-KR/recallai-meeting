"""JWT authentication helpers for the user service."""

import os
from datetime import datetime, timedelta, timezone

from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

_SECRET = os.getenv("JWT_SECRET", "change-me-in-production-please-use-a-strong-random-secret")
_ADMIN_SECRET = os.getenv("ADMIN_SECRET_KEY", "")
_ALGORITHM = "HS256"
_ACCESS_TTL_MINUTES = int(os.getenv("JWT_ACCESS_TTL_MINUTES", "60"))
_REFRESH_TTL_DAYS = int(os.getenv("JWT_REFRESH_TTL_DAYS", "7"))

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
_bearer = HTTPBearer()


def hash_password(plain: str) -> str:
    return pwd_context.hash(plain)


def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)


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
