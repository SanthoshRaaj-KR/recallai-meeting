"""Pydantic models for the user/org service."""

from __future__ import annotations

from typing import Optional
from pydantic import BaseModel


# ── Enums ──────────────────────────────────────────────────────────────────────

class OrgRole:
    CEO = "CEO"
    MANAGER = "MANAGER"
    MEMBER = "MEMBER"
    ASSOCIATE = "ASSOCIATE"
    all = ("CEO", "MANAGER", "MEMBER", "ASSOCIATE")
    managers_and_above = ("CEO", "MANAGER")


class TeamRole:
    MANAGER = "MANAGER"
    MEMBER = "MEMBER"
    ASSOCIATE = "ASSOCIATE"


# ── Auth models ────────────────────────────────────────────────────────────────

class RegisterRequest(BaseModel):
    email: str
    name: str
    password: str
    role: str = OrgRole.MEMBER
    org_id: Optional[str] = None


class LoginRequest(BaseModel):
    email: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user_id: str
    role: str
    org_id: Optional[str]


class RefreshRequest(BaseModel):
    refresh_token: str


# ── User models ────────────────────────────────────────────────────────────────

class UserOut(BaseModel):
    id: str
    email: str
    name: str
    role: str
    org_id: Optional[str]
    is_active: bool
    created_at: str


class UserUpdate(BaseModel):
    name: Optional[str] = None
    email: Optional[str] = None
    is_active: Optional[bool] = None


# ── Organization models ────────────────────────────────────────────────────────

class OrgCreate(BaseModel):
    name: str


class OrgOut(BaseModel):
    id: str
    name: str
    created_at: str


# ── Team models ────────────────────────────────────────────────────────────────

class TeamCreate(BaseModel):
    name: str
    org_id: str


class TeamOut(BaseModel):
    id: str
    name: str
    org_id: str
    created_at: str
    member_count: int = 0
    bot_assigned: bool = False


class TeamUpdate(BaseModel):
    name: Optional[str] = None


# ── Team member models ─────────────────────────────────────────────────────────

class AddMemberRequest(BaseModel):
    user_id: str
    role: str = TeamRole.MEMBER
    admin_secret: str


class RemoveMemberRequest(BaseModel):
    admin_secret: str


class MemberOut(BaseModel):
    user_id: str
    team_id: str
    role: str
    joined_at: str
    user: Optional[UserOut] = None


# ── Bot models ─────────────────────────────────────────────────────────────────

class BotCreate(BaseModel):
    name: str
    config: dict = {}


class BotOut(BaseModel):
    id: str
    team_id: str
    name: str
    config: dict
    created_at: str


# ── Hierarchy models ───────────────────────────────────────────────────────────

class HierarchyNode(BaseModel):
    user: UserOut
    direct_reports: list["HierarchyNode"] = []


HierarchyNode.model_rebuild()
