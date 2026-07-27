"""Pydantic models for the user/org service."""

from __future__ import annotations

from typing import Optional
from pydantic import BaseModel


# ── Enums ──────────────────────────────────────────────────────────────────────

class OrgRole:
    CEO = "CEO"
    ADMIN = "ADMIN"
    MANAGER = "MANAGER"
    MEMBER = "MEMBER"
    ASSOCIATE = "ASSOCIATE"
    all = ("CEO", "ADMIN", "MANAGER", "MEMBER", "ASSOCIATE")
    managers_and_above = ("CEO", "ADMIN", "MANAGER")
    admin_and_above = ("CEO", "ADMIN")


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
    job_title: Optional[str] = None
    department: Optional[str] = None
    phone: Optional[str] = None
    bio: Optional[str] = None
    avatar_url: Optional[str] = None


class LoginRequest(BaseModel):
    email: str
    password: str


class GoogleExchangeRequest(BaseModel):
    supabase_token: str


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
    job_title: Optional[str] = None
    department: Optional[str] = None
    phone: Optional[str] = None
    bio: Optional[str] = None
    avatar_url: Optional[str] = None


class UserUpdate(BaseModel):
    name: Optional[str] = None
    email: Optional[str] = None
    is_active: Optional[bool] = None
    role: Optional[str] = None
    job_title: Optional[str] = None
    department: Optional[str] = None
    phone: Optional[str] = None
    bio: Optional[str] = None
    avatar_url: Optional[str] = None


class MeetingStats(BaseModel):
    total_meetings: int
    total_minutes: int
    last_meeting_at: Optional[str] = None
    meetings_this_week: int = 0
    meetings_this_month: int = 0
    avg_meeting_duration_mins: float = 0.0


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
    description: Optional[str] = None


class TeamOut(BaseModel):
    id: str
    name: str
    org_id: str
    created_at: str
    member_count: int = 0
    bot_assigned: bool = False
    description: Optional[str] = None
    # The requesting caller's role *within this team* ("MANAGER"/"MEMBER"/…), or
    # their org role ("CEO"/"ADMIN") if they aren't a member. Lets the UI decide
    # which management controls to show without an extra round-trip.
    viewer_team_role: Optional[str] = None


class TeamUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None


# ── Team member models ─────────────────────────────────────────────────────────

class AddMemberRequest(BaseModel):
    user_id: str
    role: str = TeamRole.MEMBER


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


# ── Invite models ─────────────────────────────────────────────────────────────

class TeamInviteCreate(BaseModel):
    email: str
    role: str = TeamRole.MEMBER


class TeamInviteOut(BaseModel):
    id: str
    team_id: str
    email: str
    role: str
    code: str
    status: str
    inviter_id: Optional[str]
    created_at: str
    expires_at: str
    team_name: Optional[str] = None
    user_exists: bool = False
    # Whether the invite email actually reached the mail relay, and why not (or
    # what may still go wrong) if applicable. The invite is valid regardless —
    # the code works either way — so this drives what the UI tells the inviter
    # rather than whether the request succeeded.
    email_sent: bool = False
    email_error: Optional[str] = None


class AcceptInviteRequest(BaseModel):
    """Accepting an invite requires proving control of the invited address.

    `supabase_token` is the caller's Google/Supabase access token; the server
    verifies it and requires the resulting email to match the invite. Without
    that, possession of the 6-character code alone was enough to create an
    account under someone else's email address.

    `name` is only a fallback for the display name when Google doesn't supply one.
    There is deliberately no password field: invited users authenticate through
    Google, and a password set by whoever opened the link would reintroduce the
    same takeover.
    """
    supabase_token: str
    name: Optional[str] = None


# ── Hierarchy models ───────────────────────────────────────────────────────────

class HierarchyNode(BaseModel):
    user: UserOut
    direct_reports: list["HierarchyNode"] = []


HierarchyNode.model_rebuild()


class PersonalHierarchy(BaseModel):
    """A user-centric reporting view: the chain of managers above them (nearest
    first) and the subtree of people who report to them."""
    me: UserOut
    manager_chain: list[UserOut] = []       # direct manager → … → top
    reports: list[HierarchyNode] = []        # direct-report subtrees
