"""Tests for team-scoped meeting oversight (routes/bots.py).

Validates: any team member can list their team's meetings; non-members and other
orgs are shut out; only a team MANAGER (or ADMIN/CEO) can see per-person in-call
time; a session that belongs to a different team is rejected on the participants
route. Supabase and the authenticated user are both faked — no network or DB.

Run from the Confluence/ directory:
    python -m pytest user_service/tests/test_team_meetings.py
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from user_service.main import app
from user_service.auth import get_current_user
from user_service import rbac
from user_service.routes import bots
from user_service.routes import admin_meetings as am


# ── Fake data ────────────────────────────────────────────────────────────────

# org1 owns team tA; org2 owns team tZ.
TEAMS = [
    {"id": "tA", "name": "Engineering", "org_id": "org1"},
    {"id": "tZ", "name": "Outsiders", "org_id": "org2"},
]

# u-mem is a MEMBER of tA; u-mgr is its MANAGER. u-admin is an org ADMIN, not a member.
MEMBERSHIPS = [
    {"team_id": "tA", "user_id": "u-mem", "role": "MEMBER"},
    {"team_id": "tA", "user_id": "u-mgr", "role": "MANAGER"},
]

SESSIONS = [
    {"session_id": "s-tA", "team_id": "tA", "status": "ended",
     "meeting_url": "u1", "started_at": "2026-06-19T09:00:00Z",
     "ended_at": "2026-06-19T09:30:00Z", "summary": {"title": "Sprint sync"},
     "changes": [1, 2], "bot_id": "b1"},
    {"session_id": "s-tZ", "team_id": "tZ", "status": "ended",
     "meeting_url": "uZ", "started_at": "2026-06-19T09:00:00Z",
     "ended_at": "2026-06-19T09:10:00Z", "summary": {}, "changes": [], "bot_id": "bZ"},
    # A live meeting for the kick tests.
    {"session_id": "s-live", "team_id": "tA", "status": "in_meeting",
     "meeting_url": "uL", "started_at": "2026-06-19T10:00:00Z", "ended_at": None,
     "summary": {"title": "Standup"}, "changes": [], "bot_id": "bL"},
]

PARTICIPANTS = [
    {"id": "p1", "session_id": "s-tA", "team_id": "tA", "recall_name": "Alice",
     "user_id": "u-mem", "is_guest": False, "match_confidence": "high",
     "joined_at": "2026-06-19T09:00:00Z", "left_at": "2026-06-19T09:20:00Z",
     "duration_mins": 20.0},
    {"id": "p2", "session_id": "s-tA", "team_id": "tA", "recall_name": "Guest",
     "user_id": None, "is_guest": True, "match_confidence": "none",
     "joined_at": "2026-06-19T09:05:00Z", "left_at": "2026-06-19T09:15:00Z",
     "duration_mins": 10.0},
]

USERS = [{"id": "u-mem", "name": "Alice Member"}, {"id": "u-mgr", "name": "Maria Manager"}]


def _match_membership(filters: dict):
    tid = (filters.get("team_id") or "").replace("eq.", "")
    uid = (filters.get("user_id") or "").replace("eq.", "")
    role = filters.get("role")
    for m in MEMBERSHIPS:
        if m["team_id"] == tid and m["user_id"] == uid:
            if role and m["role"] != role.replace("eq.", ""):
                continue
            return m
    return None


def _fake_select(table, filters=None, limit=None, columns=None):
    filters = filters or {}
    if table == "jarvis_sessions":
        tid = (filters.get("team_id") or "").replace("eq.", "")
        return [s for s in SESSIONS if s["team_id"] == tid]
    if table == "meeting_participants":
        sid = (filters.get("session_id") or "").replace("eq.", "")
        return [p for p in PARTICIPANTS if p["session_id"] == sid]
    if table == "org_users":
        raw = filters.get("id", "")
        if raw.startswith("in.("):
            allowed = set(raw[4:-1].split(",")) if raw[4:-1] else set()
            return [u for u in USERS if u["id"] in allowed]
    return []


def _fake_select_one(table, filters, columns=None):
    if table == "org_teams":
        tid = (filters.get("id") or "").replace("eq.", "")
        return next((t for t in TEAMS if t["id"] == tid), None)
    if table == "org_team_members":
        return _match_membership(filters)
    if table == "jarvis_sessions":
        sid = (filters.get("session_id") or "").replace("eq.", "")
        return next((s for s in SESSIONS if s["session_id"] == sid), None)
    return None


@pytest.fixture(autouse=True)
def _patch_db(monkeypatch):
    monkeypatch.setattr(bots, "select", _fake_select)
    monkeypatch.setattr(bots, "select_one", _fake_select_one)
    # can_manage_team → is_team_manager reads via the rbac module's own select_one.
    monkeypatch.setattr(rbac, "select_one", _fake_select_one)
    yield
    app.dependency_overrides.clear()


def _as(role: str, sub: str, org_id: str = "org1"):
    app.dependency_overrides[get_current_user] = lambda: {
        "sub": sub, "role": role, "org_id": org_id,
    }
    return TestClient(app)


# ── List: any member of the team ─────────────────────────────────────────────

def test_member_can_list_team_meetings():
    client = _as("MEMBER", "u-mem")
    r = client.get("/teams/tA/meetings")
    assert r.status_code == 200
    body = r.json()
    assert {m["session_id"] for m in body} == {"s-tA", "s-live"}
    sprint = next(m for m in body if m["session_id"] == "s-tA")
    assert sprint["duration_mins"] == 30.0
    assert sprint["change_count"] == 2


def test_non_member_forbidden_on_list():
    client = _as("MEMBER", "u-outsider")  # in org1 but not a member of tA
    assert client.get("/teams/tA/meetings").status_code == 403


def test_foreign_org_team_forbidden_on_list():
    client = _as("ADMIN", "u-admin", org_id="org1")  # tZ is org2
    assert client.get("/teams/tZ/meetings").status_code == 403


# ── Participants: team manager / admin only ──────────────────────────────────

def test_manager_can_view_participants():
    client = _as("MEMBER", "u-mgr")  # org role irrelevant; team MANAGER of tA
    r = client.get("/teams/tA/meetings/s-tA/participants")
    assert r.status_code == 200
    rows = r.json()
    assert len(rows) == 2
    alice = next(x for x in rows if x["user_id"] == "u-mem")
    assert alice["name"] == "Alice Member"      # resolved from org_users
    assert alice["duration_mins"] == 20.0
    guest = next(x for x in rows if x["is_guest"])
    assert guest["name"] == "Guest"             # falls back to recall_name


def test_admin_can_view_participants():
    client = _as("ADMIN", "u-admin")  # not a team member, but org ADMIN
    assert client.get("/teams/tA/meetings/s-tA/participants").status_code == 200


def test_plain_member_forbidden_on_participants():
    client = _as("MEMBER", "u-mem")  # member of tA, but not its manager
    assert client.get("/teams/tA/meetings/s-tA/participants").status_code == 403


def test_participants_wrong_team_rejected():
    client = _as("MEMBER", "u-mgr")  # manager of tA asking for a tZ session via tA
    assert client.get("/teams/tA/meetings/s-tZ/participants").status_code == 403


# ── Kick: team manager / admin only ──────────────────────────────────────────

class _StopResp:
    ok = True
    status_code = 200

    def json(self):
        return {"status": "ended", "session_id": "s-live"}


def test_manager_can_kick(monkeypatch):
    monkeypatch.setattr(am, "_BOT_SERVICE_URL", "http://bot.test")
    captured = {}

    def _fake_post(url, headers=None, timeout=None):
        captured["url"] = url
        return _StopResp()

    monkeypatch.setattr(am.requests, "post", _fake_post)
    client = _as("MEMBER", "u-mgr")  # team MANAGER
    r = client.post("/teams/tA/meetings/s-live/kick")
    assert r.status_code == 200
    assert r.json()["status"] == "ended"
    assert captured["url"] == "http://bot.test/sessions/s-live/bot/stop"


def test_plain_member_cannot_kick():
    client = _as("MEMBER", "u-mem")  # member of tA, not its manager
    assert client.post("/teams/tA/meetings/s-live/kick").status_code == 403


def test_kick_wrong_team_rejected():
    client = _as("MEMBER", "u-mgr")  # tZ session via tA
    assert client.post("/teams/tA/meetings/s-tZ/kick").status_code == 403


def test_kick_already_ended_conflict():
    client = _as("ADMIN", "u-admin")
    assert client.post("/teams/tA/meetings/s-tA/kick").status_code == 409  # s-tA is ended
