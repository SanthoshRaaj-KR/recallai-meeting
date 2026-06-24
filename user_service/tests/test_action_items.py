"""Tests for the action-item workflow routes (action_items.py).

Validates RBAC + lifecycle (open -> submitted -> closed) with the Supabase layer
and the authenticated user both faked. Run from Confluence/:
    python -m pytest user_service/tests/test_action_items.py
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from user_service.main import app
from user_service.auth import get_current_user
from user_service.routes import action_items as ai


USERS = {
    "u-mem": {"id": "u-mem", "name": "Mia Member", "email": "mia@co.com"},
    "u-mgr": {"id": "u-mgr", "name": "Max Manager", "email": "max@co.com"},
    "u-adm": {"id": "u-adm", "name": "Ada Admin", "email": "ada@co.com"},
}
MEMBERS = [  # org_team_members rows for team tA
    {"team_id": "tA", "user_id": "u-mgr", "role": "MANAGER"},
    {"team_id": "tA", "user_id": "u-mem", "role": "MEMBER"},
]
TEAMS = [{"id": "tA", "name": "Eng", "org_id": "org1"}]
SESSIONS = [{"session_id": "s1", "team_id": "tA", "summary": {"title": "Sprint"}}]
ITEMS: list[dict] = []


def _reset():
    ITEMS.clear()
    ITEMS.append({
        "id": "ai1", "session_id": "s1", "team_id": "tA", "org_id": "org1",
        "description": "Ship the thing", "owner_user_id": "u-mem", "owner_name": None,
        "due": None, "status": "open", "member_note": None, "reviewed_by": None,
        "reviewed_at": None, "source": "ai",
    })


def _in_set(val: str) -> set[str]:
    return set(val[4:-1].split(",")) if val.startswith("in.(") and val[4:-1] else set()


def _fake_select(table, filters=None, limit=None):
    f = filters or {}
    if table == "meeting_action_items":
        rows = ITEMS
        if "owner_user_id" in f:
            rows = [r for r in rows if r["owner_user_id"] == f["owner_user_id"].replace("eq.", "")]
        if "session_id" in f and f["session_id"].startswith("eq."):
            rows = [r for r in rows if r["session_id"] == f["session_id"][3:]]
        if "team_id" in f and f["team_id"].startswith("in.("):
            allowed = _in_set(f["team_id"]); rows = [r for r in rows if r["team_id"] in allowed]
        if "status" in f and f["status"].startswith("eq."):
            rows = [r for r in rows if r["status"] == f["status"][3:]]
        return [dict(r) for r in rows]
    if table == "org_users":
        if "id" in f and f["id"].startswith("in.("):
            return [USERS[i] for i in _in_set(f["id"]) if i in USERS]
        return list(USERS.values())
    if table == "org_team_members":
        rows = MEMBERS
        if "team_id" in f and f["team_id"].startswith("eq."):
            rows = [m for m in rows if m["team_id"] == f["team_id"][3:]]
        if "user_id" in f and f["user_id"].startswith("eq."):
            rows = [m for m in rows if m["user_id"] == f["user_id"][3:]]
        if "role" in f and f["role"].startswith("eq."):
            rows = [m for m in rows if m["role"] == f["role"][3:]]
        return list(rows)
    if table == "org_teams":
        rows = TEAMS
        if "id" in f and f["id"].startswith("eq."):
            rows = [t for t in rows if t["id"] == f["id"][3:]]
        if "org_id" in f and f["org_id"].startswith("eq."):
            rows = [t for t in rows if t["org_id"] == f["org_id"][3:]]
        return list(rows)
    if table == "jarvis_sessions":
        sid = f.get("session_id", "")
        if sid.startswith("eq."):
            return [s for s in SESSIONS if s["session_id"] == sid[3:]]
        if sid.startswith("in.("):
            allowed = _in_set(sid); return [s for s in SESSIONS if s["session_id"] in allowed]
        return list(SESSIONS)
    return []


def _fake_select_one(table, filters):
    rows = _fake_select(table, filters, limit=1)
    return rows[0] if rows else None


def _fake_update(table, filters, data):
    iid = filters.get("id", "").replace("eq.", "")
    for r in ITEMS:
        if r["id"] == iid:
            r.update(data)
            return [dict(r)]
    return []


@pytest.fixture(autouse=True)
def _patch(monkeypatch):
    _reset()
    monkeypatch.setattr(ai, "select", _fake_select)
    monkeypatch.setattr(ai, "select_one", _fake_select_one)
    monkeypatch.setattr(ai, "update", _fake_update)
    yield
    app.dependency_overrides.clear()


def _as(uid, role):
    app.dependency_overrides[get_current_user] = lambda: {"sub": uid, "role": role, "org_id": "org1"}
    return TestClient(app)


def test_my_action_items_returns_only_mine():
    r = _as("u-mem", "MEMBER").get("/me/action-items")
    assert r.status_code == 200
    body = r.json()
    assert [i["id"] for i in body] == ["ai1"]
    assert body[0]["owner_display"] == "Mia Member"
    assert body[0]["meeting_title"] == "Sprint"


def test_owner_can_submit():
    r = _as("u-mem", "MEMBER").patch("/action-items/ai1/submit", json={"member_note": "done"})
    assert r.status_code == 200 and r.json()["status"] == "submitted"


def test_non_owner_cannot_submit():
    assert _as("u-adm", "ADMIN").patch("/action-items/ai1/submit", json={}).status_code == 403


def test_manager_can_close_but_member_cannot():
    ITEMS[0]["status"] = "submitted"
    assert _as("u-mgr", "MANAGER").patch("/action-items/ai1/close").status_code == 200
    assert ITEMS[0]["status"] == "closed" and ITEMS[0]["reviewed_by"] == "u-mgr"
    ITEMS[0]["status"] = "submitted"
    assert _as("u-mem", "MEMBER").patch("/action-items/ai1/close").status_code == 403


def test_manager_can_cancel_member_cannot():
    assert _as("u-mgr", "MANAGER").patch("/action-items/ai1/cancel").status_code == 200
    assert ITEMS[0]["status"] == "cancelled" and ITEMS[0]["reviewed_by"] == "u-mgr"
    ITEMS[0]["status"] = "open"
    assert _as("u-mem", "MEMBER").patch("/action-items/ai1/cancel").status_code == 403


def test_meeting_items_members_only():
    client = _as("u-mem", "MEMBER")
    r = client.get("/meetings/s1/action-items")
    assert r.status_code == 200
    assert r.json()["can_manage"] is False
    assert len(r.json()["items"]) == 1
    # an outsider (not a team member, not admin) is rejected
    assert _as("u-out", "MEMBER").get("/meetings/s1/action-items").status_code == 403


def test_admin_meeting_items_can_manage():
    r = _as("u-adm", "ADMIN").get("/meetings/s1/action-items")
    assert r.status_code == 200 and r.json()["can_manage"] is True
