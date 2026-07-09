"""Tests for team routes (routes/teams.py) after the RBAC refactor.

Covers the two behaviours the visibility/permission change introduced:
  - TeamOut.viewer_team_role reflects the caller's role within each team
    (their team role, or their org role for a non-member ADMIN/CEO).
  - Team-scoped writes go through rbac.can_manage_team: a plain member is
    denied add/remove; the team's MANAGER (regardless of org role) and
    ADMIN/CEO are allowed.

Supabase and the authenticated user are faked — no network or DB.

Run from the Confluence/ directory:
    python -m pytest user_service/tests/test_teams.py
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from user_service.main import app
from user_service.auth import get_current_user
from user_service import rbac
from user_service.routes import teams as t


# ── Fake data ────────────────────────────────────────────────────────────────

TEAMS = [
    {"id": "tA", "name": "Engineering", "org_id": "org1",
     "created_at": "2026-01-01T00:00:00Z", "description": "Eng"},
]

MEMBERSHIPS = [
    {"team_id": "tA", "user_id": "u-mem", "role": "MEMBER", "joined_at": "2026-01-02T00:00:00Z"},
    {"team_id": "tA", "user_id": "u-mgr", "role": "MANAGER", "joined_at": "2026-01-02T00:00:00Z"},
]


def _members_matching(filters: dict):
    tid = (filters.get("team_id") or "").replace("eq.", "")
    uid = (filters.get("user_id") or "").replace("eq.", "")
    role = filters.get("role")
    out = []
    for m in MEMBERSHIPS:
        if tid and m["team_id"] != tid:
            continue
        if uid and m["user_id"] != uid:
            continue
        if role and m["role"] != role.replace("eq.", ""):
            continue
        out.append(m)
    return out


def _fake_select(table, filters=None, limit=None, columns=None):
    filters = filters or {}
    if table == "org_teams":
        org = (filters.get("org_id") or "").replace("eq.", "")
        return [x for x in TEAMS if x["org_id"] == org]
    if table == "org_team_members":
        return _members_matching(filters)
    return []


def _fake_select_one(table, filters, columns=None):
    if table == "org_teams":
        tid = (filters.get("id") or "").replace("eq.", "")
        return next((x for x in TEAMS if x["id"] == tid), None)
    if table == "org_team_members":
        rows = _members_matching(filters)
        return rows[0] if rows else None
    if table == "org_team_bots":
        return None
    return None


def _fake_insert(table, data):
    return {**data, "joined_at": "2026-06-01T00:00:00Z", "id": "new-id"}


def _fake_delete(table, filters):
    return None


@pytest.fixture(autouse=True)
def _patch_db(monkeypatch):
    monkeypatch.setattr(t, "select", _fake_select)
    monkeypatch.setattr(t, "select_one", _fake_select_one)
    monkeypatch.setattr(t, "insert", _fake_insert)
    monkeypatch.setattr(t, "delete", _fake_delete)
    # Hierarchy wiring is exercised elsewhere; isolate the permission behaviour.
    monkeypatch.setattr(t, "_wire_hierarchy", lambda *a, **k: None)
    # can_manage_team → is_team_manager reads via the rbac module's own select_one.
    monkeypatch.setattr(rbac, "select_one", _fake_select_one)
    yield
    app.dependency_overrides.clear()


def _as(role: str, sub: str, org_id: str = "org1"):
    app.dependency_overrides[get_current_user] = lambda: {
        "sub": sub, "role": role, "org_id": org_id,
    }
    return TestClient(app)


# ── viewer_team_role ─────────────────────────────────────────────────────────

def test_member_list_reports_team_role():
    client = _as("MEMBER", "u-mem")
    r = client.get("/teams")
    assert r.status_code == 200
    body = r.json()
    assert [x["id"] for x in body] == ["tA"]
    assert body[0]["viewer_team_role"] == "MEMBER"
    assert body[0]["member_count"] == 2


def test_manager_list_reports_manager_role():
    client = _as("MEMBER", "u-mgr")  # org MEMBER, but team MANAGER
    body = client.get("/teams").json()
    assert body[0]["viewer_team_role"] == "MANAGER"


def test_admin_nonmember_list_reports_org_role():
    client = _as("ADMIN", "u-admin")  # not a member of tA
    body = client.get("/teams").json()
    assert body[0]["viewer_team_role"] == "ADMIN"


# ── can_manage_team enforcement ──────────────────────────────────────────────

def test_plain_member_cannot_add_member():
    client = _as("MEMBER", "u-mem")
    r = client.post("/teams/tA/members", json={"user_id": "u-new", "role": "MEMBER"})
    assert r.status_code == 403


def test_team_manager_can_add_member():
    client = _as("MEMBER", "u-mgr")  # team MANAGER
    r = client.post("/teams/tA/members", json={"user_id": "u-new", "role": "MEMBER"})
    assert r.status_code == 201
    assert r.json()["user_id"] == "u-new"


def test_admin_can_remove_member():
    client = _as("ADMIN", "u-admin")
    assert client.delete("/teams/tA/members/u-mem").status_code == 204


def test_plain_member_cannot_remove_member():
    client = _as("MEMBER", "u-mem")
    assert client.delete("/teams/tA/members/u-mgr").status_code == 403
