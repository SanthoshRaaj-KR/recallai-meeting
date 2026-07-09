"""Tests for the personal reporting tree (GET /org/hierarchy/me).

Verifies the manager chain (nearest→top) and the direct-report subtree are built
correctly from the closure table, for any authenticated user. DB is faked.

Run from the Confluence/ directory:
    python -m pytest user_service/tests/test_hierarchy.py
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from user_service.main import app
from user_service.auth import get_current_user
from user_service.routes import hierarchy as h


# ── Fake data ────────────────────────────────────────────────────────────────
# u-ceo → u-mgr → u-mem (a 3-level chain in org1).

USERS = [
    {"id": "u-ceo", "email": "ceo@x", "name": "Cee Oh", "role": "CEO",
     "org_id": "org1", "is_active": True, "created_at": "2026-01-01T00:00:00Z"},
    {"id": "u-mgr", "email": "mgr@x", "name": "Manny Ger", "role": "MANAGER",
     "org_id": "org1", "is_active": True, "created_at": "2026-01-01T00:00:00Z"},
    {"id": "u-mem", "email": "mem@x", "name": "Mem Ber", "role": "MEMBER",
     "org_id": "org1", "is_active": True, "created_at": "2026-01-01T00:00:00Z"},
]

ROWS = [
    {"ancestor_id": "u-ceo", "descendant_id": "u-ceo", "depth": 0},
    {"ancestor_id": "u-mgr", "descendant_id": "u-mgr", "depth": 0},
    {"ancestor_id": "u-mem", "descendant_id": "u-mem", "depth": 0},
    {"ancestor_id": "u-ceo", "descendant_id": "u-mgr", "depth": 1},
    {"ancestor_id": "u-mgr", "descendant_id": "u-mem", "depth": 1},
    {"ancestor_id": "u-ceo", "descendant_id": "u-mem", "depth": 2},
]


def _fake_select(table, filters=None, limit=None, columns=None):
    filters = filters or {}
    if table == "org_users":
        org = (filters.get("org_id") or "").replace("eq.", "")
        return [u for u in USERS if u["org_id"] == org]
    if table == "org_reporting_hierarchy":
        desc = filters.get("descendant_id")
        if desc:
            d = desc.replace("eq.", "")
            rows = [r for r in ROWS if r["descendant_id"] == d and r["depth"] > 0]
            return sorted(rows, key=lambda r: r["depth"])
        return list(ROWS)
    return []


def _fake_select_one(table, filters, columns=None):
    if table == "org_users":
        uid = (filters.get("id") or "").replace("eq.", "")
        return next((u for u in USERS if u["id"] == uid), None)
    return None


@pytest.fixture(autouse=True)
def _patch_db(monkeypatch):
    monkeypatch.setattr(h, "select", _fake_select)
    monkeypatch.setattr(h, "select_one", _fake_select_one)
    yield
    app.dependency_overrides.clear()


def _as(role: str, sub: str, org_id: str = "org1"):
    app.dependency_overrides[get_current_user] = lambda: {
        "sub": sub, "role": role, "org_id": org_id,
    }
    return TestClient(app)


def test_member_sees_full_manager_chain_and_no_reports():
    client = _as("MEMBER", "u-mem")
    r = client.get("/org/hierarchy/me")
    assert r.status_code == 200
    body = r.json()
    assert body["me"]["id"] == "u-mem"
    assert [u["id"] for u in body["manager_chain"]] == ["u-mgr", "u-ceo"]  # nearest first
    assert body["reports"] == []


def test_manager_sees_reports_and_own_manager():
    client = _as("MANAGER", "u-mgr")
    body = client.get("/org/hierarchy/me").json()
    assert [u["id"] for u in body["manager_chain"]] == ["u-ceo"]
    assert [n["user"]["id"] for n in body["reports"]] == ["u-mem"]


def test_ceo_has_no_managers_and_nested_reports():
    client = _as("CEO", "u-ceo")
    body = client.get("/org/hierarchy/me").json()
    assert body["manager_chain"] == []
    # CEO → mgr → mem nested
    assert [n["user"]["id"] for n in body["reports"]] == ["u-mgr"]
    assert body["reports"][0]["direct_reports"][0]["user"]["id"] == "u-mem"
