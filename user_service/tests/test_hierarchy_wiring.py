"""Reporting-hierarchy wiring: everybody in a team reports to the manager.

Covers the fix for members added to a team *before* it had a manager (previously
left reporting to nobody):
  - Adding a MANAGER retro-wires every existing MEMBER/ASSOCIATE under them.
  - A member added *after* the manager is wired under the manager as before.

DB is faked and closure-table inserts are captured — no network, no DB.

Run from the Confluence/ directory:
    python -m pytest user_service/tests/test_hierarchy_wiring.py
"""
from __future__ import annotations

import pytest

from user_service.routes import teams as t


@pytest.fixture
def env(monkeypatch):
    members = [
        {"team_id": "tA", "user_id": "u-mem1", "role": "MEMBER"},
        {"team_id": "tA", "user_id": "u-mem2", "role": "ASSOCIATE"},
    ]
    inserts: list[tuple] = []

    def fake_select(table, filters=None, limit=None, columns=None):
        filters = filters or {}
        if table == "org_team_members":
            tid = filters.get("team_id", "").replace("eq.", "")
            rows = [m for m in members if m["team_id"] == tid]
            role = filters.get("role", "")
            if role.startswith("eq."):
                rows = [m for m in rows if m["role"] == role[len("eq."):]]
            return rows
        if table == "org_reporting_hierarchy":
            desc = filters.get("descendant_id", "").replace("eq.", "")
            # The manager already rolls up to the CEO.
            return [{"ancestor_id": "ceo", "descendant_id": "u-mgr", "depth": 1}] if desc == "u-mgr" else []
        return []

    def fake_select_one(table, filters, columns=None):
        if table == "org_users" and filters.get("role") == "eq.CEO":
            return {"id": "ceo"}
        if table == "org_team_members":
            rows = fake_select(table, filters)
            return rows[0] if rows else None
        return None

    def fake_insert(table, data):
        if table == "org_reporting_hierarchy":
            inserts.append((data["ancestor_id"], data["descendant_id"], data["depth"]))
        return data

    monkeypatch.setattr(t, "select", fake_select)
    monkeypatch.setattr(t, "select_one", fake_select_one)
    monkeypatch.setattr(t, "insert", fake_insert)
    return members, inserts


def test_adding_manager_adopts_existing_members(env):
    _, inserts = env
    t._wire_hierarchy("tA", "u-mgr", "MANAGER")
    # every pre-existing member now reports to the manager …
    assert ("u-mgr", "u-mem1", 1) in inserts
    assert ("u-mgr", "u-mem2", 1) in inserts
    # … and rolls up to the CEO one level further out
    assert ("ceo", "u-mem1", 2) in inserts
    assert ("ceo", "u-mem2", 2) in inserts


def test_member_added_after_manager_reports_to_manager(env):
    members, inserts = env
    members.append({"team_id": "tA", "user_id": "u-mgr", "role": "MANAGER"})
    t._wire_hierarchy("tA", "u-new", "MEMBER")
    assert ("u-mgr", "u-new", 1) in inserts
    assert ("ceo", "u-new", 2) in inserts
