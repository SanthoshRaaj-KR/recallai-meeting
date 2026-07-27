"""Removing someone from ONE team must not erase them from the org chart.

Regression: remove_member deleted every org_reporting_hierarchy row where the
user was the descendant — including their depth-0 self-loop — and did so without
any team scoping. A user on two teams who was removed from one lost their entire
position in the hierarchy, self-loop included, so every hierarchy query stopped
seeing them at all.

Run from the Confluence/ directory:
    python -m pytest user_service/tests/test_remove_member_hierarchy.py
"""
from __future__ import annotations

import pytest

from user_service.routes import teams as t


# (ancestor_id, descendant_id, depth)
HIERARCHY: list[dict] = []
MEMBERSHIPS: list[dict] = []
WIRED: list[tuple] = []


def _fake_select(table, filters=None, limit=None, columns=None):
    filters = filters or {}
    if table == "org_team_members":
        uid = (filters.get("user_id") or "").replace("eq.", "")
        tid = (filters.get("team_id") or "").replace("eq.", "")
        rows = MEMBERSHIPS
        if uid:
            rows = [m for m in rows if m["user_id"] == uid]
        if tid:
            rows = [m for m in rows if m["team_id"] == tid]
        return rows
    if table == "org_reporting_hierarchy":
        return list(HIERARCHY)
    return []


def _fake_delete(table, filters):
    """Mimic PostgREST filter semantics for the columns this code uses."""
    if table != "org_reporting_hierarchy":
        return
    anc = filters.get("ancestor_id")
    desc = filters.get("descendant_id")
    depth = filters.get("depth")

    def matches(row: dict) -> bool:
        if anc is not None and row["ancestor_id"] != anc.replace("eq.", ""):
            return False
        if desc is not None and row["descendant_id"] != desc.replace("eq.", ""):
            return False
        if depth is not None:
            if depth.startswith("gt."):
                if not row["depth"] > int(depth[3:]):
                    return False
            elif depth.startswith("eq.") and row["depth"] != int(depth[3:]):
                return False
        return True

    HIERARCHY[:] = [r for r in HIERARCHY if not matches(r)]


@pytest.fixture(autouse=True)
def _patch(monkeypatch):
    HIERARCHY.clear()
    MEMBERSHIPS.clear()
    WIRED.clear()
    monkeypatch.setattr(t, "select", _fake_select)
    monkeypatch.setattr(t, "delete", _fake_delete)
    monkeypatch.setattr(
        t, "_wire_hierarchy",
        lambda team_id, user_id, role: WIRED.append((team_id, user_id, role)),
    )
    yield


def _h(anc, desc, depth):
    return {"ancestor_id": anc, "descendant_id": desc, "depth": depth}


def test_self_loop_survives_removal():
    """The depth-0 row is the user's own anchor — deleting it removes them from
    the chart entirely instead of merely detaching them from a manager."""
    HIERARCHY.extend([
        _h("u1", "u1", 0),      # self-loop
        _h("mgrA", "u1", 1),    # reports to team A's manager
        _h("ceo", "u1", 2),
    ])
    t._unwire_hierarchy("u1")
    assert _h("u1", "u1", 0) in HIERARCHY
    assert _h("mgrA", "u1", 1) not in HIERARCHY
    assert _h("ceo", "u1", 2) not in HIERARCHY


def test_user_on_a_second_team_is_rewired_under_that_team():
    MEMBERSHIPS.append({"team_id": "tB", "user_id": "u1", "role": "MEMBER"})
    HIERARCHY.extend([_h("u1", "u1", 0), _h("mgrA", "u1", 1)])

    t._unwire_hierarchy("u1")

    # Detached from team A's manager, then re-attached under team B.
    assert _h("mgrA", "u1", 1) not in HIERARCHY
    assert WIRED == [("tB", "u1", "MEMBER")]


def test_user_with_no_remaining_team_is_left_detached_but_present():
    HIERARCHY.extend([_h("u1", "u1", 0), _h("mgrA", "u1", 1)])
    t._unwire_hierarchy("u1")
    assert WIRED == []
    assert HIERARCHY == [_h("u1", "u1", 0)]


def test_their_own_reports_are_detached():
    """A removed manager stops managing people."""
    HIERARCHY.extend([
        _h("u1", "u1", 0),
        _h("u1", "report1", 1),
        _h("u1", "report2", 1),
    ])
    t._unwire_hierarchy("u1")
    assert _h("u1", "report1", 1) not in HIERARCHY
    assert _h("u1", "report2", 1) not in HIERARCHY
    assert _h("u1", "u1", 0) in HIERARCHY


def test_other_peoples_edges_are_untouched():
    HIERARCHY.extend([
        _h("u1", "u1", 0), _h("mgrA", "u1", 1),
        _h("u2", "u2", 0), _h("mgrA", "u2", 1),   # a colleague
    ])
    t._unwire_hierarchy("u1")
    assert _h("u2", "u2", 0) in HIERARCHY
    assert _h("mgrA", "u2", 1) in HIERARCHY


def test_rewire_uses_the_role_held_on_the_remaining_team():
    MEMBERSHIPS.append({"team_id": "tB", "user_id": "u1", "role": "MANAGER"})
    HIERARCHY.extend([_h("u1", "u1", 0), _h("mgrA", "u1", 1)])
    t._unwire_hierarchy("u1")
    assert WIRED == [("tB", "u1", "MANAGER")]
