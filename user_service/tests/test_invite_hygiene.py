"""Invite creation hygiene: case-insensitive user lookup, no duplicate live
invites, and one source of truth for the expiry.

Run from the Confluence/ directory:
    python -m pytest user_service/tests/test_invite_hygiene.py
"""
from __future__ import annotations

import datetime

import pytest
from fastapi.testclient import TestClient

from user_service.main import app
from user_service.auth import get_current_user
from user_service import rbac
from user_service.routes import teams as t


TEAM = {"id": "tA", "name": "Engineering", "org_id": "org1",
        "created_at": "2026-01-01T00:00:00Z", "description": None}

# Stored with mixed case, as an earlier manual add would have left it.
USERS = [{"id": "u-jane", "email": "Jane.Doe@Company.com", "name": "Jane Doe",
          "role": "MEMBER", "org_id": "org1", "is_active": True}]

INVITES: list[dict] = []
DELETED: list[dict] = []


def _fake_select(table, filters=None, limit=None, columns=None):
    filters = filters or {}
    if table == "org_team_invitations":
        return list(INVITES)
    if table == "org_users":
        ef = filters.get("email", "")
        if ef.startswith("eq."):
            return [u for u in USERS if u["email"] == ef[len("eq."):]]
        if ef.startswith("ilike."):
            pat = ef[len("ilike."):].replace("\\", "")
            return [u for u in USERS if u["email"].lower() == pat.lower()]
        idf = (filters.get("id") or "").replace("eq.", "")
        return [u for u in USERS if u["id"] == idf] if idf else list(USERS)
    if table == "org_teams":
        return [TEAM]
    return []


def _fake_select_one(table, filters, columns=None):
    rows = _fake_select(table, filters)
    return rows[0] if rows else None


def _fake_delete(table, filters):
    DELETED.append({"table": table, **filters})
    if table == "org_team_invitations":
        code = (filters.get("code") or "").replace("eq.", "")
        if code:
            INVITES[:] = [i for i in INVITES if i["code"] != code]


def _fake_insert(table, data):
    row = {"id": "inv-new", "created_at": "2026-07-27T10:00:00Z", **data}
    if table == "org_team_invitations":
        INVITES.append(row)
    return row


@pytest.fixture(autouse=True)
def _patch(monkeypatch):
    INVITES.clear()
    DELETED.clear()
    monkeypatch.setattr(t, "select", _fake_select)
    monkeypatch.setattr(t, "select_one", _fake_select_one)
    monkeypatch.setattr(t, "insert", _fake_insert)
    monkeypatch.setattr(t, "delete", _fake_delete)
    # find_by_text_ci reaches into database.py's own globals.
    from user_service import database as db
    monkeypatch.setattr(db, "select", _fake_select)
    monkeypatch.setattr(db, "select_one", _fake_select_one)
    monkeypatch.setattr(rbac, "can_manage_team", lambda claims, team_id: True)
    monkeypatch.setattr(t, "can_manage_team", lambda claims, team_id: True)
    app.dependency_overrides[get_current_user] = lambda: {
        "sub": "u-boss", "role": "ADMIN", "org_id": "org1",
    }
    yield
    app.dependency_overrides.clear()


def _invite(email: str, role: str = "MEMBER"):
    return TestClient(app).post(f"/teams/tA/invite", json={"email": email, "role": role})


# ── #4: case-insensitive user_exists ──────────────────────────────────────────

def test_existing_user_is_recognised_despite_different_casing():
    r = _invite("jane.doe@company.com")  # stored as Jane.Doe@Company.com
    assert r.status_code == 201, r.text
    assert r.json()["user_exists"] is True


def test_genuinely_new_address_reports_user_exists_false():
    r = _invite("brand.new@company.com")
    assert r.status_code == 201, r.text
    assert r.json()["user_exists"] is False


def test_email_is_trimmed_before_storage():
    r = _invite("  spaced@company.com  ")
    assert r.status_code == 201, r.text
    assert r.json()["email"] == "spaced@company.com"


def test_blank_email_is_rejected():
    assert _invite("   ").status_code == 400


# ── #7: one live invite per person per team ───────────────────────────────────

def test_reinviting_supersedes_the_previous_pending_code():
    first = _invite("jane.doe@company.com")
    assert first.status_code == 201
    first_code = first.json()["code"]

    second = _invite("jane.doe@company.com")
    assert second.status_code == 201
    second_code = second.json()["code"]

    assert first_code != second_code
    # The old code was deleted, so only one way in remains.
    assert any(d.get("code") == f"eq.{first_code}" for d in DELETED)
    assert [i["code"] for i in INVITES] == [second_code]


def test_superseding_matches_case_insensitively():
    _invite("Jane.Doe@Company.com")
    _invite("jane.doe@company.com")
    assert len(INVITES) == 1


def test_a_different_address_keeps_its_own_invite():
    _invite("jane.doe@company.com")
    _invite("someone.else@company.com")
    assert len(INVITES) == 2


# ── #6: one source of truth for expiry ────────────────────────────────────────

def test_expiry_comes_from_INVITE_TTL():
    assert t.INVITE_TTL == datetime.timedelta(hours=1)
    r = _invite("jane.doe@company.com")
    expires = datetime.datetime.fromisoformat(r.json()["expires_at"])
    delta = expires - datetime.datetime.now(datetime.timezone.utc)
    # Within a minute of the configured TTL.
    assert abs(delta - t.INVITE_TTL) < datetime.timedelta(minutes=1)


def test_migration_realigns_the_column_default_to_match():
    from pathlib import Path
    sql = Path(__file__).resolve().parents[1] / "migrations" / "014_invite_expiry_and_dedupe.sql"
    body = sql.read_text(encoding="utf-8")
    assert "interval '1 hour'" in body
    assert "uq_pending_invite_per_team_email" in body
