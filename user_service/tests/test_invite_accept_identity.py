"""Accepting an invite requires proving control of the invited email address.

Regression: the 6-character invite code was the ONLY thing required to accept.
Anyone holding (or guessing) a code could accept an invite addressed to someone
else — and when that person had no account yet, the accept CREATED an org_users
row under their email and returned a valid session for it. That is account
takeover of an address the caller does not control.

Accept now requires a Supabase (Google) access token whose verified email
matches the invited address.

Supabase Auth and the DB are faked — no network, no DB.

Run from the Confluence/ directory:
    python -m pytest user_service/tests/test_invite_accept_identity.py
"""
from __future__ import annotations

import datetime as dt

import pytest
from fastapi.testclient import TestClient

from user_service.main import app
from user_service import auth as auth_module
from user_service.routes import invites as invite_routes
from user_service import database as db


class _FakeResp:
    status_code = 200
    ok = True

    def __init__(self, payload: dict):
        self._payload = payload

    def json(self) -> dict:
        return self._payload


FUTURE = (dt.datetime.now(dt.timezone.utc) + dt.timedelta(hours=1)).isoformat()

INVITE = {
    "id": "inv1", "team_id": "t1", "email": "victim@company.com",
    "role": "MEMBER", "code": "A3F0C2", "status": "pending",
    "inviter_id": "boss", "created_at": "2026-01-01T00:00:00Z",
    "expires_at": FUTURE,
}
TEAM = {"id": "t1", "name": "Platform", "org_id": "org1"}

# Nobody exists yet — this is the takeover-prone case.
USERS: list[dict] = []
INSERTED: list[dict] = []


def _fake_select(table, filters=None, limit=None, columns=None):
    filters = filters or {}
    if table == "org_team_invitations":
        code = (filters.get("code") or "").replace("eq.", "")
        return [INVITE] if code == INVITE["code"] else []
    if table == "org_teams":
        return [TEAM] if (filters.get("id") or "").replace("eq.", "") == "t1" else []
    if table == "org_users":
        ef = filters.get("email", "")
        if ef.startswith("eq."):
            return [u for u in USERS if u["email"] == ef[len("eq."):]]
        if ef.startswith("ilike."):
            pat = ef[len("ilike."):].replace("\\", "")
            return [u for u in USERS if u["email"].lower() == pat.lower()]
        idf = (filters.get("id") or "").replace("eq.", "")
        return [u for u in USERS if u["id"] == idf] if idf else list(USERS)
    return []


def _fake_select_one(table, filters, columns=None):
    rows = _fake_select(table, filters, limit=1)
    return rows[0] if rows else None


def _fake_insert(table, data):
    INSERTED.append({"table": table, **data})
    row = {"id": "new-user", **data}
    if table == "org_users":
        USERS.append(row)
    return row


@pytest.fixture(autouse=True)
def _patch(monkeypatch):
    monkeypatch.setenv("SUPABASE_URL", "https://x.supabase.co")
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "svc")
    USERS.clear()
    INSERTED.clear()
    monkeypatch.setattr(db, "select", _fake_select)
    monkeypatch.setattr(db, "select_one", _fake_select_one)
    monkeypatch.setattr(invite_routes, "select_one", _fake_select_one)
    monkeypatch.setattr(invite_routes, "insert", _fake_insert)
    monkeypatch.setattr(invite_routes, "update", lambda *a, **k: None)
    monkeypatch.setattr(invite_routes, "_wire_hierarchy", lambda *a, **k: None)
    yield


def _signed_in_as(monkeypatch, email: str, full_name: str | None = None):
    monkeypatch.setattr(
        auth_module._requests, "get",
        lambda *a, **k: _FakeResp({
            "id": "sb-uid", "email": email,
            "user_metadata": {"full_name": full_name} if full_name else {},
        }),
    )


def _accept(payload: dict):
    return TestClient(app).post(f"/invites/{INVITE['code']}/accept", json=payload)


# ── The takeover this fix closes ───────────────────────────────────────────────

def test_attacker_with_the_code_cannot_claim_someone_elses_email(monkeypatch):
    _signed_in_as(monkeypatch, "attacker@evil.com")
    r = _accept({"supabase_token": "attacker-token", "name": "Not Victim"})
    assert r.status_code == 403
    assert "victim@company.com" in r.json()["detail"]
    # Critically: no account was created under the victim's address.
    assert not [i for i in INSERTED if i["table"] == "org_users"]


def test_code_alone_is_no_longer_enough(monkeypatch):
    _signed_in_as(monkeypatch, "victim@company.com")
    r = _accept({"name": "Victim"})  # no supabase_token
    assert r.status_code == 422  # rejected by the request model


# ── The legitimate path still works ────────────────────────────────────────────

def test_invited_user_accepting_with_matching_google_account(monkeypatch):
    _signed_in_as(monkeypatch, "victim@company.com", full_name="Vic Tim")
    r = _accept({"supabase_token": "victim-token"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["team_name"] == "Platform"
    assert body["role"] == "MEMBER"
    assert body["access_token"]

    created = [i for i in INSERTED if i["table"] == "org_users"]
    assert len(created) == 1
    # Name comes from Google, password is never set, supabase id is linked.
    assert created[0]["name"] == "Vic Tim"
    assert created[0]["password_hash"] is None
    assert created[0]["supabase_user_id"] == "sb-uid"


def test_email_match_is_case_and_whitespace_insensitive(monkeypatch):
    _signed_in_as(monkeypatch, "  Victim@Company.COM  ", full_name="Vic Tim")
    r = _accept({"supabase_token": "victim-token"})
    assert r.status_code == 200, r.text


def test_verified_email_is_stored_not_the_invited_string(monkeypatch):
    # Google is the source of truth for casing, so google-exchange finds the row.
    _signed_in_as(monkeypatch, "Victim@Company.com", full_name="Vic Tim")
    r = _accept({"supabase_token": "victim-token"})
    assert r.status_code == 200, r.text
    created = [i for i in INSERTED if i["table"] == "org_users"][0]
    assert created["email"] == "Victim@Company.com"


def test_body_name_is_used_when_google_supplies_none(monkeypatch):
    _signed_in_as(monkeypatch, "victim@company.com")  # no full_name
    r = _accept({"supabase_token": "victim-token", "name": "Fallback Name"})
    assert r.status_code == 200, r.text
    created = [i for i in INSERTED if i["table"] == "org_users"][0]
    assert created["name"] == "Fallback Name"


def test_missing_name_everywhere_is_rejected(monkeypatch):
    _signed_in_as(monkeypatch, "victim@company.com")  # no full_name, no body.name
    r = _accept({"supabase_token": "victim-token"})
    assert r.status_code == 400
