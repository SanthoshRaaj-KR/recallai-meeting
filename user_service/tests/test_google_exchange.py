"""google-exchange must match org users case-insensitively.

Regression: Google/Supabase returns the sign-in email in its own casing. When it
differed from the stored ``org_users.email`` (e.g. an invite typed as
``John.Doe@Company.com``), the exact ``eq.`` lookup missed and the member — even a
manager — was wrongly told ``not_in_org`` and stranded at the login screen.

Supabase Auth and the DB are faked — no network, no DB.

Run from the Confluence/ directory:
    python -m pytest user_service/tests/test_google_exchange.py
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from user_service.main import app
from user_service import auth as auth_module
from user_service.routes import auth as auth_routes
from user_service import database as db


class _FakeResp:
    status_code = 200
    ok = True

    def __init__(self, payload: dict):
        self._payload = payload

    def json(self) -> dict:
        return self._payload


# Stored with mixed case; Google will hand us the lowercased form.
USERS = [{
    "id": "u1", "email": "John.Doe@Company.com", "name": "John Doe",
    "role": "MEMBER", "org_id": "org1", "is_active": True,
    "created_at": "2026-01-01T00:00:00Z", "password_hash": None,
    "supabase_user_id": None,
}]


def _fake_select(table, filters=None, limit=None, columns=None):
    filters = filters or {}
    if table != "org_users":
        return []
    ef = filters.get("email", "")
    if ef.startswith("eq."):
        return [u for u in USERS if u["email"] == ef[len("eq."):]]
    if ef.startswith("ilike."):
        pat = ef[len("ilike."):].replace("\\", "")
        return [u for u in USERS if u["email"].lower() == pat.lower()]
    return list(USERS)


def _fake_select_one(table, filters, columns=None):
    rows = _fake_select(table, filters, limit=1)
    return rows[0] if rows else None


@pytest.fixture(autouse=True)
def _patch(monkeypatch):
    monkeypatch.setenv("SUPABASE_URL", "https://x.supabase.co")
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "svc")
    # find_by_text_ci lives in database.py and calls its own module globals.
    monkeypatch.setattr(db, "select", _fake_select)
    monkeypatch.setattr(db, "select_one", _fake_select_one)
    monkeypatch.setattr(auth_routes, "update", lambda *a, **k: None)
    yield


def _set_google_email(monkeypatch, email: str):
    # The Supabase call lives in user_service.auth.verify_supabase_token, shared
    # by google-exchange and invite-accept.
    monkeypatch.setattr(
        auth_module._requests, "get",
        lambda *a, **k: _FakeResp({"id": "sb-uid", "email": email}),
    )


def test_matches_email_case_insensitively(monkeypatch):
    _set_google_email(monkeypatch, "john.doe@company.com")  # differs in case only
    r = TestClient(app).post("/auth/google-exchange", json={"supabase_token": "tok"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["user_id"] == "u1"
    assert body["org_id"] == "org1"


def test_trims_surrounding_whitespace(monkeypatch):
    _set_google_email(monkeypatch, "  John.Doe@Company.com  ")
    r = TestClient(app).post("/auth/google-exchange", json={"supabase_token": "tok"})
    assert r.status_code == 200, r.text
    assert r.json()["user_id"] == "u1"


def test_unknown_email_still_not_in_org(monkeypatch):
    _set_google_email(monkeypatch, "stranger@nowhere.com")
    r = TestClient(app).post("/auth/google-exchange", json={"supabase_token": "tok"})
    assert r.status_code == 404
    assert r.json()["detail"] == "not_in_org"
