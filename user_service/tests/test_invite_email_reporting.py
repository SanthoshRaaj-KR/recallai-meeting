"""An invite must never report success when no mail was sent.

Regression: send_invite_email failures were swallowed into a log line and the
route still returned 201, so the UI said "Invite sent!" whether or not anything
reached the mail relay. Admins waited on an email that was never coming.

Run from the Confluence/ directory:
    python -m pytest user_service/tests/test_invite_email_reporting.py
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from user_service.main import app
from user_service.auth import get_current_user
from user_service import email as email_mod
from user_service.routes import teams as t


TEAM = {"id": "tA", "name": "Sales", "org_id": "org1",
        "created_at": "2026-01-01T00:00:00Z", "description": None}
USERS = [{"id": "u-boss", "email": "boss@company.com", "name": "The Boss",
          "role": "ADMIN", "org_id": "org1", "is_active": True}]


def _fake_select(table, filters=None, limit=None, columns=None):
    filters = filters or {}
    if table == "org_teams":
        return [TEAM]
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
    rows = _fake_select(table, filters)
    return rows[0] if rows else None


@pytest.fixture(autouse=True)
def _patch(monkeypatch):
    monkeypatch.setattr(t, "select", _fake_select)
    monkeypatch.setattr(t, "select_one", _fake_select_one)
    monkeypatch.setattr(t, "delete", lambda *a, **k: None)
    monkeypatch.setattr(
        t, "insert",
        lambda table, data: {"id": "inv1", "created_at": "2026-07-27T10:00:00Z", **data},
    )
    from user_service import database as db
    monkeypatch.setattr(db, "select", _fake_select)
    monkeypatch.setattr(db, "select_one", _fake_select_one)
    monkeypatch.setattr(t, "can_manage_team", lambda claims, team_id: True)
    app.dependency_overrides[get_current_user] = lambda: {
        "sub": "u-boss", "role": "ADMIN", "org_id": "org1",
    }
    yield
    app.dependency_overrides.clear()


def _invite(role: str = "MANAGER"):
    return TestClient(app).post(
        "/teams/tA/invite", json={"email": "newmanager@company.com", "role": role},
    )


def test_delivery_failure_is_reported_not_swallowed(monkeypatch):
    monkeypatch.setattr(
        email_mod, "send_invite_email",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("Email delivery failed: 550 sender rejected")),
    )
    r = _invite()
    assert r.status_code == 201, r.text  # invite is still valid
    body = r.json()
    assert body["email_sent"] is False
    assert "550 sender rejected" in body["email_error"]
    assert body["code"]  # the admin can still share it


def test_unconfigured_smtp_is_reported(monkeypatch):
    monkeypatch.setattr(email_mod, "send_invite_email", lambda *a, **k: False)
    r = _invite()
    assert r.status_code == 201
    body = r.json()
    assert body["email_sent"] is False
    assert "not configured" in body["email_error"].lower()


def test_successful_send_reports_sent(monkeypatch):
    monkeypatch.setattr(email_mod, "send_invite_email", lambda *a, **k: True)
    monkeypatch.setattr(email_mod, "sender_domain_warning", lambda: None)
    r = _invite()
    assert r.status_code == 201
    body = r.json()
    assert body["email_sent"] is True
    assert body["email_error"] is None


def test_sent_but_undeliverable_sender_carries_a_warning(monkeypatch):
    monkeypatch.setattr(email_mod, "send_invite_email", lambda *a, **k: True)
    monkeypatch.setattr(email_mod, "sender_domain_warning", lambda: "FROM_EMAIL is x@gmail.com...")
    r = _invite()
    body = r.json()
    assert body["email_sent"] is True
    assert "gmail.com" in body["email_error"]


# ── The sender-domain check itself ────────────────────────────────────────────

@pytest.mark.parametrize("addr", [
    "genreal.ai@gmail.com", "team@yahoo.com", "x@outlook.com", "y@GMAIL.COM",
])
def test_free_webmail_senders_are_flagged(monkeypatch, addr):
    monkeypatch.setattr(email_mod, "FROM_EMAIL", addr)
    warning = email_mod.sender_domain_warning()
    assert warning is not None
    assert addr in warning          # names the offending address
    assert "DMARC" in warning       # and says why it matters


@pytest.mark.parametrize("addr", ["noreply@genreal.ai", "invites@company.co.uk"])
def test_owned_domain_senders_are_not_flagged(monkeypatch, addr):
    monkeypatch.setattr(email_mod, "FROM_EMAIL", addr)
    assert email_mod.sender_domain_warning() is None


def test_gmail_sent_via_brevo_is_still_flagged(monkeypatch):
    # The scenario that started this: relaying a gmail.com From through a
    # third party that doesn't own gmail.com.
    monkeypatch.setattr(email_mod, "FROM_EMAIL", "santhoshraajkr.17@gmail.com")
    monkeypatch.setattr(email_mod, "SMTP_HOST", "smtp-relay.brevo.com")
    warning = email_mod.sender_domain_warning()
    assert warning is not None
    assert "smtp-relay.brevo.com" in warning


def test_gmail_sent_via_gmail_own_smtp_is_not_flagged(monkeypatch):
    # Authenticated directly to Google as that exact account: Google is
    # delivering its own domain's mail, so alignment holds.
    monkeypatch.setattr(email_mod, "FROM_EMAIL", "santhoshraajkr.17@gmail.com")
    monkeypatch.setattr(email_mod, "SMTP_HOST", "smtp.gmail.com")
    assert email_mod.sender_domain_warning() is None


def test_gmail_host_does_not_launder_an_unrelated_free_webmail_domain(monkeypatch):
    # Authenticating to smtp.gmail.com does not make a yahoo.com From aligned —
    # only gmail.com/googlemail.com are Google's own domains.
    monkeypatch.setattr(email_mod, "FROM_EMAIL", "someone@yahoo.com")
    monkeypatch.setattr(email_mod, "SMTP_HOST", "smtp.gmail.com")
    assert email_mod.sender_domain_warning() is not None


def test_smtp_host_match_is_case_insensitive(monkeypatch):
    monkeypatch.setattr(email_mod, "FROM_EMAIL", "someone@gmail.com")
    monkeypatch.setattr(email_mod, "SMTP_HOST", "SMTP.GMAIL.COM")
    assert email_mod.sender_domain_warning() is None
