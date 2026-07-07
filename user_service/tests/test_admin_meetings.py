"""Tests for admin meeting-oversight routes (admin_meetings.py).

Validates RBAC (admin/CEO only), org-scoping (one org can't see/kick another's
meetings), and the kick proxy to bot-service. The Supabase layer and the
authenticated user are both faked — no network or DB required.

Run from the Confluence/ directory:
    python -m pytest user_service/tests/test_admin_meetings.py
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from user_service.main import app
from user_service.auth import get_current_user
from user_service.routes import admin_meetings as am


# ── Fake data ────────────────────────────────────────────────────────────────

# org1 owns team A & B; org2 owns team Z.
TEAMS = [
    {"id": "tA", "name": "Engineering", "org_id": "org1"},
    {"id": "tB", "name": "Design", "org_id": "org1"},
    {"id": "tZ", "name": "Outsiders", "org_id": "org2"},
]

SESSIONS = [
    {"session_id": "s-live1", "team_id": "tA", "status": "in_meeting",
     "meeting_url": "u1", "started_at": "2026-06-20T10:00:00Z", "ended_at": None,
     "summary": {"title": "Sprint sync"}, "changes": [1, 2], "bot_id": "b1",
     "transcript": [{"participant": "x", "text": "hi"}]},
    {"session_id": "s-live2", "team_id": "tB", "status": "joining",
     "meeting_url": "u2", "started_at": "2026-06-20T11:00:00Z", "ended_at": None,
     "summary": {}, "changes": [], "bot_id": "b2", "transcript": []},
    {"session_id": "s-old1", "team_id": "tA", "status": "ended",
     "meeting_url": "u3", "started_at": "2026-06-19T09:00:00Z",
     "ended_at": "2026-06-19T09:30:00Z", "summary": {"title": "Retro"},
     "changes": [1], "bot_id": "b3", "transcript": []},
    # belongs to org2 — must be invisible/unkickable to org1
    {"session_id": "s-foreign", "team_id": "tZ", "status": "in_meeting",
     "meeting_url": "uZ", "started_at": "2026-06-20T12:00:00Z", "ended_at": None,
     "summary": {}, "changes": [], "bot_id": "bZ", "transcript": []},
]


def _fake_select(table, filters=None, limit=None, columns=None):
    filters = filters or {}
    if table == "org_teams":
        org = (filters.get("org_id") or "").replace("eq.", "")
        return [t for t in TEAMS if t["org_id"] == org]
    if table == "jarvis_sessions":
        rows = SESSIONS
        tid = filters.get("team_id", "")
        if tid.startswith("in.("):
            allowed = set(tid[4:-1].split(",")) if tid[4:-1] else set()
            rows = [s for s in rows if s["team_id"] in allowed]
        status = filters.get("status", "")
        if status.startswith("in.("):
            allowed = set(status[4:-1].split(","))
            rows = [s for s in rows if s["status"] in allowed]
        elif status.startswith("eq."):
            rows = [s for s in rows if s["status"] == status[3:]]
        return list(rows)
    return []


def _fake_select_one(table, filters, columns=None):
    if table == "jarvis_sessions":
        sid = (filters.get("session_id") or "").replace("eq.", "")
        return next((s for s in SESSIONS if s["session_id"] == sid), None)
    rows = _fake_select(table, filters, limit=1)
    return rows[0] if rows else None


@pytest.fixture(autouse=True)
def _patch_db(monkeypatch):
    monkeypatch.setattr(am, "select", _fake_select)
    monkeypatch.setattr(am, "select_one", _fake_select_one)
    yield
    app.dependency_overrides.clear()


def _as(role: str, org_id: str = "org1", sub: str = "u-admin"):
    """Override the auth dependency to act as a given role/org."""
    app.dependency_overrides[get_current_user] = lambda: {
        "sub": sub, "role": role, "org_id": org_id,
    }
    return TestClient(app)


# ── RBAC ─────────────────────────────────────────────────────────────────────

def test_member_forbidden_on_live():
    client = _as("MEMBER")
    assert client.get("/admin/meetings/live").status_code == 403


def test_member_forbidden_on_history():
    client = _as("MEMBER")
    assert client.get("/admin/meetings").status_code == 403


# ── Live ─────────────────────────────────────────────────────────────────────

def test_admin_live_only_org_and_live_status():
    client = _as("ADMIN")
    r = client.get("/admin/meetings/live")
    assert r.status_code == 200
    ids = {m["session_id"] for m in r.json()}
    assert ids == {"s-live1", "s-live2"}  # org1 live only; no ended, no foreign
    first = next(m for m in r.json() if m["session_id"] == "s-live1")
    assert first["team_name"] == "Engineering"
    assert first["change_count"] == 2
    assert "elapsed_seconds" in first


# ── History ──────────────────────────────────────────────────────────────────

def test_history_includes_ended_and_scopes_to_org():
    client = _as("CEO")
    r = client.get("/admin/meetings")
    assert r.status_code == 200
    ids = {m["session_id"] for m in r.json()}
    assert "s-old1" in ids and "s-foreign" not in ids
    old = next(m for m in r.json() if m["session_id"] == "s-old1")
    assert old["duration_mins"] == 30.0


def test_history_team_filter_rejects_foreign_team():
    client = _as("ADMIN")
    assert client.get("/admin/meetings", params={"team_id": "tZ"}).status_code == 403


# ── Detail ───────────────────────────────────────────────────────────────────

def test_detail_returns_summary_and_transcript():
    client = _as("ADMIN")
    r = client.get("/admin/meetings/s-live1")
    assert r.status_code == 200
    body = r.json()
    assert body["summary"]["title"] == "Sprint sync"
    assert body["transcript"][0]["participant"] == "x"


def test_detail_foreign_meeting_forbidden():
    client = _as("ADMIN")
    assert client.get("/admin/meetings/s-foreign").status_code == 403


# ── Kick ─────────────────────────────────────────────────────────────────────

def test_kick_foreign_meeting_forbidden(monkeypatch):
    monkeypatch.setattr(am, "_BOT_SERVICE_URL", "http://bot.test")
    client = _as("ADMIN")
    assert client.post("/admin/meetings/s-foreign/kick").status_code == 403


def test_kick_already_ended_conflict():
    client = _as("ADMIN")
    assert client.post("/admin/meetings/s-old1/kick").status_code == 409


def test_kick_happy_path_proxies_to_bot_service(monkeypatch):
    monkeypatch.setattr(am, "_BOT_SERVICE_URL", "http://bot.test")

    captured = {}

    class _Resp:
        ok = True
        status_code = 200

        def json(self):
            return {"status": "ended", "session_id": "s-live1"}

    def _fake_post(url, headers=None, timeout=None):
        captured["url"] = url
        return _Resp()

    monkeypatch.setattr(am.requests, "post", _fake_post)
    client = _as("ADMIN")
    r = client.post("/admin/meetings/s-live1/kick")
    assert r.status_code == 200
    assert r.json()["status"] == "ended"
    assert captured["url"] == "http://bot.test/sessions/s-live1/bot/stop"


def test_kick_without_bot_url_configured_returns_503(monkeypatch):
    monkeypatch.setattr(am, "_BOT_SERVICE_URL", "")
    client = _as("ADMIN")
    assert client.post("/admin/meetings/s-live1/kick").status_code == 503
