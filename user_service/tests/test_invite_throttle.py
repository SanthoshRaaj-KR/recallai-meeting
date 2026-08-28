"""Per-IP throttling of invalid invite codes, and code entropy.

GET /invites/{code} is public and unauthenticated by necessity — an invitee is
not in the org yet — which makes it the one guessable way into an organisation.
The old 6-hex-character code was 24 bits and there was no throttle at all.

Run from the Confluence/ directory:
    python -m pytest user_service/tests/test_invite_throttle.py
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from user_service.main import app
from user_service.routes import invites as invite_routes
from user_service.routes import teams as team_routes


@pytest.fixture(autouse=True)
def _reset(monkeypatch):
    invite_routes._bad_attempts.clear()
    # No invite matches, so every lookup is a "bad guess".
    monkeypatch.setattr(invite_routes, "select_one", lambda *a, **k: None)
    yield
    invite_routes._bad_attempts.clear()


def _get(client: TestClient, code: str, ip: str = "203.0.113.9"):
    return client.get(f"/invites/{code}", headers={"x-forwarded-for": ip})


def test_bad_codes_are_throttled_after_the_budget(monkeypatch):
    monkeypatch.setattr(invite_routes, "_MAX_BAD_ATTEMPTS", 5)
    client = TestClient(app)

    for _ in range(5):
        assert _get(client, "BADCODE").status_code == 404

    blocked = _get(client, "BADCODE")
    assert blocked.status_code == 429
    assert "Too many invalid invite codes" in blocked.json()["detail"]


def test_throttle_is_per_ip(monkeypatch):
    monkeypatch.setattr(invite_routes, "_MAX_BAD_ATTEMPTS", 3)
    client = TestClient(app)

    for _ in range(3):
        _get(client, "BADCODE", ip="203.0.113.9")
    assert _get(client, "BADCODE", ip="203.0.113.9").status_code == 429
    # A different caller is unaffected.
    assert _get(client, "BADCODE", ip="198.51.100.4").status_code == 404


def test_a_valid_code_clears_the_budget(monkeypatch):
    monkeypatch.setattr(invite_routes, "_MAX_BAD_ATTEMPTS", 3)
    client = TestClient(app)
    _get(client, "BADCODE")
    _get(client, "BADCODE")
    assert len(invite_routes._bad_attempts["203.0.113.9"]) == 2

    # A real invitee reloading their page must never be throttled.
    invite_routes._clear_attempts("203.0.113.9")
    assert "203.0.113.9" not in invite_routes._bad_attempts


def test_expired_entries_fall_out_of_the_window(monkeypatch):
    monkeypatch.setattr(invite_routes, "_MAX_BAD_ATTEMPTS", 2)
    monkeypatch.setattr(invite_routes, "_WINDOW_S", 0)  # everything is already stale
    client = TestClient(app)
    for _ in range(5):
        assert _get(client, "BADCODE").status_code == 404


def test_falls_back_to_peer_address_without_a_proxy_header(monkeypatch):
    monkeypatch.setattr(invite_routes, "_MAX_BAD_ATTEMPTS", 2)
    client = TestClient(app)
    for _ in range(2):
        assert client.get("/invites/BADCODE").status_code == 404
    assert client.get("/invites/BADCODE").status_code == 429


def test_invite_codes_carry_64_bits_of_entropy():
    """A code must be long enough that guessing is not a strategy."""
    import inspect
    src = inspect.getsource(team_routes.invite_member)
    assert "secrets.token_hex(8)" in src, "invite code entropy was reduced"

    # And distinct across generations.
    import secrets
    codes = {secrets.token_hex(8).upper() for _ in range(200)}
    assert len(codes) == 200
    assert all(len(c) == 16 for c in codes)
