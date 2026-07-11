"""
Supabase-backed shared session store.

Both bot_service (port 8000) and confluence_service (port 8001) read and
write session state through this module so they share the same meeting data
without coupling their processes together.

Supabase is the only store — there is no local fallback. If Supabase is
unreachable or misconfigured, calls raise instead of silently degrading to
stale/empty local state.

Required Supabase tables (run migrations/001_initial.sql):
    jarvis_sessions (session_id PK, bot_id, meeting_url, status, error,
                     changes, transcript, transcript_memory_text, summary,
                     extracted_meeting, pipeline_diagnostics,
                     started_at, ended_at, updated_at, team_id)
    session_transcript_turns (session_id, seq, participant, text, ts, source)
"""

import datetime
import logging
import os
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env.local")

logger = logging.getLogger(__name__)

_SUPABASE_URL = os.getenv("SUPABASE_URL", "").rstrip("/")
_SUPABASE_KEY = (
    os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    or os.getenv("SUPABASE_ANON_KEY")
    or ""
)
_TABLE = "jarvis_sessions"
_TURNS_TABLE = "session_transcript_turns"
_TURNS_PAGE = 1000  # PostgREST caps rows per response; page through in chunks.

if not (_SUPABASE_URL and _SUPABASE_KEY):
    raise RuntimeError(
        "session_store requires SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY "
        "(or SUPABASE_ANON_KEY) to be set — there is no local fallback."
    )


class SessionStoreError(RuntimeError):
    """Raised when a Supabase request fails or returns a non-OK response."""


def _utcnow() -> str:
    return datetime.datetime.utcnow().isoformat() + "Z"


def _headers(return_repr: bool = False) -> dict[str, str]:
    h = {
        "apikey": _SUPABASE_KEY,
        "Authorization": f"Bearer {_SUPABASE_KEY}",
        "Content-Type": "application/json",
    }
    if return_repr:
        h["Prefer"] = "return=representation"
    return h


def _rest_url() -> str:
    return f"{_SUPABASE_URL}/rest/v1/{_TABLE}"


def _raise_for_response(resp: requests.Response, action: str) -> None:
    if not resp.ok:
        raise SessionStoreError(
            f"session_store.{action} failed: {resp.status_code} {resp.text}"
        )


# ── Public API ─────────────────────────────────────────────────────────────────

def get(session_id: str) -> dict | None:
    resp = requests.get(
        _rest_url(),
        headers=_headers(),
        params={"session_id": f"eq.{session_id}"},
        timeout=5,
    )
    _raise_for_response(resp, "get")
    data = resp.json()
    return data[0] if data else None


def upsert(session_id: str, data: dict) -> dict:
    record = {**data, "session_id": session_id, "updated_at": _utcnow()}
    resp = requests.post(
        _rest_url(),
        headers={**_headers(), "Prefer": "resolution=merge-duplicates,return=representation"},
        params={"on_conflict": "session_id"},
        json=record,
        timeout=5,
    )
    _raise_for_response(resp, "upsert")
    rows = resp.json()
    return rows[0] if rows else record


def patch(session_id: str, updates: dict) -> None:
    updates = {**updates, "updated_at": _utcnow()}
    resp = requests.patch(
        _rest_url(),
        headers=_headers(),
        params={"session_id": f"eq.{session_id}"},
        json=updates,
        timeout=5,
    )
    _raise_for_response(resp, "patch")


def list_all() -> list[dict]:
    resp = requests.get(
        _rest_url(),
        headers=_headers(),
        params={"order": "updated_at.desc"},
        timeout=8,
    )
    _raise_for_response(resp, "list_all")
    return resp.json()


def require(session_id: str) -> dict:
    """Return session dict or raise KeyError if not found."""
    s = get(session_id)
    if s is None:
        raise KeyError(session_id)
    return s


# ── Transcript turns (append-only; Recall meeting transcript only) ──────────────

def append_transcript_turn(session_id: str, entry: dict) -> None:
    """Append one meeting-transcript turn — O(1) INSERT, no blob rewrite.

    ``entry`` uses the same shape callers already produce:
    ``{participant, text, timestamp, source}``. Only ``text`` is required.
    """
    text = (entry.get("text") or "").strip()
    if not text:
        return
    row = {
        "session_id": session_id,
        "participant": entry.get("participant"),
        "text": text,
        "ts": entry.get("timestamp"),
        "source": entry.get("source") or "recall",
    }
    resp = requests.post(
        f"{_SUPABASE_URL}/rest/v1/{_TURNS_TABLE}",
        headers=_headers(),
        json=row,
        timeout=5,
    )
    _raise_for_response(resp, "append_transcript_turn")


def get_transcript_turns(session_id: str) -> list[dict]:
    """Return a session's meeting transcript in chronological order.

    Shape matches the old inline transcript entries so readers are unchanged:
    ``[{participant, text, timestamp, source}, ...]``.
    """
    out: list[dict] = []
    offset = 0
    while True:
        resp = requests.get(
            f"{_SUPABASE_URL}/rest/v1/{_TURNS_TABLE}",
            headers=_headers(),
            params={
                "session_id": f"eq.{session_id}",
                "select": "participant,text,ts,source,seq",
                "order": "seq.asc",
                "limit": str(_TURNS_PAGE),
                "offset": str(offset),
            },
            timeout=8,
        )
        _raise_for_response(resp, "get_transcript_turns")
        rows = resp.json()
        out.extend(
            {"participant": r.get("participant"), "text": r.get("text"),
             "timestamp": r.get("ts"), "source": r.get("source")}
            for r in rows
        )
        if len(rows) < _TURNS_PAGE:
            return out
        offset += _TURNS_PAGE
