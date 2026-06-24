"""
Supabase-backed shared session store.

Both bot_service (port 8000) and confluence_service (port 8001) read and
write session state through this module so they share the same meeting data
without coupling their processes together.

Priority:
  1. Supabase (if SUPABASE_URL + key are set) — authoritative cross-host store
  2. SQLite file (.sessions.db next to .env.local) — cross-process shared store
     for local dev without Supabase

Required Supabase table (run migrations/001_initial.sql):
    jarvis_sessions (session_id PK, bot_id, meeting_url, status, error,
                     changes, transcript, transcript_memory_text, summary,
                     extracted_meeting, pipeline_diagnostics,
                     started_at, ended_at, updated_at, team_id)
"""

import datetime
import json
import logging
import os
import sqlite3
import threading
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

# SQLite fallback — shared file so both services see the same sessions
_DB_PATH = Path(__file__).parent.parent / ".sessions.db"
_db_lock = threading.Lock()


def _utcnow() -> str:
    return datetime.datetime.utcnow().isoformat() + "Z"


def _use_supabase() -> bool:
    return bool(_SUPABASE_URL and _SUPABASE_KEY)


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


# ── SQLite helpers ─────────────────────────────────────────────────────────────

def _db_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(str(_DB_PATH), timeout=10, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def _ensure_table() -> None:
    if _use_supabase():
        return  # Supabase is the store; no local SQLite needed
    try:
        with _db_lock:
            with _db_conn() as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS sessions (
                        session_id TEXT PRIMARY KEY,
                        data       TEXT NOT NULL,
                        updated_at TEXT
                    )
                """)
                conn.commit()
    except Exception as exc:
        logger.warning("SQLite setup failed (will rely on Supabase): %s", exc)


_ensure_table()


def _sqlite_get(session_id: str) -> dict | None:
    with _db_lock:
        with _db_conn() as conn:
            row = conn.execute(
                "SELECT data FROM sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
    if row:
        try:
            return json.loads(row[0])
        except Exception:
            return None
    return None


def _sqlite_set(session_id: str, data: dict) -> None:
    serialized = json.dumps(data, default=str)
    with _db_lock:
        with _db_conn() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO sessions (session_id, data, updated_at) VALUES (?, ?, ?)",
                (session_id, serialized, data.get("updated_at")),
            )
            conn.commit()


def _sqlite_list() -> list[dict]:
    with _db_lock:
        with _db_conn() as conn:
            rows = conn.execute(
                "SELECT data FROM sessions ORDER BY updated_at DESC"
            ).fetchall()
    result = []
    for (raw,) in rows:
        try:
            result.append(json.loads(raw))
        except Exception:
            pass
    return result


# ── Public API ─────────────────────────────────────────────────────────────────

def get(session_id: str) -> dict | None:
    if _use_supabase():
        try:
            resp = requests.get(
                _rest_url(),
                headers=_headers(),
                params={"session_id": f"eq.{session_id}"},
                timeout=5,
            )
            if resp.ok:
                data = resp.json()
                if data:
                    _sqlite_set(session_id, data[0])
                    return data[0]
                return None
        except Exception as exc:
            logger.warning("session_store.get remote failed: %s — using local cache", exc)
    return _sqlite_get(session_id)


def upsert(session_id: str, data: dict) -> dict:
    record = {**data, "session_id": session_id, "updated_at": _utcnow()}
    _sqlite_set(session_id, record)
    if _use_supabase():
        try:
            resp = requests.post(
                _rest_url(),
                headers={**_headers(), "Prefer": "resolution=merge-duplicates,return=representation"},
                params={"on_conflict": "session_id"},
                json=record,
                timeout=5,
            )
            if resp.ok:
                rows = resp.json()
                if rows:
                    _sqlite_set(session_id, rows[0])
                    return rows[0]
        except Exception as exc:
            logger.warning("session_store.upsert remote failed: %s — local only", exc)
    return record


def patch(session_id: str, updates: dict) -> None:
    updates = {**updates, "updated_at": _utcnow()}
    existing = _sqlite_get(session_id) or {}
    merged = {**existing, **updates}
    _sqlite_set(session_id, merged)
    if _use_supabase():
        try:
            requests.patch(
                _rest_url(),
                headers=_headers(),
                params={"session_id": f"eq.{session_id}"},
                json=updates,
                timeout=5,
            )
        except Exception as exc:
            logger.warning("session_store.patch remote failed: %s", exc)


def list_all() -> list[dict]:
    if _use_supabase():
        try:
            resp = requests.get(
                _rest_url(),
                headers=_headers(),
                params={"order": "updated_at.desc"},
                timeout=8,
            )
            if resp.ok:
                rows = resp.json()
                for r in rows:
                    _sqlite_set(r["session_id"], r)
                return rows
        except Exception as exc:
            logger.warning("session_store.list_all remote failed: %s — using local", exc)
    return _sqlite_list()


def require(session_id: str) -> dict:
    """Return session dict or raise KeyError if not found."""
    s = get(session_id)
    if s is None:
        raise KeyError(session_id)
    return s
