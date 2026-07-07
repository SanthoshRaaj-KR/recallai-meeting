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
                # Append-only transcript turns (mirror of the Supabase table).
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS transcript_turns (
                        session_id  TEXT NOT NULL,
                        seq         INTEGER PRIMARY KEY AUTOINCREMENT,
                        participant TEXT,
                        text        TEXT NOT NULL,
                        ts          REAL,
                        source      TEXT NOT NULL DEFAULT 'recall'
                    )
                """)
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_turns_session "
                    "ON transcript_turns(session_id, seq)"
                )
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
                # Supabase returned OK but empty — could be a transient consistency
                # gap; fall through to the local SQLite cache before giving up.
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


# ── Transcript turns (append-only; Recall meeting transcript only) ──────────────

_TURNS_TABLE = "session_transcript_turns"
_TURNS_PAGE = 1000  # PostgREST caps rows per response; page through in chunks.


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
    # Local mirror first so bot-only dev works without Supabase.
    try:
        with _db_lock:
            with _db_conn() as conn:
                conn.execute(
                    "INSERT INTO transcript_turns (session_id, participant, text, ts, source) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (session_id, row["participant"], text, row["ts"], row["source"]),
                )
                conn.commit()
    except Exception as exc:
        logger.debug("SQLite transcript append failed: %s", exc)
    if _use_supabase():
        try:
            requests.post(
                f"{_SUPABASE_URL}/rest/v1/{_TURNS_TABLE}",
                headers=_headers(),
                json=row,
                timeout=5,
            )
        except Exception as exc:
            logger.warning("append_transcript_turn remote failed: %s", exc)


def get_transcript_turns(session_id: str) -> list[dict]:
    """Return a session's meeting transcript in chronological order.

    Shape matches the old inline transcript entries so readers are unchanged:
    ``[{participant, text, timestamp, source}, ...]``.
    """
    if _use_supabase():
        try:
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
                if not resp.ok:
                    break
                rows = resp.json()
                out.extend(
                    {"participant": r.get("participant"), "text": r.get("text"),
                     "timestamp": r.get("ts"), "source": r.get("source")}
                    for r in rows
                )
                if len(rows) < _TURNS_PAGE:
                    return out
                offset += _TURNS_PAGE
            if out:
                return out
            # Remote reachable but empty — fall through to local mirror.
        except Exception as exc:
            logger.warning("get_transcript_turns remote failed: %s — using local", exc)
    with _db_lock:
        with _db_conn() as conn:
            rows = conn.execute(
                "SELECT participant, text, ts, source FROM transcript_turns "
                "WHERE session_id = ? ORDER BY seq ASC",
                (session_id,),
            ).fetchall()
    return [
        {"participant": r[0], "text": r[1], "timestamp": r[2], "source": r[3]}
        for r in rows
    ]
