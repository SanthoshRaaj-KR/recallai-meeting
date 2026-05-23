"""SQLite-backed transcript store for cross-process IPC between agent_worker and jarvis_agentic.

Replaces the HTTP POST path for transcript reads in agent_bridge tools.
agent_worker writes here directly (no HTTP); agent_bridge reads from here for
tool calls (summarize_meeting_tool, etc.) without a jarvis_agentic round-trip.

The HTTP POST to /livekit-transcript/{session_id} is ALSO kept so jarvis_agentic's
in-memory transcript_log stays populated for the post-meeting proposals pipeline.

DB path: JARVIS_TRANSCRIPT_DB env var (default /tmp/jarvis_transcripts.db).
WAL mode is enabled so concurrent readers and one writer co-exist without locking.
"""
from __future__ import annotations

import os
import sqlite3
import threading
import time
from typing import Any, Dict, List

_DB_PATH: str = os.getenv("JARVIS_TRANSCRIPT_DB", "/tmp/jarvis_transcripts.db")

# Thread-local connections: each thread (asyncio.to_thread worker) gets its own
# connection so we avoid passing connections across threads.
_local = threading.local()


def _conn(db_path: str = _DB_PATH) -> sqlite3.Connection:
    c = getattr(_local, "conn", None)
    if c is None:
        c = sqlite3.connect(db_path, check_same_thread=False)
        c.execute("PRAGMA journal_mode=WAL")
        c.execute("PRAGMA synchronous=NORMAL")
        c.execute("""
            CREATE TABLE IF NOT EXISTS transcripts (
                id       INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                speaker    TEXT NOT NULL,
                text       TEXT NOT NULL,
                timestamp  REAL NOT NULL,
                source     TEXT NOT NULL DEFAULT 'livekit',
                created_at REAL NOT NULL
            )
        """)
        c.execute(
            "CREATE INDEX IF NOT EXISTS idx_session ON transcripts(session_id)"
        )
        c.commit()
        _local.conn = c
    return c


def append_utterance(
    session_id: str,
    speaker: str,
    text: str,
    timestamp: float = 0.0,
    source: str = "livekit",
    db_path: str = _DB_PATH,
) -> None:
    """Append one utterance synchronously. Call via asyncio.to_thread from async code."""
    if not (session_id and text.strip()):
        return
    _conn(db_path).execute(
        "INSERT INTO transcripts (session_id, speaker, text, timestamp, source, created_at)"
        " VALUES (?, ?, ?, ?, ?, ?)",
        (session_id, speaker, text, timestamp or time.time(), source, time.time()),
    )
    _conn(db_path).commit()


def get_utterances(
    session_id: str,
    db_path: str = _DB_PATH,
) -> List[Dict[str, Any]]:
    """Return all utterances for session_id in insertion order. Call via asyncio.to_thread."""
    rows = _conn(db_path).execute(
        "SELECT speaker, text, timestamp, source FROM transcripts"
        " WHERE session_id = ? ORDER BY id",
        (session_id,),
    ).fetchall()
    return [
        {"participant": r[0], "text": r[1], "timestamp": r[2], "source": r[3]}
        for r in rows
    ]
