import sqlite3

import session_store


class _NoRemoteResponse:
    ok = False


def test_upsert_creates_sqlite_cache_table_when_supabase_is_enabled(
    monkeypatch,
    tmp_path,
) -> None:
    db_path = tmp_path / ".sessions.db"
    monkeypatch.setattr(session_store, "_DB_PATH", db_path)
    monkeypatch.setattr(session_store, "_SUPABASE_URL", "https://example.supabase.co")
    monkeypatch.setattr(session_store, "_SUPABASE_KEY", "test-key")
    monkeypatch.setattr(
        session_store.requests,
        "post",
        lambda *args, **kwargs: _NoRemoteResponse(),
    )

    record = session_store.upsert("session-1", {"status": "ended"})

    assert record["session_id"] == "session-1"
    with sqlite3.connect(db_path) as conn:
        stored = conn.execute(
            "SELECT data FROM sessions WHERE session_id = ?",
            ("session-1",),
        ).fetchone()
    assert stored is not None
