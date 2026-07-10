"""Durable, org-global persistence for the RAG knowledge-base sync status.

The knowledge base itself is already global — one Pinecone index/namespace fed from
one Confluence space, shared by the whole org. This module makes the *sync status*
global too: the latest sync is stored in Supabase keyed by org, so every admin/
manager sees the same "last synced" even after a bot-service restart (the in-memory
job dict resets on restart; this doesn't).

Best-effort by construction — never raises into the sync path. A missing table or a
network blip just means callers fall back to the in-memory job.
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
_SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_ANON_KEY") or ""
_TABLE = "rag_sync_status"

# Columns the table knows about. The job dict may carry extras (hybrid_changed…),
# so we whitelist before writing to avoid PostgREST rejecting unknown columns.
_COLS = (
    "job_id", "status", "total", "total_stale", "checked", "changed",
    "skipped", "failed", "deleted", "current_page", "error",
    "started_at", "finished_at", "synced_by", "synced_by_name",
)


def _configured() -> bool:
    return bool(_SUPABASE_URL and _SUPABASE_KEY)


def _headers(prefer: str | None = None) -> dict:
    h = {
        "apikey": _SUPABASE_KEY,
        "Authorization": f"Bearer {_SUPABASE_KEY}",
        "Content-Type": "application/json",
    }
    if prefer:
        h["Prefer"] = prefer
    return h


def _url() -> str:
    return f"{_SUPABASE_URL}/rest/v1/{_TABLE}"


def save(job: dict, org_id: str) -> None:
    """Upsert the latest sync for an org (one row per org). Never raises."""
    if not _configured() or not org_id:
        return
    try:
        row: dict = {"org_id": org_id}
        for c in _COLS:
            if c in job:
                row[c] = job[c]
        row["updated_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        requests.post(
            _url(),
            headers=_headers(prefer="resolution=merge-duplicates"),
            params={"on_conflict": "org_id"},
            json=row,
            timeout=6,
        )
    except Exception as exc:  # persistence must never break the sync
        logger.warning("rag_sync_store.save skipped: %s", exc)


def latest(org_id: str) -> dict | None:
    """Return the stored sync status for an org, or None. Never raises."""
    if not _configured() or not org_id:
        return None
    try:
        resp = requests.get(
            _url(),
            headers=_headers(),
            params={"org_id": f"eq.{org_id}", "limit": "1"},
            timeout=6,
        )
        if resp.ok and resp.json():
            return resp.json()[0]
    except Exception as exc:
        logger.warning("rag_sync_store.latest skipped: %s", exc)
    return None


def get(job_id: str, org_id: str) -> dict | None:
    """Return the stored sync only if it matches job_id. Never raises."""
    row = latest(org_id)
    if row and row.get("job_id") == job_id:
        return row
    return None


def user_name(user_id: str | None) -> str | None:
    """Resolve an org_user's display name (for 'last synced by …'). Never raises."""
    if not _configured() or not user_id:
        return None
    try:
        resp = requests.get(
            f"{_SUPABASE_URL}/rest/v1/org_users",
            headers=_headers(),
            params={"id": f"eq.{user_id}", "select": "name", "limit": "1"},
            timeout=5,
        )
        if resp.ok and resp.json():
            return resp.json()[0].get("name")
    except Exception as exc:
        logger.warning("rag_sync_store.user_name skipped: %s", exc)
    return None
