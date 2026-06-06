"""Supabase-backed meeting history helpers for the review API.

Expected table:

create table if not exists meeting_history (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null,
  session_id text not null unique,
  title text,
  meeting_url text,
  status text,
  started_at timestamptz,
  ended_at timestamptz,
  summary text,
  summary_json jsonb,
  transcript_compressed text,
  transcript_codec text,
  transcript_entry_count integer default 0,
  transcript_uncompressed_bytes integer default 0,
  transcript_compressed_bytes integer default 0,
  change_count integer default 0,
  stats jsonb,
  created_at timestamptz default now(),
  updated_at timestamptz default now()
);
"""
from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import requests

logger = logging.getLogger(__name__)

SUPABASE_URL = (os.getenv("SUPABASE_URL") or "").rstrip("/")
SUPABASE_ANON_KEY = (os.getenv("SUPABASE_ANON_KEY") or "").strip()
SUPABASE_SERVICE_ROLE_KEY = (os.getenv("SUPABASE_SERVICE_ROLE_KEY") or "").strip()
SUPABASE_HISTORY_TABLE = (os.getenv("SUPABASE_HISTORY_TABLE") or "meeting_history").strip()


def is_configured() -> bool:
    return bool(SUPABASE_URL and SUPABASE_ANON_KEY)


def _db_key() -> str:
    return SUPABASE_SERVICE_ROLE_KEY or SUPABASE_ANON_KEY


def _rest_headers(prefer: Optional[str] = None) -> Dict[str, str]:
    headers = {
        "apikey": _db_key(),
        "Authorization": f"Bearer {_db_key()}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    if prefer:
        headers["Prefer"] = prefer
    return headers


def user_from_bearer(token: str) -> Optional[Dict[str, Any]]:
    if not is_configured() or not token:
        return None

    try:
        response = requests.get(
            f"{SUPABASE_URL}/auth/v1/user",
            headers={
                "apikey": SUPABASE_ANON_KEY,
                "Authorization": f"Bearer {token}",
                "Accept": "application/json",
            },
            timeout=8,
        )
        if response.status_code >= 400:
            logger.info("Supabase auth lookup failed with %s", response.status_code)
            return None
        data = response.json()
        user_id = data.get("id")
        if not user_id:
            return None
        return {
            "id": user_id,
            "email": data.get("email"),
            "name": (data.get("user_metadata") or {}).get("full_name"),
            "avatar_url": (data.get("user_metadata") or {}).get("avatar_url"),
        }
    except Exception as exc:
        logger.warning("Supabase auth lookup failed: %s", exc)
        return None


def upsert_history(row: Dict[str, Any]) -> None:
    if not is_configured() or not row.get("user_id") or not row.get("session_id"):
        return

    payload = {key: value for key, value in row.items() if value is not None}
    payload["updated_at"] = datetime.now(timezone.utc).isoformat()
    try:
        response = requests.post(
            f"{SUPABASE_URL}/rest/v1/{SUPABASE_HISTORY_TABLE}?on_conflict=session_id",
            headers=_rest_headers("resolution=merge-duplicates"),
            json=payload,
            timeout=8,
        )
        response.raise_for_status()
    except Exception as exc:
        logger.warning("Supabase history upsert failed: %s", exc)


def list_history(user_id: str, limit: int = 50) -> List[Dict[str, Any]]:
    if not is_configured() or not user_id:
        return []

    try:
        response = requests.get(
            f"{SUPABASE_URL}/rest/v1/{SUPABASE_HISTORY_TABLE}",
            headers=_rest_headers(),
            params={
                "select": "session_id,title,meeting_url,status,started_at,ended_at,summary,change_count,stats,updated_at",
                "user_id": f"eq.{user_id}",
                "order": "updated_at.desc",
                "limit": str(limit),
            },
            timeout=8,
        )
        response.raise_for_status()
        data = response.json()
        return data if isinstance(data, list) else []
    except Exception as exc:
        logger.warning("Supabase history list failed: %s", exc)
        return []


def get_history_item(user_id: str, session_id: str) -> Optional[Dict[str, Any]]:
    if not is_configured() or not user_id or not session_id:
        return None

    try:
        response = requests.get(
            f"{SUPABASE_URL}/rest/v1/{SUPABASE_HISTORY_TABLE}",
            headers=_rest_headers(),
            params={
                "select": "*",
                "user_id": f"eq.{user_id}",
                "session_id": f"eq.{session_id}",
                "limit": "1",
            },
            timeout=8,
        )
        response.raise_for_status()
        data = response.json()
        if isinstance(data, list) and data:
            return data[0]
    except Exception as exc:
        logger.warning("Supabase history lookup failed: %s", exc)
    return None


def create_pipeline_job(session_id: str, user_id: str) -> Optional[str]:
    """Create a new pipeline_jobs row and return the generated job_id."""
    if not is_configured():
        return None
    try:
        response = requests.post(
            f"{SUPABASE_URL}/rest/v1/pipeline_jobs",
            headers=_rest_headers("return=representation"),
            json={
                "session_id": session_id,
                "user_id": user_id,
                "status": "pending",
                "created_at": datetime.now(timezone.utc).isoformat(),
            },
            timeout=8,
        )
        response.raise_for_status()
        data = response.json()
        if isinstance(data, list) and data:
            return str(data[0].get("job_id") or "")
    except Exception as exc:
        logger.warning("Pipeline job create failed: %s", exc)
    return None


def update_pipeline_job(
    job_id: str,
    stage: Optional[str] = None,
    status: Optional[str] = None,
    error: Optional[str] = None,
    completed_at: Optional[str] = None,
) -> None:
    """Patch a pipeline_jobs row with updated stage/status/error/completed_at."""
    if not is_configured() or not job_id:
        return
    payload: Dict[str, Any] = {}
    if stage is not None:
        payload["stage"] = stage
    if status is not None:
        payload["status"] = status
    if error is not None:
        payload["error"] = error
    if completed_at is not None:
        payload["completed_at"] = completed_at
    if not payload:
        return
    try:
        response = requests.patch(
            f"{SUPABASE_URL}/rest/v1/pipeline_jobs?job_id=eq.{job_id}",
            headers=_rest_headers(),
            json=payload,
            timeout=8,
        )
        response.raise_for_status()
    except Exception as exc:
        logger.warning("Pipeline job update failed for %s: %s", job_id, exc)


# Plan 08-03 / D-16: ``change_summary`` is persisted as a top-level column on the
# ``proposals`` table. If the column does not exist (migration not yet applied), the
# whole insert fails with a PostgREST schema error. We retry once without the column
# and emit a single warning per process so the pipeline keeps working until the manual
# SUPABASE_MIGRATION.sql is run. See .planning/phases/08-auto-proposal-quality-fix/SUPABASE_MIGRATION.sql.
_CHANGE_SUMMARY_COL_WARNED = False


def upsert_proposal(row: Dict[str, Any]) -> Optional[str]:
    """Write one verified proposal card row to the proposals table. Returns the Supabase UUID on success, None on failure."""
    global _CHANGE_SUMMARY_COL_WARNED
    if not is_configured() or not row.get("job_id") or not row.get("user_id"):
        return None
    payload = {k: v for k, v in row.items() if v is not None}
    payload["created_at"] = datetime.now(timezone.utc).isoformat()
    try:
        response = requests.post(
            f"{SUPABASE_URL}/rest/v1/proposals",
            headers=_rest_headers("return=representation"),
            json=payload,
            timeout=8,
        )
        # PostgREST returns 4xx if the schema is missing a column; retry without
        # change_summary so we degrade gracefully until the migration runs.
        if response.status_code >= 400 and "change_summary" in payload:
            body_text = (response.text or "").lower()
            if "change_summary" in body_text or "could not find" in body_text or "schema cache" in body_text:
                if not _CHANGE_SUMMARY_COL_WARNED:
                    logger.warning(
                        "Supabase proposals table is missing the 'change_summary' column — "
                        "run .planning/phases/08-auto-proposal-quality-fix/SUPABASE_MIGRATION.sql. "
                        "Dropping the field from this insert; UI will show a synthesized fallback."
                    )
                    _CHANGE_SUMMARY_COL_WARNED = True
                payload.pop("change_summary", None)
                response = requests.post(
                    f"{SUPABASE_URL}/rest/v1/proposals",
                    headers=_rest_headers("return=representation"),
                    json=payload,
                    timeout=8,
                )
        response.raise_for_status()
        data = response.json()
        if isinstance(data, list) and data:
            return str(data[0].get("id") or "") or None
    except Exception as exc:
        logger.warning("Proposal upsert failed: %s", exc)
    return None


def get_proposal_with_intent(proposal_id: str) -> Optional[Dict[str, Any]]:
    """Fetch a single proposals row by UUID for regenerate-against-current-page.

    The stored proposal contains everything the regenerate endpoint needs to
    reconstruct a minimal ``ChangeIntent``: page_id, page_title, section_heading,
    before_content, after_content, change_type, rationale, change_summary. No
    separate intents table exists today — the proposal row is itself the source
    of truth.
    """
    return get_proposal_by_id(proposal_id)


def update_proposal_full(proposal_id: str, fields: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Patch arbitrary fields on a proposals row atomically. Bumps updated_at.

    Returns the updated row dict, or None on failure. Used by the regenerate
    endpoint (Plan 08-03 / D-09) to overwrite multiple fields in a single PATCH:
    change_type, section_heading, before_content, after_content, edit_mode,
    rationale, change_summary, status, verifier_note, risk.

    Gracefully drops ``change_summary`` and retries once if the column is missing,
    mirroring upsert_proposal's behaviour.
    """
    global _CHANGE_SUMMARY_COL_WARNED
    if not is_configured() or not proposal_id or not fields:
        return None
    payload = {k: v for k, v in fields.items() if k != "id"}
    payload["updated_at"] = datetime.now(timezone.utc).isoformat()
    try:
        response = requests.patch(
            f"{SUPABASE_URL}/rest/v1/proposals",
            headers=_rest_headers("return=representation"),
            params={"id": f"eq.{proposal_id}"},
            json=payload,
            timeout=8,
        )
        if response.status_code >= 400 and "change_summary" in payload:
            body_text = (response.text or "").lower()
            if "change_summary" in body_text or "could not find" in body_text or "schema cache" in body_text:
                if not _CHANGE_SUMMARY_COL_WARNED:
                    logger.warning(
                        "Supabase proposals table is missing the 'change_summary' column — "
                        "run .planning/phases/08-auto-proposal-quality-fix/SUPABASE_MIGRATION.sql. "
                        "Dropping the field from this update; UI will show a synthesized fallback."
                    )
                    _CHANGE_SUMMARY_COL_WARNED = True
                payload.pop("change_summary", None)
                response = requests.patch(
                    f"{SUPABASE_URL}/rest/v1/proposals",
                    headers=_rest_headers("return=representation"),
                    params={"id": f"eq.{proposal_id}"},
                    json=payload,
                    timeout=8,
                )
        response.raise_for_status()
        data = response.json()
        if isinstance(data, list) and data:
            return data[0]
    except Exception as exc:
        logger.warning("update_proposal_full failed for %s: %s", proposal_id, exc)
    return None


def get_proposal_by_id(proposal_id: str) -> Optional[Dict[str, Any]]:
    """Fetch a single proposals row by its UUID primary key."""
    if not is_configured() or not proposal_id:
        return None
    try:
        response = requests.get(
            f"{SUPABASE_URL}/rest/v1/proposals",
            headers=_rest_headers(),
            params={"id": f"eq.{proposal_id}", "select": "*", "limit": "1"},
            timeout=8,
        )
        response.raise_for_status()
        data = response.json()
        if isinstance(data, list) and data:
            return data[0]
    except Exception as exc:
        logger.warning("get_proposal_by_id failed for %s: %s", proposal_id, exc)
    return None


def update_proposal_status(proposal_id: str, status: str) -> bool:
    """Patch the status field of a proposals row by UUID."""
    if not is_configured() or not proposal_id:
        return False
    try:
        response = requests.patch(
            f"{SUPABASE_URL}/rest/v1/proposals",
            headers=_rest_headers(),
            params={"id": f"eq.{proposal_id}"},
            json={"status": status},
            timeout=8,
        )
        response.raise_for_status()
        return True
    except Exception as exc:
        logger.warning("update_proposal_status failed for %s: %s", proposal_id, exc)
    return False


def list_proposals_by_session(session_id: str, user_id: str) -> List[Dict[str, Any]]:
    """Return all proposals for a session from the proposals table, ordered by creation time.

    Used by GET /sessions/{session_id}/review/changes to surface pipeline-generated
    cards after the SSE stream completes and across server restarts.
    """
    if not is_configured() or not session_id or not user_id:
        return []
    try:
        response = requests.get(
            f"{SUPABASE_URL}/rest/v1/proposals",
            headers=_rest_headers(),
            params={
                "session_id": f"eq.{session_id}",
                "user_id": f"eq.{user_id}",
                "select": "*",
                "order": "created_at.asc",
                "limit": "200",
            },
            timeout=8,
        )
        response.raise_for_status()
        data = response.json()
        return data if isinstance(data, list) else []
    except Exception as exc:
        logger.warning("list_proposals_by_session failed for %s: %s", session_id, exc)
    return []


def get_pipeline_job(job_id: str) -> Optional[Dict[str, Any]]:
    """Fetch a single pipeline_jobs row by job_id. Returns the row dict or None.

    Used by the SSE stream endpoint to verify the caller's user_id matches
    the job's user_id before streaming events.
    """
    if not is_configured() or not job_id:
        return None
    try:
        response = requests.get(
            f"{SUPABASE_URL}/rest/v1/pipeline_jobs",
            headers=_rest_headers(),
            params={"job_id": f"eq.{job_id}", "select": "*", "limit": "1"},
            timeout=8,
        )
        response.raise_for_status()
        data = response.json()
        if isinstance(data, list) and data:
            return data[0]
    except Exception as exc:
        logger.warning("get_pipeline_job failed for %s: %s", job_id, exc)
    return None
