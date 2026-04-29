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
