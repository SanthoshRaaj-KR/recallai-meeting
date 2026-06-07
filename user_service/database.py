"""Supabase REST client wrapper for the user service."""

import logging
import os
from typing import Any

import requests
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

_SUPABASE_URL = os.getenv("SUPABASE_URL", "").rstrip("/")
_SUPABASE_KEY = (
    os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    or os.getenv("SUPABASE_ANON_KEY")
    or ""
)


class DBError(Exception):
    pass


def _check_configured() -> None:
    if not _SUPABASE_URL or not _SUPABASE_KEY:
        raise DBError(
            "Supabase is not configured. Set SUPABASE_URL and "
            "SUPABASE_SERVICE_ROLE_KEY in user_service/.env"
        )


def _headers(return_repr: bool = False) -> dict[str, str]:
    h = {
        "apikey": _SUPABASE_KEY,
        "Authorization": f"Bearer {_SUPABASE_KEY}",
        "Content-Type": "application/json",
    }
    if return_repr:
        h["Prefer"] = "return=representation"
    return h


def _url(table: str) -> str:
    return f"{_SUPABASE_URL}/rest/v1/{table}"


def _raise_for(resp: requests.Response) -> None:
    if not resp.ok:
        raise DBError(f"DB {resp.status_code}: {resp.text[:200]}")


def _call(fn, *args, **kwargs):
    """Wrap any requests call so network/schema errors surface as DBError."""
    try:
        return fn(*args, **kwargs)
    except DBError:
        raise
    except Exception as exc:
        raise DBError(str(exc)) from exc


def select(table: str, filters: dict[str, str] | None = None, limit: int | None = None) -> list[dict]:
    _check_configured()
    params: dict[str, Any] = {}
    if filters:
        params.update(filters)
    if limit:
        params["limit"] = str(limit)
    resp = _call(requests.get, _url(table), headers=_headers(), params=params, timeout=8)
    _raise_for(resp)
    return resp.json()


def select_one(table: str, filters: dict[str, str]) -> dict | None:
    rows = select(table, filters, limit=1)
    return rows[0] if rows else None


def insert(table: str, data: dict) -> dict:
    _check_configured()
    resp = _call(requests.post, _url(table), headers=_headers(return_repr=True), json=data, timeout=8)
    _raise_for(resp)
    rows = resp.json()
    return rows[0] if rows else data


def upsert(table: str, data: dict, on_conflict: str) -> dict:
    _check_configured()
    resp = _call(
        requests.post, _url(table),
        headers={**_headers(), "Prefer": "resolution=merge-duplicates,return=representation"},
        params={"on_conflict": on_conflict},
        json=data, timeout=8,
    )
    _raise_for(resp)
    rows = resp.json()
    return rows[0] if rows else data


def update(table: str, filters: dict[str, str], data: dict) -> list[dict]:
    _check_configured()
    resp = _call(requests.patch, _url(table), headers=_headers(return_repr=True), params=filters, json=data, timeout=8)
    _raise_for(resp)
    return resp.json()


def delete(table: str, filters: dict[str, str]) -> None:
    _check_configured()
    resp = _call(requests.delete, _url(table), headers=_headers(), params=filters, timeout=8)
    _raise_for(resp)


def delete_all(table: str, match_col: str = "id") -> None:
    _check_configured()
    resp = _call(requests.delete, _url(table), headers=_headers(), params={match_col: "not.is.null"}, timeout=15)
    _raise_for(resp)
