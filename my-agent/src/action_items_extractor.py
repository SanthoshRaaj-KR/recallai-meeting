"""
Action-item extractor + auto-assigner.

When a meeting ends, this reads the transcript, asks the LLM (Cerebras gpt-oss-120b,
OpenAI fallback) to extract concrete action items and assign each to a specific team
member (matched against the team roster), and upserts them into `meeting_action_items`.

Safe by construction (mirrors org_activity.py): no-op if Supabase / team_id / transcript
is absent, idempotent per (session_id, description) via insert-ignore-duplicates so it
NEVER overwrites a member's progress on re-run, and never raises into the caller.
"""

import json
import logging
import os

import requests
from dotenv import load_dotenv
from pathlib import Path

try:
    from . import session_store
except ImportError:  # standalone / non-package execution
    import session_store

load_dotenv(Path(__file__).parent.parent / ".env.local")

logger = logging.getLogger(__name__)

_SUPABASE_URL = os.getenv("SUPABASE_URL", "").rstrip("/")
_SUPABASE_KEY = (
    os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_ANON_KEY") or ""
)
_CEREBRAS_API_KEY = os.getenv("CEREBRAS_API_KEY", "")
_CEREBRAS_BASE_URL = "https://api.cerebras.ai/v1"
_CEREBRAS_MODEL = "gpt-oss-120b"


def _configured() -> bool:
    return bool(_SUPABASE_URL and _SUPABASE_KEY)


def _headers(prefer: str | None = None) -> dict[str, str]:
    h = {
        "apikey": _SUPABASE_KEY,
        "Authorization": f"Bearer {_SUPABASE_KEY}",
        "Content-Type": "application/json",
    }
    if prefer:
        h["Prefer"] = prefer
    return h


def _url(table: str) -> str:
    return f"{_SUPABASE_URL}/rest/v1/{table}"


def _transcript_text(session: dict, limit_chars: int = 16000) -> str:
    # Meeting transcript now lives in session_transcript_turns (Recall-sourced only).
    entries = session_store.get_transcript_turns(session.get("session_id", ""))
    if entries:
        lines = []
        for e in entries:
            who = e.get("participant") or e.get("speaker") or "?"
            txt = e.get("text") or ""
            if txt:
                lines.append(f"{who}: {txt}")
        text = "\n".join(lines)
    else:
        text = session.get("transcript_memory_text") or ""
    return text[-limit_chars:]  # keep the tail (most recent / wrap-up) if very long


def _roster(team_id: str) -> list[dict]:
    """Return [{user_id, name, email}] for a team."""
    try:
        members = requests.get(
            _url("org_team_members"),
            headers=_headers(),
            params={"team_id": f"eq.{team_id}", "select": "user_id"},
            timeout=6,
        )
        if not members.ok:
            return []
        ids = [m["user_id"] for m in members.json() if m.get("user_id")]
        if not ids:
            return []
        users = requests.get(
            _url("org_users"),
            headers=_headers(),
            params={"id": f"in.({','.join(ids)})", "select": "id,name,email"},
            timeout=6,
        )
        if not users.ok:
            return []
        return [{"user_id": u["id"], "name": u.get("name", ""), "email": u.get("email", "")} for u in users.json()]
    except Exception as exc:
        logger.warning("action_items: roster lookup failed: %s", exc)
        return []


def _org_id_for_team(team_id: str) -> str | None:
    try:
        r = requests.get(
            _url("org_teams"),
            headers=_headers(),
            params={"id": f"eq.{team_id}", "select": "org_id", "limit": "1"},
            timeout=6,
        )
        if r.ok and r.json():
            return r.json()[0].get("org_id")
    except Exception:
        pass
    return None


def _llm_extract(transcript: str, roster: list[dict]) -> list[dict]:
    """Ask the LLM for [{description, assignee, due}]. Returns [] on any failure."""
    if not transcript.strip():
        return []
    names = [m["name"] for m in roster if m["name"]]
    sys_prompt = (
        "You extract concrete, actionable follow-up tasks from a meeting transcript. "
        "For each task return a short imperative description, the single person responsible "
        "(assignee), and a due date if one is explicitly stated (else null). "
        "The assignee MUST be exactly one of this team roster (by name) or the string "
        f"'unassigned'. Roster: {', '.join(names) if names else '(none)'}. "
        'Respond with STRICT JSON only: {"items":[{"description":"...","assignee":"...","due":null}]}. '
        "Only include real action items; if there are none, return an empty list."
    )
    try:
        from openai import OpenAI
    except Exception:
        return []

    def _call(client, model):
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": f"Transcript:\n{transcript}"},
            ],
            temperature=0.2,
            response_format={"type": "json_object"},
        )
        return resp.choices[0].message.content or "{}"

    raw = None
    if _CEREBRAS_API_KEY:
        try:
            raw = _call(OpenAI(api_key=_CEREBRAS_API_KEY, base_url=_CEREBRAS_BASE_URL), _CEREBRAS_MODEL)
        except Exception as exc:
            logger.warning("action_items: Cerebras extract failed (%s); trying OpenAI", exc)
    if raw is None:
        try:
            raw = _call(OpenAI(), "gpt-4o-mini")
        except Exception as exc:
            logger.warning("action_items: LLM extract failed: %s", exc)
            return []

    try:
        data = json.loads(raw)
        items = data.get("items") if isinstance(data, dict) else data
        return [i for i in (items or []) if isinstance(i, dict) and i.get("description")]
    except Exception as exc:
        logger.warning("action_items: bad LLM JSON: %s", exc)
        return []


def _match_owner(assignee: str | None, roster: list[dict]) -> tuple[str | None, str | None]:
    """Map an LLM assignee name to (owner_user_id, owner_name). Best-effort."""
    if not assignee or assignee.strip().lower() in ("unassigned", "none", ""):
        return None, None
    a = assignee.strip().lower()
    for m in roster:
        if m["name"].lower() == a:
            return m["user_id"], m["name"]
    # first-name / contains match
    for m in roster:
        first = m["name"].lower().split()[0] if m["name"] else ""
        if first and (first == a or a in m["name"].lower() or m["name"].lower() in a):
            return m["user_id"], m["name"]
    return None, assignee.strip()  # keep raw name for a manager to reassign


def extract_and_assign(session: dict) -> None:
    """Extract action items from a session's transcript and upsert assigned rows. Never raises."""
    try:
        if not _configured():
            return
        session_id = session.get("session_id")
        team_id = session.get("team_id")
        if not session_id or not team_id or str(session_id).startswith("seed-"):
            return

        roster = _roster(team_id)
        items = _llm_extract(_transcript_text(session), roster)
        if not items:
            logger.info("action_items: no items extracted for session %s", session_id)
            return

        org_id = _org_id_for_team(team_id)
        rows = []
        for it in items:
            owner_id, owner_name = _match_owner(it.get("assignee"), roster)
            rows.append({
                "session_id": session_id,
                "team_id": team_id,
                "org_id": org_id,
                "description": str(it["description"])[:1000],
                "owner_user_id": owner_id,
                "owner_name": owner_name,
                "due": it.get("due"),
                "status": "open",
                "source": "ai",
            })

        # insert-ignore-duplicates: never clobber existing rows (preserves member/manager progress)
        resp = requests.post(
            _url("meeting_action_items"),
            headers=_headers(prefer="resolution=ignore-duplicates"),
            json=rows,
            timeout=10,
        )
        if resp.ok or resp.status_code == 409:
            logger.info("action_items: upserted %d items for session %s", len(rows), session_id)
        else:
            logger.warning("action_items: insert failed (%s): %s", resp.status_code, resp.text[:160])
    except Exception as exc:  # analytics-style: never break meeting teardown
        logger.warning("action_items.extract_and_assign skipped: %s", exc)
