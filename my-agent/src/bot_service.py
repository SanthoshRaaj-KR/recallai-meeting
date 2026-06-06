"""
Bot Service — port 8000.

Handles Recall.ai bot lifecycle, LiveKit token minting, session management,
transcript ingestion, and session history.  All Confluence intelligence
(proposals, summary, chat) lives in confluence_service.py on port 8001.

Run:
    uv run uvicorn src.bot_service:app --host 0.0.0.0 --port 8000

Required env vars (.env.local):
    LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET
    RECALL_API_KEY, BRIDGE_SERVER_URL

Optional:
    RECALL_API_REGION    (default: us-west-2)
    BOT_NAME             (default: Meeting Assistant)
    AGENT_NAME           (default: my-agent)
    TOKEN_TTL_HOURS      (default: 8)
    CORS_ORIGINS         (default: *)
    SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY  — shared session persistence
"""

import asyncio
import datetime
import json
import logging
import os
import uuid
from pathlib import Path
from typing import Optional

import requests
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from livekit import api as livekit_api
from pydantic import BaseModel

try:
    from .memory_compaction import TranscriptCompactor
    from . import session_store
except ImportError:
    from memory_compaction import TranscriptCompactor
    import session_store

load_dotenv(Path(__file__).parent.parent / ".env.local")

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ── Config ─────────────────────────────────────────────────────────────────────
RECALL_API_KEY = os.getenv("RECALL_API_KEY", "")
RECALL_API_REGION = os.getenv("RECALL_API_REGION", "us-west-2")
RECALL_BASE_URL = f"https://{RECALL_API_REGION}.recall.ai/api/v1"

LIVEKIT_URL = os.getenv("LIVEKIT_URL", "")
LIVEKIT_API_KEY = os.getenv("LIVEKIT_API_KEY", "")
LIVEKIT_API_SECRET = os.getenv("LIVEKIT_API_SECRET", "")

SERVER_URL = os.getenv("BRIDGE_SERVER_URL", "").rstrip("/")
BOT_NAME = os.getenv("BOT_NAME", "Meeting Assistant")
AGENT_NAME = os.getenv("AGENT_NAME", "my-agent")
TOKEN_TTL_HOURS = int(os.getenv("TOKEN_TTL_HOURS", "8"))

_CORS_ORIGINS = [o.strip() for o in os.getenv("CORS_ORIGINS", "*").split(",") if o.strip()]
_BOT_HTML_PATH = Path(__file__).parent / "bot.html"

_RECALL_STATUS_MAP = {
    "joining": "joining",
    "in_call_not_recording": "in_meeting",
    "in_call_recording": "in_meeting",
    "done": "ended",
    "call_ended": "ended",
    "error": "error",
}

# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(title="Jarvis Bot Service", version="2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── In-process caches ────────────────────────────────────────────────────────
# TranscriptCompactor is not serialisable so it lives only in this process;
# the compacted text is flushed to Supabase.
_compactors: dict[str, TranscriptCompactor] = {}
# bot_id → session_id index so webhook lookups are O(1) instead of O(N).
_bot_index: dict[str, str] = {}


def _utcnow() -> str:
    return datetime.datetime.utcnow().isoformat() + "Z"


def _new_compactor() -> TranscriptCompactor:
    return TranscriptCompactor(
        window_size=2000,
        max_memory_chars=int(os.getenv("JARVIS_BRIDGE_COMPACTED_MEMORY_CHARS", "12000")),
    )


# ── LiveKit helpers ────────────────────────────────────────────────────────────
def _mint_token(room_name: str, identity: str, can_publish: bool = True) -> str:
    return (
        livekit_api.AccessToken(LIVEKIT_API_KEY, LIVEKIT_API_SECRET)
        .with_identity(identity)
        .with_name(identity)
        .with_grants(livekit_api.VideoGrants(
            room_join=True,
            room=room_name,
            can_publish=can_publish,
            can_subscribe=True,
        ))
        .with_ttl(datetime.timedelta(hours=TOKEN_TTL_HOURS))
        .to_jwt()
    )


# ── Recall helpers ─────────────────────────────────────────────────────────────
def _create_recall_bot(meeting_url: str, room_name: str) -> str:
    if not RECALL_API_KEY:
        raise RuntimeError("RECALL_API_KEY is not configured")
    if not SERVER_URL:
        raise RuntimeError("BRIDGE_SERVER_URL is not configured — set to public HTTPS URL")
    if not SERVER_URL.startswith("https://"):
        raise RuntimeError(f"BRIDGE_SERVER_URL must start with https:// (got: {SERVER_URL})")
    if not LIVEKIT_URL or not LIVEKIT_API_KEY or not LIVEKIT_API_SECRET:
        raise RuntimeError("LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET must be configured")

    publisher_token = _mint_token(room_name, f"recall-browser-{room_name}", can_publish=True)
    subscriber_token = _mint_token(room_name, f"recall-listener-{room_name}", can_publish=False)

    bot_page_url = (
        f"{SERVER_URL}/bot-page"
        f"?url={LIVEKIT_URL}"
        f"&token={subscriber_token}"
        f"&pub_token={publisher_token}"
        f"&room={room_name}"
    )

    payload = {
        "meeting_url": meeting_url,
        "bot_name": BOT_NAME,
        "metadata": {"room_name": room_name},
        "output_media": {
            "camera": {"kind": "webpage", "config": {"url": bot_page_url}},
        },
    }

    response = requests.post(
        f"{RECALL_BASE_URL}/bot/",
        headers={"Authorization": f"Token {RECALL_API_KEY}", "Content-Type": "application/json"},
        json=payload,
        timeout=15,
    )
    if not response.ok:
        logger.error("Recall bot creation failed: %s — %s", response.status_code, response.text)
        response.raise_for_status()

    bot_id = response.json()["id"]
    logger.info("Recall bot created — bot_id=%s room=%s", bot_id, room_name)
    return bot_id


async def _dispatch_agent(room_name: str) -> None:
    async with livekit_api.LiveKitAPI(
        url=LIVEKIT_URL, api_key=LIVEKIT_API_KEY, api_secret=LIVEKIT_API_SECRET,
    ) as lk:
        dispatch = await lk.agent_dispatch.create_dispatch(
            livekit_api.CreateAgentDispatchRequest(
                agent_name=AGENT_NAME,
                room=room_name,
                metadata=json.dumps({"room_name": room_name}),
            )
        )
    logger.info("Agent dispatched — dispatch_sid=%s room=%s", dispatch.sid, room_name)


# ── Request / response models ──────────────────────────────────────────────────
class StartBotRequest(BaseModel):
    meeting_url: str
    room_name: Optional[str] = None
    session_id: Optional[str] = None
    team_id: Optional[str] = None  # optional team scoping (port 8003 integration)


class StartBotResponse(BaseModel):
    status: str
    bot_id: str
    room_name: str
    session_id: str
    meeting_url: str
    change_count: int = 0


# ── Endpoints: bot lifecycle ───────────────────────────────────────────────────
@app.post("/bot/start", response_model=StartBotResponse)
async def start_bot(body: StartBotRequest) -> StartBotResponse:
    room_name = body.room_name or body.session_id or str(uuid.uuid4())

    try:
        bot_id = _create_recall_bot(body.meeting_url, room_name)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except requests.HTTPError as exc:
        err_body = exc.response.text if exc.response is not None else ""
        raise HTTPException(status_code=502, detail=f"Recall.ai error: {exc} — {err_body}")

    try:
        await _dispatch_agent(room_name)
    except Exception as exc:
        logger.warning("Agent dispatch failed (non-fatal): %s", exc)

    now = _utcnow()
    session_data = {
        "session_id": room_name,
        "bot_id": bot_id,
        "meeting_url": body.meeting_url,
        "status": "joining",
        "error": None,
        "changes": [],
        "transcript": [],
        "transcript_memory_text": "",
        "summary": None,
        "extracted_meeting": None,
        "pipeline_diagnostics": [],
        "started_at": now,
        "ended_at": None,
        "updated_at": now,
        "team_id": body.team_id,
    }
    session_store.upsert(room_name, session_data)
    _compactors[room_name] = _new_compactor()
    _bot_index[bot_id] = room_name

    return StartBotResponse(
        status="joining",
        bot_id=bot_id,
        room_name=room_name,
        session_id=room_name,
        meeting_url=body.meeting_url,
        change_count=0,
    )


@app.get("/bot-page", response_class=HTMLResponse)
async def bot_page() -> HTMLResponse:
    if not _BOT_HTML_PATH.exists():
        return HTMLResponse("<html><body>bot.html not found</body></html>", status_code=500)
    return HTMLResponse(_BOT_HTML_PATH.read_text(encoding="utf-8"))


@app.get("/health")
async def health() -> dict:
    return {
        "status": "ok",
        "service": "bot-service",
        "agent_name": AGENT_NAME,
        "recall_region": RECALL_API_REGION,
        "recall_configured": bool(RECALL_API_KEY),
        "livekit_configured": bool(LIVEKIT_URL and LIVEKIT_API_KEY and LIVEKIT_API_SECRET),
        "server_url": SERVER_URL or "<not set>",
    }


# ── Endpoints: session status ──────────────────────────────────────────────────
@app.get("/sessions/{session_id}/bot/status")
async def session_bot_status(session_id: str) -> dict:
    try:
        s = session_store.require(session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Session {session_id!r} not found")

    # Refresh from Recall.ai if bot is active
    if s.get("bot_id") and RECALL_API_KEY and s.get("status") not in ("ended", "error"):
        try:
            resp = requests.get(
                f"{RECALL_BASE_URL}/bot/{s['bot_id']}/",
                headers={"Authorization": f"Token {RECALL_API_KEY}"},
                timeout=5,
            )
            if resp.ok:
                changes = resp.json().get("status_changes") or []
                code = changes[-1].get("code", "") if changes else ""
                new_status = _RECALL_STATUS_MAP.get(code, s["status"])
                if new_status != s["status"]:
                    updates: dict = {"status": new_status}
                    if new_status in ("ended", "error") and not s.get("ended_at"):
                        updates["ended_at"] = _utcnow()
                    session_store.patch(session_id, updates)
                    s = {**s, **updates}
        except Exception as exc:
            logger.debug("Recall status poll error: %s", exc)

    return {
        "status": s.get("status"),
        "session_id": session_id,
        "bot_id": s.get("bot_id"),
        "meeting_url": s.get("meeting_url"),
        "change_count": len(s.get("changes") or []),
        "error": s.get("error"),
        "ended_at": s.get("ended_at"),
        "end_reason": None,
        "recall_status_code": None,
    }


@app.get("/bot/status")
async def bot_status_no_session() -> dict:
    return {
        "status": "idle", "session_id": None, "bot_id": None,
        "meeting_url": None, "change_count": 0, "error": None,
        "ended_at": None, "end_reason": None, "recall_status_code": None,
    }


# ── Endpoints: stop bot ────────────────────────────────────────────────────────
@app.post("/sessions/{session_id}/bot/stop")
async def stop_bot(session_id: str) -> dict:
    try:
        s = session_store.require(session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Session {session_id!r} not found")

    if s.get("bot_id") and RECALL_API_KEY:
        try:
            resp = requests.post(
                f"{RECALL_BASE_URL}/bot/{s['bot_id']}/leave_call/",
                headers={"Authorization": f"Token {RECALL_API_KEY}"},
                timeout=10,
            )
            if not resp.ok:
                logger.warning("Recall bot removal returned %s", resp.status_code)
        except Exception as exc:
            logger.warning("Failed to remove Recall bot: %s", exc)

    session_store.patch(session_id, {"status": "ended", "ended_at": _utcnow()})
    s = session_store.get(session_id) or s
    return {
        "status": "ended", "session_id": session_id, "bot_id": s.get("bot_id"),
        "meeting_url": s.get("meeting_url"), "change_count": len(s.get("changes") or []),
        "error": s.get("error"), "ended_at": s.get("ended_at"),
        "end_reason": None, "recall_status_code": None,
    }


# ── Endpoints: Recall webhook ──────────────────────────────────────────────────
@app.post("/recall-webhook")
async def recall_webhook(request: Request) -> dict:
    try:
        body = await request.json()
    except Exception:
        return {"ok": True}

    event = body.get("event", "")
    data = body.get("data", {})

    if event in ("bot.status_change", "bot.done"):
        bot_id = data.get("bot_id", "")
        if bot_id:
            # O(1) lookup via local index; fall back to Supabase scan if not cached
            session_id = _bot_index.get(bot_id)
            s = session_store.get(session_id) if session_id else None
            if s is None:
                # Process restart — rebuild index entry from Supabase
                all_sessions = session_store.list_all()
                s = next((x for x in all_sessions if x.get("bot_id") == bot_id), None)
                if s:
                    _bot_index[bot_id] = s["session_id"]
            if s:
                status_obj = data.get("status") or {}
                code = status_obj.get("code") or (data.get("code") if event == "bot.done" else "")
                new_status = _RECALL_STATUS_MAP.get(code, "")
                if new_status and new_status != s.get("status"):
                    updates: dict = {"status": new_status}
                    if new_status in ("ended", "error") and not s.get("ended_at"):
                        updates["ended_at"] = _utcnow()
                    session_store.patch(s["session_id"], updates)
                    logger.info("Webhook %s → session %s status=%s", event, s["session_id"], new_status)

    return {"ok": True}


# ── Endpoints: transcript ingestion ───────────────────────────────────────────
@app.post("/livekit-transcript/{session_id}")
async def receive_livekit_transcript(session_id: str, request: Request) -> dict:
    body = await request.json()
    text = (body.get("text") or "").strip()
    if not text:
        return {"ok": True}

    s = session_store.get(session_id)
    if s is None:
        now = _utcnow()
        s = {
            "session_id": session_id, "bot_id": None, "meeting_url": None,
            "status": "in_meeting", "error": None, "changes": [], "transcript": [],
            "transcript_memory_text": "", "summary": None, "extracted_meeting": None,
            "pipeline_diagnostics": [], "started_at": now, "ended_at": None,
            "updated_at": now, "team_id": None,
        }
        session_store.upsert(session_id, s)

    entry = {
        "participant": (body.get("speaker") or body.get("participant") or "Meeting").strip(),
        "text": text,
        "timestamp": body.get("timestamp") or datetime.datetime.utcnow().timestamp(),
        "source": body.get("source") or "livekit",
    }

    transcript = list(s.get("transcript") or [])
    transcript.append(entry)
    if len(transcript) > 2000:
        transcript = transcript[-2000:]

    if session_id not in _compactors:
        _compactors[session_id] = _new_compactor()
        for prev in transcript[:-1]:
            _compactors[session_id].observe_entry(prev)
    _compactors[session_id].observe_entry(entry)
    memory_text = _compactors[session_id].memory_text()

    session_store.patch(session_id, {
        "transcript": transcript,
        "transcript_memory_text": memory_text,
    })

    return {"ok": True, "transcript_entries": len(transcript)}


# ── Endpoints: transcript read ─────────────────────────────────────────────────
@app.get("/sessions/{session_id}/review/transcript")
async def get_transcript(session_id: str) -> list:
    try:
        s = session_store.require(session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Session {session_id!r} not found")
    return s.get("transcript") or []


# ── Endpoints: history ─────────────────────────────────────────────────────────
@app.get("/history")
async def list_history() -> list:
    sessions = session_store.list_all()
    result = []
    for s in sessions:
        summary_obj = s.get("summary") or {}
        result.append({
            "session_id": s.get("session_id"),
            "title": summary_obj.get("title") or f"Meeting {str(s.get('session_id', ''))[:8]}",
            "meeting_url": s.get("meeting_url"),
            "status": s.get("status"),
            "started_at": s.get("started_at"),
            "ended_at": s.get("ended_at"),
            "summary": summary_obj.get("summary"),
            "change_count": len(s.get("changes") or []),
            "stats": {
                "transcript_entries": len(s.get("transcript") or []),
                "topic_count": len(summary_obj.get("key_topics") or []),
                "decision_count": len(summary_obj.get("decisions") or []),
                "action_item_count": len(summary_obj.get("action_items") or []),
            },
            "updated_at": s.get("updated_at"),
        })
    return result
