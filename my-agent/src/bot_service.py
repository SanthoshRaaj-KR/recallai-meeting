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
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from livekit import api as livekit_api
from pydantic import BaseModel

try:
    from .memory_compaction import TranscriptCompactor
    from . import session_store
    from . import org_activity
    from . import rag_sync_store
    from . import action_items_extractor
    from .review_pipeline.rag import ConfluenceVectorIndex
    from .review_pipeline.confluence import RestConfluenceClient
    from .review_pipeline.confluence_pipeline.retrieval import PineconeHybridIndex
    from .review_pipeline.confluence_pipeline.chunker import chunk_markdown_text
except ImportError:
    from memory_compaction import TranscriptCompactor
    import session_store
    import org_activity
    import rag_sync_store
    import action_items_extractor
    from review_pipeline.rag import ConfluenceVectorIndex
    from review_pipeline.confluence import RestConfluenceClient
    from review_pipeline.confluence_pipeline.retrieval import PineconeHybridIndex
    from review_pipeline.confluence_pipeline.chunker import chunk_markdown_text

def _bg_extract_action_items(s: dict) -> None:
    """Run action-item extraction off the request path (LLM call is slow)."""
    import threading
    threading.Thread(
        target=action_items_extractor.extract_and_assign, args=(s,), daemon=True
    ).start()

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
# This bot-service serves a single organisation; used to key the durable, org-global
# RAG sync status so every admin/manager sees the same "last synced".
_ORG_ID = os.getenv("ORG_ID", "").strip()
TOKEN_TTL_HOURS = int(os.getenv("TOKEN_TTL_HOURS", "8"))

_CORS_ORIGINS = [o.strip() for o in os.getenv("CORS_ORIGINS", "*").split(",") if o.strip()]
_BOT_HTML_PATH = Path(__file__).parent / "bot.html"

_RECALL_STATUS_MAP = {
    "joining_call": "joining",
    "in_waiting_room": "joining",
    "in_call_not_recording": "in_meeting",
    "in_call_recording": "in_meeting",
    "recording_permission_denied": "error",
    "done": "ended",
    "call_ended": "ended",
    "fatal": "error",
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

# Observability: Prometheus /metrics + optional OTLP tracing. No-op unless the
# observability deps are installed and OTEL_* env is set — see deploy/observability/.
try:
    from .observability import setup_fastapi_observability
except ImportError:
    from observability import setup_fastapi_observability
setup_fastapi_observability(app, "bot-service")


# ── In-process caches ────────────────────────────────────────────────────────
# TranscriptCompactor is not serialisable so it lives only in this process;
# the compacted text is flushed to Supabase.
_compactors: dict[str, TranscriptCompactor] = {}
# bot_id → session_id index so webhook lookups are O(1) instead of O(N).
_bot_index: dict[str, str] = {}

# ── RAG sync job store ───────────────────────────────────────────────────────
# In-memory only; resets on process restart. Good enough for a manual sync
# operation — users can just click the button again if the server restarts.
_sync_jobs: dict[str, dict] = {}

# Singleton RAG index — lazy init on first sync call.
_rag_index: ConfluenceVectorIndex | None = None
_hybrid_index: PineconeHybridIndex | None = None
_confluence_client: RestConfluenceClient | None = None


def _get_rag_index() -> ConfluenceVectorIndex:
    global _rag_index
    if _rag_index is None:
        _rag_index = ConfluenceVectorIndex()
    return _rag_index


def _get_hybrid_index() -> PineconeHybridIndex:
    global _hybrid_index
    if _hybrid_index is None:
        _hybrid_index = PineconeHybridIndex()
    return _hybrid_index


def _get_confluence_client() -> RestConfluenceClient:
    global _confluence_client
    if _confluence_client is None:
        _confluence_client = RestConfluenceClient()
    return _confluence_client


def _utcnow() -> str:
    return datetime.datetime.utcnow().isoformat() + "Z"


def _session_for_bot(bot_id: str) -> dict | None:
    """Resolve a session dict from a Recall bot_id, via the O(1) local index with
    a Supabase-scan fallback for process restarts. Returns None if unknown."""
    if not bot_id:
        return None
    session_id = _bot_index.get(bot_id)
    s = session_store.get(session_id) if session_id else None
    if s is None:
        s = next((x for x in session_store.list_all() if x.get("bot_id") == bot_id), None)
    if s:
        _bot_index[bot_id] = s["session_id"]
    return s


def _record_participant_event(bot_id: str, participant: dict, kind: str, ts: str) -> None:
    """Merge one Recall participant join/leave into the session's presence map.

    Presence lives in jarvis_sessions.participants (jsonb) keyed by Recall
    participant id: {name, is_host, joined_at, left_at, email?}. Low-frequency
    (a few events per meeting), so a read-merge-write patch is fine here.
    Identity resolution (Recall name → org_user) happens later, at meeting end.
    """
    try:
        s = _session_for_bot(bot_id)
        if not s:
            return
        pid = str(participant.get("id", "")) or (participant.get("name") or "").strip()
        if not pid:
            return
        presence: dict = dict(s.get("participants") or {})
        entry = dict(presence.get(pid) or {})
        entry.setdefault("name", participant.get("name"))
        if participant.get("is_host") is not None:
            entry["is_host"] = participant.get("is_host")
        email = (participant.get("extra_data") or {}).get("email") or participant.get("email")
        if email and not entry.get("email"):
            entry["email"] = email
        if kind == "join" and not entry.get("joined_at"):
            entry["joined_at"] = ts
        if kind == "leave":
            entry["left_at"] = ts
        presence[pid] = entry
        session_store.patch(s["session_id"], {"participants": presence})
        logger.info(
            "participant %s → session %s pid=%s name=%s",
            kind, s["session_id"], pid, entry.get("name"),
        )
    except Exception as exc:  # presence tracking must never break the webhook
        logger.warning("participant event skipped: %s", exc)


def _backfill_participants_from_recall(s: dict) -> None:
    """On meeting end, pull the authoritative participant list (with join/leave
    timestamps) from GET /bot/{id} and merge it into the presence map — covers any
    join/leave webhook we missed. Never raises."""
    try:
        bot_id = s.get("bot_id")
        if not bot_id or not RECALL_API_KEY:
            return
        resp = requests.get(
            f"{RECALL_BASE_URL}/bot/{bot_id}/",
            headers={"Authorization": f"Token {RECALL_API_KEY}"},
            timeout=8,
        )
        if not resp.ok:
            return
        parts = resp.json().get("meeting_participants") or []
        if not parts:
            return
        presence: dict = dict(s.get("participants") or {})
        for p in parts:
            pid = str(p.get("id", "")) or (p.get("name") or "").strip()
            if not pid:
                continue
            entry = dict(presence.get(pid) or {})
            entry.setdefault("name", p.get("name"))
            if p.get("is_host") is not None:
                entry.setdefault("is_host", p.get("is_host"))
            events = p.get("events") or {}
            join_ts = (events.get("join") or {}).get("absolute") or p.get("join_at")
            leave_ts = (events.get("leave") or {}).get("absolute") or p.get("leave_at")
            if join_ts and not entry.get("joined_at"):
                entry["joined_at"] = join_ts
            if leave_ts:
                entry["left_at"] = leave_ts
            email = (p.get("extra_data") or {}).get("email") or p.get("email")
            if email and not entry.get("email"):
                entry["email"] = email
            presence[pid] = entry
        if presence:
            session_store.patch(s["session_id"], {"participants": presence})
            logger.info(
                "participant backfill → session %s (%d participants)",
                s["session_id"], len(presence),
            )
    except Exception as exc:
        logger.warning("participant backfill skipped: %s", exc)


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
        "recording_config": {
            "transcript": {
                "provider": {
                    "assembly_ai_v3_streaming": {
                        "language_code": "en",
                    },
                },
                "diarization": {
                    # Separate audio streams per participant give AssemblyAI
                    # the cleanest signal for speaker attribution.
                    "use_separate_streams_when_available": True,
                },
            },
            "realtime_endpoints": [
                {
                    "type": "webhook",
                    "url": f"{SERVER_URL}/recall-webhook",
                    # transcript.data → diarized transcript; participant_events →
                    # real per-person presence (join/leave) for attendance analytics.
                    "events": [
                        "transcript.data",
                        "participant_events.join",
                        "participant_events.leave",
                    ],
                },
            ],
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


async def _dispatch_agent(room_name: str, confluence_enabled: bool = False) -> None:
    async with livekit_api.LiveKitAPI(
        url=LIVEKIT_URL, api_key=LIVEKIT_API_KEY, api_secret=LIVEKIT_API_SECRET,
    ) as lk:
        dispatch = await lk.agent_dispatch.create_dispatch(
            livekit_api.CreateAgentDispatchRequest(
                agent_name=AGENT_NAME,
                room=room_name,
                metadata=json.dumps({"room_name": room_name, "confluence_enabled": confluence_enabled}),
            )
        )
    logger.info("Agent dispatched — dispatch_sid=%s room=%s confluence=%s", dispatch.sid, room_name, confluence_enabled)


# ── Request / response models ──────────────────────────────────────────────────
class StartBotRequest(BaseModel):
    meeting_url: str
    room_name: Optional[str] = None
    session_id: Optional[str] = None
    team_id: Optional[str] = None  # optional team scoping (port 8003 integration)
    confluence_enabled: bool = False


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

    # Write the session BEFORE calling Recall so any webhook or frontend status
    # poll that arrives in the gap between bot creation and this function
    # completing always finds an existing session rather than a 404.
    now = _utcnow()
    session_data = {
        "session_id": room_name,
        "bot_id": None,
        "meeting_url": body.meeting_url,
        "status": "pending",
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
        "confluence_enabled": body.confluence_enabled,
    }
    session_store.upsert(room_name, session_data)
    _compactors[room_name] = _new_compactor()

    try:
        bot_id = _create_recall_bot(body.meeting_url, room_name)
    except RuntimeError as exc:
        session_store.patch(room_name, {"status": "error", "error": str(exc)})
        raise HTTPException(status_code=400, detail=str(exc))
    except requests.HTTPError as exc:
        err_body = exc.response.text if exc.response is not None else ""
        session_store.patch(room_name, {"status": "error", "error": f"Recall.ai error: {exc}"})
        raise HTTPException(status_code=502, detail=f"Recall.ai error: {exc} — {err_body}")

    # Update index and session with the real bot_id immediately so the webhook
    # handler can do an O(1) lookup even if it fires before we finish the rest.
    _bot_index[bot_id] = room_name
    session_store.patch(room_name, {"bot_id": bot_id, "status": "joining"})

    try:
        await _dispatch_agent(room_name, confluence_enabled=body.confluence_enabled)
    except Exception as exc:
        logger.warning("Agent dispatch failed (non-fatal): %s", exc)

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
                    if new_status == "ended":
                        _backfill_participants_from_recall(s)
                        s = session_store.get(session_id) or s
                        org_activity.record_participants(s)  # real per-attendee attendance
                        _bg_extract_action_items(s)
        except Exception as exc:
            logger.debug("Recall status poll error: %s", exc)

    return {
        "status": s.get("status"),
        "session_id": session_id,
        "bot_id": s.get("bot_id"),
        "meeting_url": s.get("meeting_url"),
        "change_count": len(s.get("changes") or []),
        "error": s.get("error"),
        "started_at": s.get("started_at"),
        "ended_at": s.get("ended_at"),
        "end_reason": None,
        "recall_status_code": None,
    }


@app.get("/bot/status")
async def bot_status_no_session() -> dict:
    return {
        "status": "idle", "session_id": None, "bot_id": None,
        "meeting_url": None, "change_count": 0, "error": None,
        "started_at": None, "ended_at": None, "end_reason": None, "recall_status_code": None,
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
    # Authoritative participant list before crediting attendance.
    _backfill_participants_from_recall(s)
    s = session_store.get(session_id) or s
    org_activity.record_participants(s)  # real per-attendee attendance
    _bg_extract_action_items(s)
    return {
        "status": "ended", "session_id": session_id, "bot_id": s.get("bot_id"),
        "meeting_url": s.get("meeting_url"), "change_count": len(s.get("changes") or []),
        "error": s.get("error"), "started_at": s.get("started_at"), "ended_at": s.get("ended_at"),
        "end_reason": None, "recall_status_code": None,
    }


@app.post("/sessions/{session_id}/action-items/extract")
async def extract_action_items(session_id: str) -> dict:
    """Re-run transcript → action-item extraction + auto-assignment for a session.

    Runs synchronously (off the event loop) so the caller can refetch the items
    right after. Idempotent: existing items are preserved (insert-ignore-duplicates).
    """
    try:
        s = session_store.require(session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Session {session_id!r} not found")
    await asyncio.to_thread(action_items_extractor.extract_and_assign, s)
    return {"ok": True, "session_id": session_id}


# ── Endpoints: Jarvis direct call ─────────────────────────────────────────────

class JarvisCallTokenRequest(BaseModel):
    participant_name: Optional[str] = "user"
    confluence_enabled: bool = False


class JarvisCallTokenResponse(BaseModel):
    livekit_url: str
    token: str
    room_name: str


async def _dispatch_jarvis_call_agent(
    room_name: str, session_id: str, confluence_enabled: bool = False
) -> None:
    async with livekit_api.LiveKitAPI(
        url=LIVEKIT_URL, api_key=LIVEKIT_API_KEY, api_secret=LIVEKIT_API_SECRET,
    ) as lk:
        dispatch = await lk.agent_dispatch.create_dispatch(
            livekit_api.CreateAgentDispatchRequest(
                agent_name=AGENT_NAME,
                room=room_name,
                metadata=json.dumps({
                    "mode": "jarvis_call",
                    "session_id": session_id,
                    "confluence_enabled": confluence_enabled,
                }),
            )
        )
    logger.info(
        "Jarvis call agent dispatched — dispatch_sid=%s room=%s session=%s",
        dispatch.sid, room_name, session_id,
    )


@app.post("/sessions/{session_id}/jarvis-call/token", response_model=JarvisCallTokenResponse)
async def jarvis_call_token(
    session_id: str, body: JarvisCallTokenRequest
) -> JarvisCallTokenResponse:
    """Mint a LiveKit token for a one-on-one voice call with Jarvis.

    Creates (or reuses) a dedicated room for this session and dispatches the
    JarvisCallAssistant agent to it.  The agent loads the full meeting transcript
    and compacted memory from session_store so it has complete meeting context.
    """
    try:
        session_store.require(session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Session {session_id!r} not found")

    if not LIVEKIT_URL or not LIVEKIT_API_KEY or not LIVEKIT_API_SECRET:
        raise HTTPException(
            status_code=503,
            detail="LiveKit is not configured on this server.",
        )

    room_name = f"jarvis-call-{session_id}"
    identity = (body.participant_name or "user").strip() or "user"
    token = _mint_token(room_name, identity, can_publish=True)

    try:
        await _dispatch_jarvis_call_agent(room_name, session_id, body.confluence_enabled)
    except Exception as exc:
        logger.warning("Jarvis call agent dispatch failed (non-fatal): %s", exc)

    return JarvisCallTokenResponse(
        livekit_url=LIVEKIT_URL,
        token=token,
        room_name=room_name,
    )


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
        s = _session_for_bot(bot_id)
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
                if new_status == "ended":
                    ended = {**s, **updates}
                    # Pull the authoritative participant list before crediting attendance.
                    _backfill_participants_from_recall(ended)
                    ended = session_store.get(s["session_id"]) or ended
                    org_activity.record_participants(ended)  # real per-attendee attendance
                    _bg_extract_action_items(ended)

    # ── Recall participant presence (real attendance: join / leave) ──────────────
    elif event in ("participant_events.join", "participant_events.leave"):
        bot_id = (data.get("bot") or {}).get("id", "") or data.get("bot_id", "")
        inner = data.get("data") or {}
        participant = inner.get("participant") or data.get("participant") or {}
        ts = (inner.get("timestamp") or {}).get("absolute") or _utcnow()
        kind = "join" if event.endswith("join") else "leave"
        if bot_id and participant:
            _record_participant_event(bot_id, participant, kind, ts)

    # ── Recall real-time diarized transcript (AssemblyAI v3 via Recall native) ──
    # Recall fires transcript.data for each finalized utterance with participant
    # name and diarized words. We store these as the authoritative transcript
    # for the post-meeting Confluence pipeline (decisions, changes, action items).
    # The LiveKit STT pipeline (/livekit-transcript) is left untouched — it still
    # feeds the agent's real-time voice responses.
    elif event == "transcript.data":
        bot_id = (data.get("bot") or {}).get("id", "")
        if bot_id:
            s = _session_for_bot(bot_id)
            if s:
                inner = data.get("data") or {}
                participant = inner.get("participant") or {}
                speaker_name = (
                    participant.get("name")
                    or f"Speaker {participant.get('id', '?')}"
                )
                words = inner.get("words") or []
                text = " ".join(w.get("text", "") for w in words).strip()
                if text:
                    first_word = words[0] if words else {}
                    ts_obj = first_word.get("start_timestamp") or {}
                    timestamp = ts_obj.get("relative") or datetime.datetime.utcnow().timestamp()

                    entry = {
                        "participant": speaker_name,
                        "text": text,
                        "timestamp": timestamp,
                        "source": "recall",
                    }
                    # O(1) append to the turns table — no blob read-modify-write.
                    session_store.append_transcript_turn(s["session_id"], entry)
                    logger.debug(
                        "Recall transcript → session %s speaker=%s len=%d",
                        s["session_id"], speaker_name, len(words),
                    )

    return {"ok": True}


# ── Endpoints: transcript ingestion ───────────────────────────────────────────
@app.post("/livekit-transcript/{session_id}")
async def receive_livekit_transcript(session_id: str, request: Request) -> dict:
    """Live meeting transcript ingest — LiveKit STT (fast, used DURING the call).

    This is the live/real-time transcript stored on jarvis_sessions.transcript and
    shown in the live view. Post-meeting artifacts (proposals/summary/MOM/action
    items) instead use the diarized Recall transcript in session_transcript_turns.
    """
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
    """Return a session's transcript entries ({participant, text, timestamp, source}).

    Includes the diarized Recall turns (source="recall", real speaker names) — which
    the UI displays — followed by the live LiveKit STT lines (source="livekit") so any
    consumer can filter by source. Recall turns first, in chronological (seq) order.
    """
    try:
        s = session_store.require(session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Session {session_id!r} not found")
    recall_turns = session_store.get_transcript_turns(session_id)
    livekit = s.get("transcript") or []
    return recall_turns + livekit


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


# ── Endpoints: RAG incremental sync ────────────────────────────────────────────

def _all_indexes_exist(rag: ConfluenceVectorIndex, hybrid: PineconeHybridIndex) -> bool:
    """Return True only when all three Pinecone indexes are present."""
    try:
        from pinecone import Pinecone
        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY", "").strip())
        existing = {it["name"] if isinstance(it, dict) else it.name for it in pc.list_indexes()}
        return (
            rag.index_name in existing
            and hybrid.dense_index in existing
            and hybrid.sparse_index in existing
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Index existence check failed — assuming all exist: %s", exc)
        return True  # avoid accidental full reindex on transient errors


async def _run_rag_sync(job_id: str) -> None:
    """Sync all three Pinecone indexes from live Confluence pages.

    Indexes synced:
      1. ConfluenceVectorIndex  (confluence-review-rag-v2)  — in-meeting live RAG
      2. PineconeHybridIndex dense  (confluence-corpus-dense)  — proposal pipeline
      3. PineconeHybridIndex sparse (confluence-corpus-sparse) — proposal pipeline

    Version tracking is applied independently to each group. Pages are fetched
    from Confluence at most once per run — a cache dict is shared between both
    sync passes so the hybrid index reuses pages already pulled for the live-RAG
    sync rather than making a second round of Confluence API calls.

    If any of the three indexes is missing it is created automatically and every
    page is force-reindexed into all indexes so they stay in sync.
    """
    job = _sync_jobs[job_id]
    try:
        confluence = _get_confluence_client()
        rag = _get_rag_index()
        hybrid = _get_hybrid_index()

        # ── Check index existence — if any index is missing force a full reindex ──
        force_full = not await asyncio.to_thread(_all_indexes_exist, rag, hybrid)
        if force_full:
            logger.info("RAG sync %s: one or more indexes missing — creating indexes upfront", job_id)
            job["current_page"] = "Creating missing indexes…"
            # Eagerly create all 3 indexes before the upsert loop so failures are
            # surfaced now rather than silently accumulating per-page failures.
            try:
                await asyncio.to_thread(rag._index)
                logger.info("RAG sync %s: live RAG index ready (%s)", job_id, rag.index_name)
            except Exception as exc:
                logger.error("RAG sync %s: could not create live RAG index %r: %s", job_id, rag.index_name, exc)
            try:
                await asyncio.to_thread(lambda: hybrid._index("dense"))
                logger.info("RAG sync %s: dense index ready (%s)", job_id, hybrid.dense_index)
            except Exception as exc:
                logger.error("RAG sync %s: could not create dense index %r: %s", job_id, hybrid.dense_index, exc)
            try:
                await asyncio.to_thread(lambda: hybrid._index("sparse"))
                logger.info("RAG sync %s: sparse index ready (%s)", job_id, hybrid.sparse_index)
            except Exception as exc:
                logger.warning("RAG sync %s: sparse index unavailable (%s) — dense-only mode", job_id, exc)

        # ── List all pages (lightweight — version numbers only) ───────────────────
        listings = await asyncio.to_thread(confluence.list_pages, 500)
        job["total"] = len(listings)
        job["current_page"] = "Checking indexes…"

        # Shared page cache so both passes never double-fetch the same page.
        _fetched_pages: dict[str, object] = {}

        def _capturing_fetch(page_id: str):
            page = confluence.fetch_page(page_id)
            _fetched_pages[page_id] = page
            return page

        def _cached_fetch(page_id: str):
            if page_id in _fetched_pages:
                return _fetched_pages[page_id]
            return _capturing_fetch(page_id)

        def _rag_progress(done: int, total: int, title: str) -> None:
            job["checked"] = done
            job["total_stale"] = total
            job["current_page"] = title

        # ── Pass 1: sync ConfluenceVectorIndex (in-meeting RAG) ───────────────────
        result = await asyncio.to_thread(
            rag.sync_index,
            listings,
            _capturing_fetch,
            _rag_progress,
            force=force_full,
        )

        # ── Pass 2: sync PineconeHybridIndex (proposal pipeline) ─────────────────
        # hybrid.sync_index does its own version check and re-embeds only stale
        # pages. It reuses _cached_fetch so pages already pulled above aren't
        # re-requested from Confluence.
        job["current_page"] = "Syncing proposal indexes…"
        hybrid_result = await asyncio.to_thread(
            hybrid.sync_index,
            listings,
            _cached_fetch,
            force=force_full,
        )

        job.update({
            "status": "done",
            "checked": result["checked"],
            "changed": result["changed"],
            "skipped": result["skipped"],
            "failed": result["failed"],
            "deleted": result.get("deleted", 0),
            "hybrid_changed": hybrid_result["changed"],
            "hybrid_failed": hybrid_result["failed"],
            "current_page": "",
            "finished_at": _utcnow(),
        })
        logger.info(
            "RAG sync %s complete: rag_changed=%d hybrid_changed=%d skipped=%d failed=%d",
            job_id, result["changed"], hybrid_result["changed"], result["skipped"], result["failed"],
        )
        rag_sync_store.save(job, _ORG_ID)  # durable org-global status
    except Exception as exc:  # noqa: BLE001
        logger.error("RAG sync %s failed: %s", job_id, exc)
        job.update({"status": "error", "error": str(exc), "finished_at": _utcnow()})
        rag_sync_store.save(job, _ORG_ID)


# ── Knowledge-base write auth ───────────────────────────────────────────────────
# Updating the KB is a manager-and-above action. The org-service issues the JWT
# (HS256, shared JWT_SECRET); we verify it here and check the role. Fails closed:
# if JWT_SECRET is unset we refuse rather than silently allow.

_JWT_SECRET = os.getenv("JWT_SECRET", "")
_KB_EDITOR_ROLES = {"CEO", "ADMIN", "MANAGER"}


def require_kb_editor(authorization: Optional[str] = Header(default=None)) -> dict:
    if not _JWT_SECRET:
        raise HTTPException(503, "Knowledge-base auth is not configured (JWT_SECRET unset)")
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Missing bearer token")
    token = authorization.split(" ", 1)[1].strip()
    try:
        import jwt as _pyjwt
        claims = _pyjwt.decode(token, _JWT_SECRET, algorithms=["HS256"])
    except Exception as exc:  # noqa: BLE001 — any decode/expiry failure → 401
        raise HTTPException(status_code=401, detail=f"Invalid or expired token: {exc}")
    if claims.get("role") not in _KB_EDITOR_ROLES:
        raise HTTPException(status_code=403, detail="Only a manager, ADMIN, or CEO can update the knowledge base")
    return claims


@app.post("/rag/sync")
async def start_rag_sync(claims: dict = Depends(require_kb_editor)) -> dict:
    """Start an incremental Confluence → Pinecone re-index job. Manager+ only.

    Returns immediately with a ``job_id``.  Poll ``GET /rag/sync/{job_id}``
    for progress.  Only pages whose Confluence version number is higher than
    what is stored in Pinecone are re-embedded; unchanged pages are skipped.
    """
    try:
        _get_rag_index()  # Validate Pinecone is configured before starting.
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"RAG not configured: {exc}")
    try:
        _get_confluence_client()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Confluence not configured: {exc}")

    job_id = str(uuid.uuid4())
    _sync_jobs[job_id] = {
        "job_id": job_id,
        "status": "running",
        "total": 0,
        "total_stale": 0,
        "checked": 0,
        "changed": 0,
        "skipped": 0,
        "failed": 0,
        "deleted": 0,
        "current_page": "Starting…",
        "error": None,
        "started_at": _utcnow(),
        "finished_at": None,
        "synced_by": claims.get("sub"),
    }
    rag_sync_store.save(_sync_jobs[job_id], _ORG_ID)  # publish "running" org-wide
    asyncio.create_task(_run_rag_sync(job_id))
    return {"job_id": job_id, "status": "running"}


@app.get("/rag/sync/latest")
async def get_latest_rag_sync() -> dict:
    """Return the org's latest sync — durable and shared across restarts.

    Prefers the in-memory job while one is live (for real-time progress), otherwise
    reads the persisted org-global status from Supabase.
    """
    mem = max(_sync_jobs.values(), key=lambda j: j.get("started_at", "")) if _sync_jobs else None
    db = rag_sync_store.latest(_ORG_ID)
    # In-memory wins on ties so live progress is shown during an active sync.
    if mem and (not db or mem.get("started_at", "") >= db.get("started_at", "")):
        return mem
    if db:
        return db
    raise HTTPException(status_code=404, detail="No sync job found")


@app.get("/rag/sync/{job_id}")
async def get_rag_sync_status(job_id: str) -> dict:
    """Poll the status and progress of a sync job (in-memory, else persisted)."""
    if job_id in _sync_jobs:
        return _sync_jobs[job_id]
    db = rag_sync_store.get(job_id, _ORG_ID)
    if db:
        return db
    raise HTTPException(status_code=404, detail=f"Sync job {job_id!r} not found")
