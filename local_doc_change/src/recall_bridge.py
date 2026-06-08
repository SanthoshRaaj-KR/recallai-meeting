"""
Recall.ai Bridge for my-agent LiveKit integration.

This FastAPI server bridges Recall.ai meeting bots to the my-agent LiveKit agent
and exposes all endpoints consumed by the sync-sage-bot frontend.

Architecture:
  1. POST /bot/start           → creates a Recall.ai bot that joins the given meeting URL
  2. Recall bot loads GET /bot-page via output_media.camera.kind=webpage
  3. bot-page (bot.html) connects to LiveKit in two rooms:
       Publisher  → joins as "recall-browser-{room_name}", publishes meeting audio
       Subscriber → plays back agent TTS audio into the meeting via <audio> element
  4. my-agent AgentSession subscribes ONLY to "recall-browser-{room_name}" for STT,
     so it hears the mixed meeting audio (all participants combined).

Frontend endpoints (sync-sage-bot):
  GET  /health                                      Bridge health check
  POST /bot/start                                   Start Recall bot + dispatch agent
  GET  /sessions/{id}/bot/status                    Poll bot / session status
  GET  /sessions/{id}/review/changes                List proposed Confluence changes
  POST /sessions/{id}/review/execute                Approve / execute changes
  POST /sessions/{id}/review/changes/propose        Propose new Confluence changes
  GET  /sessions/{id}/review/summary                Meeting summary + MoM
  POST /sessions/{id}/review/chat                   Chat about the meeting
  POST /sessions/{id}/review/regenerate/{pid}       Re-draft a single proposal
  POST /review/pipeline/start                       Kick off the analysis pipeline
  GET  /review/pipeline/{job_id}/stream             SSE pipeline progress stream
  GET  /history                                     All past sessions

Run alongside the agent:
    uv run uvicorn src.recall_bridge:app --host 0.0.0.0 --port 8000

Port note: the sync-sage-bot Vite dev proxy routes /api/local-doc/*, /api/bot/*
and /api/sessions/* to botTarget (http://localhost:8000), and only /api/review/*
to confluenceTarget (http://localhost:8001). For the local-doc sandbox this
bridge MUST listen on :8000 (override with VITE_API_PROXY_TARGET if you need a
different port).

Required environment variables (.env.local):
    LIVEKIT_URL            wss://your-project.livekit.cloud
    LIVEKIT_API_KEY        your LiveKit API key
    LIVEKIT_API_SECRET     your LiveKit API secret
    RECALL_API_KEY         your Recall.ai API key
    BRIDGE_SERVER_URL      public HTTPS URL for this server (e.g. ngrok tunnel)

Optional:
    RECALL_API_REGION      Recall region (default: us-west-2)
    BOT_NAME               display name for the bot in the meeting (default: Meeting Assistant)
    AGENT_NAME             LiveKit agent name to dispatch (default: my-agent)
    TOKEN_TTL_HOURS        LiveKit token lifetime in hours (default: 8)
    CORS_ORIGINS           comma-separated allowed origins (default: *)
"""

import asyncio
import datetime
import json
import logging
import os
import uuid
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Optional

import requests
from pathlib import Path

# ── Local-doc pipeline ──
import sys, pathlib as _pathlib
sys.path.insert(0, str(_pathlib.Path(__file__).parent.parent))
from pipeline.run import run_pipeline, PipelineConfig, PIPELINE_STAGES
from pipeline.safe_apply import SafeApply
from models.proposals import LocalDocProposal

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from livekit import api as livekit_api
from pydantic import BaseModel

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

# Public HTTPS URL of this server — Recall requires HTTPS for output_media URLs.
SERVER_URL = os.getenv("BRIDGE_SERVER_URL", "").rstrip("/")

BOT_NAME = os.getenv("BOT_NAME", "Meeting Assistant")
AGENT_NAME = os.getenv("AGENT_NAME", "my-agent")
TOKEN_TTL_HOURS = int(os.getenv("TOKEN_TTL_HOURS", "8"))

_CORS_ORIGINS = [o.strip() for o in os.getenv("CORS_ORIGINS", "*").split(",") if o.strip()]

_BOT_HTML_PATH = Path(__file__).parent / "bot.html"

# ── App setup ──────────────────────────────────────────────────────────────────
app = FastAPI(title="Recall.ai Bridge for my-agent")

app.add_middleware(
    CORSMiddleware,
    allow_origins=_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _utcnow() -> str:
    return datetime.datetime.utcnow().isoformat() + "Z"


# ── In-memory session store ────────────────────────────────────────────────────

# Maps session_id (== room_name) → _SessionRecord.
# Lives in process memory; reset on server restart.
_sessions: dict[str, "_SessionRecord"] = {}
# Maps pipeline job_id → pipeline state dict.
_pipelines: dict[str, dict] = {}

# ── Local-doc pipeline ──
_local_doc_jobs: dict[str, dict] = {}  # job_id -> job state dict
_local_doc_proposals: dict[str, list[dict]] = {}  # session_id -> list of proposal dicts

# Recall.ai status_changes code → BotStatus
_RECALL_STATUS_MAP = {
    "joining": "joining",
    "in_call_not_recording": "in_meeting",
    "in_call_recording": "in_meeting",
    "done": "ended",
    "call_ended": "ended",
    "error": "error",
}


class _SessionRecord:
    """Lightweight in-memory state for one meeting session."""

    def __init__(self, session_id: str, bot_id: str | None, meeting_url: str | None) -> None:
        self.session_id = session_id
        self.bot_id = bot_id
        self.meeting_url = meeting_url
        self.status: str = "joining"
        self.error: str | None = None
        self.changes: list[dict] = []
        self._change_counter: int = 0
        self.summary: dict | None = None
        # Final meeting transcript, pushed once by the LiveKit agent when the
        # meeting ends (reuses the agent's existing transcript buffer — no
        # second transcription). Consumed by the local-doc pipeline.
        self.transcript: str = ""
        self.started_at: str = _utcnow()
        self.ended_at: str | None = None
        self.updated_at: str = self.started_at

    def _touch(self) -> None:
        self.updated_at = _utcnow()

    def refresh_recall_status(self) -> None:
        """Poll Recall.ai for the live bot status and update self.status in-place."""
        if not self.bot_id or not RECALL_API_KEY:
            return
        try:
            resp = requests.get(
                f"{RECALL_BASE_URL}/bot/{self.bot_id}/",
                headers={"Authorization": f"Token {RECALL_API_KEY}"},
                timeout=5,
            )
            if not resp.ok:
                return
            data = resp.json()
            changes = data.get("status_changes") or []
            code = changes[-1].get("code", "") if changes else ""
            new_status = _RECALL_STATUS_MAP.get(code, self.status)
            if new_status != self.status:
                self.status = new_status
                if new_status in ("ended", "error") and not self.ended_at:
                    self.ended_at = _utcnow()
                self._touch()
        except Exception as exc:  # noqa: BLE001
            logger.debug("Recall status poll error: %s", exc)

    def as_session_status(self) -> dict:
        return {
            "status": self.status,
            "session_id": self.session_id,
            "bot_id": self.bot_id,
            "meeting_url": self.meeting_url,
            "change_count": len(self.changes),
            "error": self.error,
            "ended_at": self.ended_at,
            "end_reason": None,
            "recall_status_code": None,
        }

    def as_history_item(self) -> dict:
        summary_obj = self.summary or {}
        return {
            "session_id": self.session_id,
            "title": summary_obj.get("title") or f"Meeting {self.session_id[:8]}",
            "meeting_url": self.meeting_url,
            "status": self.status,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "summary": summary_obj.get("summary"),
            "change_count": len(self.changes),
            "stats": {
                "transcript_entries": 0,
                "topic_count": len(summary_obj.get("key_topics", [])),
                "decision_count": len(summary_obj.get("decisions", [])),
                "action_item_count": len(summary_obj.get("action_items", [])),
            },
            "updated_at": self.updated_at,
        }


def _require_session(session_id: str) -> _SessionRecord:
    s = _sessions.get(session_id)
    if not s:
        raise HTTPException(status_code=404, detail=f"Session {session_id!r} not found")
    return s


# ── Request / response models ──────────────────────────────────────────────────

class StartBotRequest(BaseModel):
    meeting_url: str
    room_name: Optional[str] = None
    session_id: Optional[str] = None  # frontend may pass this


class StartBotResponse(BaseModel):
    status: str
    bot_id: str
    room_name: str
    # session_id mirrors room_name — required by the sync-sage-bot frontend's
    # SessionStatus interface. Without it, MeetingInput cannot pass the session
    # to MeetingLive via the URL, causing MeetingLive to see status="idle" and
    # dispatch a second Recall bot into the same meeting.
    session_id: str
    meeting_url: str
    change_count: int = 0


class ExecuteBody(BaseModel):
    ids: Optional[list[int]] = None
    proposal_id: Optional[str] = None
    proposal_ids: Optional[list[str]] = None


class ProposeBody(BaseModel):
    query: Optional[str] = None


class ChatBody(BaseModel):
    messages: list[dict]


class PipelineStartBody(BaseModel):
    session_id: str


# ── Local-doc pipeline ──
class LocalDocPipelineStartBody(BaseModel):
    session_id: str
    doc_folder: str
    use_embeddings: bool = True
    rerank: bool = True
    contextual_retrieval: bool = True
    # Optional: paste a meeting transcript directly (no Recall meeting needed).
    # When omitted, falls back to the session's transcript.
    transcript: str | None = None


class LocalDocExecuteBody(BaseModel):
    proposal_ids: list[str]
    # Optional per-proposal user edits: proposal_id -> revised after_content.
    # When present, the user's text is written instead of the AI draft.
    edited_content: dict[str, str] = {}


# ── LiveKit token minting ──────────────────────────────────────────────────────

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


# ── Recall bot creation ────────────────────────────────────────────────────────

def _create_recall_bot(meeting_url: str, room_name: str) -> str:
    if not RECALL_API_KEY:
        raise RuntimeError("RECALL_API_KEY is not configured")
    if not SERVER_URL:
        raise RuntimeError(
            "BRIDGE_SERVER_URL is not configured. "
            "Set it to your public HTTPS server URL (e.g. ngrok tunnel)."
        )
    if not SERVER_URL.startswith("https://"):
        raise RuntimeError(
            f"BRIDGE_SERVER_URL must start with https:// (got: {SERVER_URL}). "
            "Recall.ai requires HTTPS for output_media URLs."
        )
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
            "camera": {
                "kind": "webpage",
                "config": {"url": bot_page_url},
            },
        },
    }

    response = requests.post(
        f"{RECALL_BASE_URL}/bot/",
        headers={
            "Authorization": f"Token {RECALL_API_KEY}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=15,
    )

    if not response.ok:
        logger.error(
            "Recall bot creation failed: %s %s — body: %s",
            response.status_code, response.reason, response.text,
        )
        response.raise_for_status()

    bot_id = response.json()["id"]
    logger.info("Recall bot created — bot_id=%s room=%s meeting=%s", bot_id, room_name, meeting_url)
    return bot_id


# ── LiveKit agent dispatch ─────────────────────────────────────────────────────

async def _dispatch_agent(room_name: str) -> None:
    async with livekit_api.LiveKitAPI(
        url=LIVEKIT_URL,
        api_key=LIVEKIT_API_KEY,
        api_secret=LIVEKIT_API_SECRET,
    ) as lk:
        dispatch = await lk.agent_dispatch.create_dispatch(
            livekit_api.CreateAgentDispatchRequest(
                agent_name=AGENT_NAME,
                room=room_name,
                metadata=json.dumps({"room_name": room_name}),
            )
        )
    logger.info("Agent dispatched — agent=%s room=%s dispatch_sid=%s", AGENT_NAME, room_name, dispatch.sid)


# ── Endpoints: bot lifecycle ───────────────────────────────────────────────────

@app.post("/bot/start", response_model=StartBotResponse)
async def start_bot(body: StartBotRequest) -> StartBotResponse:
    """
    Start a Recall.ai bot that joins the given meeting URL and connects it to my-agent.

    Example:
        curl -X POST http://localhost:8001/bot/start \\
             -H "Content-Type: application/json" \\
             -d '{"meeting_url": "https://zoom.us/j/123456789"}'
    """
    room_name = body.room_name or body.session_id or str(uuid.uuid4())

    try:
        bot_id = _create_recall_bot(body.meeting_url, room_name)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except requests.HTTPError as exc:
        err_body = exc.response.text if exc.response is not None else ""
        raise HTTPException(status_code=502, detail=f"Recall.ai API error: {exc} — {err_body}")

    try:
        await _dispatch_agent(room_name)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Agent dispatch failed (non-fatal in dev mode): %s", exc)

    # Register session so subsequent status / review calls work
    _sessions[room_name] = _SessionRecord(room_name, bot_id, body.meeting_url)

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
    """Serve the LiveKit bridge HTML page to Recall's headless Chrome."""
    if not _BOT_HTML_PATH.exists():
        logger.error("bot.html not found at %s", _BOT_HTML_PATH)
        return HTMLResponse(
            "<html><body>bot.html not found — ensure src/bot.html exists</body></html>",
            status_code=500,
        )
    return HTMLResponse(_BOT_HTML_PATH.read_text(encoding="utf-8"))


# ── Endpoints: health ──────────────────────────────────────────────────────────

@app.get("/health")
async def health() -> dict:
    """Bridge health check — called by the frontend to verify connectivity."""
    return {
        "status": "ok",
        "agent_name": AGENT_NAME,
        "recall_region": RECALL_API_REGION,
        "recall_configured": bool(RECALL_API_KEY),
        "livekit_configured": bool(LIVEKIT_URL and LIVEKIT_API_KEY and LIVEKIT_API_SECRET),
        "server_url": SERVER_URL or "<not set>",
    }


# ── Endpoints: session status ──────────────────────────────────────────────────

@app.get("/sessions/{session_id}/bot/status")
async def session_bot_status(session_id: str) -> dict:
    """Poll the current status of a session's Recall bot."""
    s = _require_session(session_id)
    s.refresh_recall_status()
    return s.as_session_status()


@app.get("/bot/status")
async def bot_status_no_session() -> dict:
    """Fallback status check when no session is active."""
    return {
        "status": "idle",
        "session_id": None,
        "bot_id": None,
        "meeting_url": None,
        "change_count": 0,
        "error": None,
        "ended_at": None,
        "end_reason": None,
        "recall_status_code": None,
    }


# ── Endpoints: review — changes ────────────────────────────────────────────────

@app.get("/sessions/{session_id}/review/changes")
async def list_changes(session_id: str) -> list:
    """Return all proposed Confluence changes for this session."""
    return _require_session(session_id).changes


@app.post("/sessions/{session_id}/review/execute")
async def execute_changes(session_id: str, body: ExecuteBody) -> dict:
    """
    Approve / execute one or more proposed changes.

    Accepts three shapes (matching the three callers in api.ts):
      - {proposal_id: str}        → executeProposal()
      - {proposal_ids: [str,...]}  → executeProposalsForPage()
      - {ids: [int,...]}           → executeChanges()
    """
    s = _require_session(session_id)

    if body.proposal_id is not None:
        pid = str(body.proposal_id)
        for ch in s.changes:
            if str(ch.get("id")) == pid:
                ch["status"] = "executed"
                s._touch()
                return {"success": True}
        raise HTTPException(status_code=404, detail=f"Proposal {pid!r} not found")

    if body.proposal_ids is not None:
        results = []
        for pid in body.proposal_ids:
            matched = next((ch for ch in s.changes if str(ch.get("id")) == pid), None)
            if matched:
                matched["status"] = "executed"
                results.append({"page": matched.get("page_title", pid), "success": True})
            else:
                results.append({"page": pid, "success": False, "message": "Not found"})
        s._touch()
        return {"results": results}

    if body.ids is not None:
        results = []
        for cid in body.ids:
            matched = next((ch for ch in s.changes if ch.get("id") == cid), None)
            if matched:
                matched["status"] = "executed"
                results.append({"id": cid, "success": True})
            else:
                results.append({"id": cid, "success": False, "error": "Not found"})
        s._touch()
        return {"results": results}

    raise HTTPException(status_code=422, detail="Provide ids, proposal_id, or proposal_ids")


# ── Endpoints: review — propose ────────────────────────────────────────────────

@app.post("/sessions/{session_id}/review/changes/propose")
async def propose_changes(session_id: str, body: ProposeBody) -> dict:
    """
    Propose new Confluence changes for this session.

    Placeholder: returns existing pending changes.
    Wire up to confluence_logic agents for real AI-generated proposals.
    """
    s = _require_session(session_id)
    pending = [ch for ch in s.changes if ch.get("status") == "pending"]
    return {"changes": pending, "generated_count": 0}


# ── Endpoints: review — summary ────────────────────────────────────────────────

@app.get("/sessions/{session_id}/review/summary")
async def get_summary(session_id: str) -> dict:
    """Return the meeting summary (MoM, decisions, action items, etc.)."""
    s = _require_session(session_id)
    if s.summary:
        return s.summary
    today = datetime.datetime.utcnow().strftime("%B %d, %Y")
    return {
        "title": f"Meeting {s.session_id[:8]}",
        "session_id": s.session_id,
        "date": today,
        "summary": (
            "Transcript is being captured. "
            "The summary will appear once the meeting ends and the pipeline completes."
        ),
        "key_topics": [],
        "action_items": [],
        "decisions": [],
        "participants": [],
        "mom": [],
        "transcript_highlights": [],
        "stats": {
            "transcript_entries": 0,
            "topic_count": 0,
            "decision_count": 0,
            "action_item_count": 0,
        },
    }


# ── Endpoints: review — chat ───────────────────────────────────────────────────

@app.post("/sessions/{session_id}/review/chat")
async def chat_with_meeting(session_id: str, body: ChatBody) -> dict:
    """
    Answer a question about the meeting using its transcript + summary.

    Placeholder: returns a canned response.
    Wire up to confluence_logic meeting_responder / graph_rag for real answers.
    """
    s = _require_session(session_id)
    return {
        "answer": (
            "Transcript processing is still in progress. "
            "Full chat will be available once the pipeline completes."
        ),
        "session_id": session_id,
        "context": {
            "transcript_entries": 0,
            "has_summary": s.summary is not None,
        },
    }


# ── Endpoints: review — regenerate proposal ────────────────────────────────────

@app.post("/sessions/{session_id}/review/regenerate/{proposal_id}")
async def regenerate_proposal(session_id: str, proposal_id: str) -> dict:
    """Re-draft a single proposal against the current live page."""
    s = _require_session(session_id)
    for ch in s.changes:
        if str(ch.get("id")) == proposal_id:
            ch["status"] = "pending"
            ch["regenerate_available"] = True
            s._touch()
            return ch
    raise HTTPException(status_code=404, detail=f"Proposal {proposal_id!r} not found")


# ── Endpoints: pipeline ────────────────────────────────────────────────────────

@app.post("/review/pipeline/start")
async def pipeline_start(body: PipelineStartBody) -> dict:
    """Kick off the post-meeting analysis pipeline for a session."""
    if body.session_id not in _sessions:
        raise HTTPException(status_code=404, detail=f"Session {body.session_id!r} not found")
    job_id = str(uuid.uuid4())
    _pipelines[job_id] = {
        "job_id": job_id,
        "session_id": body.session_id,
        "status": "running",
        "stage": None,
        "created_at": _utcnow(),
    }
    logger.info("Pipeline started — job_id=%s session=%s", job_id, body.session_id)
    return {"job_id": job_id, "status": "running"}


async def _pipeline_sse(job_id: str) -> AsyncGenerator[str, None]:
    """
    Emit SSE events through each pipeline stage.

    Placeholder: completes immediately with 0 proposals.
    Wire up to confluence_logic review/api.py for real AI-generated proposals.
    """
    stages = [
        "transcript_source",
        "fact_extraction",
        "rag_retrieval",
        "drafting",
        "verification",
    ]
    for stage in stages:
        if job_id in _pipelines:
            _pipelines[job_id]["stage"] = stage
        yield f"event: stage_start\ndata: {json.dumps({'stage': stage})}\n\n"
        await asyncio.sleep(0.4)

    if job_id in _pipelines:
        _pipelines[job_id]["status"] = "completed"
        _pipelines[job_id]["stage"] = None

    yield f"event: pipeline_complete\ndata: {json.dumps({'proposal_count': 0})}\n\n"


@app.get("/review/pipeline/{job_id}/stream")
async def pipeline_stream(job_id: str, token: str = "") -> StreamingResponse:
    """SSE stream for pipeline progress — consumed by PipelinePage."""
    if job_id not in _pipelines:
        raise HTTPException(status_code=404, detail=f"Pipeline job {job_id!r} not found")
    return StreamingResponse(
        _pipeline_sse(job_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# ── Endpoints: history ─────────────────────────────────────────────────────────

@app.get("/history")
async def list_history() -> list:
    """Return all sessions as history items (most recent first)."""
    items = [s.as_history_item() for s in _sessions.values()]
    items.sort(key=lambda x: x.get("updated_at") or "", reverse=True)
    return items


# ── Endpoints: local-doc pipeline ─────────────────────────────────────────────

class TranscriptIngestBody(BaseModel):
    transcript: str


@app.post("/sessions/{session_id}/transcript")
async def ingest_session_transcript(session_id: str, body: TranscriptIngestBody) -> dict:
    """Store a meeting transcript for a session (pushed once by the LiveKit agent).

    This is the single transcript source reused by the local-doc pipeline — the
    agent already transcribed the meeting for live Q&A, so we do not transcribe
    again. Creates the session record if it does not exist yet.
    """
    session = _sessions.get(session_id)
    if session is None:
        session = _SessionRecord(session_id, bot_id=None, meeting_url=None)
        _sessions[session_id] = session
    session.transcript = body.transcript or ""
    logger.info("Stored transcript for session %s (%d chars)", session_id, len(session.transcript))
    return {"session_id": session_id, "length": len(session.transcript)}


@app.get("/sessions/{session_id}/transcript")
async def get_session_transcript(session_id: str) -> dict:
    """Report whether a captured transcript is available for a session."""
    session = _sessions.get(session_id)
    text = getattr(session, "transcript", "") if session else ""
    return {"session_id": session_id, "available": bool(text.strip()), "length": len(text)}


@app.post("/local-doc/pipeline/start")
async def local_doc_pipeline_start(body: LocalDocPipelineStartBody) -> dict:
    """Kick off the local document change pipeline for a session."""
    # T-12-09: Validate doc_folder path — resolve and check existence
    import pathlib as _pl
    folder = _pl.Path(body.doc_folder).resolve()
    if not folder.exists() or not folder.is_dir():
        raise HTTPException(status_code=400, detail=f"doc_folder does not exist: {body.doc_folder}")
    job_id = str(uuid.uuid4())
    q: asyncio.Queue = asyncio.Queue()
    _local_doc_jobs[job_id] = {
        "job_id": job_id,
        "session_id": body.session_id,
        "status": "running",
        "stage": None,
        "queue": q,
        "proposals": [],
        "created_at": _utcnow(),
    }
    # Prefer a transcript pasted directly in the request (sandbox / no-meeting
    # flow); otherwise fall back to the session's transcript.
    transcript = (body.transcript or "").strip()
    if not transcript:
        session = _sessions.get(body.session_id)
        if session and hasattr(session, "transcript"):
            transcript = getattr(session, "transcript", "") or ""

    config = PipelineConfig(
        session_id=body.session_id,
        doc_folder=str(folder),
        use_embeddings=body.use_embeddings,
        rerank=body.rerank,
        contextual_retrieval=body.contextual_retrieval,
    )

    async def _run():
        try:
            proposals = await run_pipeline(transcript=transcript, config=config, progress_queue=q)
            proposal_dicts = [p.model_dump() for p in proposals]
            _local_doc_jobs[job_id]["proposals"] = proposal_dicts
            _local_doc_jobs[job_id]["status"] = "completed"
            _local_doc_proposals.setdefault(body.session_id, []).extend(proposal_dicts)
        except Exception as exc:
            logger.error("Local doc pipeline failed: %s", exc)
            _local_doc_jobs[job_id]["status"] = "error"
        finally:
            await q.put(None)  # sentinel to close SSE stream (T-12-12)

    asyncio.create_task(_run())
    logger.info("Local doc pipeline started — job_id=%s session=%s", job_id, body.session_id)
    return {"job_id": job_id, "status": "running"}


@app.get("/local-doc/pipeline/{job_id}/stream")
async def local_doc_pipeline_stream(job_id: str) -> StreamingResponse:
    """SSE stream for local-doc pipeline progress.

    Emits 'stage_start' events for each of the 8 PIPELINE_STAGES, then a
    final 'pipeline_complete' event. T-12-12: auto-closes after 5 minutes
    (asyncio.wait_for timeout=300).
    """
    job = _local_doc_jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Local-doc job {job_id!r} not found")
    q: asyncio.Queue = job["queue"]

    async def _generate() -> AsyncGenerator[str, None]:
        while True:
            stage = await asyncio.wait_for(q.get(), timeout=300.0)
            if stage is None:
                job_data = _local_doc_jobs.get(job_id, {})
                proposals = job_data.get("proposals", [])
                yield f"event: pipeline_complete\ndata: {json.dumps({'proposal_count': len(proposals)})}\n\n"
                break
            job["stage"] = stage
            yield f"event: stage_start\ndata: {json.dumps({'stage': stage})}\n\n"

    return StreamingResponse(
        _generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/sessions/{session_id}/local-doc/changes")
async def list_local_doc_changes(session_id: str) -> list:
    """Return all local-doc proposals for this session."""
    return _local_doc_proposals.get(session_id, [])


@app.post("/sessions/{session_id}/local-doc/execute")
async def execute_local_doc_changes(session_id: str, body: LocalDocExecuteBody) -> dict:
    """Accept and write back local-doc proposals by proposal_id.

    T-12-11: Only explicitly accepted proposal_ids are written; no auto-apply.
    T-12-10: SafeApply.apply() always creates backup before writing.
    """
    proposals = _local_doc_proposals.get(session_id, [])
    applier = SafeApply()
    results = []
    for pid in body.proposal_ids:
        proposal = next((p for p in proposals if p.get("proposal_id") == pid), None)
        if not proposal:
            results.append({"proposal_id": pid, "success": False, "message": "Not found"})
            continue
        chunk = proposal.get("source_chunk", {})
        # Honor a user edit from the card's textarea, if provided.
        override = body.edited_content.get(pid)
        content = override if override is not None else proposal.get("after_content", "")
        # A user edit always means "write this text" — never a section delete.
        edit_type = proposal.get("edit_type", "replace")
        if override is not None:
            edit_type = "replace"
        try:
            backup_path = await asyncio.to_thread(
                applier.apply,
                file_path=chunk.get("source_path", ""),
                section_heading=chunk.get("section_heading", ""),
                new_content=content,
                session_id=session_id,
                proposal_id=pid,
                edit_type=edit_type,
            )
            proposal["status"] = "accepted"
            if override is not None:
                # Reflect the user's edit in the stored proposal and flag it so
                # the UI no longer presents stale AI verification as authoritative.
                proposal["after_content"] = override
                proposal["user_edited"] = True
            results.append({"proposal_id": pid, "success": True, "backup_path": backup_path})
        except Exception as exc:
            logger.error("Safe apply failed for proposal %s: %s", pid, exc)
            results.append({"proposal_id": pid, "success": False, "message": str(exc)})
    return {"results": results}
