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
    uv run uvicorn src.recall_bridge:app --host 0.0.0.0 --port 8001

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

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from livekit import api as livekit_api
from pydantic import BaseModel

try:
    from .review_pipeline import ProposalPipeline
except ImportError:  # Allows `uvicorn recall_bridge:app` from my-agent/src.
    from review_pipeline import ProposalPipeline

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
        self.transcript: list[dict] = []
        self._change_counter: int = 0
        self.summary: dict | None = None
        self.extracted_meeting = None
        self.pipeline_diagnostics: list[dict] = []
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
                "transcript_entries": len(self.transcript),
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


def _pipeline() -> ProposalPipeline:
    return ProposalPipeline()


async def _execute_change_dict(change: dict) -> dict:
    """Execute one proposal and fold exceptions into the API response shape."""
    try:
        result = await _pipeline().execute(change)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Proposal execution failed")
        result = {"success": False, "message": str(exc)}
    change["status"] = "executed" if result.get("success") else "failed"
    if not result.get("success"):
        change["last_error"] = result.get("message") or result.get("error")
    return result


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


# ── Endpoints: Recall webhook ──────────────────────────────────────────────────

def _session_for_bot(bot_id: str) -> "_SessionRecord | None":
    """Return the session whose Recall bot_id matches, or None."""
    return next((s for s in _sessions.values() if s.bot_id == bot_id), None)


@app.post("/recall-webhook")
async def recall_webhook(request: Request) -> dict:
    """Receive Recall.ai project-level webhook events.

    Register BRIDGE_SERVER_URL/recall-webhook in the Recall dashboard under
    Webhooks → Events: bot.status_change (and optionally bot.participant_events).

    Handles:
      bot.status_change   → transitions session status (joining → in_meeting → ended)
      bot.done            → marks session ended (legacy event name)
    """
    try:
        body = await request.json()
    except Exception:
        return {"ok": True}

    event = body.get("event", "")
    data = body.get("data", {})

    # bot.status_change: {"event": "bot.status_change", "data": {"bot_id": "...", "status": {"code": "..."}}}
    if event in ("bot.status_change", "bot.done"):
        bot_id = data.get("bot_id", "")
        s = _session_for_bot(bot_id) if bot_id else None
        if s:
            status_obj = data.get("status") or {}
            code = status_obj.get("code") or (data.get("code") if event == "bot.done" else "")
            new_status = _RECALL_STATUS_MAP.get(code, "")
            if new_status and new_status != s.status:
                s.status = new_status
                if new_status in ("ended", "error") and not s.ended_at:
                    s.ended_at = _utcnow()
                s._touch()
                logger.info("Recall webhook %s → session %s status=%s", event, s.session_id, new_status)
        return {"ok": True}

    # bot.participant_events: track current speaker for per-participant transcript attribution
    if event == "bot.participant_events":
        bot_id = data.get("bot_id", "")
        s = _session_for_bot(bot_id) if bot_id else None
        if s:
            for evt in data.get("events", []):
                etype = evt.get("type", "")
                participant_name = (evt.get("participant") or {}).get("name", "").strip()
                if not participant_name:
                    continue
                if etype == "speech_on":
                    logger.debug("Speaker ON  [%s] → %s", s.session_id, participant_name)
                elif etype == "speech_off":
                    logger.debug("Speaker OFF [%s] → %s", s.session_id, participant_name)
        return {"ok": True}

    return {"ok": True}


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


@app.post("/livekit-transcript/{session_id}")
async def receive_livekit_transcript(session_id: str, request: Request) -> dict:
    """Receive final STT utterances from my-agent.

    The LiveKit agent process and this FastAPI bridge commonly run as separate
    processes, so the transcript is posted over localhost rather than shared in
    memory.
    """
    body = await request.json()
    text = (body.get("text") or "").strip()
    if not text:
        return {"ok": True}
    session = _sessions.get(session_id)
    if session is None:
        session = _SessionRecord(session_id, bot_id=None, meeting_url=None)
        session.status = "in_meeting"
        _sessions[session_id] = session
    session.transcript.append(
        {
            "participant": (body.get("speaker") or body.get("participant") or "Meeting").strip(),
            "text": text,
            "timestamp": body.get("timestamp") or datetime.datetime.utcnow().timestamp(),
            "source": body.get("source") or "livekit",
        }
    )
    if len(session.transcript) > 2000:
        session.transcript = session.transcript[-2000:]
    session._touch()
    return {"ok": True, "transcript_entries": len(session.transcript)}


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
                ch["status"] = "executing"
                result = await _execute_change_dict(ch)
                s._touch()
                return result
        raise HTTPException(status_code=404, detail=f"Proposal {pid!r} not found")

    if body.proposal_ids is not None:
        results = []
        for pid in body.proposal_ids:
            matched = next((ch for ch in s.changes if str(ch.get("id")) == pid), None)
            if matched:
                matched["status"] = "executing"
                result = await _execute_change_dict(matched)
                results.append(
                    {
                        "page": matched.get("page_title", pid),
                        "success": bool(result.get("success")),
                        "message": result.get("message") or result.get("error"),
                    }
                )
            else:
                results.append({"page": pid, "success": False, "message": "Not found"})
        s._touch()
        return {"results": results}

    if body.ids is not None:
        results = []
        for cid in body.ids:
            matched = next((ch for ch in s.changes if str(ch.get("id")) == str(cid)), None)
            if matched:
                matched["status"] = "executing"
                result = await _execute_change_dict(matched)
                results.append(
                    {
                        "id": cid,
                        "success": bool(result.get("success")),
                        "error": None if result.get("success") else result.get("message") or result.get("error"),
                    }
                )
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

    Runs the new my-agent pipeline synchronously and returns generated cards.
    """
    s = _require_session(session_id)
    pipeline = _pipeline()
    meeting, proposals = await pipeline.run(
        session_id=session_id,
        transcript=s.transcript,
        query=body.query,
    )
    s.extracted_meeting = meeting
    s.summary = pipeline.summary_response(session_id, meeting, s.transcript)
    s.summary["proposal_diagnostics"] = pipeline.last_diagnostics
    s.pipeline_diagnostics = pipeline.last_diagnostics
    s.changes = proposals
    s._touch()
    return {"changes": s.changes, "generated_count": len(proposals)}


# ── Endpoints: review — summary ────────────────────────────────────────────────

@app.get("/sessions/{session_id}/review/summary")
async def get_summary(session_id: str) -> dict:
    """Return the meeting summary (MoM, decisions, action items, etc.)."""
    s = _require_session(session_id)
    if s.summary:
        return s.summary
    return _pipeline().summary_response(session_id, None, s.transcript)


# ── Endpoints: review — chat ───────────────────────────────────────────────────

@app.post("/sessions/{session_id}/review/chat")
async def chat_with_meeting(session_id: str, body: ChatBody) -> dict:
    """
    Answer a question about the meeting using its transcript + summary.

    Lightweight transcript-grounded chat endpoint.
    """
    s = _require_session(session_id)
    last_user = next(
        (m.get("content", "") for m in reversed(body.messages) if m.get("role") == "user"),
        "",
    )
    recent = "\n".join(f"{e.get('participant', 'Speaker')}: {e.get('text', '')}" for e in s.transcript[-20:])
    return {
        "answer": (
            f"I found {len(s.transcript)} transcript entries for this meeting. "
            f"Most recent context: {recent[-600:] or 'no transcript captured yet.'}"
        ),
        "session_id": session_id,
        "context": {
            "transcript_entries": len(s.transcript),
            "has_summary": s.summary is not None,
            "last_question": last_user,
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
        "events": [],
        "completed_at": None,
        "error": None,
    }
    logger.info("Pipeline started — job_id=%s session=%s", job_id, body.session_id)
    asyncio.create_task(_run_pipeline_job(job_id), name=f"review-pipeline-{job_id}")
    return {"job_id": job_id, "status": "running"}


def _record_pipeline_event(job_id: str, event: dict) -> None:
    job = _pipelines.get(job_id)
    if not job:
        return
    if event.get("type") == "stage_start":
        job["stage"] = event.get("stage")
    job.setdefault("events", []).append(event)


async def _run_pipeline_job(job_id: str) -> None:
    job = _pipelines[job_id]
    session_id = job["session_id"]
    s = _require_session(session_id)
    pipeline = _pipeline()

    async def emit(event: dict) -> None:
        _record_pipeline_event(job_id, event)

    try:
        meeting, proposals = await pipeline.run(
            session_id=session_id,
            transcript=s.transcript,
            emit=emit,
        )
        s.extracted_meeting = meeting
        s.summary = pipeline.summary_response(session_id, meeting, s.transcript)
        s.summary["proposal_diagnostics"] = pipeline.last_diagnostics
        s.pipeline_diagnostics = pipeline.last_diagnostics
        s.changes = proposals
        s._touch()
        job["status"] = "completed"
        job["stage"] = None
        job["completed_at"] = _utcnow()
        _record_pipeline_event(
            job_id,
            {
                "type": "pipeline_complete",
                "proposal_count": len(proposals),
                "diagnostic_count": len(pipeline.last_diagnostics),
            },
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Pipeline failed — job_id=%s", job_id)
        job["status"] = "failed"
        job["stage"] = None
        job["error"] = str(exc)
        job["completed_at"] = _utcnow()
        _record_pipeline_event(job_id, {"type": "pipeline_error", "detail": str(exc)})


def _sse(event: dict) -> str:
    event_type = event.get("type", "message")
    data = {k: v for k, v in event.items() if k != "type"}
    return f"event: {event_type}\ndata: {json.dumps(data)}\n\n"


async def _pipeline_sse(job_id: str) -> AsyncGenerator[str, None]:
    """Stream buffered and live pipeline events."""
    sent = 0
    while True:
        job = _pipelines.get(job_id)
        if not job:
            yield _sse({"type": "pipeline_error", "detail": f"Pipeline job {job_id!r} not found"})
            return
        events = job.get("events", [])
        while sent < len(events):
            event = events[sent]
            sent += 1
            yield _sse(event)
            if event.get("type") in {"pipeline_complete", "pipeline_error"}:
                return
        if job.get("status") in {"completed", "failed"}:
            # Defensive fallback in case terminal event was not appended.
            terminal = (
                {"type": "pipeline_complete", "proposal_count": len(_sessions[job["session_id"]].changes)}
                if job.get("status") == "completed"
                else {"type": "pipeline_error", "detail": job.get("error") or "Pipeline failed"}
            )
            yield _sse(terminal)
            return
        await asyncio.sleep(0.25)


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


# ── Endpoints: stop bot ────────────────────────────────────────────────────────

@app.post("/sessions/{session_id}/bot/stop")
async def stop_bot(session_id: str) -> dict:
    """Remove the Recall bot from the meeting and mark the session ended.

    Calls the Recall.ai DELETE /bot/{id}/ endpoint to eject the bot, then
    updates session status to 'ended' so the frontend poll sees the change
    immediately without waiting for the webhook.
    """
    s = _require_session(session_id)
    if s.bot_id and RECALL_API_KEY:
        try:
            resp = requests.post(
                f"{RECALL_BASE_URL}/bot/{s.bot_id}/leave_call/",
                headers={"Authorization": f"Token {RECALL_API_KEY}"},
                timeout=10,
            )
            if not resp.ok:
                logger.warning(
                    "Recall bot removal returned %s for bot_id=%s — marking session ended anyway",
                    resp.status_code, s.bot_id,
                )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to remove Recall bot %s: %s", s.bot_id, exc)
    s.status = "ended"
    if not s.ended_at:
        s.ended_at = _utcnow()
    s._touch()
    return s.as_session_status()


# ── Endpoints: review — reject proposal ────────────────────────────────────────

@app.post("/sessions/{session_id}/review/changes/{change_id}/reject")
async def reject_proposal(session_id: str, change_id: str) -> dict:
    """Mark a single proposal as rejected without executing it.

    Rejected proposals are preserved in the session store so the audit trail
    is complete, but they are excluded from future execute-all calls.
    """
    s = _require_session(session_id)
    for ch in s.changes:
        if str(ch.get("id")) == change_id:
            ch["status"] = "rejected"
            s._touch()
            return ch
    raise HTTPException(status_code=404, detail=f"Proposal {change_id!r} not found")


# ── Endpoints: review — transcript ─────────────────────────────────────────────

@app.get("/sessions/{session_id}/review/transcript")
async def get_transcript(session_id: str) -> list:
    """Return the full raw transcript captured for this session.

    Each entry has: participant, text, timestamp (epoch float), source.
    """
    return _require_session(session_id).transcript


# ── Endpoints: history ─────────────────────────────────────────────────────────

@app.get("/history")
async def list_history() -> list:
    """Return all sessions as history items (most recent first)."""
    items = [s.as_history_item() for s in _sessions.values()]
    items.sort(key=lambda x: x.get("updated_at") or "", reverse=True)
    return items
