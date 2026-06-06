"""
Confluence Service — port 8001.

Handles all meeting intelligence: proposal generation, Confluence writes,
meeting summary / MOM, and post-meeting chat.  Bot lifecycle (start/stop/
status) lives in bot_service.py on port 8000.

Session state is read from the shared Supabase store (session_store.py) so
both services see the same data regardless of which process wrote it.

<<<<<<< HEAD
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
    CONFLUENCE_LOGIC_URL   internal URL of the jarvis_agentic service (default: http://localhost:8001)
                           Set this to the Azure internal service DNS name when deploying on separate pods
                           e.g. http://confluence-logic-service:8001
=======
Run:
    uv run uvicorn src.recall_bridge:app --host 0.0.0.0 --port 8001

Required env vars (.env.local):
    OPENAI_API_KEY (or CEREBRAS_API_KEY)
    ATLASSIAN_USER_EMAIL, ATLASSIAN_API_TOKEN, ATLASSIAN_DOMAIN
    SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY  (shared session persistence)
>>>>>>> confluence
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

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

try:
    from .review_pipeline import ProposalPipeline
    from . import session_store
except ImportError:
    from review_pipeline import ProposalPipeline
    import session_store

load_dotenv(Path(__file__).parent.parent / ".env.local")

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ── Config ─────────────────────────────────────────────────────────────────────
_CEREBRAS_API_KEY = os.getenv("CEREBRAS_API_KEY", "")
_CEREBRAS_BASE_URL = "https://api.cerebras.ai/v1"
_CHAT_MODEL = "gpt-oss-120b"
_OPENAI_FALLBACK_MODEL = os.getenv("JARVIS_GENERAL_MODEL", "gpt-4o-mini")

_CORS_ORIGINS = [o.strip() for o in os.getenv("CORS_ORIGINS", "*").split(",") if o.strip()]

<<<<<<< HEAD
# URL of the jarvis_agentic (confluence logic) service.
# When both services run on the same host this is http://localhost:8001.
# In Azure set this to the internal service DNS, e.g. http://confluence-logic:8001.
CONFLUENCE_LOGIC_URL = os.getenv("CONFLUENCE_LOGIC_URL", "http://localhost:8001").rstrip("/")

_BOT_HTML_PATH = Path(__file__).parent / "bot.html"

# ── App setup ──────────────────────────────────────────────────────────────────
app = FastAPI(title="Recall.ai Bridge for my-agent")
=======
# In-process pipeline job registry (ephemeral — jobs live only for this request).
_pipelines: dict[str, dict] = {}
>>>>>>> confluence

# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(title="Jarvis Confluence Service", version="2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _utcnow() -> str:
    return datetime.datetime.utcnow().isoformat() + "Z"


def _pipeline() -> ProposalPipeline:
    return ProposalPipeline()


def _require_session(session_id: str) -> dict:
    try:
        return session_store.require(session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Session {session_id!r} not found")


async def _execute_change_dict(change: dict) -> dict:
    try:
        result = await _pipeline().execute(change)
    except Exception as exc:
        logger.exception("Proposal execution failed")
        result = {"success": False, "message": str(exc)}
    change["status"] = "executed" if result.get("success") else "failed"
    if not result.get("success"):
        change["last_error"] = result.get("message") or result.get("error")
    return result


# ── Request / response models ──────────────────────────────────────────────────

class ExecuteBody(BaseModel):
    ids: Optional[list[int]] = None
    proposal_id: Optional[str] = None
    proposal_ids: Optional[list[str]] = None


class ProposeBody(BaseModel):
    query: Optional[str] = None
    create_new_page: bool = False


class ChatBody(BaseModel):
    messages: list[dict]


class PipelineStartBody(BaseModel):
    session_id: str


# ── Helpers: persist changes back to session store ────────────────────────────

def _save_changes(session_id: str, changes: list[dict]) -> None:
    session_store.patch(session_id, {"changes": changes})


<<<<<<< HEAD
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
        curl -X POST http://localhost:8000/bot/start \\
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


def _forward_to_confluence_logic(path: str, body: dict) -> None:
    """Fire-and-forget HTTP POST to the jarvis_agentic service at CONFLUENCE_LOGIC_URL.

    Called from async endpoints via asyncio.to_thread so it never blocks the event loop.
    Failures are silently swallowed — the confluence-logic service is optional.
    """
    if not CONFLUENCE_LOGIC_URL:
        return
    try:
        requests.post(
            f"{CONFLUENCE_LOGIC_URL}{path}",
            json=body,
            timeout=3,
        )
    except Exception as exc:  # noqa: BLE001
        logger.debug("Confluence-logic forward failed (non-fatal): %s", exc)


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

    # Forward to the confluence-logic service so jarvis_agentic can update its
    # own session state when running as a separate Azure pod.
    asyncio.create_task(
        asyncio.to_thread(_forward_to_confluence_logic, "/recall-webhook", body)
    )

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
=======
def _save_summary(session_id: str, summary: dict, extracted_meeting: object, diagnostics: list) -> None:
    session_store.patch(session_id, {
        "summary": summary,
        "extracted_meeting": extracted_meeting,
        "pipeline_diagnostics": diagnostics,
    })
>>>>>>> confluence


# ── Endpoints: health ──────────────────────────────────────────────────────────

@app.get("/health")
async def health() -> dict:
<<<<<<< HEAD
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
    entry = {
        "participant": (body.get("speaker") or body.get("participant") or "Meeting").strip(),
        "text": text,
        "timestamp": body.get("timestamp") or datetime.datetime.utcnow().timestamp(),
        "source": body.get("source") or "livekit",
    }
    session.transcript.append(entry)
    session.transcript_memory.observe_entry(entry)
    if len(session.transcript) > 2000:
        session.transcript = session.transcript[-2000:]
    session._touch()
    # Forward to confluence-logic service so jarvis_agentic can use the transcript
    # for its voice pipeline when running as a separate Azure pod.
    asyncio.create_task(
        asyncio.to_thread(
            _forward_to_confluence_logic,
            f"/livekit-transcript/{session_id}",
            body,
        )
    )
    return {"ok": True, "transcript_entries": len(session.transcript)}
=======
    return {"status": "ok", "service": "confluence-service"}
>>>>>>> confluence


# ── Endpoints: review — changes ────────────────────────────────────────────────

@app.get("/sessions/{session_id}/review/changes")
async def list_changes(session_id: str) -> list:
    return _require_session(session_id).get("changes") or []


@app.post("/sessions/{session_id}/review/execute")
async def execute_changes(session_id: str, body: ExecuteBody) -> dict:
    s = _require_session(session_id)
    changes: list[dict] = list(s.get("changes") or [])

    if body.proposal_id is not None:
        pid = str(body.proposal_id)
        for ch in changes:
            if str(ch.get("id")) == pid:
                ch["status"] = "executing"
                result = await _execute_change_dict(ch)
                _save_changes(session_id, changes)
                return result
        raise HTTPException(status_code=404, detail=f"Proposal {pid!r} not found")

    if body.proposal_ids is not None:
        results = []
        for pid in body.proposal_ids:
            matched = next((ch for ch in changes if str(ch.get("id")) == pid), None)
            if matched:
                matched["status"] = "executing"
                result = await _execute_change_dict(matched)
                results.append({
                    "page": matched.get("page_title", pid),
                    "success": bool(result.get("success")),
                    "message": result.get("message") or result.get("error"),
                })
            else:
                results.append({"page": pid, "success": False, "message": "Not found"})
        _save_changes(session_id, changes)
        return {"results": results}

    if body.ids is not None:
        results = []
        for cid in body.ids:
            matched = next((ch for ch in changes if str(ch.get("id")) == str(cid)), None)
            if matched:
                matched["status"] = "executing"
                result = await _execute_change_dict(matched)
                results.append({
                    "id": cid,
                    "success": bool(result.get("success")),
                    "error": None if result.get("success") else result.get("message") or result.get("error"),
                })
            else:
                results.append({"id": cid, "success": False, "error": "Not found"})
        _save_changes(session_id, changes)
        return {"results": results}

    raise HTTPException(status_code=422, detail="Provide ids, proposal_id, or proposal_ids")


# ── Endpoints: review — propose ────────────────────────────────────────────────

@app.post("/sessions/{session_id}/review/changes/propose")
async def propose_changes(session_id: str, body: ProposeBody) -> dict:
    s = _require_session(session_id)
    transcript = s.get("transcript") or []
    memory_context = s.get("transcript_memory_text") or ""
    changes: list[dict] = list(s.get("changes") or [])

    pipeline = _pipeline()
    if body.create_new_page:
        meeting, proposals = await pipeline.propose_custom_new_page(
            session_id=session_id,
            transcript=transcript,
            query=body.query or "",
            memory_context=memory_context,
        )
    else:
        meeting, proposals = await pipeline.run(
            session_id=session_id,
            transcript=transcript,
            query=body.query,
            memory_context=memory_context,
        )

    summary = pipeline.summary_response(session_id, meeting, transcript)
    summary["proposal_diagnostics"] = pipeline.last_diagnostics
    _save_summary(session_id, summary, meeting, pipeline.last_diagnostics)

    existing_ids = {str(ch.get("id")) for ch in changes}
    new_proposals = [p for p in proposals if str(p.get("id")) not in existing_ids]
    changes.extend(new_proposals)
    _save_changes(session_id, changes)

    return {"changes": changes, "generated_count": len(new_proposals)}


# ── Endpoints: review — summary ────────────────────────────────────────────────

@app.get("/sessions/{session_id}/review/summary")
async def get_summary(session_id: str) -> dict:
    s = _require_session(session_id)
    if s.get("summary"):
        return s["summary"]
    transcript = s.get("transcript") or []
    if transcript:
        try:
            summary = await _pipeline().generate_meeting_summary(session_id, transcript)
            session_store.patch(session_id, {"summary": summary})
            return summary
        except Exception as exc:
            logger.warning("On-demand summary generation failed for %s: %s", session_id, exc)
    return _pipeline().summary_response(session_id, None, transcript)


# ── Endpoints: review — chat ───────────────────────────────────────────────────

@app.post("/sessions/{session_id}/review/chat")
async def chat_with_meeting(session_id: str, body: ChatBody) -> dict:
    from openai import AsyncOpenAI

    s = _require_session(session_id)
    transcript = s.get("transcript") or []

    transcript_lines = [
        f"{e.get('participant') or e.get('speaker') or 'Speaker'}: {e.get('text', '')}"
        for e in transcript
        if e.get("text", "").strip()
    ]
    transcript_text = "\n".join(transcript_lines) or "No transcript captured yet."

    summary_obj = s.get("summary") or {}
    summary_ctx = ""
    if summary_obj:
        summary_ctx = (
            f"\nMeeting summary: {summary_obj.get('summary', '')}"
            f"\nKey decisions: {'; '.join(summary_obj.get('decisions', []))}"
            f"\nAction items: {'; '.join(item.get('description', '') for item in summary_obj.get('action_items', []))}"
        )

    system_prompt = (
        "You are a meeting assistant. Answer questions about the meeting below. "
        "Be concise and accurate. Only use information from the transcript."
        f"{summary_ctx}\n\n[Full transcript]\n{transcript_text}"
    )
    messages = [{"role": "system", "content": system_prompt}] + [
        {"role": m["role"], "content": m["content"]}
        for m in body.messages
        if m.get("role") in ("user", "assistant") and m.get("content")
    ]

    if not _CEREBRAS_API_KEY:
        return {"answer": "Chat is not configured — CEREBRAS_API_KEY is missing.", "session_id": session_id}

    last_exc: Exception | None = None
    cerebras_client = AsyncOpenAI(api_key=_CEREBRAS_API_KEY, base_url=_CEREBRAS_BASE_URL)
    for attempt in range(3):
        try:
            response = await cerebras_client.chat.completions.create(
                model=_CHAT_MODEL, messages=messages, max_tokens=512, temperature=0.2,
            )
            return {"answer": response.choices[0].message.content or "", "session_id": session_id}
        except Exception as exc:
            last_exc = exc
            if attempt < 2:
                await asyncio.sleep(2 ** attempt)

    logger.warning("Cerebras chat failed (%s); falling back to OpenAI for session %s", last_exc, session_id)
    try:
        openai_client = AsyncOpenAI()
        response = await openai_client.chat.completions.create(
            model=_OPENAI_FALLBACK_MODEL, messages=messages, max_tokens=512, temperature=0.2,
        )
        return {"answer": response.choices[0].message.content or "", "session_id": session_id}
    except Exception as exc:
        logger.error("OpenAI fallback failed for session %s: %s", session_id, exc)
        return {"answer": "Both Cerebras and OpenAI are unavailable right now.", "session_id": session_id}


# ── Endpoints: review — regenerate / reject ────────────────────────────────────

@app.post("/sessions/{session_id}/review/regenerate/{proposal_id}")
async def regenerate_proposal(session_id: str, proposal_id: str) -> dict:
    s = _require_session(session_id)
    changes: list[dict] = list(s.get("changes") or [])
    for ch in changes:
        if str(ch.get("id")) == proposal_id:
            ch["status"] = "pending"
            ch["regenerate_available"] = True
            _save_changes(session_id, changes)
            return ch
    raise HTTPException(status_code=404, detail=f"Proposal {proposal_id!r} not found")


@app.post("/sessions/{session_id}/review/changes/{change_id}/reject")
async def reject_proposal(session_id: str, change_id: str) -> dict:
    s = _require_session(session_id)
    changes: list[dict] = list(s.get("changes") or [])
    for ch in changes:
        if str(ch.get("id")) == change_id:
            ch["status"] = "rejected"
            _save_changes(session_id, changes)
            return ch
    raise HTTPException(status_code=404, detail=f"Proposal {change_id!r} not found")


# ── Endpoints: pipeline ────────────────────────────────────────────────────────

@app.post("/review/pipeline/start")
async def pipeline_start(body: PipelineStartBody) -> dict:
    _require_session(body.session_id)  # validate session exists before queuing
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
    # Use session_store.get directly — _require_session raises HTTPException
    # which is wrong in a background task context.
    s = session_store.get(session_id)
    if s is None:
        job["status"] = "failed"
        job["error"] = f"Session {session_id!r} not found"
        job["completed_at"] = _utcnow()
        _record_pipeline_event(job_id, {"type": "pipeline_error", "detail": job["error"]})
        return
    pipeline = _pipeline()

    async def emit(event: dict) -> None:
        _record_pipeline_event(job_id, event)

    try:
        meeting, proposals = await pipeline.run(
            session_id=session_id,
            transcript=s.get("transcript") or [],
            memory_context=s.get("transcript_memory_text") or "",
            emit=emit,
        )
        summary = pipeline.summary_response(session_id, meeting, s.get("transcript") or [])
        summary["proposal_diagnostics"] = pipeline.last_diagnostics
        _save_summary(session_id, summary, meeting, pipeline.last_diagnostics)

        changes: list[dict] = list(s.get("changes") or [])
        existing_ids = {str(ch.get("id")) for ch in changes}
        new_proposals = [p for p in proposals if str(p.get("id")) not in existing_ids]
        changes.extend(new_proposals)
        _save_changes(session_id, changes)

        job["status"] = "completed"
        job["stage"] = None
        job["completed_at"] = _utcnow()
        _record_pipeline_event(job_id, {
            "type": "pipeline_complete",
            "proposal_count": len(proposals),
            "diagnostic_count": len(pipeline.last_diagnostics),
        })
    except Exception as exc:
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
            s = session_store.get(job["session_id"]) or {}
            terminal = (
                {"type": "pipeline_complete", "proposal_count": len(s.get("changes") or [])}
                if job.get("status") == "completed"
                else {"type": "pipeline_error", "detail": job.get("error") or "Pipeline failed"}
            )
            yield _sse(terminal)
            return
        await asyncio.sleep(0.25)


@app.get("/review/pipeline/{job_id}/stream")
async def pipeline_stream(job_id: str, token: str = "") -> StreamingResponse:
    if job_id not in _pipelines:
        raise HTTPException(status_code=404, detail=f"Pipeline job {job_id!r} not found")
    return StreamingResponse(
        _pipeline_sse(job_id),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
