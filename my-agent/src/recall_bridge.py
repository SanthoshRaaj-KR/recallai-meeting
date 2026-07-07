"""
Confluence Service — port 8001.

Handles all meeting intelligence: proposal generation, Confluence writes,
meeting summary / MOM, and post-meeting chat.  Bot lifecycle (start/stop/
status) lives in bot_service.py on port 8000.

Session state is read from the shared Supabase store (session_store.py) so
both services see the same data regardless of which process wrote it.

Run:
    uv run uvicorn src.recall_bridge:app --host 0.0.0.0 --port 8001

Required env vars (.env.local):
    OPENAI_API_KEY (or CEREBRAS_API_KEY)
    ATLASSIAN_USER_EMAIL, ATLASSIAN_API_TOKEN, ATLASSIAN_DOMAIN
    SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY  (shared session persistence)
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
    from . import session_store
    from .review_pipeline import ProposalPipeline
    from .review_pipeline import confluence_proposal_adapter
except ImportError:
    import session_store
    from review_pipeline import ProposalPipeline
    from review_pipeline import confluence_proposal_adapter


async def _propose(
    *,
    session_id: str,
    transcript: list,
    memory_context: str,
    emit=None,
):
    """Generate Confluence change proposals via PineconeHybridIndex pipeline."""
    return await confluence_proposal_adapter.run_confluence_pipeline(
        session_id=session_id,
        transcript=transcript,
        memory_context=memory_context,
        emit=emit,
    )

load_dotenv(Path(__file__).parent.parent / ".env.local")

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ── Config ─────────────────────────────────────────────────────────────────────
_CEREBRAS_API_KEY = os.getenv("CEREBRAS_API_KEY", "")
_CEREBRAS_BASE_URL = "https://api.cerebras.ai/v1"
_CHAT_MODEL = "gpt-oss-120b"
_OPENAI_FALLBACK_MODEL = os.getenv("JARVIS_GENERAL_MODEL", "gpt-4o-mini")

_CORS_ORIGINS = [o.strip() for o in os.getenv("CORS_ORIGINS", "*").split(",") if o.strip()]

# In-process pipeline job registry (ephemeral — jobs live only for this request).
_pipelines: dict[str, dict] = {}

# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(title="Jarvis Confluence Service", version="2.0")
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
setup_fastapi_observability(app, "confluence-service")


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


class PipelineStartWithTranscriptBody(BaseModel):
    transcript: str


# ── Helpers: persist changes back to session store ────────────────────────────

def _save_changes(session_id: str, changes: list[dict]) -> None:
    session_store.patch(session_id, {"changes": changes})


def _save_summary(session_id: str, summary: dict, extracted_meeting: object, diagnostics: list) -> None:
    session_store.patch(session_id, {
        "summary": summary,
        "extracted_meeting": extracted_meeting,
        "pipeline_diagnostics": diagnostics,
    })


# ── Endpoints: health ──────────────────────────────────────────────────────────

@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "service": "confluence-service"}


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
    transcript = session_store.get_transcript_turns(session_id)
    memory_context = ""  # post-meeting uses the Recall transcript only (no LiveKit memory)
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
        meeting, proposals = await _propose(
            session_id=session_id,
            transcript=transcript,
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


# ── Endpoint: paste-a-transcript test (no meeting/session needed) ──────────────

class TestTranscriptBody(BaseModel):
    transcript: str


@app.post("/review/test/propose")
async def test_propose_from_transcript(body: TestTranscriptBody) -> dict:
    """Paste a transcript → get Confluence proposal cards back directly.

    Convenience test endpoint: runs the Confluence proposal adapter on the given
    transcript (no bot/meeting/session required) and returns the cards synchronously.
    Proposals are always generated via PineconeHybridIndex (confluence_proposal_adapter).
    """
    text = (body.transcript or "").strip()
    if not text:
        return {"proposal_count": 0, "intents_extracted": 0, "proposals": []}
    meeting, proposals = await _propose(
        session_id=f"test-{uuid.uuid4().hex[:8]}",
        transcript=[{"participant": "Meeting", "text": text}],
        memory_context="",
    )
    return {
        "meeting_title": meeting.title,
        "intents_extracted": len(meeting.change_intents),
        "proposal_count": len(proposals),
        "proposals": proposals,
    }


# ── Endpoints: review — summary ────────────────────────────────────────────────

@app.get("/sessions/{session_id}/review/summary")
async def get_summary(session_id: str) -> dict:
    s = _require_session(session_id)
    transcript = session_store.get_transcript_turns(session_id)
    cached = s.get("summary")
    # Don't serve a cached summary that has empty decisions when there is transcript
    # content — it means a prior LLM call failed silently; retry so the UI doesn't
    # stay stuck on "Extracting decisions…" forever.
    if cached and (cached.get("decisions") or not transcript):
        return cached
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
    transcript = session_store.get_transcript_turns(session_id)

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


@app.post("/review/pipeline/start-with-transcript")
async def pipeline_start_with_transcript(body: PipelineStartWithTranscriptBody) -> dict:
    session_id = str(uuid.uuid4())
    session_store.upsert(session_id, {
        "status": "ended",
        "transcript_memory_text": "",
        "changes": [],
    })
    # Store the supplied transcript as a single Recall-sourced turn (turns table).
    session_store.append_transcript_turn(session_id, {
        "participant": "Meeting", "text": body.transcript.strip(), "source": "recall",
    })
    job_id = str(uuid.uuid4())
    _pipelines[job_id] = {
        "job_id": job_id,
        "session_id": session_id,
        "status": "running",
        "stage": None,
        "created_at": _utcnow(),
        "events": [],
        "completed_at": None,
        "error": None,
    }
    logger.info("Transcript pipeline started — job_id=%s session=%s", job_id, session_id)
    asyncio.create_task(_run_pipeline_job(job_id), name=f"review-pipeline-{job_id}")
    return {"job_id": job_id, "session_id": session_id, "status": "running"}


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
        transcript = session_store.get_transcript_turns(session_id)
        meeting, proposals = await _propose(
            session_id=session_id,
            transcript=transcript,
            memory_context="",  # post-meeting uses the Recall transcript only
            emit=emit,
        )
        summary = pipeline.summary_response(session_id, meeting, transcript)
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
