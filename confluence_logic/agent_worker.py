"""
LiveKit Native Agent Worker for Jarvis (Phase 04).

Standalone process — run with:
    python -m confluence_logic.agent_worker dev      # development (hot reload)
    python -m confluence_logic.agent_worker start    # production

Designed to coexist with the FastAPI server (uvicorn confluence_logic.jarvis_agentic:app).
Phase 4 (D-02 / D-06 / D-09): Native voice agent — hears via Deepgram Nova-3 STT
(linked to the recall-relay-{session_id} participant), thinks via inference.LLM,
speaks via Cartesia TTS. Recall transcripts remain in jarvis_agentic.py for the
Confluence post-meeting review pipeline (untouched). IPC dispatch path removed.

REQ-11 (worker entrypoint), D-02/D-09 (Deepgram STT), D-06 (no IPC), D-08 (tools wired), D-09 (room_options).
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    JobProcess,
    TurnHandlingOptions,
    cli,
    inference,
)
from livekit.agents.voice import room_io
from livekit.plugins import silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel

from confluence_logic.agent_bridge import JARVIS_TOOLS

_MODULE_DIR = Path(__file__).resolve().parent
load_dotenv(_MODULE_DIR.parent / ".env")
load_dotenv(_MODULE_DIR / ".env", override=True)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# --- Config (REQ-20) ----------------------------------------------------------------
JARVIS_AGENT_WORKER_NAME = os.getenv("JARVIS_AGENT_WORKER_NAME", "jarvis-agent").strip()
JARVIS_LK_TTS_PROVIDER = os.getenv("JARVIS_LK_TTS_PROVIDER", "cartesia").strip().lower()
JARVIS_LK_TTS_VOICE = os.getenv(
    "JARVIS_LK_TTS_VOICE",
    # Default = the Cartesia voice ID confirmed in Plan 01 checkpoint. Override in .env.
    "9626c31c-bec5-4cca-baa8-f8ba9e84c8bc",
).strip()
JARVIS_LK_LLM = os.getenv("JARVIS_LK_LLM", "openai/gpt-4o-mini").strip()


# --- Agent definition ---------------------------------------------------------------
class JarvisAgent(Agent):
    """Persona-only agent — instructions + tools added in Plan 04."""

    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "You are Jarvis, an AI meeting assistant. You help teams update Confluence "
                "documentation based on what is discussed in meetings. Keep responses concise "
                "— you are speaking aloud in a meeting. Do not use markdown, asterisks, "
                "bullet points, or emojis."
            ),
            tools=JARVIS_TOOLS,
        )

    async def on_enter(self) -> None:
        # No greeting — Recall.ai bot handles wake-word ACK on the meeting side.
        logger.info("JarvisAgent joined the LiveKit room.")


# --- Worker registration ------------------------------------------------------------
server = AgentServer()


def prewarm(proc: JobProcess) -> None:
    """Load Silero VAD once per worker process (Pitfall 2 in RESEARCH)."""
    proc.userdata["vad"] = silero.VAD.load()


server.setup_fnc = prewarm


@server.rtc_session(agent_name=JARVIS_AGENT_WORKER_NAME)
async def entrypoint(ctx: JobContext) -> None:
    """Per-room session: build the AgentSession and start it on ctx.room.

    Phase 4: STT is now active (Deepgram Nova-3 via LiveKit Inference). The session
    links its STT pipeline to the 'recall-relay-{session_id}' participant published
    by jarvis_agentic.py's /recall-audio-mixed/{session_id} relay (Pitfall 1: avoids
    transcribing Jarvis's own TTS output on jarvis-publisher-{session_id}).
    """
    session = AgentSession(
        stt=inference.STT(model="deepgram/nova-3", language="multi"),              # D-02 / D-09
        llm=inference.LLM(JARVIS_LK_LLM),                                          # REQ-15 backbone
        tts=inference.TTS(f"{JARVIS_LK_TTS_PROVIDER}/sonic-3", voice=JARVIS_LK_TTS_VOICE),  # REQ-14
        vad=ctx.proc.userdata["vad"],                                              # Pitfall 2
        turn_handling=TurnHandlingOptions(
            turn_detection=MultilingualModel(),
            interruption={
                "resume_false_interruption": True,
                "false_interruption_timeout": 1.0,
            },
        ),
        preemptive_generation=False,                                               # Pitfall 7
        tts_text_transforms=["filter_emoji", "filter_markdown"],
    )

    # Resolve session_id from job metadata so STT links to the right relay participant
    # (Pitfall 1 / RESEARCH §participant_identity coordination).
    session_id = ""
    try:
        metadata = json.loads(ctx.job.metadata or "{}")
        session_id = (metadata.get("session_id") or "").strip()
    except (ValueError, TypeError) as exc:
        logger.warning("agent_worker: failed to parse ctx.job.metadata (%s) — STT will link to first participant", exc)

    logger.info(
        "AgentSession built — agent=%s stt=deepgram/nova-3 tts=%s/sonic-3 voice=%s llm=%s session_id=%s",
        JARVIS_AGENT_WORKER_NAME, JARVIS_LK_TTS_PROVIDER, JARVIS_LK_TTS_VOICE, JARVIS_LK_LLM,
        session_id or "<missing — falling back to first participant>",
    )

    if session_id:
        await session.start(
            agent=JarvisAgent(),
            room=ctx.room,
            room_options=room_io.RoomOptions(
                participant_identity=f"recall-relay-{session_id}",
            ),
        )
    else:
        # Dev / manual dispatch without metadata — STT links to first participant.
        # Acceptable for local testing; production always sets session_id (Plan 005 / review/api.py).
        await session.start(agent=JarvisAgent(), room=ctx.room)


if __name__ == "__main__":
    cli.run_app(server)
