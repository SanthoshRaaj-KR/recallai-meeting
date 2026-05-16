"""
LiveKit Native Agent Worker for Jarvis (Phase 03).

Standalone process — run with:
    python -m confluence_logic.agent_worker dev      # development (hot reload)
    python -m confluence_logic.agent_worker start    # production

Designed to coexist with the FastAPI server (uvicorn confluence_logic.jarvis_agentic:app).
Recall.ai still provides meeting transcripts; this worker only handles LLM + TTS via the
LiveKit AgentSession framework. Dispatched explicitly from review/api.py per session.

REQ-11 (worker entrypoint), REQ-13 (stt=None pattern), REQ-14 (Cartesia Sonic-3),
REQ-15 (tools wired in Plan 03+04), REQ-20 (env var configuration).
"""
from __future__ import annotations

import asyncio
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
    """Per-room session: build the AgentSession and start it on ctx.room."""
    session = AgentSession(
        stt=None,                                                                  # REQ-13
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
    logger.info(
        "AgentSession built — agent=%s tts=%s/sonic-3 voice=%s llm=%s",
        JARVIS_AGENT_WORKER_NAME, JARVIS_LK_TTS_PROVIDER, JARVIS_LK_TTS_VOICE, JARVIS_LK_LLM,
    )

    # --- IPC: FastAPI publishes wake-word queries to this room via publish_data() (REQ-18 / Plan 05).
    # Payload schema (locked in agent_worker.py and review/api.py — keep in sync):
    #   {"type": "user_query", "query": "<text>", "session_id": "<uuid>"}
    # Handler is a sync def (livekit-rtc emits events synchronously); generate_reply runs as a task.
    @ctx.room.on("data_received")
    def _on_data(packet) -> None:
        try:
            data = json.loads(packet.data.decode("utf-8") if isinstance(packet.data, (bytes, bytearray)) else packet.data)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            logger.warning("agent_worker IPC: failed to decode data packet (%d bytes): %s", len(packet.data or b""), exc)
            return
        if not isinstance(data, dict):
            logger.warning("agent_worker IPC: non-dict payload dropped: %r", data)
            return
        if data.get("type") != "user_query":
            logger.debug("agent_worker IPC: ignoring non-user_query type=%r", data.get("type"))
            return
        query = (data.get("query") or "").strip()
        if not query:
            logger.warning("agent_worker IPC: user_query with empty query — dropping")
            return
        logger.info("agent_worker IPC: dispatching user_query (%d chars) to generate_reply", len(query))
        asyncio.create_task(session.generate_reply(user_input=query))

    await session.start(agent=JarvisAgent(), room=ctx.room)


if __name__ == "__main__":
    cli.run_app(server)
