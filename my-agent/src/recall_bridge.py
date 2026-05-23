"""
Recall.ai Bridge for my-agent LiveKit integration.

This FastAPI server bridges Recall.ai meeting bots to the my-agent LiveKit agent.

Architecture:
  1. POST /bot/start  → creates a Recall.ai bot that joins the given meeting URL
  2. Recall bot loads GET /bot-page via output_media.camera.kind=webpage
  3. bot-page (bot.html) connects to LiveKit in two rooms:
       Publisher  → joins as "recall-browser-{room_name}", publishes meeting audio
       Subscriber → plays back agent TTS audio into the meeting via <audio> element
  4. my-agent AgentSession subscribes ONLY to "recall-browser-{room_name}" for STT,
     so it hears the mixed meeting audio (all participants combined).

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
"""

import datetime
import json
import logging
import os
import uuid
from pathlib import Path
from typing import Optional

import requests
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from livekit import api as livekit_api
from pydantic import BaseModel

load_dotenv(".env.local")

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ── Config ────────────────────────────────────────────────────────────────────
RECALL_API_KEY = os.getenv("RECALL_API_KEY", "")
RECALL_API_REGION = os.getenv("RECALL_API_REGION", "us-west-2")
RECALL_BASE_URL = f"https://{RECALL_API_REGION}.recall.ai/api/v1"

LIVEKIT_URL = os.getenv("LIVEKIT_URL", "")
LIVEKIT_API_KEY = os.getenv("LIVEKIT_API_KEY", "")
LIVEKIT_API_SECRET = os.getenv("LIVEKIT_API_SECRET", "")

# Public HTTPS URL of this server — Recall requires HTTPS for output_media URLs.
# When developing locally, set this to your ngrok tunnel URL:
#   ngrok http 8001
#   BRIDGE_SERVER_URL=https://xxxx.ngrok.io
SERVER_URL = os.getenv("BRIDGE_SERVER_URL", "").rstrip("/")

BOT_NAME = os.getenv("BOT_NAME", "Meeting Assistant")
AGENT_NAME = os.getenv("AGENT_NAME", "my-agent")
TOKEN_TTL_HOURS = int(os.getenv("TOKEN_TTL_HOURS", "8"))

_BOT_HTML_PATH = Path(__file__).parent / "bot.html"

app = FastAPI(title="Recall.ai Bridge for my-agent")


# ── Request/response models ───────────────────────────────────────────────────

class StartBotRequest(BaseModel):
    meeting_url: str
    room_name: Optional[str] = None  # auto-generated UUID if not provided


class StartBotResponse(BaseModel):
    status: str
    bot_id: str
    room_name: str
    meeting_url: str


# ── LiveKit token minting ─────────────────────────────────────────────────────

def _mint_token(room_name: str, identity: str, can_publish: bool = True) -> str:
    """Mint a LiveKit JWT for the given room and participant identity."""
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


# ── Recall bot creation ───────────────────────────────────────────────────────

def _create_recall_bot(meeting_url: str, room_name: str) -> str:
    """
    Call Recall.ai to create a bot that joins `meeting_url`.

    The bot uses output_media.camera.kind=webpage to load bot.html.
    bot.html then:
      - Publisher room  → publishes meeting audio to LiveKit as recall-browser-{room_name}
      - Subscriber room → plays back agent TTS audio so the meeting hears the agent

    Returns the Recall bot_id.
    """
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

    # IDENTITY CONTRACT (must match agent.py participant_identity filter):
    #   Publisher identity = "recall-browser-{room_name}"
    #   This is the participant the AgentSession STT subscribes to.
    #   Subscriber identity = "recall-listener-{room_name}" (just needs room access)
    publisher_token = _mint_token(room_name, f"recall-browser-{room_name}", can_publish=True)
    subscriber_token = _mint_token(room_name, f"recall-listener-{room_name}", can_publish=False)

    # Build the URL for the bot-page that Recall's headless Chrome will load.
    # All LiveKit connection details are passed as query params so bot.html
    # can read them from window.location.search (no server-side rendering needed).
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
        # output_media: Recall loads this URL in a headless Chrome tab.
        # The page captures meeting audio (getUserMedia) and publishes it to LiveKit,
        # and plays back agent TTS audio from LiveKit into the meeting.
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
    """
    Dispatch my-agent to the LiveKit room via the agent dispatch API.

    Without this call the agent process is registered but never receives a job,
    so it never joins the room and never hears the Recall bot's audio.

    The metadata JSON string is parsed by agent.py to extract room_name,
    which is then used to set participant_identity in RoomOptions.
    """
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


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.post("/bot/start", response_model=StartBotResponse)
async def start_bot(body: StartBotRequest) -> StartBotResponse:
    """
    Start a Recall.ai bot that joins the given meeting URL and connects it to my-agent.

    Steps:
      1. Generate a room_name (UUID) that both the Recall bot and the agent share.
      2. Create the Recall bot — it will join the meeting and load /bot-page.
      3. Dispatch my-agent to the LiveKit room so it's ready when the bot connects.

    Example:
        curl -X POST http://localhost:8001/bot/start \\
             -H "Content-Type: application/json" \\
             -d '{"meeting_url": "https://zoom.us/j/123456789"}'
    """
    room_name = body.room_name or str(uuid.uuid4())

    try:
        bot_id = _create_recall_bot(body.meeting_url, room_name)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except requests.HTTPError as exc:
        body = exc.response.text if exc.response is not None else ""
        raise HTTPException(status_code=502, detail=f"Recall.ai API error: {exc} — {body}")

    try:
        await _dispatch_agent(room_name)
    except Exception as exc:
        # Non-fatal: in dev mode the agent worker picks up rooms via room watch.
        logger.warning("Agent dispatch failed (non-fatal in dev mode): %s", exc)

    return StartBotResponse(
        status="joining",
        bot_id=bot_id,
        room_name=room_name,
        meeting_url=body.meeting_url,
    )


@app.get("/bot-page", response_class=HTMLResponse)
async def bot_page() -> HTMLResponse:
    """
    Serve the LiveKit bridge HTML page to Recall's headless Chrome.

    Recall's bot loads this URL (with LiveKit credentials as query params).
    The page:
      - Publisher room: captures meeting audio via getUserMedia() and publishes
        it to LiveKit as "recall-browser-{room_name}"
      - Subscriber room: plays back agent TTS audio into the meeting
    """
    if not _BOT_HTML_PATH.exists():
        logger.error("bot.html not found at %s", _BOT_HTML_PATH)
        return HTMLResponse(
            "<html><body>bot.html not found — ensure src/bot.html exists</body></html>",
            status_code=500,
        )
    return HTMLResponse(_BOT_HTML_PATH.read_text(encoding="utf-8"))


@app.get("/health")
async def health() -> dict:
    """Quick health check for the bridge server."""
    return {
        "status": "ok",
        "agent_name": AGENT_NAME,
        "recall_region": RECALL_API_REGION,
        "recall_configured": bool(RECALL_API_KEY),
        "livekit_configured": bool(LIVEKIT_URL and LIVEKIT_API_KEY and LIVEKIT_API_SECRET),
        "server_url": SERVER_URL or "<not set>",
    }
