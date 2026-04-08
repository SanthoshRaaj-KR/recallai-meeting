"""
Jarvis Meeting Assistant (Agentic Version)
===========================================
A wake-word activated AI assistant that joins meetings via Recall.ai.
This version uses the OpenAI Agents SDK for Confluence actions, AssemblyAI
transcription through Recall, and hosted TTS.

Usage:
  uvicorn confluence_logic.jarvis_agentic:app --host 0.0.0.0 --port 8000
"""

import asyncio
import base64
import logging
import os
import re
import sys
import time
from io import BytesIO
from threading import Thread
from typing import Optional

import requests
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from gtts import gTTS
from openai import OpenAI

from .agents.editor_agent import EditorAgent

load_dotenv()

RECALL_API_KEY = os.getenv("RECALL_API_KEY")
RECALL_API_REGION = os.getenv("RECALL_API_REGION", "ap-northeast-1")
RECALL_BASE_URL = f"https://{RECALL_API_REGION}.recall.ai/api/v1"
WEBHOOK_URL = os.getenv("WEBHOOK_URL")
MEETING_URL = os.getenv("MEETING_URL")
BOT_NAME = os.getenv("BOT_NAME", "Jarvis")
LANGUAGE_CODE = os.getenv("LANGUAGE_CODE", "en")
STREAMING_MODE = os.getenv("STREAMING_MODE", "prioritize_low_latency")
APP_HOST = os.getenv("APP_HOST", "0.0.0.0")
APP_PORT = int(os.getenv("APP_PORT", "8000"))
JARVIS_AGENT_MODEL = os.getenv("JARVIS_AGENT_MODEL", "gpt-5-mini")
RECALL_TRANSCRIPT_PROVIDER = os.getenv("RECALL_TRANSCRIPT_PROVIDER", "recallai_streaming").strip()
ASSEMBLY_API = (os.getenv("ASSEMBLY_API") or "").strip()

JARVIS_TTS_PROVIDER = os.getenv("JARVIS_TTS_PROVIDER", "openai").strip().lower()
JARVIS_TTS_MODEL = os.getenv("JARVIS_TTS_MODEL", "gpt-4o-mini-tts").strip()
JARVIS_TTS_VOICE = os.getenv("JARVIS_TTS_VOICE", "echo").strip()
JARVIS_TTS_SPEED = float(os.getenv("JARVIS_TTS_SPEED", "1.0"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI()
session_agent = EditorAgent(model=JARVIS_AGENT_MODEL)
logger.info("Jarvis meeting agent using model: %s", JARVIS_AGENT_MODEL)
logger.info("Jarvis transcript provider: %s", RECALL_TRANSCRIPT_PROVIDER)
logger.info("Jarvis TTS provider: %s", JARVIS_TTS_PROVIDER)
if RECALL_TRANSCRIPT_PROVIDER == "assembly_ai_v3_streaming" and ASSEMBLY_API:
    logger.warning(
        "ASSEMBLY_API is set locally, but Recall BYOB transcription still requires the AssemblyAI key "
        "to be configured in the Recall transcription credentials dashboard."
    )

meeting_state = {
    "bot_id": None,
    "transcript_log": [],
    "is_active": False,
    "jarvis_listening": False,
}

_openai_client: Optional[OpenAI] = None
_WAKE_PATTERN = re.compile(r"(?:hey\s+)?jarvis[,.]?\s*(.*)", re.IGNORECASE)


def get_openai_client() -> OpenAI:
    global _openai_client
    if _openai_client is None:
        _openai_client = OpenAI()
    return _openai_client


def build_transcript_provider_config() -> dict:
    if RECALL_TRANSCRIPT_PROVIDER == "recallai_streaming":
        return {
            "recallai_streaming": {
                "mode": STREAMING_MODE,
                "language_code": LANGUAGE_CODE,
            }
        }

    if RECALL_TRANSCRIPT_PROVIDER == "assembly_ai_v3_streaming":
        return {
            "assembly_ai_v3_streaming": {}
        }

    return {
        RECALL_TRANSCRIPT_PROVIDER: {
            "mode": STREAMING_MODE,
            "language_code": LANGUAGE_CODE,
        }
    }


def build_create_bot_payload(meeting_url: str) -> dict:
    ws_url = WEBHOOK_URL.replace("https://", "wss://") + "/recall-audio-stream"
    return {
        "meeting_url": meeting_url,
        "bot_name": BOT_NAME,
        "recording_config": {
            "transcript": {
                "provider": build_transcript_provider_config()
            },
            "realtime_endpoints": [
                {
                    "type": "websocket",
                    "url": ws_url,
                    "events": ["transcript.data"],
                }
            ],
        },
    }


def create_bot(meeting_url: str) -> Optional[str]:
    payload = build_create_bot_payload(meeting_url)
    try:
        response = requests.post(
            f"{RECALL_BASE_URL}/bot/",
            headers={"Authorization": f"Token {RECALL_API_KEY}", "Content-Type": "application/json"},
            json=payload,
            timeout=10,
        )
        if response.status_code in [200, 201]:
            bot_id = response.json()["id"]
            logger.info("Bot created: %s", bot_id)
            return bot_id
        logger.error("Bot creation failed: %s", response.text)
        return None
    except Exception as e:
        logger.error("Bot creation exception: %s", e)
        return None


def synthesize_speech(text: str) -> bytes:
    if JARVIS_TTS_PROVIDER == "gtts":
        buffer = BytesIO()
        gTTS(text, lang=LANGUAGE_CODE).write_to_fp(buffer)
        return buffer.getvalue()

    response = get_openai_client().audio.speech.create(
        model=JARVIS_TTS_MODEL,
        voice=JARVIS_TTS_VOICE,
        input=text,
        response_format="mp3",
        speed=JARVIS_TTS_SPEED,
    )

    if hasattr(response, "read"):
        return response.read()
    if hasattr(response, "content"):
        return response.content
    if hasattr(response, "iter_bytes"):
        return b"".join(response.iter_bytes())

    raise TypeError("Unexpected OpenAI TTS response type.")


def speak(text: str, bot_id: str) -> bool:
    try:
        audio_bytes = synthesize_speech(text)
        audio_b64 = base64.b64encode(audio_bytes).decode()
        resp = requests.post(
            f"{RECALL_BASE_URL}/bot/{bot_id}/output_audio/",
            headers={"Authorization": f"Token {RECALL_API_KEY}", "Content-Type": "application/json"},
            json={"kind": "mp3", "b64_data": audio_b64},
            timeout=10,
        )
        return resp.status_code == 200
    except Exception as e:
        logger.error("speak() error: %s", e)
        if JARVIS_TTS_PROVIDER != "gtts":
            try:
                buffer = BytesIO()
                gTTS(text, lang=LANGUAGE_CODE).write_to_fp(buffer)
                resp = requests.post(
                    f"{RECALL_BASE_URL}/bot/{bot_id}/output_audio/",
                    headers={"Authorization": f"Token {RECALL_API_KEY}", "Content-Type": "application/json"},
                    json={"kind": "mp3", "b64_data": base64.b64encode(buffer.getvalue()).decode()},
                    timeout=10,
                )
                return resp.status_code == 200
            except Exception as fallback_error:
                logger.error("gTTS fallback failed: %s", fallback_error)
        return False


async def handle_query(query: str, bot_id: str) -> None:
    logger.info("Query String Detected: %r", query)
    try:
        answer = await session_agent.handle_query(query)
        logger.info("Jarvis Agentic: %s", answer)
        await asyncio.to_thread(speak, answer, bot_id)
    except Exception as e:
        logger.error("handle_query error: %s", e)
        await asyncio.to_thread(speak, "Sorry, I ran into an error connecting to my agent brain.", bot_id)


def extract_wake_and_query(text: str) -> Optional[str]:
    m = _WAKE_PATTERN.search(text)
    if m:
        return m.group(1).strip()
    return None


def _extract_sentence(data_block: dict) -> str:
    words = data_block.get("words", []) or []
    if words:
        return " ".join(word.get("text", "") for word in words).strip()
    return (data_block.get("text") or "").strip()


def process_transcript_event(sentence: str, timestamp: float) -> Optional[str]:
    if not sentence:
        return None
    query = extract_wake_and_query(sentence)
    if query is not None:
        if query:
            meeting_state["jarvis_listening"] = False
            return query
        meeting_state["jarvis_listening"] = True
        return None

    if meeting_state["jarvis_listening"]:
        meeting_state["jarvis_listening"] = False
        return sentence or None
    return None


@app.websocket("/recall-audio-stream")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("WebSocket connected")

    try:
        while True:
            payload = await websocket.receive_json()
            event_name = payload.get("event")
            if event_name != "transcript.data":
                continue

            data_block = payload["data"]["data"]
            participant = data_block["participant"]["name"]
            sentence = _extract_sentence(data_block)

            if not sentence or BOT_NAME.lower() in participant.lower():
                continue

            logger.info("Transcript %s: %s", participant, sentence)

            bot_id = meeting_state.get("bot_id")
            if not bot_id and os.path.exists("bot_id.txt"):
                with open("bot_id.txt", "r") as f:
                    bot_id = f.read().strip()
                    meeting_state["bot_id"] = bot_id

            if not bot_id:
                continue

            meeting_state["transcript_log"].append({
                "participant": participant,
                "text": sentence,
                "timestamp": time.time(),
            })

            query = process_transcript_event(sentence, time.time())
            if query:
                asyncio.create_task(handle_query(query, bot_id))
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as e:
        logger.error("WebSocket error: %s", e)


@app.get("/health")
async def health():
    return {
        "status": "healthy_agentic",
        "bot_id": meeting_state["bot_id"],
        "transcript_provider": RECALL_TRANSCRIPT_PROVIDER,
        "tts_provider": JARVIS_TTS_PROVIDER,
    }


def _start_server():
    uvicorn.run(app, host=APP_HOST, port=APP_PORT, log_level="warning")


def main():
    print("=" * 60)
    print(f"{BOT_NAME} AGENTIC Editor Bot")
    print("=" * 60)

    Thread(target=_start_server, daemon=True).start()
    time.sleep(2)

    meeting_url = MEETING_URL or input("\nMeeting URL (Google Meet / Zoom / Teams): ").strip()
    if not meeting_url:
        print("No meeting URL provided")
        sys.exit(1)

    print(f"Joining: {meeting_url}")
    bot_id = create_bot(meeting_url)
    if not bot_id:
        print("Failed to spawn bot")
        sys.exit(1)

    meeting_state["bot_id"] = bot_id
    meeting_state["is_active"] = True

    with open("bot_id.txt", "w") as f:
        f.write(bot_id)

    print(f"Bot joined! ID: {bot_id}")
    print(f'Listening for "Hey {BOT_NAME}"...')

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nshutting down.")


if __name__ == "__main__":
    main()
