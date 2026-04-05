"""
Jarvis Meeting Assistant (Agentic Version)
===========================================
A wake-word activated AI assistant that joins meetings via Recall.ai.
This version uses standard OpenAI Chat Completions SDK with Function Calling 
(Acting as an Agent) for RAG interactions with Pinecone & Confluence.

Usage:
  uvicorn confluence_logic.jarvis_agentic:app --host 0.0.0.0 --port 8000
"""

import os
import re
import sys
import time
import base64
import logging
from threading import Thread
from typing import Optional

import requests
import uvicorn
from dotenv import load_dotenv
from gtts import gTTS
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

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

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI()
session_agent = EditorAgent(model=JARVIS_AGENT_MODEL)
logger.info("Jarvis meeting agent using model: %s", JARVIS_AGENT_MODEL)

meeting_state = {
    "bot_id": None,
    "transcript_log": [],
    "is_active": False,
    "jarvis_listening": False,
}

def create_bot(meeting_url: str) -> Optional[str]:
    ws_url = WEBHOOK_URL.replace("https://", "wss://") + "/recall-audio-stream"
    payload = {
        "meeting_url": meeting_url,
        "bot_name": BOT_NAME,
        "recording_config": {
            "transcript": {
                "provider": {
                    "recallai_streaming": {
                        "mode": STREAMING_MODE,
                        "language_code": LANGUAGE_CODE,
                    }
                }
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
    
    try:
        response = requests.post(
            f"{RECALL_BASE_URL}/bot/",
            headers={"Authorization": f"Token {RECALL_API_KEY}", "Content-Type": "application/json"},
            json=payload,
            timeout=10,
        )
        if response.status_code in [200, 201]:
            bot_id = response.json()["id"]
            logger.info(f"✅ Bot created: {bot_id}")
            return bot_id
        logger.error(f"❌ Bot creation failed: {response.text}")
        return None
    except Exception as e:
        logger.error(f"❌ Bot creation exception: {e}")
        return None

def speak(text: str, bot_id: str) -> bool:
    tmp = "temp_jarvis_agentic.mp3"
    try:
        tts = gTTS(text, lang=LANGUAGE_CODE)
        tts.save(tmp)
        with open(tmp, "rb") as f:
            audio_b64 = base64.b64encode(f.read()).decode()

        resp = requests.post(
            f"{RECALL_BASE_URL}/bot/{bot_id}/output_audio/",
            headers={"Authorization": f"Token {RECALL_API_KEY}", "Content-Type": "application/json"},
            json={"kind": "mp3", "b64_data": audio_b64},
            timeout=10,
        )
        return resp.status_code == 200
    except Exception as e:
        logger.error(f"❌ speak() error: {e}")
        return False
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)

import asyncio

async def handle_query(query: str, bot_id: str) -> None:
    logger.info(f"🧠 Query String Detected: {query!r}")
    try:
        answer = await session_agent.handle_query(query)
        logger.info(f"🤖 Jarvis Agentic: {answer}")
        await asyncio.to_thread(speak, answer, bot_id)
        
    except Exception as e:
        error_str = str(e)
        logger.error(f"❌ handle_query error: {error_str}")
        await asyncio.to_thread(speak, "Sorry, I ran into an error connecting to my agent brain.", bot_id)

_WAKE_PATTERN = re.compile(r"(?:hey\s+)?jarvis[,.]?\s*(.*)", re.IGNORECASE)

def extract_wake_and_query(text: str) -> Optional[str]:
    m = _WAKE_PATTERN.search(text)
    if m:
        return m.group(1).strip()
    return None

@app.websocket("/recall-audio-stream")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("🔌 WebSocket connected")

    try:
        while True:
            data = await websocket.receive_json()

            if data.get("event") != "transcript.data":
                continue

            data_block = data["data"]["data"]
            participant = data_block["participant"]["name"]
            words = data_block.get("words", [])
            sentence = " ".join(w["text"] for w in words).strip()

            if not sentence or BOT_NAME.lower() in participant.lower():
                continue
                
            meeting_state["transcript_log"].append({
                "participant": participant,
                "text": sentence,
                "timestamp": time.time(),
            })

            bot_id = meeting_state.get("bot_id")
            if not bot_id:
                # Fallback to local bot_id.txt if restarting uvicorn
                if os.path.exists("bot_id.txt"):
                    with open("bot_id.txt", "r") as f:
                        bot_id = f.read().strip()
                        meeting_state["bot_id"] = bot_id

            if not bot_id:
                continue

            query = extract_wake_and_query(sentence)

            if query is not None:
                if query:
                    meeting_state["jarvis_listening"] = False
                    asyncio.create_task(handle_query(query, bot_id))
                else:
                    meeting_state["jarvis_listening"] = True
            elif meeting_state["jarvis_listening"]:
                meeting_state["jarvis_listening"] = False
                asyncio.create_task(handle_query(sentence, bot_id))

    except WebSocketDisconnect:
        logger.info("🔌 WebSocket disconnected")
    except Exception as e:
        logger.error(f"❌ WebSocket error: {e}")

@app.get("/health")
async def health():
    return {
        "status": "healthy_agentic",
        "bot_id": meeting_state["bot_id"]
    }

def _start_server():
    uvicorn.run(app, host=APP_HOST, port=APP_PORT, log_level="warning")

def main():
    print("=" * 60)
    print(f"🤖 {BOT_NAME} AGENTIC Editor Bot")
    print("=" * 60)

    Thread(target=_start_server, daemon=True).start()
    time.sleep(2)

    meeting_url = MEETING_URL or input("\nMeeting URL (Google Meet / Zoom / Teams): ").strip()
    if not meeting_url:
        print("❌ No meeting URL provided")
        sys.exit(1)

    print(f"🚀 Joining: {meeting_url}")
    bot_id = create_bot(meeting_url)
    if not bot_id:
        print("❌ Failed to spawn bot")
        sys.exit(1)

    meeting_state["bot_id"] = bot_id
    meeting_state["is_active"] = True

    with open("bot_id.txt", "w") as f:
        f.write(bot_id)

    print(f"✅ Bot joined! ID: {bot_id}")
    print(f'👂 Listening for "Hey {BOT_NAME}"...')
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print(f"\n⚠️ shutting down.")

if __name__ == "__main__":
    main()
