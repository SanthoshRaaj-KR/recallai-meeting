"""
Jarvis Meeting Assistant
========================
A wake-word activated AI assistant that joins meetings via Recall.ai.

Usage:
  python jarvis.py

Wake word: "Hey Jarvis" or "Jarvis"
Example commands:
  "Hey Jarvis, what's the weather in Paris?"
  "Hey Jarvis, summarize what we've discussed so far."
  "Hey Jarvis, who is the current CEO of Apple?"
"""

import os
import re
import sys
import json
import time
import base64
import logging
from threading import Thread
from typing import Optional

import requests
import uvicorn
from dotenv import load_dotenv
from gtts import gTTS
from openai import OpenAI
from fastapi import FastAPI, WebSocket, WebSocketDisconnect


# ============================================================================
# CONFIGURATION
# ============================================================================

load_dotenv()

RECALL_API_KEY = os.getenv("RECALL_API_KEY")
RECALL_API_REGION = os.getenv("RECALL_API_REGION", "ap-northeast-1")
RECALL_BASE_URL = f"https://{RECALL_API_REGION}.recall.ai/api/v1"
WEBHOOK_URL = os.getenv("WEBHOOK_URL")
MEETING_URL = os.getenv("MEETING_URL")
BOT_NAME = os.getenv("BOT_NAME", "Jarvis")
LANGUAGE_CODE = os.getenv("LANGUAGE_CODE", "en")
STREAMING_MODE = os.getenv("STREAMING_MODE", "prioritize_low_latency")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
APP_HOST = os.getenv("APP_HOST", "0.0.0.0")
APP_PORT = int(os.getenv("APP_PORT", "8000"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

client = OpenAI()
app = FastAPI()

# ============================================================================
# MEETING STATE
# ============================================================================

meeting_state = {
    "bot_id": None,
    "transcript_log": [],   # [{participant, text, timestamp}, ...]
    "is_active": False,
    "jarvis_listening": False,  # True after bare "Hey Jarvis" with no query
}

# ============================================================================
# RECALL.AI HELPERS
# ============================================================================

def create_bot(meeting_url: str) -> Optional[str]:
    """Spawn a Recall.ai bot and return its bot_id."""
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
        logger.error(f"❌ Bot creation failed: {response.status_code} — {response.text}")
        return None
    except Exception as e:
        logger.error(f"❌ Bot creation exception: {e}")
        return None


def speak(text: str, bot_id: str) -> bool:
    """Convert text to speech and play it in the meeting."""
    tmp = "temp_jarvis.mp3"
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

# ============================================================================
# TOOLS AVAILABLE TO JARVIS
# ============================================================================

def _fetch_weather(city: str) -> str:
    """Fetch weather from wttr.in (no API key required)."""
    try:
        resp = requests.get(f"https://wttr.in/{city}?format=3", timeout=5)
        return resp.text.strip() if resp.status_code == 200 else f"No weather data for {city}."
    except Exception as e:
        return f"Weather service unavailable: {e}"


def _get_meeting_transcript() -> str:
    """Return the accumulated meeting transcript as a readable string."""
    if not meeting_state["transcript_log"]:
        return "[No transcript yet]"
    lines = [f"{e['participant']}: {e['text']}" for e in meeting_state["transcript_log"]]
    return "\n".join(lines)


# OpenAI function-calling tool definitions
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a city or location.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name, e.g. 'Paris' or 'New York'"},
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_meeting_summary",
            "description": "Get the transcript of the current meeting to answer questions about it.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
]


def _run_tool(name: str, args: dict) -> str:
    if name == "get_weather":
        return _fetch_weather(args.get("city", ""))
    if name == "get_meeting_summary":
        return _get_meeting_transcript()
    return f"Unknown tool: {name}"

# ============================================================================
# JARVIS QUERY HANDLER
# ============================================================================

def handle_query(query: str, bot_id: str) -> None:
    """Send a user query to Jarvis and speak the response back in the meeting."""
    logger.info(f"🧠 Query: {query!r}")

    system_prompt = (
        f"You are {BOT_NAME}, an AI assistant attending a meeting. "
        "Answer questions concisely (1-3 sentences). "
        "Use your tools for weather or meeting-related questions. "
        "For general knowledge questions, answer directly without tools."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": query},
    ]

    try:
        # Agentic loop with tool calling
        for _ in range(5):  # max 5 tool-call rounds
            response = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",
                max_tokens=300,
            )

            msg = response.choices[0].message
            finish_reason = response.choices[0].finish_reason

            if finish_reason == "tool_calls" and msg.tool_calls:
                messages.append(msg)
                for tc in msg.tool_calls:
                    args = json.loads(tc.function.arguments)
                    result = _run_tool(tc.function.name, args)
                    logger.info(f"🔧 Tool {tc.function.name}({args}) → {result[:80]}")
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result,
                    })
            else:
                answer = (msg.content or "").strip()
                logger.info(f"🤖 {BOT_NAME}: {answer}")
                speak(answer, bot_id)
                return

        speak("I couldn't complete that request. Please try again.", bot_id)

    except Exception as e:
        error_str = str(e)
        logger.error(f"❌ handle_query error: {error_str}")
        if "insufficient_quota" in error_str or "429" in error_str:
            speak("Sorry, the AI service is out of credits. Please check the OpenAI billing.", bot_id)
        elif "401" in error_str or "invalid_api_key" in error_str:
            speak("Sorry, the AI service API key is invalid.", bot_id)
        else:
            speak("Sorry, I ran into an error. Please try again.", bot_id)

# ============================================================================
# WAKE WORD DETECTION
# ============================================================================

# Matches "hey jarvis", "jarvis", with optional punctuation, captures trailing query
_WAKE_PATTERN = re.compile(
    r"(?:hey\s+)?jarvis[,.]?\s*(.*)",
    re.IGNORECASE,
)


def extract_wake_and_query(text: str) -> Optional[str]:
    """
    Check if text contains the wake word.
    Returns the query string (may be empty) if wake word found, None otherwise.
    """
    m = _WAKE_PATTERN.search(text)
    if m:
        return m.group(1).strip()
    return None

# ============================================================================
# WEBSOCKET HANDLER
# ============================================================================

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

            if not sentence:
                continue

            # Ignore bot's own speech
            if BOT_NAME.lower() in participant.lower():
                continue

            logger.info(f"💬 {participant}: {sentence}")

            # Append to meeting log
            meeting_state["transcript_log"].append({
                "participant": participant,
                "text": sentence,
                "timestamp": time.time(),
            })

            bot_id = meeting_state["bot_id"]
            if not bot_id:
                continue

            query = extract_wake_and_query(sentence)

            if query is not None:
                # Wake word detected in this chunk
                if query:
                    # Full query in same sentence: "Hey Jarvis, what's the weather in Paris?"
                    meeting_state["jarvis_listening"] = False
                    Thread(target=handle_query, args=(query, bot_id), daemon=True).start()
                else:
                    # Bare wake word: "Hey Jarvis" — acknowledge and wait for next chunk
                    meeting_state["jarvis_listening"] = True
                    Thread(target=speak, args=("Yes?", bot_id), daemon=True).start()

            elif meeting_state["jarvis_listening"]:
                # Previous chunk was just the wake word; this chunk is the query
                meeting_state["jarvis_listening"] = False
                Thread(target=handle_query, args=(sentence, bot_id), daemon=True).start()

    except WebSocketDisconnect:
        logger.info("🔌 WebSocket disconnected")
    except Exception as e:
        logger.error(f"❌ WebSocket error: {e}")


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "bot_id": meeting_state["bot_id"],
        "active": meeting_state["is_active"],
        "transcript_lines": len(meeting_state["transcript_log"]),
    }

# ============================================================================
# ENTRY POINT
# ============================================================================

def _start_server():
    uvicorn.run(app, host=APP_HOST, port=APP_PORT, log_level="warning")


def main():
    print("=" * 60)
    print(f"🤖 {BOT_NAME} Meeting Assistant")
    print(f'   Wake word: "Hey {BOT_NAME}" or "{BOT_NAME}"')
    print("=" * 60)

    missing = [k for k in ("RECALL_API_KEY", "OPENAI_API_KEY", "WEBHOOK_URL") if not os.getenv(k)]
    if missing:
        for k in missing:
            print(f"❌ {k} not set in .env")
        sys.exit(1)

    # Start WebSocket server
    Thread(target=_start_server, daemon=True).start()
    print(f"✅ Listening on {APP_HOST}:{APP_PORT}")
    time.sleep(2)

    # Resolve meeting URL
    meeting_url = MEETING_URL or input("\nMeeting URL (Google Meet / Zoom / Teams): ").strip()
    if not meeting_url:
        print("❌ No meeting URL provided")
        sys.exit(1)

    # Spawn bot
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
    print("   Ctrl+C to stop\n")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print(f"\n⚠️  {BOT_NAME} shutting down.")
        meeting_state["is_active"] = False


if __name__ == "__main__":
    main()
