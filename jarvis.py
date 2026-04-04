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

import asyncio
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
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request

from meeting_state import MeetingState
from storage.metadata_store import MetadataStore
from storage.pinecone_client import PineconeClient
from storage.models import MeetingRecord
from agents.summarizer import SummarizerAgent
from agents.retriever import RetrieverAgent
from agents.date_resolver import DateResolutionAgent
from agents.answer_agent import AnswerAgent
from agents.orchestrator import OrchestratorAgent, OrchestratorResult
from slack_bolt.async_app import AsyncApp
from slack_bolt.adapter.fastapi.async_handler import AsyncSlackRequestHandler


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

# Ingestion pipeline env vars
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "meeting-memory")
SLACK_BOT_TOKEN = os.getenv("SLACK_BOT_TOKEN")
SLACK_SIGNING_SECRET = os.getenv("SLACK_SIGNING_SECRET")
SLACK_CHANNEL_ID = os.getenv("SLACK_CHANNEL_ID", "")
MEETING_CHANNEL_NAME = os.getenv("MEETING_CHANNEL_NAME", "general")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

client = OpenAI()
app = FastAPI()

# ============================================================================
# MEETING STATE
# ============================================================================

state = MeetingState()

# ============================================================================
# INGESTION SINGLETONS
# ============================================================================

metadata_store = MetadataStore(base_dir="./meetings")

pinecone_client: Optional[PineconeClient] = None
if PINECONE_API_KEY:
    pinecone_client = PineconeClient(
        api_key=PINECONE_API_KEY,
        index_name=PINECONE_INDEX_NAME,
    )

summarizer = SummarizerAgent()

slack_app: Optional[AsyncApp] = None
slack_handler: Optional[AsyncSlackRequestHandler] = None
if SLACK_BOT_TOKEN and SLACK_SIGNING_SECRET:
    slack_app = AsyncApp(token=SLACK_BOT_TOKEN, signing_secret=SLACK_SIGNING_SECRET)
    slack_handler = AsyncSlackRequestHandler(slack_app)

# ============================================================================
# ORCHESTRATION SINGLETONS
# ============================================================================

_pending_disambig: dict[str, list[dict]] = {}  # keyed by user_id

retriever_agent: Optional[RetrieverAgent] = None
orchestrator: Optional[OrchestratorAgent] = None
if pinecone_client is not None:
    retriever_agent = RetrieverAgent(pinecone_client=pinecone_client)
    orchestrator = OrchestratorAgent(
        retriever=retriever_agent,
        date_resolver=DateResolutionAgent(),
        answer_agent=AnswerAgent(),
    )

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
            logger.info(f"Bot created: {bot_id}")
            return bot_id
        logger.error(f"Bot creation failed: {response.status_code} -- {response.text}")
        return None
    except Exception as e:
        logger.error(f"Bot creation exception: {e}")
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
        logger.error(f"speak() error: {e}")
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


async def _get_meeting_transcript() -> str:
    """Return the accumulated meeting transcript as a readable string."""
    return await state.get_transcript()


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


async def _run_tool(name: str, args: dict) -> str:
    if name == "get_weather":
        return _fetch_weather(args.get("city", ""))
    if name == "get_meeting_summary":
        return await _get_meeting_transcript()
    return f"Unknown tool: {name}"

# ============================================================================
# INGESTION PIPELINE
# ============================================================================

async def run_ingestion_pipeline(transcript: str, meeting_meta: dict) -> MeetingRecord:
    """
    Run the full ingestion pipeline: summarize -> write disk -> conditionally upsert Pinecone.

    Args:
        transcript: Raw formatted transcript string ("Speaker: text\n...").
        meeting_meta: Dict with meeting_id, channel_id, channel_name, start_ts,
                      end_ts, duration_seconds, participants.

    Returns:
        The stored MeetingRecord.
    """
    record = await summarizer.run(transcript=transcript, meeting_meta=meeting_meta)
    await metadata_store.write(record)
    logger.info("Wrote meeting record to disk: %s", record.meeting_id)

    if record.status == "complete" and pinecone_client is not None:
        pinecone_client.upsert_meeting(record)
        logger.info("Upserted meeting to Pinecone: %s", record.meeting_id)
    elif record.status == "partial":
        logger.info(
            "Skipping Pinecone upsert for partial summary: %s (chars=%s)",
            record.meeting_id,
            record.raw_transcript_chars,
        )

    return record


def _is_memory_query(query: str) -> bool:
    """Return True if the query looks like a memory/history question.

    Used by handle_query() to route wake-word queries through the orchestrator
    rather than the existing weather/live-transcript tool-calling loop.
    """
    ql = query.lower()
    patterns = [
        "what did we", "what happened", "what was decided",
        "last week", "last monday", "last tuesday", "last wednesday",
        "last thursday", "last friday", "yesterday", "two weeks",
        "action items", "what did i commit", "my tasks",
        "summarize last", "recap of", "tell me about the meeting",
    ]
    return any(p in ql for p in patterns)


async def _handle_memory_query(
    query: str,
    user_id: str,
    channel_id: str,
) -> OrchestratorResult:
    """Route a memory query through the OrchestratorAgent.

    Returns OrchestratorResult. If orchestrator is not configured (no Pinecone),
    returns a static OrchestratorResult with a configuration error message.
    """
    if orchestrator is None:
        return OrchestratorResult(
            query=query,
            query_type="memory_query",
            answer="Memory queries are not configured — PINECONE_API_KEY is missing.",
            source_meeting_ids=[],
            confidence="low",
        )
    return await orchestrator.run(query=query, user_id=user_id, channel_id=channel_id)


async def _handle_ask(ack, say, client, command):
    """Slack slash command: /ask <question about past meetings>

    Defined at module level so it can be imported in tests.
    Registered with slack_app below if Slack is configured.
    """
    await ack()

    query = (command.get("text") or "").strip()
    user_id = command.get("user_id", "unknown")
    channel_id = command.get("channel_id", SLACK_CHANNEL_ID)

    if not query:
        await say("Usage: `/ask <your question about past meetings>`\n"
                  "Example: `/ask what did we decide about the API design?`")
        return

    try:
        result = await _handle_memory_query(query, user_id, channel_id)
    except Exception as e:
        logger.error("OrchestratorAgent error in /ask: %s", e)
        await say(f"Sorry, I encountered an error: {e}")
        return

    if result.needs_disambiguation:
        lines = ["*Multiple meetings found. Please reply with the number of the meeting you mean:*\n"]
        for opt in result.disambiguation_options:
            lines.append(f"{opt['index']}. *{opt['title']}* — #{opt['channel']} on {opt['date']}")
        _pending_disambig[user_id] = result.disambiguation_options
        await say("\n".join(lines))
    else:
        confidence_tag = f" _(confidence: {result.confidence})_" if result.confidence != "high" else ""
        await say(f"{result.answer}{confidence_tag}")


async def _handle_message_disambig(message, say, client):
    """Handle user replies to disambiguation prompts.

    Defined at module level so it can be imported in tests.
    Registered with slack_app below if Slack is configured.

    If a user has a pending disambiguation and sends a digit, resolve
    to the selected meeting and re-run the query scoped to that meeting_id.
    """
    user_id = message.get("user", "")
    text = (message.get("text") or "").strip()
    channel_id = message.get("channel", "")

    pending = _pending_disambig.get(user_id)
    if not pending:
        return  # No pending disambiguation for this user

    if not text.isdigit():
        return  # Not a number reply — ignore

    selection = int(text)
    if selection < 1 or selection > len(pending):
        await say(f"Please reply with a number between 1 and {len(pending)}.")
        return

    chosen = pending[selection - 1]
    del _pending_disambig[user_id]

    # Re-run the query scoped to the chosen meeting
    scoped_query = f"Tell me about meeting {chosen['meeting_id']}"
    try:
        result = await _handle_memory_query(scoped_query, user_id, channel_id)
    except Exception as e:
        await say(f"Error retrieving that meeting: {e}")
        return

    await say(result.answer)


# Register /summarize, /ask, and message handlers when Slack is configured
if slack_app is not None:
    @slack_app.command("/summarize")
    async def handle_summarize_command(ack, say, client, command):
        """Slack slash command: trigger ingestion pipeline and post structured summary."""
        await ack()  # Acknowledge within 3 seconds — Slack requirement

        transcript = await state.get_transcript()
        if not transcript or transcript == "[No transcript yet]":
            await say("No meeting transcript available yet. Is the bot in a meeting?")
            return

        meeting_id = await state.get_bot_id() or f"mtg-{int(time.time())}"
        meeting_meta = {
            "meeting_id": meeting_id,
            "channel_id": command.get("channel_id", SLACK_CHANNEL_ID),
            "channel_name": command.get("channel_name", MEETING_CHANNEL_NAME),
            "start_ts": int(time.time()) - 3600,
            "end_ts": int(time.time()),
            "duration_seconds": None,
            "participants": [],
        }

        try:
            record = await run_ingestion_pipeline(transcript, meeting_meta)
        except Exception as e:
            logger.error("Ingestion pipeline failed: %s", e)
            await say(f"Summary failed: {e}")
            return

        # Format structured summary for Slack
        action_items_text = "\n".join(
            f"  \u2022 {item.owner}: {item.task}" + (f" (due: {item.due})" if item.due else "")
            for item in record.action_items
        ) or "  None identified"

        decisions_text = "\n".join(f"  \u2022 {d}" for d in record.decisions) or "  None identified"
        topics_text = "\n".join(f"  \u2022 {t}" for t in record.topics_covered) or "  None identified"
        participants_text = ", ".join(record.participants) or "Unknown"

        status_tag = " _(partial \u2014 meeting may still be in progress)_" if record.status == "partial" else ""

        message = (
            f"*Meeting Summary*{status_tag}\n\n"
            f"*Participants:* {participants_text}\n\n"
            f"*Topics Discussed:*\n{topics_text}\n\n"
            f"*Decisions:*\n{decisions_text}\n\n"
            f"*Action Items:*\n{action_items_text}"
        )

        target_channel = command.get("channel_id", SLACK_CHANNEL_ID)
        await client.chat_postMessage(channel=target_channel, text=message)

    slack_app.command("/ask")(_handle_ask)
    slack_app.event("message")(_handle_message_disambig)

# ============================================================================
# JARVIS QUERY HANDLER
# ============================================================================

async def handle_query(query: str, bot_id: str) -> None:
    """Send a user query to Jarvis and speak the response back in the meeting."""
    logger.info(f"Query: {query!r}")

    # Route memory questions through the orchestrator if configured
    if orchestrator is not None and _is_memory_query(query):
        try:
            result = await _handle_memory_query(
                query=query,
                user_id="voice",
                channel_id=SLACK_CHANNEL_ID,
            )
            if not result.needs_disambiguation:
                await asyncio.to_thread(speak, result.answer, bot_id)
                return
            # Disambiguation in voice context: speak the options aloud
            spoken = "Multiple meetings found. " + " ".join(
                f"Option {opt['index']}: {opt['title']} from {opt['date']}."
                for opt in result.disambiguation_options
            )
            await asyncio.to_thread(speak, spoken, bot_id)
            return
        except Exception as e:
            logger.error("Orchestrator error in handle_query: %s", e)
            # Fall through to existing tool-calling loop on error

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
                    result = await _run_tool(tc.function.name, args)
                    logger.info(f"Tool {tc.function.name}({args}) -> {result[:80]}")
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result,
                    })
            else:
                answer = (msg.content or "").strip()
                logger.info(f"{BOT_NAME}: {answer}")
                await asyncio.to_thread(speak, answer, bot_id)
                return

        await asyncio.to_thread(speak, "I couldn't complete that request. Please try again.", bot_id)

    except Exception as e:
        error_str = str(e)
        logger.error(f"handle_query error: {error_str}")
        if "insufficient_quota" in error_str or "429" in error_str:
            await asyncio.to_thread(speak, "Sorry, the AI service is out of credits. Please check the OpenAI billing.", bot_id)
        elif "401" in error_str or "invalid_api_key" in error_str:
            await asyncio.to_thread(speak, "Sorry, the AI service API key is invalid.", bot_id)
        else:
            await asyncio.to_thread(speak, "Sorry, I ran into an error. Please try again.", bot_id)

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
    logger.info("WebSocket connected")

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

            logger.info(f"{participant}: {sentence}")

            # Append to meeting log
            await state.add_transcript(participant, sentence, time.time())

            bot_id = await state.get_bot_id()
            if not bot_id:
                continue

            query = extract_wake_and_query(sentence)

            if query is not None:
                # Wake word detected in this chunk
                if query:
                    # Full query in same sentence: "Hey Jarvis, what's the weather in Paris?"
                    await state.set_listening(False)
                    asyncio.create_task(handle_query(query, bot_id))
                else:
                    # Bare wake word: "Hey Jarvis" -- acknowledge and wait for next chunk
                    await state.set_listening(True)
                    asyncio.create_task(asyncio.to_thread(speak, "Yes?", bot_id))

            elif await state.is_listening():
                # Previous chunk was just the wake word; this chunk is the query
                await state.set_listening(False)
                asyncio.create_task(handle_query(sentence, bot_id))

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected — triggering ingestion pipeline")
        transcript = await state.get_transcript()
        if transcript and transcript != "[No transcript yet]":
            meeting_id = await state.get_bot_id() or f"mtg-{int(time.time())}"
            meeting_meta = {
                "meeting_id": meeting_id,
                "channel_id": SLACK_CHANNEL_ID,
                "channel_name": MEETING_CHANNEL_NAME,
                "start_ts": int(time.time()) - 3600,  # approximate; real start_ts from state if available
                "end_ts": int(time.time()),
                "duration_seconds": None,
                "participants": [],  # SummarizerAgent extracts participants from transcript
            }
            asyncio.create_task(run_ingestion_pipeline(transcript, meeting_meta))
        else:
            logger.info("No transcript to summarize on disconnect")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")


@app.get("/health")
async def health():
    return await state.get_health_snapshot()


@app.post("/slack/events")
async def slack_events(req: Request):
    """Route all Slack events (slash commands, actions) through the Bolt handler."""
    if slack_handler is None:
        return {"error": "Slack not configured"}
    return await slack_handler.handle(req)

# ============================================================================
# ENTRY POINT
# ============================================================================

def _start_server():
    uvicorn.run(app, host=APP_HOST, port=APP_PORT, log_level="warning")


def _sync_set_state(coro):
    """Run an async state operation from synchronous main()."""
    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(coro)
    finally:
        loop.close()


def main():
    print("=" * 60)
    print(f"{BOT_NAME} Meeting Assistant")
    print(f'   Wake word: "Hey {BOT_NAME}" or "{BOT_NAME}"')
    print("=" * 60)

    missing = [k for k in ("RECALL_API_KEY", "OPENAI_API_KEY", "WEBHOOK_URL") if not os.getenv(k)]
    if missing:
        for k in missing:
            print(f"{k} not set in .env")
        sys.exit(1)

    # Start WebSocket server
    Thread(target=_start_server, daemon=True).start()
    print(f"Listening on {APP_HOST}:{APP_PORT}")
    time.sleep(2)

    # Resolve meeting URL
    meeting_url = MEETING_URL or input("\nMeeting URL (Google Meet / Zoom / Teams): ").strip()
    if not meeting_url:
        print("No meeting URL provided")
        sys.exit(1)

    # Spawn bot
    print(f"Joining: {meeting_url}")
    bot_id = create_bot(meeting_url)
    if not bot_id:
        print("Failed to spawn bot")
        sys.exit(1)

    _sync_set_state(state.set_bot_id(bot_id))
    _sync_set_state(state.set_active(True))

    with open("bot_id.txt", "w") as f:
        f.write(bot_id)

    print(f"Bot joined! ID: {bot_id}")
    print(f'Listening for "Hey {BOT_NAME}"...')
    print("   Ctrl+C to stop\n")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print(f"\n{BOT_NAME} shutting down.")
        _sync_set_state(state.set_active(False))


if __name__ == "__main__":
    main()
