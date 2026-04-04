# External Integrations

**Analysis Date:** 2026-04-04

## APIs & External Services

**Meeting Recording & Streaming:**
- Recall.ai - Meeting bot and transcript streaming service
  - SDK/Client: `requests` (HTTP REST client)
  - Auth: `RECALL_API_KEY` (token-based auth in Authorization header)
  - Endpoints:
    - `POST /{RECALL_API_REGION}.recall.ai/api/v1/bot/` - Create bot instance to join meeting
    - `POST /{RECALL_API_REGION}.recall.ai/api/v1/bot/{bot_id}/output_audio/` - Send audio output to meeting
  - Real-time: WebSocket callbacks receive `transcript.data` events with streaming transcript

**AI/LLM:**
- OpenAI - GPT language model for query responses and tool calling
  - SDK/Client: `openai` Python SDK (Official OpenAI client)
  - Auth: `OPENAI_API_KEY` (environment variable, used by SDK)
  - Model: `OPENAI_MODEL` (configurable, default: gpt-4o-mini)
  - Features: Function calling (tools), system prompts, message roles
  - Located in: `jarvis.py` lines 30, 55, 218

**Text-to-Speech:**
- Google Translate/gTTS API - Text-to-speech conversion
  - SDK/Client: `gTTS` Python library (unofficial Google API wrapper)
  - Auth: None required (public service)
  - Language: `LANGUAGE_CODE` (configurable, default: en)
  - Output format: MP3 (base64 encoded to Recall.ai)
  - Located in: `jarvis.py` lines 29, 117-138

**Weather Data:**
- wttr.in - Weather API (no authentication)
  - SDK/Client: `requests` HTTP client
  - Auth: None required (public service)
  - Endpoint: `https://wttr.in/{city}?format=3`
  - Used by: Tool `get_weather` for meeting queries
  - Located in: `jarvis.py` lines 144-150

## Data Storage

**Databases:**
- None - In-memory state only

**File Storage:**
- Local filesystem only
  - Temporary: MP3 audio files created during speech synthesis (cleaned up after use)
  - Persistent: `bot_id.txt` stores current bot ID for reference

**Caching:**
- None - In-memory transcript log only (session-based, not persisted)

## Authentication & Identity

**Auth Provider:**
- Custom token-based auth
  - Recall.ai: Token in `Authorization: Token {RECALL_API_KEY}` header
  - OpenAI: Automatic via `openai` SDK (reads `OPENAI_API_KEY` environment variable)
  - Other services: None (public APIs)

## Monitoring & Observability

**Error Tracking:**
- None (no external error tracking service)
- Local logging via Python `logging` module (stdout/stderr)

**Logs:**
- Python `logging` module with INFO level
- Format: `%(asctime)s - %(levelname)s - %(message)s`
- Output: Console (stdout)
- Located in: `jarvis.py` lines 52-53

## CI/CD & Deployment

**Hosting:**
- Local development or custom deployment (no built-in hosting)
- Requires ngrok or similar tunnel for public webhook URL during development

**CI Pipeline:**
- None detected

**Network Tunnel:**
- ngrok - Exposes local HTTP server to public internet for Recall.ai webhooks
  - Auth: `ngrok` library installed but no direct SDK usage (CLI-based)
  - Usage: Run `ngrok http 8000`, paste URL in `WEBHOOK_URL` env var

## Environment Configuration

**Required env vars:**
- `RECALL_API_KEY` - Mandatory for bot creation and control
- `OPENAI_API_KEY` - Mandatory for AI responses
- `WEBHOOK_URL` - Mandatory for receiving Recall.ai transcript events

**Optional env vars:**
- `RECALL_API_REGION` - Recall.ai region endpoint (default: ap-northeast-1)
- `MEETING_URL` - Meeting to join (prompted at runtime if not set)
- `BOT_NAME` - Bot display name (default: Jarvis)
- `OPENAI_MODEL` - LLM to use (default: gpt-4o-mini)
- `STREAMING_MODE` - Recall.ai transcription mode (default: prioritize_low_latency)
- `LANGUAGE_CODE` - TTS/transcription language (default: en)
- `APP_HOST` - Server bind address (default: 0.0.0.0)
- `APP_PORT` - Server port (default: 8000)

**Secrets location:**
- `.env` file in project root (git-ignored)
- Example template: `.env.example`

## Webhooks & Callbacks

**Incoming:**
- WebSocket endpoint: `wss://{WEBHOOK_URL}/recall-audio-stream`
  - Source: Recall.ai service
  - Events: `transcript.data` (real-time meeting transcript chunks)
  - Handler: `websocket_endpoint()` function in `jarvis.py` lines 283-341

**Outgoing:**
- None - Application does not send outbound webhooks
- Direct API calls to Recall.ai for bot control (HTTP REST, not webhooks)

## API Call Patterns

**Recall.ai Bot Creation:**
- HTTP POST to create bot with meeting URL and WebSocket callback
- Includes recording config with realtime transcript endpoint
- Located in: `jarvis.py` lines 73-114

**Speech Output:**
- HTTP POST with base64-encoded MP3 audio to bot's output_audio endpoint
- Located in: `jarvis.py` lines 117-138

**OpenAI Function Calling Loop:**
- Chat completions API with `tools` parameter for agentic behavior
- Iterates up to 5 times for multi-turn tool calling (weather, meeting summary)
- Located in: `jarvis.py` lines 215-246

---

*Integration audit: 2026-04-04*
