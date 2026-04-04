# Technology Stack

**Analysis Date:** 2026-04-04

## Languages

**Primary:**
- Python 3.x - Core application backend

## Runtime

**Environment:**
- Python 3.x (standard CPython environment)

**Package Manager:**
- pip
- Lockfile: Not detected (requirements.txt used for dependency pinning)

## Frameworks

**Core:**
- FastAPI - Web framework for HTTP/WebSocket server
- Uvicorn - ASGI server for running FastAPI application

**API & Communication:**
- OpenAI Python SDK - Integration with OpenAI GPT models for AI responses
- requests - HTTP client for making REST API calls to Recall.ai and external services

**Audio & TTS:**
- gTTS (Google Text-to-Speech) - Text-to-speech conversion for bot responses
- pyaudio - Audio playback/recording capabilities

**Real-time Communication:**
- websockets - WebSocket support built into FastAPI for streaming transcript data

## Key Dependencies

**Critical:**
- `openai` - AI model inference (GPT-4o-mini or configured model); enables agentic tool calling
- `fastapi` - Web framework; handles HTTP endpoints and WebSocket connections for meeting integration
- `uvicorn` - ASGI server; serves the FastAPI application on configurable host/port
- `gTTS` - Text-to-speech conversion; speaks bot responses back into meeting
- `requests` - HTTP client; communicates with Recall.ai REST API for bot lifecycle management

**Infrastructure:**
- `python-dotenv` - Environment variable loading from .env files
- `ngrok` - Network tunneling; exposes local server to public internet for webhook callbacks
- `pyaudio` - Audio system integration
- `websockets` - WebSocket protocol support (used by FastAPI)
- `openai-agents` - Agent framework for tool-calling workflows (installed but not directly imported in main code)

## Configuration

**Environment:**
- Environment variables loaded from `.env` file at runtime via `python-dotenv`
- Key configurations:
  - `RECALL_API_KEY` - API authentication for Recall.ai service
  - `RECALL_API_REGION` - Recall.ai regional endpoint (default: ap-northeast-1)
  - `OPENAI_API_KEY` - API key for OpenAI service
  - `WEBHOOK_URL` - Public URL (typically ngrok) for receiving webhook callbacks
  - `MEETING_URL` - Target meeting URL (Google Meet, Zoom, Teams)
  - `OPENAI_MODEL` - LLM model selection (default: gpt-4o-mini)
  - `STREAMING_MODE` - Recall.ai transcript streaming mode (default: prioritize_low_latency)
  - `LANGUAGE_CODE` - Language for TTS and transcription (default: en)
  - `BOT_NAME` - Name of the bot (default: Jarvis)
  - `APP_HOST` - Server bind address (default: 0.0.0.0)
  - `APP_PORT` - Server port (default: 8000)

**Build:**
- `requirements.txt` - Static dependency list at `./requirements.txt`

## Platform Requirements

**Development:**
- Python 3.x installed
- Audio hardware (microphone/speaker or passthrough to meeting platform)
- Network access to:
  - OpenAI API (api.openai.com)
  - Recall.ai API (regional endpoint, e.g., ap-northeast-1.recall.ai)
  - Google Translate API (for gTTS)
  - wttr.in (for weather tool)
- ngrok or similar tunnel service for public webhook URL

**Production:**
- Python 3.x runtime environment
- Network connectivity to OpenAI, Recall.ai, and meeting platforms
- Public IP or tunnel service (ngrok) for receiving webhooks from Recall.ai
- Audio capabilities (real or virtual audio devices)

---

*Stack analysis: 2026-04-04*
