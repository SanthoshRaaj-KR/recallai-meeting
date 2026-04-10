# Technology Stack

**Analysis Date:** 2026-04-10

## Languages

**Primary:**
- Python 3.13.9 - Core application language, used throughout `confluence_logic/`, meeting integration, and agents

**Secondary:**
- HTML/CSS - Confluence storage format for document editing
- Markdown - Generated from HTML during document ingestion pipeline

## Runtime

**Environment:**
- Python 3.13.9

**Package Manager:**
- pip - Manages Python dependencies
- Lockfile: requirements.txt (present, pinned versions)

## Frameworks

**Core:**
- FastAPI - Web framework for webhook handling and WebSocket connections in `jarvis.py` and `confluence_logic/jarvis_agentic.py`
- Uvicorn - ASGI server for FastAPI applications

**Agent Framework:**
- openai-agents - Custom OpenAI SDK for agentic workflows, used in `confluence_logic/agents/editor_agent.py`, `reframer_agent.py`, tools decorators

**Testing:**
- pytest - Test runner with async support for agent workflows

**Document Processing:**
- docling - Document converter for HTML to Markdown transformation in `confluence_logic/ingestion/doc_pipeline.py`
- BeautifulSoup4 (beautifulsoup4) - HTML parsing and DOM manipulation for section editing in `confluence_logic/utils/html_parser.py`

**Build/Dev:**
- python-dotenv - Environment variable loading from `.env` files

## Key Dependencies

**Critical:**
- openai-agents - Provides `Agent`, `Runner`, and `@function_tool` decorator for building agentic Confluence editing workflows
- requests - HTTP client for Atlassian Confluence API and Recall.ai API calls
- pinecone-client - Vector database client for semantic search over Confluence page content
- gTTS (Google Text-to-Speech) - Text-to-speech synthesis for bot voice responses

**Infrastructure:**
- fastapi - Request/response handling, WebSocket support for real-time meeting transcripts
- uvicorn - Production ASGI server
- websockets - WebSocket protocol support for streaming meeting audio
- pyaudio - Audio capture from system/microphone (referenced in requirements)
- ngrok - Tunneling utility for exposing local webhooks to Recall.ai (configured via WEBHOOK_URL)

**External APIs:**
- openai - OpenAI API client for GPT models, embeddings, and TTS
- requests - Confluence REST API, Recall.ai REST API

## Configuration

**Environment:**
- Loaded from `.env` file via python-dotenv
- Environment variables required for:
  - Atlassian Confluence: `ATLASSIAN_USER_EMAIL`, `ATLASSIAN_API_TOKEN`, `ATLASSIAN_DOMAIN`, `ATLASSIAN_SPACE_KEY`
  - OpenAI: `OPENAI_API_KEY`, `OPENAI_MODEL`, `OPENAI_EMBEDDING_MODEL`, `OPENAI_EMBEDDING_DIMENSIONS`
  - Pinecone: `PINECONE_API_KEY`, `PINECONE_INDEX_NAME`, `PINECONE_INDEX_DIMENSION`
  - Recall.ai: `RECALL_API_KEY`, `RECALL_API_REGION`, `WEBHOOK_URL`, `MEETING_URL`
  - Server: `APP_HOST`, `APP_PORT`

**Build:**
- requirements.txt: Pinned dependency versions
- Default models:
  - `OPENAI_MODEL=gpt-4o-mini` (configurable via env)
  - `JARVIS_AGENT_MODEL=gpt-5-mini` (in agents)
  - `OPENAI_EMBEDDING_MODEL=text-embedding-3-small`

## Platform Requirements

**Development:**
- Python 3.13.9
- Local ngrok/tunnel for webhook development (Recall.ai requires public HTTPS URL)
- System audio support (pyaudio) for microphone input

**Production:**
- HTTP(S) server capable of serving FastAPI applications
- Persistent vector database access (Pinecone SaaS)
- Atlassian Cloud Confluence workspace with API token auth
- Recall.ai account with meeting URLs

## Module Structure

**confluence_logic** - Main Python module containing:
- `connectors/confluence.py` - Atlassian Confluence REST API client wrapper
- `db/vector_store.py` - Pinecone vector database integration
- `agents/` - OpenAI agent framework implementations (editor, reframer, tools)
- `ingestion/doc_pipeline.py` - Document processing pipeline (HTML → Markdown → chunks → Pinecone)
- `core/` - Pydantic models and interfaces
- `utils/` - HTML parsing and DOM manipulation

**Root entry points:**
- `jarvis.py` - Meeting bot with Recall.ai integration, WebSocket streaming, wake-word detection
- `confluence_logic/jarvis_agentic.py` - FastAPI service for agentic Confluence editing workflows
- `confluence_logic/local_repl.py` - Local REPL for testing agent workflows

---

*Stack analysis: 2026-04-10*
