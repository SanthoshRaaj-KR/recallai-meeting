# External Integrations

**Analysis Date:** 2026-04-10

## APIs & External Services

**Atlassian Confluence:**
- Service: Enterprise document management and collaboration
- What it's used for: Primary document system for page creation, editing, deletion, searching, and metadata management
  - SDK/Client: `confluence_logic/connectors/confluence.py` (custom HTTP wrapper using `requests` library)
  - Auth: HTTPBasicAuth with `ATLASSIAN_USER_EMAIL` and `ATLASSIAN_API_TOKEN`
  - Endpoint: `https://{ATLASSIAN_DOMAIN}/wiki/rest/api`
  - Key API calls:
    - `GET /content/{page_id}` - Fetch metadata and page content (HTML storage format)
    - `GET /content/search` - CQL-based page search with title/text matching
    - `PUT /content/{page_id}` - Update page with version locking
    - `POST /content` - Create new page with optional parent/space hierarchy

**OpenAI API:**
- Service: Large language model inference and embeddings
- What it's used for: Agent reasoning, document editing decisions, embedding generation for semantic search, and text-to-speech
  - SDK/Client: `openai` Python SDK (OpenAI() client)
  - Auth: `OPENAI_API_KEY` environment variable
  - Models used:
    - `gpt-4o-mini` (or configured `OPENAI_MODEL`) - Agent reasoning in editor/reframer agents
    - `gpt-5-mini` (or `JARVIS_AGENT_MODEL`) - Agentic loop execution
    - `text-embedding-3-small` (or `OPENAI_EMBEDDING_MODEL`) - Document chunk embeddings
    - `gpt-4o-mini-tts` (or configured TTS model) - Text-to-speech synthesis
  - Usage in code:
    - `confluence_logic/agents/editor_agent.py` - Agent initialization with model parameter
    - `confluence_logic/agents/reframer_agent.py` - Resolver agent for ambiguous requests
    - `confluence_logic/db/vector_store.py` - Embedding generation via `client.embeddings.create()`
    - `jarvis.py` - TTS and chat completions for meeting bot responses

**Recall.ai:**
- Service: Meeting bot platform for joining video conferences and capturing transcripts
- What it's used for: Meeting recording, bot joining, audio streaming, and real-time transcript delivery
  - SDK/Client: Custom HTTP integration (no SDK, direct API calls in `jarvis.py`)
  - Auth: `RECALL_API_KEY` header authentication
  - Base URL: `https://{RECALL_API_REGION}.recall.ai/api/v1` (default region: `ap-northeast-1`)
  - Key endpoints:
    - `POST /bot` - Create bot instance to join meeting
    - `GET /bot/{bot_id}` - Poll bot status
    - `DELETE /bot/{bot_id}` - Terminate bot
    - WebSocket: `wss://{WEBHOOK_URL}/recall-audio-stream` - Receive real-time audio/transcripts
  - Configuration:
    - `RECALL_API_REGION` - AWS region (default: `ap-northeast-1`)
    - `MEETING_URL` - Meeting link (e.g., Google Meet, Zoom)
    - `BOT_NAME` - Name displayed in meeting (e.g., "Jarvis")
    - `WEBHOOK_URL` - Public URL for receiving streaming data (requires ngrok or tunnel)

## Data Storage

**Databases:**
- Pinecone (Vector Database):
  - Connection: `pinecone-client` library, initialized with `PINECONE_API_KEY`
  - Index: `PINECONE_INDEX_NAME` (default: `confluence-kb`)
  - Dimensions: `PINECONE_INDEX_DIMENSION` (must match embedding dimension, typically 1024 for `text-embedding-3-small`)
  - Client: Custom `PineconeStore` class in `confluence_logic/db/vector_store.py`
  - Operations:
    - `upsert()` - Store document chunks with metadata and embeddings
    - `query()` - Semantic search for relevant content using embedding similarity
    - `fetch()` - Retrieve specific vectors by page_id
    - `delete()` - Remove stale sections after updates

**File Storage:**
- Local filesystem only
  - Temporary files: `tempfile` for HTML-to-Markdown conversion in `confluence_logic/ingestion/doc_pipeline.py`
  - Meeting transcripts: JSON stored in `meetings/` directory
  - Metadata: `bot_id.txt` for Recall.ai bot tracking

**Caching:**
- Pinecone vector cache (de facto caching via embeddings)
- Version tracking: Document version numbers cached in Pinecone metadata to avoid re-indexing stale pages

## Authentication & Identity

**Auth Providers:**
- Atlassian: Custom HTTP Basic Auth with email + API token (no external provider)
- OpenAI: API key-based authentication
- Recall.ai: API key header authentication
- Pinecone: API key-based authentication

**Implementation:**
- `ATLASSIAN_USER_EMAIL` + `ATLASSIAN_API_TOKEN` → HTTPBasicAuth in `confluence_logic/connectors/confluence.py`
- Environment variables validated at startup in:
  - `ConfluenceConnector.__init__()` - Raises ValueError if missing
  - `PineconeStore.__init__()` - Graceful degradation if missing (vector search disabled)
- No session/JWT tokens; all authentication is credential-based per request

## Monitoring & Observability

**Error Tracking:**
- None (no Sentry/Rollbar integration detected)
- Errors logged via Python `logging` module to console/file

**Logs:**
- Standard Python logging with format: `"%(asctime)s - %(levelname)s - %(message)s"`
- Configured in:
  - `jarvis.py` - `logging.basicConfig()`
  - All modules use `logger = logging.getLogger(__name__)`
- Log levels: DEBUG, INFO, WARNING, ERROR used throughout
- Key logged events:
  - Confluence API errors in `confluence_logic/connectors/confluence.py`
  - Pinecone upsert/query failures in `confluence_logic/db/vector_store.py`
  - Tool execution state in `confluence_logic/agents/tools.py`

## CI/CD & Deployment

**Hosting:**
- Recall.ai - Meeting bot platform (API-driven)
- Custom deployment (no explicit platform specified) - Requires Python 3.13 runtime
- ngrok or similar tunnel for local development/testing (WEBHOOK_URL must be public HTTPS)

**CI Pipeline:**
- None detected - No GitHub Actions, GitLab CI, or similar configuration

## Environment Configuration

**Required env vars (from `.env.example`):**

Atlassian Confluence:
- `ATLASSIAN_USER_EMAIL` - Email for Confluence API authentication
- `ATLASSIAN_API_TOKEN` - API token for Confluence (from Atlassian account settings)
- `ATLASSIAN_DOMAIN` - Confluence instance domain (e.g., `company.atlassian.net`)
- `ATLASSIAN_SPACE_KEY` - Default space for page creation (optional override per request)

OpenAI:
- `OPENAI_API_KEY` - API key from OpenAI account
- `OPENAI_MODEL` - Chat model (default: `gpt-4o-mini`)
- `OPENAI_EMBEDDING_MODEL` - Embedding model (default: `text-embedding-3-small`)
- `OPENAI_EMBEDDING_DIMENSIONS` - Embedding dimensions (default: `1024`)
- `JARVIS_AGENT_MODEL` - Agentic model (default: `gpt-5-mini`)
- `JARVIS_TTS_PROVIDER` - TTS provider (default: `openai`)
- `JARVIS_TTS_MODEL` - TTS model (default: `gpt-4o-mini-tts`)
- `JARVIS_TTS_VOICE` - Voice option (default: `alloy`)
- `JARVIS_TTS_SPEED` - Speech speed (default: `1.0`)

Pinecone:
- `PINECONE_API_KEY` - API key for vector database
- `PINECONE_INDEX_NAME` - Vector index name (default: `confluence-kb`)
- `PINECONE_INDEX_DIMENSION` - Index dimension (must match embeddings, default: `1024`)

Recall.ai:
- `RECALL_API_KEY` - API key for meeting bot
- `RECALL_API_REGION` - AWS region (default: `ap-northeast-1`)
- `RECALL_WEBHOOK_SECRET` - Webhook signature verification (if used)
- `MEETING_URL` - Meeting link to join
- `BOT_NAME` - Display name in meeting
- `STREAMING_MODE` - Audio streaming priority (default: `prioritize_low_latency`)
- `LANGUAGE_CODE` - Language for transcription (default: `en`)
- `ASSEMBLY_API` - AssemblyAI key (optional, for BYOB transcription via Recall)

Server:
- `APP_HOST` - FastAPI host (default: `0.0.0.0`)
- `APP_PORT` - FastAPI port (default: `8000`)
- `OUTPUT_MEDIA_BASE_URL` - Base URL for media files (ngrok URL)
- `WEBHOOK_URL` - Public webhook URL for Recall.ai callbacks

**Secrets location:**
- `.env` file (committed as `.gitignore`-d to prevent accidental leaks)
- Runtime: Environment variables read via `load_dotenv()` in startup code

## Webhooks & Callbacks

**Incoming:**
- `POST /recall-audio-stream` (WebSocket) - Recall.ai sends real-time meeting transcripts and audio packets
  - Handler: `@app.websocket("/recall-audio-stream")` in `jarvis.py` and `confluence_logic/jarvis_agentic.py`
  - Payload format: JSON frames containing participant transcript, audio chunks, metadata
  - Processing: Real-time keyword detection ("Hey Jarvis"), speech-to-text, agent response

**Outgoing:**
- Recall.ai endpoints called directly (no webhooks):
  - `POST /bot` - Create bot
  - `GET /bot/{bot_id}` - Poll status
  - `DELETE /bot/{bot_id}` - Terminate bot
- OpenAI endpoints called directly (streaming completions)
- Confluence API endpoints called directly (read/write operations)
- Pinecone endpoints called directly (upsert/query operations)

## Integration Data Flow

**Document Editing Workflow:**
1. User voice command via meeting → Jarvis WebSocket receives transcript
2. ReframerAgent (via tools) → searches live Confluence and lists recent pages
3. EditorAgent (with search/fetch/preview/commit tools) → reads live page, previews DOM changes, commits update
4. IngestionPipeline → automatically re-indexes updated page to Pinecone
5. Pinecone vector store now has updated embeddings for future searches

**Knowledge Search Workflow:**
1. User asks question via Jarvis → EditorAgent tools trigger search
2. `search_workspace_knowledge()` → live CQL search + Pinecone semantic search
3. Results deduplicated and scored by title similarity + Pinecone relevance
4. Top 5 candidates returned to agent for context

**Document Ingestion Workflow:**
1. Page created/updated via Confluence API
2. `IngestionPipeline.process_page()` triggered manually or post-commit
3. Fetches HTML from Confluence → Docling converts to Markdown → splits by headings
4. OpenAI embeddings generated for each section
5. Upserted to Pinecone with metadata (page_id, version, heading, title)
6. Stale sections (higher indices) deleted to keep index clean

---

*Integration audit: 2026-04-10*
