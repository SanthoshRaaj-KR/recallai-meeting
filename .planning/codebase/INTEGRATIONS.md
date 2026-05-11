# External Integrations

**Analysis Date:** 2026-05-11

## APIs & Services

**Recall.ai (Meeting Bot Platform):**
- Purpose: Joins video meetings (Google Meet, Zoom, etc.) as a bot, delivers real-time transcription events via WebSocket, and receives audio playback
- SDK/Client: `requests` (direct REST calls)
- Base URL: `https://{RECALL_API_REGION}.recall.ai/api/v1` (region configurable, default `ap-northeast-1`)
- Auth env var: `RECALL_API_KEY`
- Webhook env var: `WEBHOOK_URL` — Recall posts transcript events here
- Interaction files: `confluence_logic/jarvis_agentic.py`, `local_office_logic/jarvis_agentic.py`, `confluence_logic/review/api.py`
- Key operations: bot creation, bot deletion, audio streaming endpoint registration, WebSocket transcript ingestion, bot status polling

**Atlassian Confluence (Document Source & Target):**
- Purpose: Fetch, search, update, and create Confluence pages (knowledge base for RAG and meeting action output)
- SDK/Client: `requests` with `HTTPBasicAuth`
- Base URL: `https://{ATLASSIAN_DOMAIN}/wiki/rest/api`
- Auth env vars: `ATLASSIAN_USER_EMAIL`, `ATLASSIAN_API_TOKEN`, `ATLASSIAN_DOMAIN`
- Space env var: `ATLASSIAN_SPACE_KEY`
- Interaction file: `confluence_logic/connectors/confluence.py`
- Key operations: fetch page HTML (storage format), search pages via CQL, list pages, push updates (versioned), create pages

**OpenAI (LLM, Embeddings, TTS):**
- Purpose: Chat completions for meeting Q&A / summarization / agent orchestration; text embeddings for vector search; TTS for bot speech
- SDK/Client: `openai` Python SDK (`openai.OpenAI`, `openai.agents.Agent/Runner`)
- Auth env var: `OPENAI_API_KEY`
- Models (configurable):
  - Agent/editor: `JARVIS_AGENT_MODEL` (default `gpt-5-mini` — **non-existent model**)
  - Review/proposals: `JARVIS_REVIEW_MODEL` (default `gpt-5-mini` — **non-existent model**)
  - General responses: `JARVIS_GENERAL_MODEL` (default `gpt-4o-mini`)
  - TTS: `JARVIS_TTS_MODEL` (default `tts-1`), voice `JARVIS_TTS_VOICE` (default `echo`)
  - Embeddings: `OPENAI_EMBEDDING_MODEL` (default `text-embedding-3-small`), dims `OPENAI_EMBEDDING_DIMENSIONS`
- Interaction files: `confluence_logic/meeting_responder.py`, `confluence_logic/general_responder.py`, `confluence_logic/db/vector_store.py`, `confluence_logic/graph_rag.py`, `confluence_logic/jarvis_agentic.py`, `confluence_logic/review/api.py`, `confluence_logic/agents/proposed_changes_agent.py`

**AssemblyAI (Alternative Transcription):**
- Purpose: BYOB (Bring Your Own Bot) transcription provider passed through Recall.ai — an alternative to Recall's native streaming transcription
- Auth env var: `ASSEMBLY_API` (key registered in Recall's transcription credentials dashboard, not sent directly by the app)
- Activation: `RECALL_TRANSCRIPT_PROVIDER=assembly_ai_v3` or `assembly_ai_v3_streaming`
- Interaction files: `confluence_logic/jarvis_agentic.py`, `local_office_logic/jarvis_agentic.py`

**Tavily (Web Search — optional):**
- Purpose: Real-time web search to answer questions requiring live data (weather, stock prices, current events)
- SDK/Client: direct REST call to `https://api.tavily.com/search`
- Auth env var: `TAVILY_API_KEY` (optional — feature gracefully disabled when absent)
- Interaction file: `confluence_logic/general_responder.py` — `_quick_web_search()`
- Routing: `_needs_web_search()` uses GPT to classify whether a question needs live data; only then calls Tavily

## Text-to-Speech Providers

**Edge TTS (Microsoft, primary default for confluence_logic):**
- Purpose: Low-latency free TTS for bot speech output
- Library: `edge-tts` (Python)
- Activation: `JARVIS_TTS_PROVIDER=edge_tts`
- Falls back to OpenAI TTS if not installed

**Google TTS (gTTS):**
- Purpose: TTS fallback imported directly in `jarvis_agentic.py`
- Library: `gTTS`
- Used for gap-filler and cached audio generation (`local_office_logic/scripts/generate_wav_assets.py`, `confluence_logic/scripts/generate_wav_assets.py`)

**OpenAI TTS (primary default for local_office_logic):**
- Purpose: High-quality TTS synthesis for bot speech
- Activation: `JARVIS_TTS_PROVIDER=openai`
- Model: `JARVIS_TTS_MODEL` (default `tts-1`)

## Databases & Storage

**Pinecone (Vector Database):**
- Purpose: Stores OpenAI embeddings of Confluence page sections for semantic search (RAG)
- SDK/Client: `pinecone` Python SDK (`pinecone.Pinecone`)
- Auth env var: `PINECONE_API_KEY`
- Index env vars: `PINECONE_INDEX_NAME` (default `confluence-kb`), `PINECONE_INDEX_DIMENSION`
- Interaction files: `confluence_logic/db/vector_store.py`, `local_office_logic/db/vector_store.py`
- Key operations: upsert embeddings, vector similarity search, fetch by ID, delete stale vectors

**Neo4j AuraDB (Graph Database):**
- Purpose: Knowledge graph over meeting transcript entities (Topics, Persons, Decisions) and Confluence page relationships for Graph RAG
- SDK/Client: `neo4j==6.1.0` async driver (`neo4j.AsyncGraphDatabase`) — **NOTE: version 6.1.0 does not exist on PyPI**
- Auth env vars: `NEO4J_URI`, `NEO4J_USER` / `NEO4J_USERNAME`, `NEO4J_PASSWORD`
- Interaction files: `confluence_logic/graph_rag.py`, `confluence_logic/confluence_page_graph.py`
- Transcript graph node types: `Topic`, `Person`, `Decision`
- Transcript graph edge types: `MENTIONED_BY`, `RELATED_TO`, `DECIDED_IN`
- Confluence page graph: separate labels, scoped by `user_id` + `graph_kind="confluence_pages"`; managed in `confluence_logic/confluence_page_graph.py`
- In-memory fallback: module-level `_local_nodes` / `_local_edges` in `graph_rag.py` used when Neo4j is unavailable

**Supabase (Auth + Postgres REST):**
- Purpose: User authentication and persistent meeting history storage (`meeting_history` table)
- SDK/Client (backend): `requests` calling Supabase REST and Auth APIs directly (no supabase-py SDK)
- SDK/Client (frontend): `@supabase/supabase-js` v2 in `sync-sage-bot/src/lib/supabase.ts`
- Auth env vars: `SUPABASE_URL`, `SUPABASE_ANON_KEY`, `SUPABASE_SERVICE_ROLE_KEY`
- Frontend env vars: `VITE_SUPABASE_URL`, `VITE_SUPABASE_ANON_KEY`
- Table env var: `SUPABASE_HISTORY_TABLE` (default `meeting_history`)
- Interaction files: `confluence_logic/review/supabase_store.py`, `sync-sage-bot/src/lib/supabase.ts`, `sync-sage-bot/src/lib/api.ts`
- Key operations: bearer-token user lookup (`user_from_bearer`), upsert meeting history rows, list history, fetch single session
- Schema file: `confluence_logic/review/supabase_schema.sql` — defines `public.meeting_history` table with RLS policies

**`meeting_history` Table Schema (Supabase Postgres):**

| Column | Type | Notes |
|--------|------|-------|
| `id` | uuid PK | auto-generated |
| `user_id` | uuid | foreign key to auth.users |
| `session_id` | text UNIQUE | Recall.ai bot session identifier |
| `title` | text | meeting title |
| `meeting_url` | text | original meeting URL |
| `status` | text | `in_meeting`, `ended`, etc. |
| `started_at` / `ended_at` | timestamptz | meeting time bounds |
| `summary` | text | plain text summary |
| `summary_json` | jsonb | structured summary (topics, decisions, action items) |
| `transcript_compressed` | text | gzip+base64 encoded transcript |
| `transcript_codec` | text | compression codec used |
| `transcript_entry_count` | integer | number of transcript entries |
| `change_count` | integer | number of proposed Confluence changes |
| `stats` | jsonb | arbitrary session stats |

Row-level security is enabled — users can only read/insert/update their own rows.

**Local Filesystem (Local Office logic):**
- Purpose: Sandboxed read/write of local office documents (DOCX, XLSX, ODS, etc.)
- Root env vars: `LOCAL_OFFICE_SANDBOX_ROOT`, `LOCAL_OFFICE_STAGING_ROOT`
- Allowed extensions: `LOCAL_OFFICE_ALLOWED_EXTENSIONS`
- Interaction files: `local_office_logic/utils/sandbox.py`, `local_office_logic/utils/document_adapter.py`, `local_office_logic/utils/spreadsheet_adapter.py`

## Authentication

**Supabase Auth:**
- The backend validates user identity by calling `{SUPABASE_URL}/auth/v1/user` with the bearer token from the `Authorization` header
- Implementation: `confluence_logic/review/supabase_store.py` — `user_from_bearer(token)`
- The sync-sage-bot frontend calls `supabase.auth.getSession()` to retrieve and forward the session token to the backend API
- The review-ui (`review-ui/`) does **not** currently implement auth — it calls the backend directly without a token

## Cloud & Infrastructure

**ngrok (Tunnel):**
- Purpose: Exposes the local FastAPI server to the public internet so Recall.ai can deliver webhook events and audio stream callbacks
- Library: `ngrok` (Python) listed in `requirements.txt`
- Env vars: `WEBHOOK_URL` / `OUTPUT_MEDIA_BASE_URL` (set to the ngrok public URL)
- No docker-compose or cloud deployment config detected in the repo

**LibreOffice Headless:**
- Purpose: Office format conversion (e.g., ODT → DOCX, legacy format round-trips) for local office documents
- Binary env var: `LOCAL_OFFICE_OFFICE_BINARY` (defaults to `soffice` / `libreoffice` on PATH)
- Interaction file: `local_office_logic/utils/office_runtime.py`

## Third-party SDKs

| SDK | Language | Purpose |
|-----|----------|---------|
| `openai` (Python) | Python | LLM, embeddings, TTS |
| `openai-agents` (`agents` import) | Python | Agentic orchestration (Agent/Runner/function_tool) |
| `pinecone` | Python | Vector database client |
| `neo4j` 6.1.0 | Python | Graph database async driver |
| `docling` | Python | HTML → Markdown document conversion |
| `gTTS` | Python | Google Text-to-Speech |
| `edge-tts` | Python | Microsoft Edge TTS |
| `requests` | Python | HTTP client for Recall, Confluence, Supabase, Tavily |
| `@supabase/supabase-js` v2 | TypeScript | Supabase auth + data (sync-sage-bot frontend) |
| `@tanstack/react-query` v5 | TypeScript | Server state management (sync-sage-bot) |
| Radix UI (full suite) | TypeScript | Headless UI components (sync-sage-bot) |

## Webhooks & Callbacks

**Incoming (from Recall.ai):**
- Recall.ai posts real-time transcript events to `WEBHOOK_URL` (must be publicly reachable — exposed via ngrok in dev)
- A WebSocket endpoint `wss://{WEBHOOK_URL}/recall-audio-stream` is registered with Recall for audio streaming
- Handled in: `confluence_logic/jarvis_agentic.py`, `local_office_logic/jarvis_agentic.py`

**Outgoing:**
- Bot REST calls to `{RECALL_BASE_URL}/bot` (create, delete, get status)
- REST calls to Confluence API for page read/write
- REST calls to Supabase for auth validation and history persistence
- REST/streaming calls to OpenAI for completions, embeddings, and TTS
- REST calls to Tavily for web search (when `TAVILY_API_KEY` is set and query needs live data)

---

*Integration audit: 2026-05-11*
