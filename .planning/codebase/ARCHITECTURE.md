<!-- refreshed: 2026-05-11 -->
# Architecture

## System Overview

A meeting intelligence platform with two parallel domain modules plus a standalone review UI:

1. **`confluence_logic/`** — AI assistant integrated with Atlassian Confluence (cloud knowledge base)
2. **`local_office_logic/`** — AI assistant integrated with local Office documents (Word, Excel, LibreOffice)
3. **`review-ui/`** — Next.js 16 frontend for reviewing and approving proposed Confluence changes

Both domain modules share the same layered architecture: a REPL/event loop at the top, a multi-agent orchestrator in the middle, and RAG + external service connectors at the bottom.

```
┌────────────────────────────────────────────────────────────────┐
│              review-ui/ (Next.js 16, App Router)               │
│   page.tsx (home/join)     results/page.tsx (review changes)   │
│   src/lib/api.ts (fetch client)   src/types.ts (TypeScript types)│
└──────────────────────────┬─────────────────────────────────────┘
                           │ /api/* proxy → localhost:8000
┌──────────────────────────▼─────────────────────────────────────┐
│  FastAPI app (confluence_logic/jarvis_agentic.py)              │
│  ┌──────────────────────────────────────────────┐              │
│  │  review/api.py  (APIRouter — /review/*, /bot/*)│             │
│  │  - GET  /review/summary                      │              │
│  │  - GET  /review/changes                      │              │
│  │  - POST /review/execute                      │              │
│  │  - POST /bot/start   GET /bot/status         │              │
│  └──────────────────────────────────────────────┘              │
│                                                                 │
│  jarvis_agentic.py (orchestrator + WebSocket + TTS)            │
│  OpenAI Agents SDK — multi-agent layer                         │
│  ┌──────────────┐  ┌───────────────┐  ┌─────────────────────┐ │
│  │ editor_agent │  │reframer_agent │  │proposed_changes_agt │ │
│  └──────────────┘  └───────────────┘  └─────────────────────┘ │
└──────┬──────────────────────┬──────────────────────────────────┘
       │                      │
┌──────▼──────────┐  ┌────────▼──────────────────────────────────┐
│  graph_rag.py   │  │              Connectors                    │
│  (transcript    │  │  confluence.py / local_office.py           │
│   graph)        │  │  supabase_store.py  (Supabase REST)        │
│                 │  └───────────────────────────────────────────-┘
│  confluence_    │
│  page_graph.py  │
│  (Confluence    │
│   workspace     │
│   graph)        │
│                 │
│  Pinecone       │
│  Neo4j AuraDB   │
└─────────────────┘
```

## Component Responsibilities

| Component | Responsibility | File |
|-----------|----------------|------|
| review-ui home | Meeting URL input, bot join, live change count polling | `review-ui/src/app/page.tsx` |
| review-ui results | Display summary, change list, approve/execute changes | `review-ui/src/app/results/page.tsx` |
| review-ui API client | Typed fetch wrappers for all backend endpoints | `review-ui/src/lib/api.ts` |
| review-ui types | TypeScript interfaces for SessionStatus, ChangeItem, MeetingSummary | `review-ui/src/types.ts` |
| review/api.py | FastAPI APIRouter — bot lifecycle, change management, summary endpoints | `confluence_logic/review/api.py` |
| supabase_store.py | Supabase REST persistence for meeting history and proposals | `confluence_logic/review/supabase_store.py` |
| jarvis_agentic.py | Core orchestrator — wake-word loop, WebSocket, TTS, intent routing | `confluence_logic/jarvis_agentic.py` |
| local_repl.py | Interactive terminal / simulated meeting entry point | `confluence_logic/local_repl.py` |
| classifier.py | Intent classification — routes to general/meeting/web_search/confluence | `confluence_logic/classifier.py` |
| general_responder.py | General Q&A with optional Tavily web search routing | `confluence_logic/general_responder.py` |
| meeting_responder.py | Meeting summarization, opinion generation, action item extraction | `confluence_logic/meeting_responder.py` |
| graph_rag.py | Dual-graph RAG — Pinecone vector search + Neo4j transcript graph | `confluence_logic/graph_rag.py` |
| confluence_page_graph.py | Neo4j graph of Confluence page relationships, scoped by user_id | `confluence_logic/confluence_page_graph.py` |
| audio_cache.py | In-memory cache of pre-generated TTS MP3 acknowledgements | `confluence_logic/audio_cache.py` |
| editor_agent.py | Proposes and applies document edits to Confluence pages | `confluence_logic/agents/editor_agent.py` |
| reframer_agent.py | Rewrites or reframes content sections | `confluence_logic/agents/reframer_agent.py` |
| proposed_changes_agent.py | Generates structured JSON change proposals for human review | `confluence_logic/agents/proposed_changes_agent.py` |
| tools.py | `@function_tool` definitions exposed to OpenAI Agents SDK | `confluence_logic/agents/tools.py` |
| ingestion/doc_pipeline.py | Document chunking, embedding, upsert to Pinecone + Neo4j | `confluence_logic/ingestion/doc_pipeline.py` |

## Pattern Overview

**Overall:** Layered event-driven pipeline with multi-agent orchestration

**Key Characteristics:**
- Wake-word activated async event loop (`local_repl.py` → `jarvis_agentic.py`)
- Intent classification gates routing to specialized handlers or agents
- Dual-graph RAG combines semantic vector search (Pinecone) with relationship traversal (Neo4j)
- Human-in-the-loop review: changes are proposed (JSON), persisted (Supabase), then approved via UI before committing to Confluence
- Two parallel domain packages (`confluence_logic/`, `local_office_logic/`) with identical layering but diverging implementations

## Data Flows

### 1. Live Meeting Wake-Word → TTS Response
```
Microphone → audio_cache → wake-word detection → local_repl
  → jarvis_agentic → classifier → [general_responder | meeting_responder | editor_agent]
  → OpenAI TTS / edge-tts / gTTS → audio output to Recall.ai
```

### 2. General Question with Optional Web Search
```
Wake-word detected question → classifier → "general" or "web_search" intent
  → general_responder._needs_web_search() [GPT router]
  → if yes: Tavily REST API → web context injected into LLM prompt
  → LLM response → TTS → audio output
```

### 3. Post-Meeting Review → Confluence Commit
```
Meeting transcript → ingestion/doc_pipeline → Pinecone + Neo4j
  → User opens review-ui (Next.js) → GET /review/summary, GET /review/changes
  → proposed_changes_agent → structured JSON proposals → stored in Supabase
  → Human selects changes → POST /review/execute
  → editor_agent → Confluence REST API commit
```

### 4. Document Ingestion Pipeline
```
Confluence REST API / LibreOffice local files
  → doc_pipeline → chunking → embeddings (OpenAI)
  → Pinecone (vector store) + Neo4j (graph store)
```

### 5. Review UI Bot Lifecycle
```
User enters meeting URL → POST /bot/start → Recall.ai bot created
  → UI polls GET /bot/status every 5s → tracks change_count
  → Bot leaves meeting → status == "ended" → UI navigates to /results
  → /results loads summary + changes via parallel fetch → user reviews → execute
```

## Layers

**Entry / Interaction:**
- Purpose: Top-level user interfaces — terminal REPL and Next.js web UI
- Location: `confluence_logic/local_repl.py`, `review-ui/src/app/`
- Depends on: orchestrator (jarvis_agentic), API client (api.ts)

**Orchestration:**
- Purpose: Intent routing, agent delegation, TTS, WebSocket/meeting event loop
- Location: `confluence_logic/jarvis_agentic.py`
- Contains: FastAPI app, meeting state machine, wake-word handling

**Review API:**
- Purpose: REST endpoints for the review-ui frontend; bot lifecycle and change management
- Location: `confluence_logic/review/api.py`
- Mounted into: `jarvis_agentic.py` FastAPI app via `app.include_router(router)`

**Classification:**
- Purpose: Determine intent from spoken query to route to correct handler
- Location: `confluence_logic/classifier.py`
- Intents: `general`, `web_search`, `meeting_opinion`, `meeting_summary`, `confluence`, `action_items`

**Response Handlers:**
- Purpose: Specialized handlers for each intent class
- Location: `confluence_logic/general_responder.py`, `confluence_logic/meeting_responder.py`

**Agents:**
- Purpose: Specialized task agents using OpenAI Agents SDK
- Location: `confluence_logic/agents/`
- Contains: editor, reframer, proposed_changes agents + tool definitions

**RAG:**
- Purpose: Retrieval-augmented generation context building
- Location: `confluence_logic/graph_rag.py`, `confluence_logic/confluence_page_graph.py`, `confluence_logic/db/vector_store.py`

**Connectors:**
- Purpose: External service clients with thin wrappers
- Location: `confluence_logic/connectors/confluence.py`

**Persistence:**
- Purpose: Supabase meeting history persistence
- Location: `confluence_logic/review/supabase_store.py`

## Key Abstractions

**`ChangeItem` (TypeScript):**
- Purpose: Represents a single proposed Confluence change
- Definition: `review-ui/src/types.ts`
- Fields: `change_type` (create/edit/delete/title), `page_id`, `section_heading`, `before_content`, `after_content`, `status`

**`SessionStatus` (TypeScript):**
- Purpose: Current bot session state used for UI polling
- Definition: `review-ui/src/types.ts`
- Fields: `status` (idle/joining/in_meeting/ended/error), `bot_id`, `change_count`

**`MeetingStateProxy`:**
- Purpose: Thread-safe meeting state container using Python `contextvars` for session isolation
- Location: `confluence_logic/jarvis_agentic.py`
- Replaces: bare module-level dict used in `local_office_logic`

**`ProposedChangesAgent`:**
- Purpose: Takes meeting context + Confluence page retrieval → outputs structured JSON change proposals
- Location: `confluence_logic/agents/proposed_changes_agent.py`
- Output schema: `{"changes": [{change_type, page_id, page_title, section_heading, before_content, after_content, rationale}]}`

**`confluence_page_graph` module:**
- Purpose: Maintains a per-user Neo4j graph of Confluence page sections indexed by content
- Location: `confluence_logic/confluence_page_graph.py`
- Key env vars: `JARVIS_CONFLUENCE_GRAPH_MAX_PAGES` (500), `JARVIS_CONFLUENCE_GRAPH_TTL_SECONDS` (7200)
- User scoping: `ContextVar("confluence_graph_user_id")` — nodes tagged with `user_id` and `graph_kind="confluence_pages"`

## Entry Points

| Command | Module |
|---------|--------|
| `python -m confluence_logic.local_repl` | Confluence interactive terminal session |
| `python -m local_office_logic.local_repl` | Local Office interactive terminal session |
| `uvicorn confluence_logic.jarvis_agentic:app` | Full FastAPI server (includes review router) |
| `uvicorn confluence_logic.review.api:app` | Review API standalone (not recommended — misses WebSocket handlers) |
| `cd review-ui && npm run dev` | Next.js dev server (proxies /api/* to :8000) |
| `python confluence_logic/scripts/generate_wav_assets.py` | Pre-generate TTS audio assets |
| `python -m tests.e2e_pipeline_eval` | End-to-end pipeline quality evaluation |
| `python -m tests.pipeline_comprehensive_eval` | Comprehensive quality + TTS latency evaluation |

## Architectural Constraints

- **Threading:** Single-threaded asyncio event loop in `jarvis_agentic.py`; CPU-bound tasks delegated via `asyncio.to_thread()`
- **Global state:** Module-level `meeting_state` (a `MeetingStateProxy` in `confluence_logic`, a plain dict in `local_office_logic`); both are shared mutable singletons
- **Circular imports:** Deferred imports inside functions used to break cycles (e.g., `confluence_logic/agents/tools.py` defers `html_builder` import)
- **review-ui proxy:** All `/api/*` calls are rewritten to `http://localhost:8000/*` via `next.config.ts` — the backend must be running for the UI to function
- **No auth on review-ui:** The Next.js frontend has no Supabase auth integration; it calls the backend unauthenticated (bot start is gated only optionally on auth)

## Anti-Patterns

### Module-level singleton clients
**What happens:** OpenAI, Pinecone, Neo4j clients instantiated at import time in `graph_rag.py`, `review/api.py`, `classifier.py`.
**Why it's wrong:** Credentials consumed at import; no lazy initialization. Unit tests must patch before import or face real network calls.
**Do this instead:** Use the lazy singleton pattern already established in `agents/tools.py` — `_store = None; def get_store(): ...`

### Untyped `meeting_state` dict (local_office_logic)
**What happens:** `local_office_logic/jarvis_agentic.py` uses a plain dict for shared session state.
**Why it's wrong:** No type safety, concurrent session bleed possible, no clear contract.
**Do this instead:** Adopt `MeetingStateProxy` with `contextvars` as implemented in `confluence_logic/jarvis_agentic.py`

### Parallel module duplication
**What happens:** `confluence_logic/` and `local_office_logic/` share ~70% of structure with diverging implementations for audio_cache, classifier, general_responder, meeting_responder, graph_rag, etc.
**Why it's wrong:** Changes must be made in two places; modules have already diverged (e.g., `confluence_logic` has `confluence_page_graph`, Tavily web search, `MeetingStateProxy`; `local_office_logic` does not).
**Do this instead:** Extract shared logic into a `shared/` or `jarvis_core/` base package.

## Error Handling

**Strategy:** Catch-log-return-safe-value at all I/O boundaries

**Patterns:**
- Broad `except Exception` → `logger.error(...)` → return typed failure object (`CommitResponse(success=False, ...)`, empty list, `None`)
- `ValueError` for domain-specific failures (version conflicts, duplicate headings)
- Graceful degradation: Pinecone/Neo4j failures fall back to live Confluence API; TTS provider failures fall back to gTTS; Tavily failures return empty web context

## Cross-Cutting Concerns

**Logging:** `logging.getLogger(__name__)` per module; `%`-style formatting in logger calls; f-strings for non-logger interpolation
**Validation:** Pydantic `BaseModel` for all agent input/output schemas in `core/schemas.py`; FastAPI Pydantic models in `review/api.py`
**Authentication:** Optional Supabase bearer token validation via `supabase_store.user_from_bearer()` in `review/api.py`; `review-ui` frontend sends no auth token

---

*Architecture analysis: 2026-05-11*
