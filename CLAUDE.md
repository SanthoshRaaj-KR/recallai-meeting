<!-- GSD:project-start source:PROJECT.md -->
## Project

**Jarvis — Meeting Intelligence Platform**

An AI-powered meeting assistant that joins meetings via a Recall.ai bot, captures transcripts in real-time, and at the end of each meeting generates proposed Confluence page changes based on what was discussed. Proposed changes are presented as review cards in the sync-sage-bot UI where users can accept or reject each one individually — only accepted changes are applied to Confluence.

**Core Value:** Users never have to manually update Confluence after a meeting — the system proposes the right changes to the right pages, and the user just approves or rejects.

### Constraints

- **Model ceiling:** GPT-5 maximum — no GPT-5 turbo or above
- **Model preference:** GPT-5 mini / GPT-5.4 nano wherever sufficient; GPT-5 only for orchestration and hard verification
- **Latency:** Pipeline may take up to 20 minutes — must show progress; do not time out
- **Safety:** No Confluence edits without explicit per-card user approval
- **RAG-first:** All Confluence page lookup must use pre-indexed graph/RAG, not live API scans
- **Focus scope:** Confluence connector only; local_office_logic is out of scope
<!-- GSD:project-end -->

<!-- GSD:stack-start source:codebase/STACK.md -->
## Technology Stack

## Runtime & Language
- Python 3 (asyncio-native) — all server-side logic in `confluence_logic/` and `local_office_logic/`
- TypeScript 5 — `review-ui/` Next.js app
- TypeScript 5 — `sync-sage-bot/` Vite/React SPA
## Frameworks & Libraries
- FastAPI — HTTP + WebSocket server; main app at `confluence_logic/jarvis_agentic.py`, review router at `confluence_logic/review/api.py`
- Uvicorn — ASGI server; launched with `uvicorn confluence_logic.jarvis_agentic:app`
- `openai` — LLM chat completions, streaming, embeddings, TTS (gpt-4o-mini, gpt-5-mini defaults)
- `openai-agents` (import name `agents`) — OpenAI Agents SDK; used for `Agent`, `Runner`, `function_tool` in `confluence_logic/agents/` and `local_office_logic/agents/`
- Next.js 16 (^16.2.4) with App Router — `review-ui/src/app/`
- React 18 — UI rendering
- Tailwind CSS 3 — styling
- TypeScript 5
- No state management library — local `useState`/`useEffect` only
- `next.config.ts` rewrites `/api/*` to `http://localhost:8000/*` (FastAPI proxy)
- Vite 7 + `@vitejs/plugin-react-swc` — build tooling
- React 18.3 — UI rendering
- React Router DOM 6 — client-side routing
- TanStack React Query 5 — server state / data fetching
- Radix UI (full component suite, ^1.x–^2.x per component) — headless UI primitives
- shadcn/ui conventions (Radix + Tailwind + CVA)
- Zod 3 — schema validation
- React Hook Form 7 + `@hookform/resolvers` — form management
- `@supabase/supabase-js` 2 — Supabase client
- Recharts 2 — charts
- Tailwind CSS 3 + `tailwindcss-animate` + `@tailwindcss/typography`
- `date-fns` 3, `lucide-react`, `sonner`, `cmdk`, `vaul`, `embla-carousel-react`
## Package Management
- pip — `requirements.txt` at repo root (no lockfile committed)
- npm — `package-lock.json` present in both `review-ui/` and `sync-sage-bot/`
## Build & Tooling
- `python-dotenv` — environment loading via `.env` files
- `pytest` + `pytest-asyncio` — test runner
- `next build` / `next dev` — standard Next.js build
- ESLint 8 + `eslint-config-next`
- PostCSS + Autoprefixer
- Vite 7 — dev server and production bundler
- `vitest` 3 + `@testing-library/react` 16 + `jsdom` — unit/component tests
- ESLint 9 + `typescript-eslint` 8 + `eslint-plugin-react-hooks`
- `lovable-tagger` — Lovable platform integration tag
- `docling` — HTML-to-Markdown document conversion in `confluence_logic/ingestion/doc_pipeline.py`
- LibreOffice headless (`soffice`) — office format conversion in `local_office_logic/utils/office_runtime.py`
## Key Dependencies
| Package | Version (requirements.txt) | Purpose |
|---------|---------------------------|---------|
| fastapi | latest | HTTP + WebSocket API server |
| uvicorn | latest | ASGI production server |
| openai | latest | LLM completions, embeddings, TTS |
| openai-agents | latest | Agentic tool orchestration (Agent/Runner) |
| requests | latest | HTTP calls to Recall.ai, Confluence, Supabase REST, Tavily |
| python-dotenv | latest | `.env` config loading |
| pinecone | latest | Vector store client (`confluence_logic/db/vector_store.py`) |
| neo4j | 6.1.0 | AuraDB graph database driver (Graph RAG) — **NOTE: this version does not exist on PyPI** |
| gTTS | latest | Google Text-to-Speech (fallback TTS provider) |
| edge-tts | latest | Microsoft Edge TTS (low-latency TTS provider) |
| websockets | latest | WebSocket protocol support |
| pyaudio | latest | Audio I/O |
| pydantic | latest | Data models and schema validation |
| docling | latest | HTML document conversion to Markdown |
| beautifulsoup4 | latest | HTML parsing (`confluence_logic/confluence_page_graph.py`) |
| ngrok | latest | Tunnel for local webhook exposure |
| python-docx | latest | DOCX file manipulation |
| openpyxl | latest | XLSX file manipulation |
| odfpy | latest | ODF (ODT/ODS) file manipulation |
| lxml | latest | XML/HTML parsing |
| pytest | latest | Test framework |
| pytest-asyncio | latest | Async test support |
## Environment Variables Reference
| Variable | Default | Used by |
|----------|---------|---------|
| `RECALL_API_KEY` | — | `jarvis_agentic.py` |
| `OPENAI_API_KEY` | — | all LLM/TTS calls |
| `SUPABASE_URL` | — | `review/supabase_store.py` |
| `SUPABASE_ANON_KEY` | — | `review/supabase_store.py` |
| `SUPABASE_SERVICE_ROLE_KEY` | — | `review/supabase_store.py` |
| `PINECONE_API_KEY` | — | `db/vector_store.py` |
| `NEO4J_URI` | — | `graph_rag.py`, `confluence_page_graph.py` |
| `TAVILY_API_KEY` | `""` | `confluence_logic/general_responder.py` (optional web search) |
| `JARVIS_AGENT_MODEL` | `gpt-5-mini` | agent constructors (**invalid model name**) |
| `JARVIS_GENERAL_MODEL` | `gpt-4o-mini` | `meeting_responder.py`, `general_responder.py` |
| `JARVIS_REVIEW_MODEL` | `gpt-5-mini` | `review/api.py` (**invalid model name**) |
| `JARVIS_TTS_PROVIDER` | `edge_tts` | `jarvis_agentic.py` |
| `APP_HOST` | `0.0.0.0` | `jarvis_agentic.py` |
| `APP_PORT` | `8000` | `jarvis_agentic.py` |
<!-- GSD:stack-end -->

<!-- GSD:conventions-start source:CONVENTIONS.md -->
## Conventions

## Naming Patterns
- `snake_case` for all Python module files: `jarvis_agentic.py`, `graph_rag.py`, `editor_agent.py`, `html_parser.py`, `confluence_page_graph.py`, `general_responder.py`
- Test files prefixed with `test_`: `test_flow.py`, `test_jarvis_agentic.py`, `test_review_api.py`, `test_classifier.py`, `test_confluence_page_graph.py`, `test_general_responder.py`
- Directories use `snake_case`: `confluence_logic/`, `local_office_logic/`, `agents/`, `connectors/`, `db/`
- `camelCase.ts` for utility modules: `api.ts`, `types.ts`
- `PascalCase.tsx` is not explicitly used for files — Next.js App Router pages are named `page.tsx` by convention; the exported component name uses PascalCase (`HomePage`, `ResultsPage`)
- `globals.css`, `layout.tsx` follow Next.js App Router naming conventions
- `PascalCase`: `EditorAgent`, `ReframerAgent`, `ProposedChangesAgent`, `ConfluenceConnector`, `PineconeStore`, `IngestionPipeline`
- Pydantic model classes `PascalCase`: `CandidatePage`, `SearchResponse`, `MasterVoiceDecision`, `CommitResponse`
- Abstract base classes use `PascalCase` without `Abstract` prefix: `DocumentFetcher`, `DocumentPusher`
- `snake_case` for all functions: `search_workspace_knowledge`, `fetch_live_page`, `classify_intent`, `user_from_bearer`, `upsert_history`, `query_user_confluence_graph`
- Private helpers prefixed with `_`: `_candidate_from_metadata`, `_commit_with_retry`, `_is_version_conflict`, `_reindex_in_background`, `_rest_headers`, `_db_key`
- Private async helpers follow same `_` convention: `_run_editor`, `_handle_general_question`, `_speak_guarded`, `_needs_web_search`
- Async coroutines named descriptively with verbs: `ingest_transcript_entry`, `query_context`, `answer_general_question`
- `snake_case` for local variables and module-level mutable state
- `UPPER_SNAKE_CASE` for module-level constants and env-var-derived config:
- Module-level singletons use `_` prefix: `_store`, `_connector`, `_driver`, `_openai_client`
- ContextVar names use `snake_case` string labels: `ContextVar("tool_run_state")`, `ContextVar("confluence_graph_user_id")`
- `PascalCase` for interfaces and type aliases: `SessionStatus`, `ChangeItem`, `MeetingSummary`, `ActionItem`, `BotStatus`, `ChangeType`
- `camelCase` for properties: `bot_id`, `change_count`, `page_id`, `section_heading` (snake_case mirroring the Python API response)
- Union string types used for discriminated state: `BotStatus = "idle" | "joining" | "in_meeting" | "ended" | "error"`
## Code Style
- No linting config files detected (`.flake8`, `.pylintrc`, `pyproject.toml`, `ruff.toml` are absent)
- Indentation: 4 spaces (standard Python)
- Line length: not enforced by config; lines up to ~120 characters observed in practice
- Trailing commas: not consistently used in function calls
- No `.prettierrc` detected; ESLint via `eslint-config-next` provides baseline rules
- 2-space indentation (Next.js default)
- `"use client"` directive at top of all interactive pages (`page.tsx`, `results/page.tsx`) — required for `useState`/`useEffect`
- No `isort` or `black` config detected; formatting is manual
- Blank lines between top-level functions and classes: 2 blank lines (standard PEP 8)
- Within classes: 1 blank line between methods
- Module-level docstrings used for key modules — plain triple-quoted strings describing module purpose and design decisions:
- Test file docstrings include ticket/spec IDs: `"""Tests for graph_rag — Neo4j Graph RAG module (GRAPHRAG-01)."""`
- Individual functions: short single-line docstrings where present, or no docstring for simple helpers
- No Google/NumPy/Sphinx docstring style enforced — informal prose only
- f-strings used universally for interpolation: `f"Version conflict max retries exceeded for page {page_id}: {ve}"`
- `%`-style used exclusively for `logger` calls: `logger.warning("Conflict on attempt %d/%d for page %s", attempt, max, page_id)`
## Module Organization
- `confluence_logic/` — Confluence-specific pipeline (agents, connectors, ingestion, review)
- `local_office_logic/` — Local Office files pipeline (parallel structure to `confluence_logic/`)
- `tests/` — top-level evaluation harnesses (not pytest unit tests)
## review-ui API Client Pattern
## Error Handling
- Pinecone/Neo4j unavailability caught with `logger.warning` — falls back to live Confluence API results
- TTS provider failures fall back to gTTS
- Classifier LLM failures default to `"confluence"` intent
- Tavily web search failures return empty context (non-fatal, logged at DEBUG level)
## Type Annotations
- `from __future__ import annotations` used in `confluence_logic/review/api.py` and `supabase_store.py` only
- Not yet migrated to PEP 604 union syntax (`X | Y`) or built-in generics (`list[X]`)
- Function signatures in `agents/tools.py` and `agents/editor_agent.py` are fully annotated
- Return types annotated for public functions: `-> SearchResponse`, `-> CommitResponse`, `-> str`, `-> bool`
- Private helpers sometimes annotated, sometimes not
- `confluence_logic/core/models.py` and `schemas.py` are fully typed via Pydantic `BaseModel`
- `confluence_logic/core/interfaces.py` uses ABCs with annotated abstract methods
- `review/supabase_store.py` is fully annotated
- All agent input/output schemas defined as `pydantic.BaseModel` subclasses in `core/schemas.py`
- `Optional` fields use `= None` default: `heading: Optional[str] = None`
- `model_validate()` used for structured LLM output: `MasterVoiceDecision.model_validate(result.final_output)`
## Import Organization
- Types imported with `import type` syntax: `import type { MeetingSummary, ChangeItem } from "@/types"`
- Path alias `@/` maps to `src/` (configured in `tsconfig.json`)
- API functions imported by name from `@/lib/api`: `import { startBot, getSession } from "@/lib/api"`
<!-- GSD:conventions-end -->

<!-- GSD:architecture-start source:ARCHITECTURE.md -->
## Architecture

## System Overview
```
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
- Wake-word activated async event loop (`local_repl.py` → `jarvis_agentic.py`)
- Intent classification gates routing to specialized handlers or agents
- Dual-graph RAG combines semantic vector search (Pinecone) with relationship traversal (Neo4j)
- Human-in-the-loop review: changes are proposed (JSON), persisted (Supabase), then approved via UI before committing to Confluence
- Two parallel domain packages (`confluence_logic/`, `local_office_logic/`) with identical layering but diverging implementations
## Data Flows
### 1. Live Meeting Wake-Word → TTS Response
```
```
### 2. General Question with Optional Web Search
```
```
### 3. Post-Meeting Review → Confluence Commit
```
```
### 4. Document Ingestion Pipeline
```
```
### 5. Review UI Bot Lifecycle
```
```
## Layers
- Purpose: Top-level user interfaces — terminal REPL and Next.js web UI
- Location: `confluence_logic/local_repl.py`, `review-ui/src/app/`
- Depends on: orchestrator (jarvis_agentic), API client (api.ts)
- Purpose: Intent routing, agent delegation, TTS, WebSocket/meeting event loop
- Location: `confluence_logic/jarvis_agentic.py`
- Contains: FastAPI app, meeting state machine, wake-word handling
- Purpose: REST endpoints for the review-ui frontend; bot lifecycle and change management
- Location: `confluence_logic/review/api.py`
- Mounted into: `jarvis_agentic.py` FastAPI app via `app.include_router(router)`
- Purpose: Determine intent from spoken query to route to correct handler
- Location: `confluence_logic/classifier.py`
- Intents: `general`, `web_search`, `meeting_opinion`, `meeting_summary`, `confluence`, `action_items`
- Purpose: Specialized handlers for each intent class
- Location: `confluence_logic/general_responder.py`, `confluence_logic/meeting_responder.py`
- Purpose: Specialized task agents using OpenAI Agents SDK
- Location: `confluence_logic/agents/`
- Contains: editor, reframer, proposed_changes agents + tool definitions
- Purpose: Retrieval-augmented generation context building
- Location: `confluence_logic/graph_rag.py`, `confluence_logic/confluence_page_graph.py`, `confluence_logic/db/vector_store.py`
- Purpose: External service clients with thin wrappers
- Location: `confluence_logic/connectors/confluence.py`
- Purpose: Supabase meeting history persistence
- Location: `confluence_logic/review/supabase_store.py`
## Key Abstractions
- Purpose: Represents a single proposed Confluence change
- Definition: `review-ui/src/types.ts`
- Fields: `change_type` (create/edit/delete/title), `page_id`, `section_heading`, `before_content`, `after_content`, `status`
- Purpose: Current bot session state used for UI polling
- Definition: `review-ui/src/types.ts`
- Fields: `status` (idle/joining/in_meeting/ended/error), `bot_id`, `change_count`
- Purpose: Thread-safe meeting state container using Python `contextvars` for session isolation
- Location: `confluence_logic/jarvis_agentic.py`
- Replaces: bare module-level dict used in `local_office_logic`
- Purpose: Takes meeting context + Confluence page retrieval → outputs structured JSON change proposals
- Location: `confluence_logic/agents/proposed_changes_agent.py`
- Output schema: `{"changes": [{change_type, page_id, page_title, section_heading, before_content, after_content, rationale}]}`
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
### Untyped `meeting_state` dict (local_office_logic)
### Parallel module duplication
## Error Handling
- Broad `except Exception` → `logger.error(...)` → return typed failure object (`CommitResponse(success=False, ...)`, empty list, `None`)
- `ValueError` for domain-specific failures (version conflicts, duplicate headings)
- Graceful degradation: Pinecone/Neo4j failures fall back to live Confluence API; TTS provider failures fall back to gTTS; Tavily failures return empty web context
## Cross-Cutting Concerns
<!-- GSD:architecture-end -->

<!-- GSD:skills-start source:skills/ -->
## Project Skills

No project skills found. Add skills to any of: `.claude/skills/`, `.agents/skills/`, `.cursor/skills/`, `.github/skills/`, or `.codex/skills/` with a `SKILL.md` index file.
<!-- GSD:skills-end -->

<!-- GSD:workflow-start source:GSD defaults -->
## GSD Workflow Enforcement

Before using Edit, Write, or other file-changing tools, start work through a GSD command so planning artifacts and execution context stay in sync.

Use these entry points:
- `/gsd-quick` for small fixes, doc updates, and ad-hoc tasks
- `/gsd-debug` for investigation and bug fixing
- `/gsd-execute-phase` for planned phase work

Do not make direct repo edits outside a GSD workflow unless the user explicitly asks to bypass it.
<!-- GSD:workflow-end -->



<!-- GSD:profile-start -->
## Developer Profile

> Profile not yet configured. Run `/gsd-profile-user` to generate your developer profile.
> This section is managed by `generate-claude-profile` -- do not edit manually.
<!-- GSD:profile-end -->
