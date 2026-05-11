# Project Structure

**Analysis Date:** 2026-05-11

## Directory Layout

```
recallai-meeting/
├── confluence_logic/          # Confluence-integrated AI assistant
│   ├── agents/                # OpenAI Agents SDK sub-agents
│   │   ├── editor_agent.py    # Document edit proposals + commits
│   │   ├── reframer_agent.py  # Content rewriting
│   │   ├── proposed_changes_agent.py  # Change proposals for human review
│   │   └── tools.py           # @function_tool definitions (OpenAI Agents SDK)
│   ├── connectors/
│   │   └── confluence.py      # Atlassian Confluence REST API client
│   ├── core/                  # Shared domain types
│   │   ├── interfaces.py      # Abstract base classes / protocols
│   │   ├── models.py          # Pydantic / dataclass models
│   │   └── schemas.py         # Request/response schemas
│   ├── db/
│   │   └── vector_store.py    # Pinecone vector store wrapper
│   ├── ingestion/
│   │   └── doc_pipeline.py    # Document chunking + embedding pipeline
│   ├── review/                # Human-in-the-loop review UI backend
│   │   ├── api.py             # FastAPI APIRouter (bot lifecycle + review endpoints)
│   │   ├── supabase_store.py  # Supabase REST persistence for meeting history
│   │   ├── supabase_schema.sql  # Postgres DDL for meeting_history + RLS policies
│   │   └── __init__.py
│   ├── scripts/
│   │   ├── generate_wav_assets.py  # Pre-generate TTS audio
│   │   └── __init__.py
│   ├── tests/                 # Unit + integration tests
│   │   ├── test_flow.py
│   │   ├── test_classifier.py        # Classifier intent routing tests
│   │   ├── test_general_responder.py # Web search LLM router tests
│   │   ├── test_graph_rag.py
│   │   ├── test_jarvis_agentic.py
│   │   ├── test_review_api.py        # Review API + session state machine tests
│   │   ├── test_confluence_page_graph.py  # Page graph + API helper tests
│   │   └── __init__.py
│   ├── utils/
│   │   ├── html_builder.py    # HTML construction utilities
│   │   └── html_parser.py     # HTML parsing utilities
│   ├── audio_cache.py         # In-memory TTS MP3 cache
│   ├── classifier.py          # Intent classifier (general/web_search/meeting/confluence)
│   ├── confluence_page_graph.py  # Neo4j Confluence workspace graph (per user_id)
│   ├── general_responder.py   # General Q&A + Tavily web search routing
│   ├── graph_rag.py           # Dual-graph RAG (Pinecone + Neo4j transcript graph)
│   ├── jarvis_agentic.py      # Main agentic orchestrator + FastAPI app
│   ├── local_repl.py          # Interactive terminal REPL / meeting event loop
│   └── meeting_responder.py   # Meeting summarization, opinion, action items
│
├── local_office_logic/        # Local Office document AI assistant
│   ├── agents/                # Mirrors confluence_logic/agents/ (no proposed_changes_agent)
│   │   ├── editor_agent.py
│   │   ├── reframer_agent.py
│   │   └── tools.py
│   ├── connectors/
│   │   └── local_office.py    # LibreOffice headless connector
│   ├── core/                  # Mirrors confluence_logic/core/
│   │   ├── interfaces.py
│   │   ├── models.py
│   │   └── schemas.py
│   ├── db/
│   │   └── vector_store.py
│   ├── ingestion/
│   │   └── doc_pipeline.py
│   ├── scripts/
│   │   ├── generate_wav_assets.py
│   │   └── __init__.py
│   ├── tests/
│   │   ├── test_local_office_flow.py
│   │   └── __init__.py
│   ├── utils/
│   │   ├── html_builder.py
│   │   ├── html_parser.py
│   │   ├── sandbox.py         # Sandboxed file operations
│   │   ├── office_runtime.py  # LibreOffice runtime management
│   │   ├── document_adapter.py   # Word document adapter
│   │   └── spreadsheet_adapter.py  # Excel/Calc adapter
│   ├── audio_cache.py
│   ├── classifier.py
│   ├── general_responder.py
│   ├── graph_rag.py
│   ├── jarvis_agentic.py
│   ├── local_repl.py
│   └── meeting_responder.py
│
├── review-ui/                 # Next.js 16 frontend for reviewing proposed changes
│   ├── src/
│   │   ├── app/               # App Router pages
│   │   │   ├── layout.tsx     # Root layout (Tailwind CSS base, metadata)
│   │   │   ├── page.tsx       # Home page — meeting URL input + bot join + polling
│   │   │   ├── globals.css    # Global Tailwind CSS imports
│   │   │   └── results/
│   │   │       └── page.tsx   # Results page — summary display + change review + execute
│   │   ├── lib/
│   │   │   └── api.ts         # Typed fetch wrappers: startBot, getSession, getChanges, executeChanges, getMeetingSummary
│   │   └── types.ts           # TypeScript interfaces: SessionStatus, ChangeItem, MeetingSummary, etc.
│   ├── next.config.ts         # Rewrites /api/* → http://localhost:8000/* (FastAPI proxy)
│   ├── package.json           # next ^16.2.4, react ^18, tailwindcss ^3
│   ├── tsconfig.json
│   └── package-lock.json
│
├── tests/                     # Root-level E2E evaluation harnesses (not pytest-collected)
│   ├── e2e_pipeline_eval.py       # Synthetic meeting transcript → full pipeline quality eval
│   ├── pipeline_comprehensive_eval.py  # Quality + TTS latency eval (--quality-only / --full modes)
│   └── test_meeting_responder.py
│
├── sync-sage-bot/             # Untracked git submodule (external Vite/React bot UI)
├── requirements.txt           # Python dependencies (no version pins on most packages)
└── .planning/                 # GSD planning artifacts
```

## Module Organization

Both domain packages (`confluence_logic/`, `local_office_logic/`) follow identical layering:

| Layer | Directory | Responsibility |
|-------|-----------|----------------|
| Entry | `local_repl.py` | Top-level loop, user interaction |
| Orchestration | `jarvis_agentic.py` | Intent routing, agent delegation, TTS, WebSocket |
| Review API | `review/api.py` | (confluence only) Bot lifecycle + change management endpoints |
| Classification | `classifier.py` | Input intent detection |
| Response handlers | `general_responder.py`, `meeting_responder.py` | Per-intent LLM response generation |
| Agents | `agents/` | Specialized task agents (editor, reframer, proposed_changes) |
| RAG | `graph_rag.py`, `confluence_page_graph.py`, `db/` | Retrieval and knowledge graph |
| Connectors | `connectors/` | External service clients |
| Ingestion | `ingestion/` | Document chunking + embedding |
| Core types | `core/` | Shared models, interfaces, schemas |
| Utilities | `utils/` | HTML, parsing, file adapters |
| Persistence | `review/supabase_store.py` | (confluence only) Supabase meeting history |

**`confluence_logic/` additions not present in `local_office_logic/`:**
- `review/` — full review backend (api.py, supabase_store.py, supabase_schema.sql)
- `confluence_page_graph.py` — Confluence workspace graph
- `agents/proposed_changes_agent.py` — structured change proposals
- Tavily web search in `general_responder.py`
- `MeetingStateProxy` with `contextvars` in `jarvis_agentic.py`

## Naming Conventions

| Pattern | Example | Meaning |
|---------|---------|---------|
| `*_agent.py` | `editor_agent.py`, `proposed_changes_agent.py` | OpenAI Agents SDK agent class |
| `*_responder.py` | `meeting_responder.py`, `general_responder.py` | Response handler for a query type |
| `*_store.py` | `supabase_store.py`, `vector_store.py` | Persistence layer wrapper |
| `*_pipeline.py` | `doc_pipeline.py` | Multi-step processing pipeline |
| `*_adapter.py` | `document_adapter.py` | Adapter to an external format |
| `*_graph.py` | `confluence_page_graph.py` | Graph data structure or query layer |
| `tools.py` | `agents/tools.py` | `@function_tool` definitions |
| `snake_case` | all Python files | File and module names |
| `PascalCase.tsx` | `ResultsPage`, `HomePage` | Next.js page components (Next.js convention) |
| `camelCase.ts` | `api.ts`, `types.ts` | TypeScript utility files |

## Where to Place New Code

| Task | Location |
|------|----------|
| New intent handler | `classifier.py` + new `*_responder.py` |
| New sub-agent | `agents/*_agent.py` + register tools in `agents/tools.py` |
| New external connector | `connectors/new_service.py` |
| New document adapter | `local_office_logic/utils/*_adapter.py` |
| New shared model | `core/models.py` or `core/schemas.py` |
| New ingestion source | `ingestion/doc_pipeline.py` or new `ingestion/` file |
| New review API endpoint | `confluence_logic/review/api.py` (add to `router`) |
| New Supabase table/schema | `confluence_logic/review/supabase_schema.sql` |
| New review-ui page | `review-ui/src/app/<page-name>/page.tsx` |
| New review-ui API call | `review-ui/src/lib/api.ts` + `review-ui/src/types.ts` |
| New unit test (confluence) | `confluence_logic/tests/test_<module>.py` |
| New unit test (local office) | `local_office_logic/tests/test_<module>.py` |
| New E2E evaluation script | `tests/<eval_name>.py` |

## Special Directories

**`.planning/`:**
- Purpose: GSD planning artifacts (phases, codebase maps)
- Generated: No (hand-maintained + GSD tooling)
- Committed: Yes

**`review-ui/`:**
- Purpose: Standalone Next.js app — run separately from the Python backend
- Generated: No
- Committed: Yes (node_modules excluded)

**`sync-sage-bot/`:**
- Purpose: External bot integration (Vite/React SPA) — previously a git submodule
- Generated: No
- Committed: Currently untracked (submodule removed, directory remains)

---

*Structure analysis: 2026-05-11*
