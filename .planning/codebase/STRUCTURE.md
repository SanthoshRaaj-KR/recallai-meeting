# Codebase Structure

**Analysis Date:** 2026-04-10

## Directory Layout

```
recallai-meeting/
├── confluence_logic/              # Core agentic Confluence document editing system
│   ├── agents/                    # Agent definitions and tool definitions
│   │   ├── editor_agent.py        # Main agent orchestrator (4 agents: edit, delete, create, list)
│   │   ├── reframer_agent.py      # Intent resolver for vague/ambiguous requests
│   │   └── tools.py               # 15+ @function_tool decorated operations
│   ├── connectors/
│   │   └── confluence.py           # ConfluenceConnector: REST API wrapper for Confluence
│   ├── core/
│   │   ├── interfaces.py           # DocumentFetcher, DocumentPusher abstract base classes
│   │   ├── schemas.py              # Pydantic response schemas (SearchResponse, LivePageResponse, etc.)
│   │   └── models.py               # DocumentChunk data model for vector index
│   ├── db/
│   │   └── vector_store.py         # PineconeStore: semantic search + embedding operations
│   ├── ingestion/
│   │   └── doc_pipeline.py         # IngestionPipeline: HTML → markdown → chunks → vector store
│   ├── utils/
│   │   ├── html_parser.py          # HTML section extraction, editing, deletion (BeautifulSoup)
│   │   └── html_builder.py         # HTML generation for new pages
│   ├── tests/
│   │   ├── __init__.py
│   │   ├── test_flow.py            # Unit tests for tools (mocked connectors)
│   │   └── test_jarvis_agentic.py  # Tests for agentic meeting flow
│   ├── jarvis_agentic.py           # Main agentic meeting assistant (WebSocket + FastAPI)
│   └── local_repl.py               # Interactive REPL for testing agent flows locally
│
├── jarvis.py                        # Legacy meeting assistant (simpler tool-calling pattern)
├── agents/                          # (Untracked, appears empty or external)
├── meetings/                        # Meeting metadata and transcripts (meeting_index.json)
├── storage/                         # Local storage (untracked)
├── tests/                           # (Untracked, appears empty or external)
│
├── requirements.txt                 # Python dependencies
├── .env                             # Environment configuration (SECRETS - NOT COMMITTED)
├── .env.example                     # Template for required env vars
├── bot_id.txt                       # Written at runtime: current bot ID for Recall.ai
├── .gitignore                       # Specifies git exclusions
└── .planning/                       # GSD documentation
    └── codebase/                    # Architecture/structure documents (generated)
        ├── ARCHITECTURE.md          # (This analysis)
        └── STRUCTURE.md             # (This document)
```

## Directory Purposes

**`confluence_logic/`:**
- Purpose: Main module containing all agentic Confluence editing logic
- Contains: Agent orchestration, tool definitions, connectors, vector store, utilities
- Key files: `jarvis_agentic.py` (entry point), `agents/editor_agent.py` (main agent)

**`confluence_logic/agents/`:**
- Purpose: Agent definitions and tool implementations
- Contains: EditorAgent, ReframerAgent, specialized agents (Edit/Delete/Create/List), all tools
- Key files:
  - `editor_agent.py`: Main coordinator; wraps 4 agents + reframer
  - `reframer_agent.py`: Resolves vague requests into concrete page edit intents
  - `tools.py`: All @function_tool functions called by agents

**`confluence_logic/connectors/`:**
- Purpose: Third-party service integrations
- Contains: ConfluenceConnector class
- Key files: `confluence.py` (Confluence REST API wrapper)

**`confluence_logic/core/`:**
- Purpose: Type definitions, interfaces, schemas
- Contains: Abstract interfaces (DocumentFetcher, DocumentPusher), Pydantic schemas, data models
- Key files:
  - `interfaces.py`: Abstract contracts for extensibility
  - `schemas.py`: Response/request Pydantic models
  - `models.py`: DocumentChunk model for vector store

**`confluence_logic/db/`:**
- Purpose: Vector database operations
- Contains: PineconeStore class for semantic search + embeddings
- Key files: `vector_store.py` (Pinecone integration)

**`confluence_logic/ingestion/`:**
- Purpose: Document processing pipeline
- Contains: IngestionPipeline for converting Confluence pages to embeddings
- Key files: `doc_pipeline.py` (HTML → markdown → chunks → Pinecone)

**`confluence_logic/utils/`:**
- Purpose: HTML manipulation and document building utilities
- Contains: BeautifulSoup-based parsing/editing, page HTML generation
- Key files:
  - `html_parser.py`: get_section_html, edit_block_in_section, delete_content_in_section
  - `html_builder.py`: build_page_html for new page creation

**`confluence_logic/tests/`:**
- Purpose: Unit and integration tests
- Contains: Mocked tests for tools, agent flow tests
- Key files:
  - `test_flow.py`: 10+ unit tests for search, edit, delete flows
  - `test_jarvis_agentic.py`: Agentic meeting flow tests

**`meetings/`:**
- Purpose: Runtime meeting data storage
- Contains: meeting_index.json (metadata), individual meeting JSON/MD files
- Usage: Logged during meeting execution; not cleaned up automatically

**`storage/`:**
- Purpose: Local file storage (untracked in git)
- Contains: Temporary files, embeddings cache (optional)
- Usage: Depends on environment configuration

## Key File Locations

**Entry Points:**
- `confluence_logic/jarvis_agentic.py`: Main meeting assistant; run via `uvicorn confluence_logic.jarvis_agentic:app --host 0.0.0.0 --port 8000`
- `confluence_logic/local_repl.py`: Interactive REPL; run via `python -m confluence_logic.local_repl`
- `jarvis.py`: Legacy meeting assistant; run via `python jarvis.py`

**Configuration:**
- `.env`: Runtime environment variables (SECRETS — never committed)
- `.env.example`: Template showing required variables
- `requirements.txt`: Python package dependencies

**Core Logic:**
- `confluence_logic/agents/editor_agent.py`: Main EditorAgent class; coordinates all agent operations
- `confluence_logic/agents/reframer_agent.py`: ReframerAgent for intent resolution
- `confluence_logic/agents/tools.py`: All tool implementations (search, fetch, preview, commit, create)

**Connectors:**
- `confluence_logic/connectors/confluence.py`: ConfluenceConnector; REST API to Confluence
- `confluence_logic/db/vector_store.py`: PineconeStore; semantic search + embeddings

**Utilities:**
- `confluence_logic/utils/html_parser.py`: HTML section extraction/modification (BeautifulSoup)
- `confluence_logic/utils/html_builder.py`: HTML generation for new pages

**Testing:**
- `confluence_logic/tests/test_flow.py`: Tool mocking + flow tests
- `confluence_logic/tests/test_jarvis_agentic.py`: Agentic flow tests

## Naming Conventions

**Files:**
- `*_agent.py`: Agent class definitions (EditorAgent, ReframerAgent)
- `*.py` in agents/: Tool definitions or agent classes
- Connector classes: `{Service}Connector.py` (e.g., ConfluenceConnector in `confluence.py`)
- Utility files: `{domain}_{purpose}.py` (e.g., `html_parser.py`, `html_builder.py`)

**Directories:**
- `agents/`: Agent and tool definitions
- `connectors/`: Backend service integrations
- `core/`: Interfaces, schemas, base models
- `db/`: Vector store and data persistence
- `utils/`: Reusable utilities (parsing, building)
- `tests/`: Test files (co-located with source module)

**Functions/Methods:**
- Tool functions: `@function_tool` decorated, PascalCase or snake_case, match OpenAI function names
  - Examples: `search_workspace_knowledge`, `fetch_live_page`, `commit_document_edit`
- Agent methods: `handle_query()`, `as_tool()` (standard for Agents SDK)
- Helpers: Prefixed with `_` if private/internal (e.g., `_resolve_target_html`, `_is_full_page_mode`)

**Classes:**
- Agent classes: Suffix `-Agent` (EditorAgent, ReframerAgent)
- Connector classes: Suffix `-Connector` (ConfluenceConnector)
- Store classes: Suffix `-Store` (PineconeStore)
- Data models: Suffix `-Response` for API responses, `-Decision` for agent decisions
  - Examples: SearchResponse, CandidatePage, ResolverDecision, MasterVoiceDecision

## Where to Add New Code

**New Agent (decision-making component):**
- Primary code: `confluence_logic/agents/{agent_name}_agent.py`
- Integration: Import and wrap in `EditorAgent.__init__()` as `.as_tool()`, store as `self.{agent_name}_tool`
- Tools it calls: Reference existing tools from `confluence_logic/agents/tools.py`, or create new ones

**New Tool (atomic operation):**
- Implementation: `confluence_logic/agents/tools.py`
- Pattern: Decorate with `@function_tool`, return Pydantic response schema (e.g., SearchResponse)
- Connector calls: Use `get_connector()` and `get_store()` globals; they lazy-initialize singletons

**New Connector (integration with external service):**
- Implementation: `confluence_logic/connectors/{service}.py`
- Pattern: Create class inheriting from `DocumentFetcher` and/or `DocumentPusher` if applicable
- Usage: Import in tools.py, initialize as singleton in `get_connector()` equivalent

**New Utility (reusable helper):**
- HTML manipulation: `confluence_logic/utils/html_parser.py`
- HTML generation: `confluence_logic/utils/html_builder.py`
- New domain: Create `confluence_logic/utils/{domain}.py`

**New Schema/Model:**
- API responses: `confluence_logic/core/schemas.py` (Pydantic BaseModel with `.model_validate()`)
- Data models: `confluence_logic/core/models.py` (DocumentChunk pattern)
- Agent outputs: Define in `schemas.py`, reference in agent `output_type=MySchema`

**New Test:**
- Location: `confluence_logic/tests/test_{module}.py`
- Pattern: Use `@patch` for connectors/stores; create Mock return values
- See `test_flow.py` for mocking patterns

**Entry point / meeting flow:**
- Agentic meeting: Modify `confluence_logic/jarvis_agentic.py` (WebSocket, task queue, TTS)
- Local testing: Modify `confluence_logic/local_repl.py` (stdin/stdout loop)

## Special Directories

**`confluence_logic/tests/`:**
- Purpose: Unit and integration tests
- Generated: No (hand-written)
- Committed: Yes
- Pattern: Tests use `@patch` to mock ConfluenceConnector and PineconeStore; verify tool behavior

**`meetings/`:**
- Purpose: Runtime meeting metadata and transcript storage
- Generated: Yes (created at runtime)
- Committed: Selectively (git-ignored in most cases)
- Structure: meeting_index.json (array of metadata), individual {timestamp}_{uuid}.json/md files

**`.planning/codebase/`:**
- Purpose: GSD documentation (this analysis)
- Generated: Yes (by GSD map-codebase orchestrator)
- Committed: Yes
- Contents: ARCHITECTURE.md, STRUCTURE.md, CONVENTIONS.md, TESTING.md, etc. (one per focus area)

**`.env` (environment configuration):**
- Purpose: Local runtime secrets and configuration
- Generated: Manual creation from `.env.example`
- Committed: NO (in .gitignore)
- Contains: RECALL_API_KEY, ATLASSIAN_API_TOKEN, OPENAI_API_KEY, PINECONE_API_KEY, etc.

**`bot_id.txt`:**
- Purpose: Stores current bot ID after successful Recall.ai bot creation
- Generated: Yes (written at runtime)
- Committed: No (in .gitignore)
- Usage: Allows restart scripts to kill/manage the bot without re-reading stdout

---

*Structure analysis: 2026-04-10*
