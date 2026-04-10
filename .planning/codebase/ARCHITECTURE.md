# Architecture

**Analysis Date:** 2026-04-10

## Pattern Overview

**Overall:** Multi-layered agent-driven architecture with separation between meeting I/O (WebSocket), agentic request handling, document operations, and knowledge retrieval.

**Key Characteristics:**
- **Agent-based execution**: Uses OpenAI Agents SDK for autonomous decision-making and tool orchestration
- **Tool-driven operations**: Confluence edits, deletes, and creates are performed via discrete tool calls
- **Workspace awareness**: Continuously queries live Confluence data to avoid stale context
- **Layered dependencies**: Core interfaces abstract Confluence/vector store; agents depend on tools; tools depend on connectors

## Layers

**Presentation/I/O Layer:**
- Purpose: Handle WebSocket connections from Recall.ai, receive meeting audio transcripts, emit bot speech
- Location: `confluence_logic/jarvis_agentic.py` (FastAPI WebSocket endpoints)
- Contains: WebSocket connection handler, transcript ingestion, speech queueing
- Depends on: OpenAI Agents SDK, agents.EditorAgent, Recall.ai API
- Used by: External Recall.ai service; initiates voice task lifecycle

**Request Orchestration Layer:**
- Purpose: Coordinate handling of voice requests, manage concurrent task queues, handle clarifications
- Location: `confluence_logic/jarvis_agentic.py` (task management, state machines)
- Contains: VoiceTask dataclass, meeting_state dict, task phase transitions (queued → resolving → executing → complete)
- Depends on: AsyncIO, EditorAgent
- Used by: WebSocket handler (receives raw transcripts, enqueues tasks)

**Agent Coordination Layer:**
- Purpose: High-level intent resolution and request dispatching to specialized agents
- Location: `confluence_logic/agents/editor_agent.py` (EditorAgent class)
- Contains: ReframerAgent (resolves ambiguous requests), Edit/Delete/Create/List agents (execute operations)
- Depends on: OpenAI Agents SDK, agents.tools, reframer_agent.py
- Used by: jarvis_agentic (calls handle_query), orchestration layer (task execution)

**Agent Execution Layer:**
- Purpose: Specialized agents that perform Confluence operations (edit, delete, create, list pages)
- Location: `confluence_logic/agents/editor_agent.py` (lines 17–100, agent definitions)
- Contains: 4 agents defined as Agent instances with specific instructions + tools
  - `edit_agent`: modifies existing page content (add, update, rewrite, rename)
  - `delete_agent`: removes sections or blocks from pages
  - `create_agent`: creates new Confluence pages
  - `list_agent`: lists available pages in workspace
- Depends on: agents.tools (function_tool decorated functions), Pydantic schemas
- Used by: EditorAgent.handle_query orchestrates them via resolve_tool/edit_tool/delete_tool/create_tool

**Tool/Connector Layer:**
- Purpose: Discrete, function-tool decorated operations that agents call; bridge between agents and backends
- Location: `confluence_logic/agents/tools.py`
- Contains: ~15 function_tool decorated functions
  - Search: `search_workspace_knowledge` (Pinecone + live Confluence search)
  - Fetch: `fetch_live_page` (get page HTML + metadata)
  - Preview: `preview_edit`, `preview_delete` (dry-run diffs before commit)
  - Commit: `commit_document_edit`, `commit_delete` (write to Confluence)
  - Helpers: `update_page_title`, `list_workspace_pages`, `create_confluence_page`
- Depends on: ConfluenceConnector, PineconeStore, html_parser utilities
- Used by: All agents call these functions during execution

**Backend Connectors:**
- Purpose: Abstract communication with external services (Confluence, Pinecone)
- Location:
  - `confluence_logic/connectors/confluence.py` (ConfluenceConnector class)
  - `confluence_logic/db/vector_store.py` (PineconeStore class)
- Contains:
  - ConfluenceConnector: fetch_page_html, search_pages, list_pages, get_page_metadata, push_update
  - PineconeStore: upsert_chunks, search (similarity search), get_page_version, get_embeddings
- Depends on: requests (HTTP), pinecone-client, OpenAI embeddings API
- Used by: tools layer

**Core Data/Interface Layer:**
- Purpose: Type definitions, schemas, abstract interfaces for extensibility
- Location:
  - `confluence_logic/core/interfaces.py` (DocumentFetcher, DocumentPusher abstractions)
  - `confluence_logic/core/schemas.py` (Pydantic response schemas)
  - `confluence_logic/core/models.py` (DocumentChunk data model)
- Contains: Type contracts and validation objects
- Depends on: Pydantic for validation
- Used by: Tools and connectors

**Utilities Layer:**
- Purpose: HTML manipulation, document building, parsing
- Location:
  - `confluence_logic/utils/html_parser.py` (get_section_html, edit_block_in_section, delete_content_in_section)
  - `confluence_logic/utils/html_builder.py` (build_page_html for new page creation)
- Contains: BeautifulSoup-based HTML section extraction/modification, heading detection
- Depends on: BeautifulSoup4
- Used by: tools layer (for preview/commit operations)

**Ingestion Pipeline:**
- Purpose: Embed Confluence pages into vector store for semantic search
- Location: `confluence_logic/ingestion/doc_pipeline.py` (IngestionPipeline class)
- Contains: process_page (fetch HTML → convert to markdown → chunk → embed → upsert)
- Depends on: docling for HTML→markdown, PineconeStore, OpenAI embeddings
- Used by: Standalone batch jobs (not currently called in meeting flow)

**Entry Points:**

Two main entry points exist:
1. **Meeting Mode (agentic):**
   - Location: `confluence_logic/jarvis_agentic.py::main()` or `uvicorn confluence_logic.jarvis_agentic:app`
   - Triggers: Direct execution; starts WebSocket server on localhost:8000
   - Responsibilities: Join Recall.ai meeting, listen for "Hey Jarvis" wake word, enqueue voice requests, route through agent system

2. **Local REPL:**
   - Location: `confluence_logic/local_repl.py::main()`
   - Triggers: `python -m confluence_logic.local_repl`
   - Responsibilities: Interactive prompt for testing agent flows without meeting context

3. **Legacy Meeting Assistant:**
   - Location: `jarvis.py::main()`
   - Triggers: `python jarvis.py`
   - Responsibilities: Earlier implementation with simpler tool-calling (not using Agents SDK)

## Data Flow

**Voice Request Flow (agentic):**

1. **Transcript Arrival**: Recall.ai WebSocket sends `{"event": "transcript.data", "data": {...}}`
2. **Wake Word Detection**: `websocket_endpoint()` checks transcript against `_WAKE_PATTERN` regex
3. **Task Enqueueing**: If wake word detected, wrap request in VoiceTask, add to `meeting_state["pending_requests"]`
4. **Request Processing**: `_process_request_queue()` dequeues, calls `session_agent.handle_query(request, history)`
5. **Agent Orchestration**: `EditorAgent.handle_query()` → ReframerAgent (resolve intent) → specialized agent (edit/delete/create)
6. **Tool Execution**: Specialized agent calls tools (search_workspace_knowledge, fetch_live_page, etc.)
7. **Backend Operations**: Tools invoke Confluence API + Pinecone search
8. **Response Generation**: Agent formats final response (success/error)
9. **Speech Output**: Response played back via TTS (OpenAI or gTTS), sent to Recall bot output audio endpoint

**Page Edit Flow (example):**

1. User: "Hey Jarvis, update the Q2 roadmap with new timelines"
2. Wake word extracted → task queued
3. ReframerAgent searches for "roadmap" pages, identifies likely target
4. EditorAgent (edit_agent) calls `search_workspace_knowledge("roadmap")`
5. Returns CandidatePage objects with page_id, heading options
6. Agent calls `fetch_live_page(page_id, "Q2 Roadmap")` → returns HTML + available headings
7. Agent calls `preview_edit(page_id, "Q2 Roadmap", old_html, new_html)` → shows diff
8. If satisfied, agent calls `commit_document_edit(page_id, version, heading, old, new)`
9. ConfluenceConnector makes REST PUT to Confluence API
10. Success response returned to agent, then spoken to user

**Search Flow:**

- User query → `search_workspace_knowledge(query)`
- Tool first tries Pinecone semantic search (if index exists)
- Falls back to live Confluence CQL search (title + text)
- Deduplicates results by page_id
- Returns top candidates with snippet previews
- Results sorted by relevance (exact match > partial match)

**State Management:**

- `meeting_state`: Global dict tracking current bot state
- Per-task: VoiceTask objects track individual request lifecycle
- Per-tool: ContextVar `_tool_run_state` persists state during tool invocation (allows agents to retry with context)
- ContextVar `_mutation_observer`: Optional callback for tracking when mutations start (debugging)

## Key Abstractions

**DocumentFetcher Interface:**
- Purpose: Abstract page HTML retrieval (allows different backend sources)
- Examples: `ConfluenceConnector.fetch_page_html()`
- Pattern: ABC with `fetch_page_html(page_id: str) → str` contract

**DocumentPusher Interface:**
- Purpose: Abstract page update operations
- Examples: `ConfluenceConnector.push_update()`
- Pattern: ABC with `push_update(page_id: str, content: str) → bool` contract

**VoiceTask Dataclass:**
- Purpose: Encapsulates a single user request lifecycle
- Fields: task_id, request, bot_id, phase, intent, cancel_requested, clarification context
- Pattern: Mutable dataclass for stateful tracking through async operations

**CandidatePage Schema:**
- Purpose: Standardized page search result representation
- Fields: page_id, title, heading (optional), is_root_section, space_key, snippet
- Pattern: Pydantic BaseModel for validation + serialization

**ResolverDecision Schema:**
- Purpose: Output of ReframerAgent; communicates intent + target page to execution agents
- Fields: action ("edit"/"delete"/"create"/"clarify"), page_title, page_id, heading, reframed_request
- Pattern: Structured agent output to avoid free-text parsing

**DocumentChunk Model:**
- Purpose: Atomic chunk stored in vector index
- Fields: id, page_id, space_key, title, heading, section_order, version, text_summary, markdown_content
- Pattern: Pydantic model; upserted in batches to Pinecone

## Error Handling

**Strategy:** Graceful degradation; prefer approximate execution over failure.

**Patterns:**

1. **Pinecone Unavailable**: Fall back from semantic search to live Confluence CQL search
   - Location: `confluence_logic/agents/tools.py::search_workspace_knowledge()`
   - Code: Try Pinecone, catch exception, use `connector.search_pages()` instead

2. **Version Conflict on Commit**: Detect "Version Conflict" in error message, request fresh fetch
   - Location: `confluence_logic/agents/tools.py` commit functions
   - Code: Check `_is_version_conflict(error)`, re-invoke fetch, retry commit

3. **Agent Task Timeout**: Supersede or cancel if task hangs beyond threshold
   - Location: `confluence_logic/jarvis_agentic.py` task management
   - Code: VoiceTask.cancel_requested flag + timeout checking in runner loop

4. **HTML Parsing Failure**: Return visible-text blocks instead of exact HTML; agent can target by text
   - Location: `confluence_logic/utils/html_parser.py::_resolve_visible_text_target()`
   - Code: Search for text in page, match first/best block

5. **Ambiguous Page References**: Reframer returns "clarify" action if page lookup yields 0 results
   - Location: `confluence_logic/agents/reframer_agent.py`
   - Code: Lists recent pages, asks user for explicit page name if no plausible match

6. **Tool Invocation Errors**: Agents catch tool failures, return "ERROR: ..." message instead of silent failure
   - Location: Agent instructions in `confluence_logic/agents/editor_agent.py`
   - Pattern: "Do NOT ask questions. If search returns zero results, return 'ERROR: Could not find target page.'"

## Cross-Cutting Concerns

**Logging:**
- Uses Python standard logging with INFO level
- All logger calls use module-scoped logger: `logger = logging.getLogger(__name__)`
- Key logging: bot lifecycle events (bot creation, connection), tool invocations (search, commit), errors

**Validation:**
- Pydantic BaseModel schemas for all data crossing boundaries
- Schema validation in: SearchResponse, LivePageResponse, PreviewResponse, CommitResponse, ResolverDecision
- Tool functions validate input types; Confluence API responses parsed/validated before return

**Authentication:**
- Environment variables for credentials: ATLASSIAN_USER_EMAIL, ATLASSIAN_API_TOKEN, ATLASSIAN_DOMAIN, RECALL_API_KEY, PINECONE_API_KEY
- HTTP Basic Auth for Confluence API (HTTPBasicAuth with email + token)
- Token bearer for Recall.ai API (Authorization: Token header)

**Concurrency:**
- AsyncIO for WebSocket handling + agent task execution
- AsyncLock (`_state_lock`, `_output_lock`) protects shared state (meeting_state dict) during concurrent task updates
- Thread-based TTS/speak operations (separate threads don't block main event loop)

**Context Variables:**
- `_tool_run_state`: ContextVar maintaining tool invocation state across async boundaries
- `_mutation_observer`: ContextVar for optional debugging callback on page mutations
- Enable per-task isolation in multi-request scenarios

---

*Architecture analysis: 2026-04-10*
