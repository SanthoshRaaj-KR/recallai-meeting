# Code Conventions

**Analysis Date:** 2026-05-11

## Naming Patterns

**Python Files:**
- `snake_case` for all Python module files: `jarvis_agentic.py`, `graph_rag.py`, `editor_agent.py`, `html_parser.py`, `confluence_page_graph.py`, `general_responder.py`
- Test files prefixed with `test_`: `test_flow.py`, `test_jarvis_agentic.py`, `test_review_api.py`, `test_classifier.py`, `test_confluence_page_graph.py`, `test_general_responder.py`
- Directories use `snake_case`: `confluence_logic/`, `local_office_logic/`, `agents/`, `connectors/`, `db/`

**TypeScript Files (review-ui):**
- `camelCase.ts` for utility modules: `api.ts`, `types.ts`
- `PascalCase.tsx` is not explicitly used for files — Next.js App Router pages are named `page.tsx` by convention; the exported component name uses PascalCase (`HomePage`, `ResultsPage`)
- `globals.css`, `layout.tsx` follow Next.js App Router naming conventions

**Classes:**
- `PascalCase`: `EditorAgent`, `ReframerAgent`, `ProposedChangesAgent`, `ConfluenceConnector`, `PineconeStore`, `IngestionPipeline`
- Pydantic model classes `PascalCase`: `CandidatePage`, `SearchResponse`, `MasterVoiceDecision`, `CommitResponse`
- Abstract base classes use `PascalCase` without `Abstract` prefix: `DocumentFetcher`, `DocumentPusher`

**Functions and Methods:**
- `snake_case` for all functions: `search_workspace_knowledge`, `fetch_live_page`, `classify_intent`, `user_from_bearer`, `upsert_history`, `query_user_confluence_graph`
- Private helpers prefixed with `_`: `_candidate_from_metadata`, `_commit_with_retry`, `_is_version_conflict`, `_reindex_in_background`, `_rest_headers`, `_db_key`
- Private async helpers follow same `_` convention: `_run_editor`, `_handle_general_question`, `_speak_guarded`, `_needs_web_search`
- Async coroutines named descriptively with verbs: `ingest_transcript_entry`, `query_context`, `answer_general_question`

**Variables and Constants:**
- `snake_case` for local variables and module-level mutable state
- `UPPER_SNAKE_CASE` for module-level constants and env-var-derived config:
  ```python
  JARVIS_TTS_PROVIDER = os.getenv("JARVIS_TTS_PROVIDER", "edge_tts").strip().lower()
  JARVIS_REVIEW_MODEL = os.getenv("JARVIS_REVIEW_MODEL", "gpt-5-mini").strip()
  _RECALL_STATUS_CACHE_SECONDS = float(os.getenv("RECALL_STATUS_CACHE_SECONDS", "4.0"))
  GENERAL_RESPONDER_MODEL = os.getenv("JARVIS_GENERAL_MODEL", "gpt-4o-mini").strip()
  TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "").strip()
  ```
- Module-level singletons use `_` prefix: `_store`, `_connector`, `_driver`, `_openai_client`
- ContextVar names use `snake_case` string labels: `ContextVar("tool_run_state")`, `ContextVar("confluence_graph_user_id")`

**TypeScript Types (review-ui):**
- `PascalCase` for interfaces and type aliases: `SessionStatus`, `ChangeItem`, `MeetingSummary`, `ActionItem`, `BotStatus`, `ChangeType`
- `camelCase` for properties: `bot_id`, `change_count`, `page_id`, `section_heading` (snake_case mirroring the Python API response)
- Union string types used for discriminated state: `BotStatus = "idle" | "joining" | "in_meeting" | "ended" | "error"`

## Code Style

**Python Formatting:**
- No linting config files detected (`.flake8`, `.pylintrc`, `pyproject.toml`, `ruff.toml` are absent)
- Indentation: 4 spaces (standard Python)
- Line length: not enforced by config; lines up to ~120 characters observed in practice
- Trailing commas: not consistently used in function calls

**TypeScript / React Formatting (review-ui):**
- No `.prettierrc` detected; ESLint via `eslint-config-next` provides baseline rules
- 2-space indentation (Next.js default)
- `"use client"` directive at top of all interactive pages (`page.tsx`, `results/page.tsx`) — required for `useState`/`useEffect`

**Python Imports:**
- No `isort` or `black` config detected; formatting is manual
- Blank lines between top-level functions and classes: 2 blank lines (standard PEP 8)
- Within classes: 1 blank line between methods

**Docstrings:**
- Module-level docstrings used for key modules — plain triple-quoted strings describing module purpose and design decisions:
  ```python
  """
  Graph RAG module — Neo4j knowledge graph over meeting transcript.

  Owns: Neo4j connection lifecycle, real-time entity ingestion from transcript entries,
  and context query for general question answering.

  Per D-10: Neo4j AuraDB (cloud), configured via NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD env vars.
  """
  ```
- Test file docstrings include ticket/spec IDs: `"""Tests for graph_rag — Neo4j Graph RAG module (GRAPHRAG-01)."""`
- Individual functions: short single-line docstrings where present, or no docstring for simple helpers
- No Google/NumPy/Sphinx docstring style enforced — informal prose only

**String Formatting:**
- f-strings used universally for interpolation: `f"Version conflict max retries exceeded for page {page_id}: {ve}"`
- `%`-style used exclusively for `logger` calls: `logger.warning("Conflict on attempt %d/%d for page %s", attempt, max, page_id)`

## Module Organization

**Python package structure mirrors logical domains:**
- `confluence_logic/` — Confluence-specific pipeline (agents, connectors, ingestion, review)
- `local_office_logic/` — Local Office files pipeline (parallel structure to `confluence_logic/`)
- `tests/` — top-level evaluation harnesses (not pytest unit tests)

**Within each domain package:**
```
<domain>/
├── agents/          # Agent classes, tool functions decorated with @function_tool
├── connectors/      # External API clients (Confluence API, local file system)
├── core/            # interfaces.py (ABCs), models.py (Pydantic data models), schemas.py (request/response schemas)
├── db/              # vector_store.py — Pinecone wrapper
├── ingestion/       # doc_pipeline.py — ingestion pipeline
├── review/          # api.py (FastAPI APIRouter), supabase_store.py, supabase_schema.sql
├── scripts/         # standalone utility scripts
├── tests/           # pytest unit tests for this domain
└── utils/           # html_builder.py, html_parser.py, sandbox.py, etc.
```

**review-ui App Router structure:**
```
review-ui/src/
├── app/             # Next.js App Router pages
│   ├── layout.tsx   # Root layout with metadata + Tailwind body classes
│   ├── page.tsx     # Home page (bot join, polling)
│   ├── globals.css  # Tailwind @tailwind directives
│   └── results/
│       └── page.tsx # Results page (summary + change review)
├── lib/
│   └── api.ts       # Typed fetch wrappers — all API calls go through here
└── types.ts         # All shared TypeScript types
```

**Singleton/lazy init pattern for expensive clients:**
```python
_store = None
def get_store():
    global _store
    if _store is None:
        _store = PineconeStore()
    return _store
```
This pattern is used consistently in `confluence_logic/agents/tools.py`, `confluence_logic/classifier.py`, `confluence_logic/graph_rag.py`, `confluence_logic/general_responder.py`, and `confluence_logic/review/api.py`.

**`@function_tool` decorator for agent tools:**
All callable tools exposed to OpenAI Agents SDK are decorated with `@function_tool` from `agents` package.
Docstrings on `@function_tool` functions serve as the LLM-visible tool description — keep them accurate and concise.

**ContextVar for per-request state:**
Tool run state, mutation observers, and the Confluence graph user ID use `ContextVar` to remain thread/task-safe:
```python
_tool_run_state: ContextVar[dict] = ContextVar("tool_run_state")
_mutation_observer: ContextVar[Optional[Callable]] = ContextVar("mutation_observer", default=None)
_current_graph_user_id: ContextVar[str] = ContextVar("confluence_graph_user_id", default="")
```

## review-ui API Client Pattern

All API calls in the review-ui go through the typed `request<T>()` helper in `review-ui/src/lib/api.ts`:
```typescript
async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const res = await fetch(`/api${path}`, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}: ...`);
  return res.json() as Promise<T>;
}
```
All exported functions (`startBot`, `getSession`, `getChanges`, `executeChanges`, `getMeetingSummary`) are thin wrappers over this helper. Do not call `fetch` directly in page components.

## Error Handling

**Python — Pattern: catch-log-return-safe-value** — functions catch broad `Exception`, log it, and return a typed failure object rather than raising:
```python
except Exception as e:
    logger.error(f"Search failed: {e}")
    return SearchResponse(candidates=[], message=f"Error: {e}")
```

**Python — ValueError for domain-specific failures** (e.g. version conflicts, duplicate headings):
```python
if _is_version_conflict(ve):
    message = f"ConflictError: {str(ve)} Please re-fetch_live_page."
return CommitResponse(success=False, version=None, message=message)
```

**Python — Graceful degradation for optional integrations:**
- Pinecone/Neo4j unavailability caught with `logger.warning` — falls back to live Confluence API results
- TTS provider failures fall back to gTTS
- Classifier LLM failures default to `"confluence"` intent
- Tavily web search failures return empty context (non-fatal, logged at DEBUG level)

**TypeScript (review-ui) — try/catch in async handlers:**
```typescript
try {
  await startBot(meetingUrl.trim());
  setPageState("active");
} catch (err) {
  setErrorMessage(err instanceof Error ? err.message : "Failed to start bot");
  setPageState("error");
}
```
Polling errors in `useEffect` intervals are silently swallowed to keep retrying.

**No custom exception classes** — Python code uses builtins (`ValueError`, `Exception`) and communicates failure through typed response objects.

**`logger` used at module level** via `logging.getLogger(__name__)` — every module that performs I/O or agent calls has its own logger.

## Type Annotations

**Python Style:** Python 3.9+ style using `typing` module imports (`List`, `Optional`, `Dict`, `Callable`, `Tuple`)
- `from __future__ import annotations` used in `confluence_logic/review/api.py` and `supabase_store.py` only
- Not yet migrated to PEP 604 union syntax (`X | Y`) or built-in generics (`list[X]`)

**Coverage:**
- Function signatures in `agents/tools.py` and `agents/editor_agent.py` are fully annotated
- Return types annotated for public functions: `-> SearchResponse`, `-> CommitResponse`, `-> str`, `-> bool`
- Private helpers sometimes annotated, sometimes not
- `confluence_logic/core/models.py` and `schemas.py` are fully typed via Pydantic `BaseModel`
- `confluence_logic/core/interfaces.py` uses ABCs with annotated abstract methods
- `review/supabase_store.py` is fully annotated

**Pydantic usage:**
- All agent input/output schemas defined as `pydantic.BaseModel` subclasses in `core/schemas.py`
- `Optional` fields use `= None` default: `heading: Optional[str] = None`
- `model_validate()` used for structured LLM output: `MasterVoiceDecision.model_validate(result.final_output)`

## Import Organization

**Python Order (observed, not enforced by tooling):**
1. Standard library: `import asyncio`, `import logging`, `import os`, `from typing import ...`
2. Third-party: `from fastapi import ...`, `from openai import OpenAI`, `from pydantic import BaseModel`
3. Internal (relative): `from ..core.schemas import ...`, `from ..agents.tools import ...`

**Relative imports** used for intra-package imports:
```python
from ..db.vector_store import PineconeStore
from ..connectors.confluence import ConfluenceConnector
from ..core.schemas import SearchResponse, CandidatePage
from ..utils.html_parser import delete_content_in_section, edit_block_in_section
```

**Absolute imports** used for inter-domain and top-level module references:
```python
from confluence_logic import jarvis_agentic
from confluence_logic.agents.proposed_changes_agent import ProposedChangesAgent
from confluence_logic import confluence_page_graph
```

**Deferred imports inside functions** used to avoid circular imports or large startup costs:
```python
def create_confluence_page(...):
    from ..utils.html_builder import build_page_html
    ...
```
Also seen in test files for test-scoped imports:
```python
def test_create_confluence_page_tool(...):
    from confluence_logic.agents.tools import create_confluence_page
```

**`dotenv` loading** done at module top-level in files that read env vars:
```python
from dotenv import load_dotenv
load_dotenv()
```

**TypeScript imports (review-ui):**
- Types imported with `import type` syntax: `import type { MeetingSummary, ChangeItem } from "@/types"`
- Path alias `@/` maps to `src/` (configured in `tsconfig.json`)
- API functions imported by name from `@/lib/api`: `import { startBot, getSession } from "@/lib/api"`
