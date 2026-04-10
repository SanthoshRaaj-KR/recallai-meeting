# Coding Conventions

**Analysis Date:** 2026-04-10

## Naming Patterns

**Files:**
- Module files use snake_case: `confluence.py`, `html_parser.py`, `editor_agent.py`
- Test files follow pytest convention: `test_flow.py`, `test_jarvis_agentic.py`
- Special files: `__init__.py` for packages

**Functions:**
- All functions use snake_case: `fetch_page_html()`, `search_workspace_knowledge()`, `edit_block_in_section()`
- Private/internal functions prefixed with underscore: `_normalize()`, `_resolve_target_html()`, `_is_full_page_mode()`, `_emit_mutation_started()`
- Tool functions decorated with `@function_tool`: `search_workspace_knowledge`, `fetch_live_page`, `commit_document_edit`
- Handler functions follow pattern: `handle_query()`, `handle_voice_query()`, `handle_prepared_query()`

**Variables:**
- Regular variables use snake_case: `page_id`, `expected_version`, `heading_string`, `candidates_by_page`
- Module-level constants use UPPERCASE: `FULL_PAGE_SENTINELS`, `JARVIS_BUSY_ACK`, `QUEUE_ACK`
- Context variables prefix with underscore: `_store`, `_connector`, `_tool_run_state`, `_mutation_observer`

**Types:**
- Pydantic models use PascalCase: `CandidatePage`, `SearchResponse`, `LivePageResponse`, `CommitResponse`, `MasterVoiceDecision`, `ResolverDecision`
- Abstract base classes use PascalCase: `DocumentFetcher`, `DocumentPusher`
- Agent classes use PascalCase: `EditorAgent`, `ReframerAgent`, `ConfluenceConnector`, `PineconeStore`

## Code Style

**Formatting:**
- No explicit formatter configured (black/autopep8 not in requirements.txt)
- 4-space indentation observed throughout
- Line length follows Python convention (typically 100-120 characters)
- Docstrings use triple double-quotes: `"""..."""`

**Linting:**
- No explicit linter config (eslint/flake8 not in requirements.txt)
- Code follows PEP 8 conventions implicitly

**Import Organization:**
1. Standard library imports: `import os`, `import asyncio`, `import logging`
2. Third-party imports: `import requests`, `from pydantic import BaseModel`, `from agents import Agent, Runner`
3. Relative imports: `from ..core.schemas import ...`, `from ..connectors.confluence import ...`

Path organization in imports:
- Two-level relative imports for module references: `from ..db.vector_store import PineconeStore`
- Import specific classes/functions rather than modules

**Path Aliases:**
- No explicit path aliases configured
- Standard relative imports used throughout: `from ..core.schemas`, `from ..agents.tools`, `from ..utils.html_parser`

## Error Handling

**Patterns:**
- Try-except blocks wrap externally-dependent operations
- Specific exception catching when needed: `except ValueError as ve`, `except HTTPError`, `except Exception as e`
- Version conflict handling is explicit: `_is_version_conflict(error)` checks for "Version Conflict" string in error message (see `confluence_logic/agents/tools.py` lines 56-57)
- Recovery pattern for version conflicts: fetch fresh metadata, retry operation (see `commit_document_edit()` lines 373-390)
- Fallback handling: Pinecone search failures fall back to Confluence-only results (lines 141-145)
- Graceful degradation: gTTS TTS falls back when OpenAI TTS fails (`jarvis_agentic.py`)

**Common patterns:**
```python
try:
    result = get_connector().fetch_page_html(page_id)
except Exception as e:
    logger.error(f"Fetch failed: {e}")
    return LivePageResponse(..., message=str(e))
```

```python
try:
    success = get_connector().push_update(page_id, content, expected_version=expected_version)
except ValueError as ve:
    if not _is_version_conflict(ve):
        raise
    # Handle version conflict with retry logic
    refreshed_metadata = get_connector().get_page_metadata(page_id)
```

## Logging

**Framework:** Python `logging` module

**Usage:**
- Logger created per module: `logger = logging.getLogger(__name__)` (standard pattern)
- Log levels used: `logger.info()`, `logger.warning()`, `logger.error()`
- Format: f-string messages with contextual info: `logger.error(f"Search failed: {e}")`

**Patterns:**
- Warnings for non-fatal issues: `logger.warning("Pinecone search unavailable, continuing...")` (tools.py:144)
- Errors for failures: `logger.error(f"Fetch live page failed: {e}")` (tools.py:198)
- Info for key transitions: `logger.info("Bot created: %s", bot_id)` (jarvis_agentic.py:341)
- Error context includes operation and reason: `logger.error(f"Version Lock conflict during title update: {ve}")`

## Comments

**When to Comment:**
- Docstrings required on all public functions and tool functions
- Comments explain non-obvious logic or business rules
- Comments used to clarify complex HTML parsing logic (see `html_parser.py`)
- Comments document agent instructions and behavioral rules

**JSDoc/TSDoc:**
- Not applicable (Python codebase)
- Docstrings follow single-line or multi-line format depending on complexity
- Tool function docstrings are brief, single-line (within docstring): `"""Lists recent Confluence pages..."""`

**Example docstring patterns:**
```python
def fetch_page_html(self, page_id: str) -> str:
    """Get the HTML/Storage format of the associated page."""

def search_workspace_knowledge(query: str) -> SearchResponse:
    """Searches live Confluence pages first, then supplements with Pinecone if available."""

def format_page_titles_for_user(candidates: List[CandidatePage]) -> str:
    """Renders user-facing page titles without leaking internal metadata by default."""
```

## Function Design

**Size:**
- Functions typically 10-50 lines
- Tool functions (decorated with `@function_tool`) range 5-30 lines
- Complex functions like `search_workspace_knowledge()` up to 60 lines for multiple fallback paths
- Agent handler methods range 20-60 lines

**Parameters:**
- Use type hints: `def fetch_live_page(page_id: str, heading_string: Optional[str] = None) -> LivePageResponse`
- Optional parameters have default values: `heading_string: Optional[str] = None`, `limit: int = 100`
- Parameters organized: required first, then optional with defaults

**Return Values:**
- Response objects used consistently: tools return Pydantic models (`SearchResponse`, `CommitResponse`, etc.)
- Boolean returns indicate success/failure: `push_update() -> bool`, `success` field in response objects
- Multiple return values use response objects with typed fields, not tuples
- Error information returned via message field in response objects

## Module Design

**Exports:**
- Explicit function exports via `@function_tool` decorator for agent integration
- Classes exported directly (no `__all__` patterns observed)
- Module boundaries: agents, connectors, core, db, utils, ingestion, tests

**Barrel Files:**
- `__init__.py` files present but minimal/empty (standard practice)
- No explicit re-exports or barrel patterns observed
- Imports via explicit relative paths

**Package structure:**
```
confluence_logic/
├── agents/           # Agent orchestration and tool definitions
├── connectors/       # External service integrations (Confluence)
├── core/            # Interfaces, schemas, models
├── db/              # Vector store integration (Pinecone)
├── ingestion/       # Document processing pipelines
├── utils/           # HTML parsing and building utilities
└── tests/           # Unit and integration tests
```

---

*Convention analysis: 2026-04-10*
