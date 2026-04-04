# Coding Conventions

**Analysis Date:** 2026-04-04

## Naming Patterns

**Files:**
- All lowercase with underscores: `jarvis.py`
- Single-word primary module names

**Functions:**
- Lowercase with underscores (snake_case): `create_bot()`, `extract_wake_and_query()`, `_start_server()`
- Private/internal functions prefixed with underscore: `_fetch_weather()`, `_get_meeting_transcript()`, `_run_tool()`, `_start_server()`
- Verb-first naming indicating action: `create_bot()`, `handle_query()`, `speak()`

**Variables:**
- Lowercase with underscores for regular variables: `meeting_state`, `bot_id`, `RECALL_API_KEY`, `audio_b64`
- Dictionary keys use lowercase with underscores: `"participant"`, `"text"`, `"timestamp"`, `"status"`
- UPPERCASE for constants and environment configuration: `RECALL_API_KEY`, `RECALL_BASE_URL`, `WEBHOOK_URL`, `LANGUAGE_CODE`, `STREAMING_MODE`, `OPENAI_MODEL`, `APP_HOST`, `APP_PORT`, `BOT_NAME`

**Types:**
- Type hints used selectively: `Optional[str]` (from `typing` module)
- Return type annotations: `-> Optional[str]`, `-> bool`, `-> None`
- Dictionary keys not explicitly typed; structure inferred from context

## Code Style

**Formatting:**
- 4-space indentation (Python standard)
- Line length approximately 100-120 characters (some lines at 120, most under 100)
- String quotes: Double quotes for most strings, double quotes for f-strings
- Multiline dictionaries indented and formatted for readability (see TOOLS list, lines 162-185)

**Linting:**
- No linting tool configured (no `.flake8`, `.pylintrc`, or similar)
- No formatter configured (no `black` config, no `yapf` config)
- Follows PEP 8 conventions informally

**Logging:**
- `logging` module imported (line 22)
- Standard logger configured in CONFIGURATION section (lines 52-53)
- Pattern: `logger.info()`, `logger.error()` with emoji prefixes for clarity
- Examples: `logger.info(f"✅ Bot created: {bot_id}")`, `logger.error(f"❌ Bot creation failed: ...")`
- Console-style logging with ASCII emoji indicators for visual scanning

## Import Organization

**Order:**
1. Standard library imports (lines 16-23): `os`, `re`, `sys`, `json`, `time`, `base64`, `logging`, `threading.Thread`, `typing.Optional`
2. Third-party imports (lines 25-31): `requests`, `uvicorn`, `dotenv`, `gtts`, `openai`, `fastapi`
3. No relative imports within project (single-file structure)

**Path Aliases:**
- Not used (single-file codebase)

**Import Style:**
- Specific imports preferred: `from typing import Optional`, `from threading import Thread`, `from fastapi import FastAPI, WebSocket, WebSocketDisconnect`
- Module imports less common: `import requests`, `import uvicorn`, `import json`, `import os`

## Error Handling

**Patterns:**
- Try-except blocks with specific exception handling (line 100-114, 126-138, 216-256)
- Generic `Exception` catch with detailed logging of error context
- Error messages use lowercase prefixes with emoji: `logger.error(f"❌ Bot creation failed: ...")`
- Status code checking preferred over exceptions where possible (line 106-107: `if response.status_code in [200, 201]`)
- Graceful degradation: Functions return `None` or `False` on error rather than raising exceptions (lines 73-114 `create_bot()`, 117-138 `speak()`)
- Specific error message handling for API quota/auth issues (lines 251-256)
- Finally blocks for cleanup: Temporary files removed after use (lines 136-138)

## Logging

**Framework:** Standard Python `logging` module

**Patterns:**
- Logger configured once at module level (line 53): `logger = logging.getLogger(__name__)`
- All logs include context with emoji indicators:
  - `✅` for success messages
  - `❌` for errors
  - `🤖` for bot responses
  - `💬` for participant speech
  - `🔧` for tool execution
  - `🧠` for query processing
  - `🔌` for WebSocket events
  - `🚀` for startup/initialization
  - `👂` for listening state
  - `⚠️` for shutdown
- Info level for normal operations, error level for failures
- Detailed logging of API calls, tool execution, and state changes

## Comments

**When to Comment:**
- Regex pattern explained with comment (lines 262-266): Wake word detection pattern documented
- Section separators used extensively (e.g., `# ============================================================================`) for organization
- Function docstrings used for public APIs (lines 73-74, 117-118, 153-154, 269-272)
- Inline comments for complex logic: State transition explanation (lines 323-336)
- State structure documented with inline comments (lines 62-67): Dictionary structure explained

**JSDoc/TSDoc:**
- Not applicable (Python project, uses docstrings)
- Python docstrings used: Triple-quoted strings immediately after function definition
- Examples: `create_bot()` (lines 73-74), `speak()` (lines 117-118), `extract_wake_and_query()` (lines 269-272)
- Docstrings document purpose but not parameters/returns comprehensively

## Function Design

**Size:**
- Most functions between 15-40 lines
- `handle_query()` is longest at 58 lines (lines 199-256) - implements agentic loop with tool calling
- Helper functions are compact and focused: `_fetch_weather()` (7 lines), `_get_meeting_transcript()` (5 lines)

**Parameters:**
- Minimal parameters: `create_bot(meeting_url: str)`, `speak(text: str, bot_id: str)`
- Functions accessing global state rather than extensive parameter passing (e.g., `handle_query()` accesses `meeting_state`, `BOT_NAME`, `OPENAI_MODEL`)
- Type hints on all function parameters

**Return Values:**
- Explicit return types: `Optional[str]` for functions that may return None, `bool` for success indicators, `None` for void operations
- Consistent error returns: `None` or `False` indicate failure
- Successful operations return meaningful data (e.g., `bot_id` string from `create_bot()`)

## Module Design

**Exports:**
- No explicit `__all__` definition
- Single entry point: `if __name__ == "__main__": main()` (lines 409-410)
- All functions callable from module level (no class-based organization)
- Private functions prefixed with underscore to indicate internal usage

**Global State:**
- Single global state dictionary: `meeting_state` (lines 62-67)
- Global configuration constants from environment (lines 40-50)
- Global FastAPI app instance: `app = FastAPI()` (line 56)
- Global OpenAI client: `client = OpenAI()` (line 55)
- Section comments group related globals

## Code Organization

**Sections:**
- Clear section headers with full-width separators (lines 34-35, 58-59, 69-70, 141-142, etc.)
- Sections: CONFIGURATION, MEETING STATE, RECALL.AI HELPERS, TOOLS, JARVIS QUERY HANDLER, WAKE WORD DETECTION, WEBSOCKET HANDLER, ENTRY POINT
- Each section logically groups related functionality
- Sequential organization: Configuration → State → Helpers → Tools → Handlers → Entry point

---

*Convention analysis: 2026-04-04*
