# Codebase Structure

**Analysis Date:** 2026-04-04

## Directory Layout

```
recallai-meeting/
├── jarvis.py                          # Main application (single-file monolith)
├── requirements.txt                   # Python dependencies
├── .env.example                       # Environment variable template
├── .env                               # Actual environment variables (secrets, not committed)
├── .gitignore                         # Git ignore rules
├── bot_id.txt                         # Stores bot ID from last successful run
├── interview_report_20260404_091721.md # Interview/analysis report
└── .planning/                         # GSD documentation
    └── codebase/                      # Generated codebase analysis
        ├── ARCHITECTURE.md            # Architecture patterns and layers
        └── STRUCTURE.md               # This file

```

## Directory Purposes

**Project Root:**
- Purpose: Application entry point and configuration
- Contains: Single Python application file, dependency manifest, environment config
- Key files: `jarvis.py`, `requirements.txt`, `.env`

**.planning/codebase/:**
- Purpose: Auto-generated GSD codebase analysis documents
- Contains: Architecture, structure, conventions, testing, concerns analysis
- Key files: ARCHITECTURE.md, STRUCTURE.md, CONVENTIONS.md, TESTING.md, CONCERNS.md
- Generated: Yes (by GSD mapping agents)
- Committed: Yes (documentation, not code)

## Key File Locations

**Entry Points:**
- `jarvis.py`: Single entry point via `if __name__ == "__main__": main()` (line 409)
  - Invoked as: `python jarvis.py`
  - Starts: WebSocket server, bot spawning, event loop

**Configuration:**
- `.env`: Environment variables (never committed, contains secrets)
  - Required: RECALL_API_KEY, OPENAI_API_KEY, WEBHOOK_URL
  - Optional: MEETING_URL (can be provided via stdin)
  - Server config: APP_HOST, APP_PORT
  - See `.env.example` for template

- `.env.example`: Template showing all configuration options
  - Not a real config; used as documentation
  - Includes comments explaining each variable

**Core Logic:**
- `jarvis.py`: All application logic in single file
  - Lines 34-56: Configuration & initialization
  - Lines 62-67: Meeting state definition
  - Lines 73-114: Recall.ai integration (create_bot)
  - Lines 117-138: Output layer (speak via TTS)
  - Lines 144-193: Tools (weather, meeting summary)
  - Lines 199-256: Query handler with agentic loop
  - Lines 263-277: Wake-word detection
  - Lines 283-341: WebSocket event handler
  - Lines 344-351: Health check endpoint
  - Lines 357-410: Server startup & main orchestration

**Artifacts:**
- `bot_id.txt`: Runtime artifact storing the bot ID (line 395)
  - Generated: During bot creation
  - Committed: No (in .gitignore)
  - Purpose: Allows external tools to reference the bot

**Dependencies:**
- `requirements.txt`: Python package manifest
  - Contains: fastapi, uvicorn, requests, python-dotenv, gTTS, websockets, pyaudio, openai, openai-agents
  - Install: `pip install -r requirements.txt`

## Naming Conventions

**Files:**
- Pattern: snake_case.py for Python modules
- Example: `jarvis.py` (lowercase with underscores, no spaces)
- Config: `.env*` files for environment (standard convention)
- Artifacts: `bot_id.txt` (descriptive, all lowercase)

**Functions:**
- Pattern: snake_case for private functions
- Example: `_fetch_weather()`, `_get_meeting_transcript()`, `_run_tool()`
- Pattern: snake_case for public functions
- Example: `create_bot()`, `speak()`, `handle_query()`, `extract_wake_and_query()`

**Variables:**
- Pattern: UPPER_CASE for constants (environment-derived)
- Example: RECALL_API_KEY, WEBHOOK_URL, OPENAI_MODEL, APP_HOST, APP_PORT
- Pattern: snake_case for module-level mutable state
- Example: `meeting_state` (line 62)
- Pattern: snake_case for local variables
- Example: `query`, `bot_id`, `response`, `participant`

**Constants:**
- Pattern: Uppercase with underscores
- Location: Lines 40-50 (all configuration constants)
- Examples: LANGUAGE_CODE, STREAMING_MODE, BOT_NAME

**Regex Patterns:**
- Pattern: _WAKE_PATTERN for compiled regex (line 263)
- Naming: Prefix with underscore to indicate internal

**Dictionaries & Data Structures:**
- meeting_state: Global dictionary for session state (line 62)
- Structure: keys are snake_case ("bot_id", "transcript_log", "is_active", "jarvis_listening")
- Transcript entries: {participant, text, timestamp} dicts (line 310-314)

## Where to Add New Code

**New Feature (Small):**
- Primary code: Add to `jarvis.py` directly
- If feature is a tool: Add function (e.g., `_fetch_data()`) then register in TOOLS array (line 162) and _run_tool() dispatch (line 188)
- If feature is an event handler: Add endpoint to FastAPI app object (line 56) with @app.route() decorator
- Tests: Create `test_jarvis.py` in root directory

**New Component/Module (Large):**
- If extracting code becomes necessary: Create new Python file (e.g., `tools.py`, `handlers.py`)
- Pattern: Single-file preferred; split only if file exceeds ~500 lines of logic
- Import: Import from new module at top of `jarvis.py`
- Example structure:
  ```python
  # tools.py
  def get_weather(city: str) -> str: ...
  def get_meeting_summary() -> str: ...
  TOOLS = [...]

  # jarvis.py
  from tools import TOOLS, run_tool
  ```

**Utilities & Helpers:**
- Shared helpers: Keep in `jarvis.py` unless used by multiple modules
- Preference: Internal functions with _ prefix (e.g., `_fetch_weather`)
- If needed across modules: Extract to `utils.py`

**Configuration:**
- New required env vars: Add to .env.example with comment (lines 1-22)
- Add fallback with `os.getenv("VAR_NAME", default)` pattern (see line 41 for example)
- Add to configuration section at top of jarvis.py (lines 40-50)

**State Management:**
- New session state: Add to meeting_state dict (line 62)
- Follow existing pattern: Use snake_case keys
- Reset on shutdown: Unset in main() except handler (lines 404-406)

## Special Directories

**.git/:**
- Purpose: Version control metadata
- Generated: Yes (git init)
- Committed: N/A (directory excluded from commits)

**.env:**
- Purpose: Local environment secrets (RECALL_API_KEY, OPENAI_API_KEY, etc.)
- Generated: Manually by developer from .env.example
- Committed: No (in .gitignore)
- Critical: Never commit this file

**.planning/:**
- Purpose: GSD (Getting Stuff Done) planning and codebase documentation
- Generated: Yes (by GSD agents)
- Committed: Yes (documentation, no sensitive data)
- Contains: Analysis documents used by code generation agents

## Dependency Installation

**Install all dependencies:**
```bash
pip install -r requirements.txt
```

**Key dependencies and their roles:**
- fastapi: WebSocket server framework
- uvicorn: ASGI server implementation
- requests: HTTP client for Recall.ai and weather API
- python-dotenv: Load .env file into environment
- gTTS: Text-to-speech conversion
- websockets: WebSocket protocol support
- pyaudio: Audio handling (optional, for future enhancements)
- openai: OpenAI API client
- openai-agents: Framework for agentic patterns (currently unused but in requirements)

## Running the Application

**Start the bot:**
```bash
python jarvis.py
```

**Expected flow:**
1. Loads .env file (line 38)
2. Validates required env vars (lines 367-371)
3. Starts WebSocket server on background thread (line 374)
4. Prompts for meeting URL if not in MEETING_URL env var (line 379)
5. Creates bot via Recall.ai API (line 386)
6. Saves bot_id.txt (line 395)
7. Enters infinite loop awaiting Ctrl+C (lines 402-406)

**Stop the bot:**
- Press Ctrl+C in terminal
- Logs "⚠️ Jarvis shutting down." (line 405)
- Sets is_active = False (line 406)

**Health check:**
```bash
curl http://localhost:8000/health
```

Returns JSON with bot_id, active status, and transcript line count.

---

*Structure analysis: 2026-04-04*
