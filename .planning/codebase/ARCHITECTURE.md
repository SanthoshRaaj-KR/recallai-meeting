# Architecture

**Analysis Date:** 2026-04-04

## Pattern Overview

**Overall:** Event-driven monolithic agent architecture with real-time streaming and tool-calling intelligence.

**Key Characteristics:**
- Single-file Python application with clear functional separation
- Asynchronous WebSocket-based event handling
- Agentic loop with OpenAI function-calling tools
- Multi-threaded request handling for long-running operations
- Stateful in-memory session management

## Layers

**API & Transport Layer:**
- Purpose: Accept incoming WebSocket connections from Recall.ai, expose health endpoint
- Location: `jarvis.py` (lines 283-351)
- Contains: FastAPI WebSocket handler, HTTP health endpoint
- Depends on: FastAPI, WebSocket protocol
- Used by: Recall.ai bot streaming service, external health checks

**Event Processing & State Management:**
- Purpose: Parse incoming transcript events, maintain meeting state, coordinate wake-word detection
- Location: `jarvis.py` (lines 62-67, 283-336)
- Contains: Meeting state dictionary, WebSocket event loop, transcript accumulation
- Depends on: API layer for event input
- Used by: Wake word detection, query routing

**AI Intelligence Layer:**
- Purpose: Process queries through agentic loop with tool calling
- Location: `jarvis.py` (lines 199-256)
- Contains: Agentic loop with OpenAI tool-calling integration, system prompt management
- Depends on: OpenAI API, tool definitions
- Used by: WebSocket handler after wake-word detection

**Tool Execution Layer:**
- Purpose: Execute available tools (weather, meeting summary) on behalf of the agent
- Location: `jarvis.py` (lines 144-193)
- Contains: Tool implementations (_fetch_weather, _get_meeting_transcript), tool registry (TOOLS), tool router (_run_tool)
- Depends on: External services (wttr.in), meeting state
- Used by: AI Intelligence layer during agentic loop

**Output & Speech Layer:**
- Purpose: Convert AI responses to speech and inject into meeting
- Location: `jarvis.py` (lines 117-138)
- Contains: Text-to-speech conversion (gTTS), audio encoding, Recall.ai audio API client
- Depends on: gTTS library, Recall.ai API, file I/O
- Used by: AI Intelligence layer, WebSocket handler for acknowledgments

**Integration Layer:**
- Purpose: Interface with external services (Recall.ai, OpenAI)
- Location: `jarvis.py` (lines 73-114, 100-131)
- Contains: Bot creation (create_bot), speech output (speak), API authentication headers
- Depends on: requests library, environment variables
- Used by: Main entry point, AI layer, output layer

**Orchestration & Configuration Layer:**
- Purpose: Bootstrap the application, coordinate component initialization
- Location: `jarvis.py` (lines 34-56, 357-410)
- Contains: Environment loading, configuration constants, main entry point, server startup
- Depends on: All other layers
- Used by: Python runtime entry point

## Data Flow

**Query Processing Flow:**

1. Recall.ai bot joins meeting, WebSocket connects to `/recall-audio-stream`
2. WebSocket handler receives `transcript.data` event (line 292)
3. Extract participant name and sentence from nested data structure (lines 296-298)
4. Append to meeting state transcript log (lines 310-314)
5. Run wake-word regex pattern on sentence (line 320)
6. If wake-word detected with query: spawn thread to handle_query (line 327)
7. If bare wake-word only: set jarvis_listening flag and speak "Yes?" (lines 330-331)
8. If jarvis_listening true and new sentence arrives: spawn thread to handle_query on that sentence (line 336)

**Agentic Loop (within handle_query):**

1. Create system prompt defining Jarvis behavior (lines 203-208)
2. Initialize messages list with system prompt and user query (lines 210-213)
3. Enter max-5-iteration tool-calling loop (line 217)
4. Call OpenAI chat API with TOOLS definition (lines 218-224)
5. Check finish_reason: if "tool_calls", extract and execute tool (lines 229-239)
6. Append tool result back to messages as "tool" role (line 235-239)
7. Loop continues until finish_reason is "stop" (lines 240-244)
8. Extract final answer text from response (line 241)
9. Speak answer into meeting via speak() function (line 243)
10. Return from handle_query

**State Management:**

Meeting state is a global dictionary (lines 62-67):
- `bot_id`: ID of spawned bot (set during create_bot, used throughout)
- `transcript_log`: Array of {participant, text, timestamp} objects (accumulated, queried by tools)
- `is_active`: Flag indicating bot is in meeting (set by main, unset on shutdown)
- `jarvis_listening`: Temporary flag for multi-chunk queries (set/unset in WebSocket handler)

## Key Abstractions

**Wake-Word Pattern:**
- Purpose: Detect "Hey Jarvis" or "Jarvis" in any sentence, extract trailing query
- Location: `jarvis.py` (lines 263-277)
- Pattern: Regex using named group capturing (re.IGNORECASE)
- Examples: "hey jarvis" → query="", "Hey Jarvis, what's the weather?" → query="what's the weather?"

**Tool Registry:**
- Purpose: Define OpenAI function-calling interface for two capabilities
- Location: `jarvis.py` (lines 162-185)
- Pattern: Array of tool objects with function definitions matching OpenAI schema
- Tools: get_weather (city parameter), get_meeting_summary (no parameters)

**Tool Router:**
- Purpose: Dispatch tool calls from OpenAI to implementation functions
- Location: `jarvis.py` (lines 188-193)
- Pattern: Simple string dispatch on function name
- Extensibility: Add new if/elif branch for new tools

**Transcript Accumulation:**
- Purpose: Maintain running log of all meeting participants' speech
- Location: `jarvis.py` (lines 153-158, 310-314)
- Pattern: Append-only list; query via _get_meeting_transcript()
- Used for: Meeting summary tool, context for AI

## Entry Points

**Main Process Entry:**
- Location: `jarvis.py` (lines 361-410)
- Triggers: `python jarvis.py`
- Responsibilities:
  - Validate environment variables (RECALL_API_KEY, OPENAI_API_KEY, WEBHOOK_URL)
  - Start FastAPI WebSocket server on background thread (line 374)
  - Accept meeting URL (from env or stdin)
  - Call create_bot() to spawn Recall.ai bot (line 386)
  - Store bot_id in memory and bot_id.txt file (lines 391-395)
  - Enter infinite sleep loop for signal handling (lines 402-406)

**WebSocket Entry:**
- Location: `jarvis.py` (lines 283-341)
- Triggers: Recall.ai bot connects to /recall-audio-stream
- Responsibilities:
  - Accept WebSocket connection
  - Loop on incoming JSON events
  - Filter for transcript.data events
  - Extract participant + sentence
  - Ignore bot's own speech
  - Append to transcript log
  - Detect wake word
  - Spawn handle_query threads

**Health Check Entry:**
- Location: `jarvis.py` (lines 344-351)
- Triggers: GET /health
- Responsibilities:
  - Return meeting state snapshot (bot_id, is_active, transcript length)

## Error Handling

**Strategy:** Defensive with fallback responses and logging

**Patterns:**

1. **Bot Creation Errors (lines 99-114):**
   - Wrap requests.post in try/except
   - Check status code (200 or 201)
   - Log error and return None on failure
   - Main entry calls sys.exit(1) if bot_id is None

2. **Tool Execution Errors (lines 144-150, 188-193):**
   - Wrap external calls (wttr.in) in try/except
   - Return error message string instead of raising
   - Agentic loop passes error string to OpenAI for recovery

3. **Query Handling Errors (lines 215-256):**
   - Wrap entire agentic loop in try/except
   - Detect quota errors via string contains (lines 251-252)
   - Detect auth errors via string contains (lines 253-254)
   - Speak appropriate error message back to meeting
   - Log full exception for debugging

4. **WebSocket Errors (lines 338-341):**
   - Catch WebSocketDisconnect separately for clean shutdown
   - Catch generic Exception for unexpected errors
   - Log error but don't crash (allow reconnect)

5. **Speech Errors (lines 120-138):**
   - Try/except around TTS conversion and file I/O
   - Check response status code (200 required)
   - Cleanup temp file in finally block
   - Return boolean success status

## Cross-Cutting Concerns

**Logging:**
- Framework: Python logging module configured at INFO level (line 52)
- Pattern: All significant events logged with descriptive emoji prefixes (✅, ❌, 🧠, 🔧, 🤖, 💬, 🔌)
- Usage: Configuration validation, bot lifecycle, query processing, errors

**Validation:**
- Configuration: Environment variable checks in main() (lines 367-371)
- Event data: Check event type == "transcript.data" (line 292), extract participant name safely (line 296)
- Input: Wake-word regex pattern (line 320), empty sentence filtering (lines 300-301)

**Authentication:**
- Recall.ai: Token in Authorization header: f"Token {RECALL_API_KEY}" (lines 102, 128)
- OpenAI: API key from environment, used by OpenAI() client (line 55)
- Pattern: Keys from environment variables, not hardcoded

**Thread Safety:**
- State: meeting_state is global dict, accessed from main thread and WebSocket event threads
- Pattern: No explicit locking; assumes GIL protects simple dict ops
- Risk: Non-atomic multi-field updates could race (e.g., bot_id + is_active at line 391-392)

---

*Architecture analysis: 2026-04-04*
