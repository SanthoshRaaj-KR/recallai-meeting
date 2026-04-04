# Codebase Concerns

**Analysis Date:** 2026-04-04

## Tech Debt

**Global mutable state for meeting management:**
- Issue: Meeting state is stored in a plain dictionary (`meeting_state`) at module level, modified across multiple threads without synchronization
- Files: `jarvis.py` (lines 62-67)
- Impact: Race conditions possible when multiple transcript events or queries execute concurrently. State corruption could cause bot to miss wake words, respond at wrong times, or lose transcript data
- Fix approach: Replace dict with a thread-safe class using locks or use `asyncio.Queue` for transcript events; consider Redis for distributed state if scaling

**Bare exception handlers everywhere:**
- Issue: Five instances of generic `except Exception as e` that catch all errors and log them, but don't distinguish between recoverable and fatal errors
- Files: `jarvis.py` (lines 112, 133, 149, 248, 340)
- Impact: Unknown errors get masked as "try again" messages. Hard to debug production failures. API errors, network timeouts, and permission issues all treated identically
- Fix approach: Catch specific exception types (requests.Timeout, json.JSONDecodeError, OpenAI exceptions). Re-raise or fail fast for unrecoverable errors

**Hardcoded maximum tool-call loops:**
- Issue: Tool-calling loop in `handle_query()` has fixed limit of 5 iterations with no timeout or cost tracking
- Files: `jarvis.py` (line 217)
- Impact: Runaway tool loops could consume unlimited OpenAI tokens. No safeguard against infinite loops if a tool keeps requesting itself
- Fix approach: Add token budget tracking, implement exponential backoff for tool retries, set max cost per query

**Temp file cleanup relies on exception handling:**
- Issue: Temp audio file cleanup in `speak()` only happens in finally block, but file path is hardcoded
- Files: `jarvis.py` (lines 119-138)
- Impact: If multiple concurrent speak() calls happen, they all use same `temp_jarvis.mp3` filename. Last call wins, first file may be deleted prematurely or overwritten
- Fix approach: Use unique temp filenames (uuid4 or tempfile module), ensure atomic write-delete

**No validation of API responses:**
- Issue: Code assumes Recall.ai and OpenAI responses have expected structure. No validation of response JSON before accessing nested keys
- Files: `jarvis.py` (lines 107, 226, 295-298)
- Impact: If Recall.ai changes response format or returns error response, bot crashes on KeyError
- Fix approach: Add response schema validation, check status codes before parsing, use `.get()` with defaults

## Known Bugs

**WebSocket handler race condition on bot_id:**
- Symptoms: Bot may receive transcript before `bot_id` is set in state, causing speech calls to fail silently
- Files: `jarvis.py` (lines 316-318)
- Trigger: Transcript data arrives during the gap between bot creation and state assignment (lines 391-392)
- Workaround: Brief delay in `main()` before meeting_state["bot_id"] = bot_id, but not guaranteed to prevent race

**Jarvis listening state not reset on timeout:**
- Symptoms: After bare "Hey Jarvis" wake word, if user doesn't speak within a sentence, bot stays in listening mode. Next speech triggers incorrectly as query continuation
- Files: `jarvis.py` (lines 330, 333-336)
- Trigger: User says "Hey Jarvis" then pauses for several seconds before speaking naturally (not directed at bot)
- Workaround: User must explicitly say another wake word or restart bot

**Tool result truncation in logs only, not user-facing:**
- Symptoms: Weather results or meeting summaries truncated in logs at 80 chars (line 234), but full result sent to OpenAI - inconsistent for debugging
- Files: `jarvis.py` (line 234)
- Trigger: Any tool that returns result > 80 chars
- Workaround: Manual inspection of API calls

## Security Considerations

**API credentials exposed in environment:**
- Risk: RECALL_API_KEY, OPENAI_API_KEY stored in .env file. If .env is committed or copied insecurely, credentials leak
- Files: `.env` file (not read by scanner, exists in root), jarvis.py loads via `load_dotenv()`
- Current mitigation: .gitignore should exclude .env (verify in codebase), .env.example provided
- Recommendations:
  - Verify .gitignore contains `.env*` entries
  - Use AWS Secrets Manager or HashiCorp Vault in production
  - Rotate API keys immediately if .env is ever exposed
  - Add pre-commit hook to prevent .env commits

**WebSocket endpoint not authenticated:**
- Risk: `/recall-audio-stream` WebSocket endpoint accepts any connection without verifying sender is legitimate Recall.ai
- Files: `jarvis.py` (lines 283-285)
- Current mitigation: Recall.ai should be authenticating the connection, but no validation on bot side
- Recommendations:
  - Verify RECALL_WEBHOOK_SECRET and validate signature of incoming WebSocket messages
  - Implement token-based auth on WebSocket (pass token in query param or header)
  - Whitelist Recall.ai IP ranges if possible

**No rate limiting on OpenAI API calls:**
- Risk: Any transcript that contains wake word triggers OpenAI call. Malicious actor in meeting could spam "Hey Jarvis" to exhaust API quota and run up bills
- Files: `jarvis.py` (lines 320-327, 333-336)
- Current mitigation: OpenAI quota/billing alert
- Recommendations:
  - Implement per-user rate limiting (deduplicate wake words in same second)
  - Add cooldown period between queries
  - Set OpenAI API spend limits
  - Log all queries for audit trail

**Transcript log grows unbounded in memory:**
- Risk: `meeting_state["transcript_log"]` appends all spoken words from entire meeting without pruning. Memory leak for long meetings
- Files: `jarvis.py` (lines 310-314)
- Current mitigation: None
- Recommendations:
  - Implement circular buffer or max-size queue (keep last N entries)
  - Persist transcript to disk periodically
  - Add configurable transcript retention policy

## Performance Bottlenecks

**Synchronous OpenAI API calls block WebSocket handler:**
- Problem: `handle_query()` runs synchronously in a daemon thread but makes blocking HTTP request to OpenAI (line 218). Multiple simultaneous queries will create thread pool exhaustion
- Files: `jarvis.py` (lines 199-256, 218-240)
- Cause: No connection pooling, thread-per-request model, no async client
- Improvement path:
  - Use OpenAI async client or asyncio+aiohttp
  - Implement request queue with worker pool
  - Add concurrent request limiting

**Text-to-speech generation blocks WebSocket for ~1-2s per response:**
- Problem: gTTS network request in `speak()` blocks the thread that's handling it, freezing WebSocket message processing
- Files: `jarvis.py` (lines 117-138, lines 121-124)
- Cause: Network I/O not async
- Improvement path:
  - Pre-cache common responses (acknowledgments, error messages)
  - Use async gTTS or switch to faster TTS provider
  - Implement response streaming instead of waiting for full audio generation

**Transcript log search is O(n) for every response:**
- Problem: `_get_meeting_transcript()` joins entire transcript list every time it's called (line 157)
- Files: `jarvis.py` (lines 153-158)
- Cause: No indexing or caching
- Improvement path:
  - Cache transcript as a single string, append only on new entries
  - Implement semantic search for relevant context instead of full transcript

## Fragile Areas

**Tool calling loop can enter infinite state:**
- Files: `jarvis.py` (lines 216-244)
- Why fragile: If OpenAI keeps returning tool_calls (finish_reason="tool_calls") and the tool keeps returning data, loop continues until max iterations. No cost or complexity limits
- Safe modification:
  - Always test tool definitions carefully (avoid circular tool chains)
  - Add token cost tracking and abort if exceeds threshold
  - Test with malformed tool responses
- Test coverage: No tests present; untested edge cases

**Meeting state initialization race:**
- Files: `jarvis.py` (lines 386-392)
- Why fragile: Bot creation and state assignment happen in main thread, but WebSocket listening starts immediately in daemon thread (line 374). Transcript events received before state is ready
- Safe modification:
  - Wrap bot_id assignment in a lock or use asyncio.Event
  - Ensure WebSocket handler checks for bot_id existence before using it
  - Add integration test that triggers WebSocket before bot_id set
- Test coverage: No tests present

**Wake word detection regex is fragile:**
- Files: `jarvis.py` (lines 263-266)
- Why fragile: Regex only matches "hey jarvis" or "jarvis" in English. Won't catch variations like "Jarvis?" or speech-to-text transcription errors (common with names)
- Safe modification:
  - Add fuzzy matching for name detection (Levenshtein distance)
  - Handle punctuation variations explicitly
  - Add language-specific detection rules
- Test coverage: No tests present; no test fixtures for transcription edge cases

**Hardcoded temp filename collision:**
- Files: `jarvis.py` (lines 119-138)
- Why fragile: `temp_jarvis.mp3` is global. Two concurrent speak() calls will overwrite each other's file
- Safe modification: Use `tempfile.NamedTemporaryFile()` with auto-cleanup
- Test coverage: No concurrent tests

## Scaling Limits

**Single-threaded WebSocket, multi-threaded queries:**
- Current capacity: Limited by number of threads Python can spawn. ~100-200 concurrent tool calls before thread pool exhaustion
- Limit: One meeting per bot_id. Multiple simultaneous meeting URLs require separate bot instances
- Scaling path:
  - Migrate to async/await with asyncio
  - Use FastAPI dependency injection for connection pooling
  - Implement multi-bot manager that spawns separate processes

**In-memory transcript storage:**
- Current capacity: Transcript_log is a Python list. For 1-hour meeting with continuous speech, ~10KB-100KB
- Limit: 8+ hour meetings approach memory limits on resource-constrained machines (< 512MB)
- Scaling path:
  - Persist to SQLite or PostgreSQL
  - Implement sliding window of last N minutes in memory
  - Archive old meetings to S3

**OpenAI token costs unbounded:**
- Current capacity: No limits set. GPT-4o-mini at ~$0.01 per 1M tokens
- Limit: Malicious actor could run up $1000+ bill in hours with repeated wake words and large transcripts
- Scaling path:
  - Implement token budget per meeting/user
  - Add OpenAI cost tracking
  - Set hard spending limits in OpenAI dashboard

## Dependencies at Risk

**pyaudio (unmaintained):**
- Risk: Project includes `pyaudio` in requirements.txt but it's not actually used (all audio comes from Recall.ai WebSocket)
- Impact: Adds build dependency that often fails on macOS (requires Portaudio C library)
- Migration plan: Remove pyaudio from requirements.txt. If audio input needed in future, use `sounddevice` or `pydub`

**openai-agents (experimental):**
- Risk: Requirements.txt includes `openai-agents` but code uses raw `openai` client, not agents library. Unused dependency
- Impact: Bloats requirements, adds maintenance burden
- Migration plan: Remove openai-agents from requirements.txt unless agent framework will be used

**ngrok dependency (implicit):**
- Risk: .env.example requires ngrok URL for webhook tunneling, but ngrok is not in requirements.txt and setup instructions missing
- Impact: New developers won't know to install ngrok, setup will fail with cryptic "WEBHOOK_URL not found"
- Migration plan: Add ngrok to requirements.txt or add explicit setup instructions in README

## Missing Critical Features

**No meeting lifecycle management:**
- Problem: Bot joins meeting but has no way to gracefully leave. Ctrl+C terminates Python process, but doesn't tell Recall.ai to stop the bot
- Blocks: Long-lived meeting bots can't be cleanly shut down without API call
- Recommendation: Implement DELETE /bot/{bot_id} call on shutdown

**No error recovery:**
- Problem: If Recall.ai service is down, bot creation fails and whole program exits
- Blocks: Can't handle temporary service outages
- Recommendation: Implement exponential backoff retry for bot creation

**No conversation context between queries:**
- Problem: Each query is stateless. User asks "What is X?" then "Is it the same as Y?" — bot doesn't know X was discussed
- Blocks: Multi-turn conversations impossible
- Recommendation: Maintain conversation history in meeting_state, pass to OpenAI messages

**No logging to persistent storage:**
- Problem: All logs printed to stdout only. If process dies, logs are lost
- Blocks: Can't debug production issues or audit what happened
- Recommendation: Add file-based logging with rotation, or send to cloud logging service

## Test Coverage Gaps

**No unit tests:**
- What's not tested: Individual functions (weather fetching, transcript extraction, wake word detection, tool execution)
- Files: `jarvis.py` (entire file)
- Risk: Regressions introduced with confidence. Wake word regex changes could silently break detection
- Priority: High - at minimum, test wake word pattern and transcript parsing

**No integration tests:**
- What's not tested: Full end-to-end flow (WebSocket message → wake word detection → tool call → response → speak)
- Files: `jarvis.py` (entire file)
- Risk: Deploy broken bots to production. No test fixtures for various Recall.ai message formats
- Priority: High - add fixtures for Recall.ai API responses and test message handling

**No concurrent query tests:**
- What's not tested: Multiple simultaneous "Hey Jarvis" queries, race conditions on meeting_state
- Files: `jarvis.py` (lines 62-67, 316-318, 333-336)
- Risk: Concurrency bugs only surface in production under load
- Priority: Medium - add thread-based stress tests

**No mock for external services:**
- What's not tested: Behavior when Recall.ai, OpenAI, or weather service fail
- Files: `jarvis.py` (lines 73-114, 144-150, 218-240)
- Risk: No way to test error messages or fallback behavior without real API keys
- Priority: Medium - add mocking framework (unittest.mock) and error response fixtures

---

*Concerns audit: 2026-04-04*
