# Codebase Concerns

**Analysis Date:** 2026-04-10

## Tech Debt

**Silent Exception Handling in Vector Store:**
- Issue: Bare `except Exception: pass` blocks swallow errors without logging in critical code paths
- Files: `confluence_logic/db/vector_store.py` (lines 28-30, 39-40)
- Impact: Pinecone failures, stale section deletions, and embedding operations fail silently. Difficult to diagnose production issues.
- Fix approach: Replace silent exception handlers with proper logging. At minimum, log the error class and message using `logger.error()`. Consider propagating exceptions for upsert operations since they should fail loudly.

**Global Mutable State in Confluence Tools:**
- Issue: Global variables `_store`, `_connector`, `_tool_run_state`, and `_mutation_observer` are initialized with custom logic but lack thread-safety guarantees
- Files: `confluence_logic/agents/tools.py` (lines 14-23, 25-44)
- Impact: Concurrent requests from multiple meeting sessions may corrupt shared state. Tool state mutations could be lost or mixed between parallel executions.
- Fix approach: Replace globals with dependency injection or thread-local storage. Use `contextvars` consistently (already used for `_tool_run_state` and `_mutation_observer`) for `_store` and `_connector`.

**Brittle HTML Parsing with BeautifulSoup:**
- Issue: `_resolve_visible_text_target()` and `find_bounded_section()` use text normalization that removes whitespace, making ambiguous matches more likely
- Files: `confluence_logic/utils/html_parser.py` (lines 29-61, 63-115)
- Impact: When multiple DOM elements have similar normalized text (e.g., "key goals", "Key Goals", "key goals  "), the parser raises "more than 1 block matches" error, preventing edits to pages with naturally repeated content.
- Fix approach: Improve matching heuristics to prefer exact case-preserving matches first, then normalize. Include tag type in matching logic to disambiguate repeated text in different containers.

**Version Conflict Retry Logic is Unsafe:**
- Issue: When version conflicts occur, code refreshes metadata and retries automatically without limit (commit_document_edit, commit_delete, update_page_title)
- Files: `confluence_logic/agents/tools.py` (lines 268-281, 322-339, 378-390)
- Impact: If a page is under continuous edit (e.g., multiple agents competing), retry loops could hammer Confluence API and ultimately fail with user-facing error. No exponential backoff or max attempt limits.
- Fix approach: Add max retry count (e.g., 3) and exponential backoff. Return specific error message for max retries exceeded vs. hard conflict.

**Ingestion Pipeline Lacks Transaction Semantics:**
- Issue: Document processing splits into HTML → Markdown → chunks → Pinecone upsert with no rollback if Pinecone fails after chunks are created
- Files: `confluence_logic/ingestion/doc_pipeline.py` (lines 18-82)
- Impact: If upsert fails, the cached version check (line 24-27) will be out of sync. Next run skips re-indexing incorrectly if error was transient.
- Fix approach: Wrap upsert in try-catch, clear cached version on upsert failure, or store version only after successful upsert.

**Agent Model Name Inconsistency:**
- Issue: EditorAgent initialized with hardcoded default `model="gpt-5-mini"` (line 11 in editor_agent.py), but no GPT-5 exists as of knowledge cutoff
- Files: `confluence_logic/agents/editor_agent.py` (line 11)
- Impact: Agent will fail at runtime if JARVIS_AGENT_MODEL env var is not set. Default model name is invalid.
- Fix approach: Change default to valid model like `"gpt-4o"` or `"gpt-4o-mini"`. Document fallback behavior.

**Missing Environment Variable Validation:**
- Issue: Confluence connector raises ValueError during init if credentials missing, but PineconeStore silently initializes without index if API key missing
- Files: `confluence_logic/connectors/confluence.py` (lines 16-17), `confluence_logic/db/vector_store.py` (lines 16-18)
- Impact: Inconsistent error handling. Pinecone operations gracefully degrade (by checking `hasattr(self, 'index')`), but this silent failure makes it unclear when vector search is unavailable.
- Fix approach: Standardize init behavior: either fail loudly for both or both gracefully degrade. If graceful, log clear warning on first use of vector operations.

## Known Bugs

**Bare Exception Catch in Pinecone Query:**
- Symptoms: Search queries fail silently with vague error messages
- Files: `confluence_logic/db/vector_store.py` (lines 88-105)
- Trigger: When OPENAI_EMBEDDING_DIMENSIONS or PINECONE_INDEX_DIMENSION mismatch occurs, error is raised but caller only sees logged message
- Workaround: Check dimension configuration in .env against Pinecone index settings manually

**TTS Fallback Logic May Not Be Invoked as Expected:**
- Symptoms: Bot may not speak responses if OpenAI TTS fails
- Files: `confluence_logic/jarvis_agentic.py` - TTS provider fallback logic
- Trigger: OpenAI TTS exception is caught and gTTS fallback is called, but if gTTS also fails, no second fallback exists
- Workaround: Ensure gTTS dependencies (ffmpeg) are available on system

**HTML Section Boundary Detection Assumes Well-Formed Markup:**
- Symptoms: Edits to pages with malformed or nested heading structures may target wrong section
- Files: `confluence_logic/utils/html_parser.py` (lines 93-115)
- Trigger: Pages with h1 → h3 (skipping h2) or h2 → h1 (level decrease) confuse section boundary detection
- Workaround: Ensure page content uses properly nested heading hierarchy (h1 → h2 → h3, etc.)

## Security Considerations

**Confluence API Token Passed in Memory Without Encryption:**
- Risk: API token stored in plaintext in ConfluenceConnector instance and environment variables
- Files: `confluence_logic/connectors/confluence.py` (lines 11-19)
- Current mitigation: Token in .env file (should be .gitignored) and only loaded at runtime
- Recommendations: Consider using secrets manager (e.g., AWS Secrets Manager, Vault) instead of .env. Add token expiration/rotation policy.

**User Input Not Validated for Injection Attacks in HTML Parsing:**
- Risk: Heading strings, HTML blocks from agents are used directly in DOM operations without sanitization
- Files: `confluence_logic/utils/html_parser.py` (entire file), `confluence_logic/agents/tools.py` (edit/delete operations)
- Current mitigation: BeautifulSoup HTML parser is forgiving but not injection-proof
- Recommendations: Validate heading strings are not HTML/JS injection attempts. Sanitize agent-generated HTML before committing to Confluence.

**No Rate Limiting on Confluence API Calls:**
- Risk: Multiple agents or concurrent requests could hit Confluence rate limits and cause cascading failures
- Files: `confluence_logic/agents/tools.py`, `confluence_logic/connectors/confluence.py`
- Current mitigation: None
- Recommendations: Implement token bucket or circuit breaker for Confluence API calls. Queue requests if rate limit approached.

**Bot ID Stored in Plaintext File:**
- Risk: `bot_id.txt` written unencrypted and may expose bot lifecycle management details
- Files: `jarvis.py` (line 394-395) and `confluence_logic/jarvis_agentic.py` (similar pattern)
- Current mitigation: File is not committed (check .gitignore)
- Recommendations: Use temporary file or delete after bot cleanup. Consider storing in secure session state instead.

## Performance Bottlenecks

**Synchronous HTTP Requests Block Event Loop:**
- Problem: `requests` library used for Confluence and Recall API calls without async wrappers
- Files: `confluence_logic/connectors/confluence.py` (entire file), `jarvis.py` (lines 100-138)
- Cause: Blocking I/O in FastAPI async handlers causes thread pool exhaustion under high load
- Improvement path: Use `httpx` async client or wrap requests in `asyncio.to_thread()`. Prioritize Confluence connector (most frequent calls).

**Pinecone Upsert Embeds All Chunks Synchronously:**
- Problem: `get_embeddings()` calls OpenAI API for all chunks in sequence, then upserts
- Files: `confluence_logic/db/vector_store.py` (lines 53-86)
- Cause: Large documents (100+ sections) cause 5-10 second delays per page update
- Improvement path: Batch embed requests. Call OpenAI API once with all texts instead of per-chunk. Parallelize embedding + upsert.

**HTML Parsing Rebuilds DOM for Every Edit Preview:**
- Problem: `preview_edit()` and `preview_delete()` re-parse full HTML, generate diff, and re-parse again
- Files: `confluence_logic/agents/tools.py` (lines 201-252)
- Cause: Two document fetches + two parses + DOM operations for preview. Same again for commit.
- Improvement path: Cache parsed DOM in tool session state. Reuse in subsequent operations on same page.

**Search Results Combined from Three Sources Without Deduplication Optimization:**
- Problem: `search_workspace_knowledge()` queries live Confluence, lists pages, and searches Pinecone separately, then deduplicates in Python
- Files: `confluence_logic/agents/tools.py` (lines 110-182)
- Cause: Three API calls per search, each with latency. Deduplication loop iterates all results.
- Improvement path: Use parallel requests (asyncio). Implement server-side deduplication if Pinecone + Confluence APIs support it.

## Fragile Areas

**EditorAgent Router Logic Depends on Exact Tool Naming:**
- Files: `confluence_logic/agents/editor_agent.py` (lines 85-130)
- Why fragile: Hardcoded tool names ("resolve_request", "edit_existing_page", etc.) must match agent tool registrations. Renaming breaks routing silently.
- Safe modification: Extract tool names to class constants. Add assertions in __init__ to verify tools are registered.
- Test coverage: No unit tests verifying tool routing. Add tests that call each router path with mock agents.

**Version Conflict Detection String Matching:**
- Files: `confluence_logic/agents/tools.py` (line 57)
- Why fragile: `_is_version_conflict()` checks if error string contains "Version Conflict". Changing error message in Confluence connector breaks detection.
- Safe modification: Define error constant and reuse. Raise custom exception class instead of ValueError.
- Test coverage: Test file does not cover version conflict scenarios.

**Full Page Mode Heading Sentinels:**
- Files: `confluence_logic/utils/html_parser.py` (line 5)
- Why fragile: FULL_PAGE_SENTINELS set is hardcoded and case-insensitive matching may collide with actual page headings
- Safe modification: Use distinct marker (e.g., `__FULL_PAGE__`) that can't occur naturally. Add validation in heading resolution.
- Test coverage: No tests for full-page mode edge cases.

**Meeting State Global Dictionary:**
- Files: `confluence_logic/jarvis_agentic.py` (lines 125+)
- Why fragile: Untyped dict with multiple agents reading/writing concurrently. Missing keys cause KeyError.
- Safe modification: Replace with dataclass or TypedDict. Add property accessors with defaults.
- Test coverage: Test file resets keys manually in `_reset_meeting_state()`, indicating fragility.

## Scaling Limits

**Single EditorAgent Instance Per Meeting:**
- Current capacity: One session_agent handles all requests sequentially (queued)
- Limit: If meeting has multiple participants requesting edits simultaneously, all requests queue and responses feel slow
- Scaling path: Implement request batching for independent operations. Use worker pool for agent operations. Consider read-only operations (search, list) as high-priority non-blocking.

**Pinecone Index Dimensionality Coupling:**
- Current capacity: Index dimension hardcoded per deployment via env var
- Limit: Changing embedding model requires recreating Pinecone index (downtime)
- Scaling path: Support multiple indices with different dimensions. Route queries based on content type/metadata.

**Document Chunk Limit of 100 Sections Per Page:**
- Current capacity: Up to 100 sections cached per page (line 36 in vector_store.py)
- Limit: Pages with more sections lose stale data incorrectly
- Scaling path: Query stale chunks by version number instead of hardcoded range.

## Dependencies at Risk

**OpenAI Agents SDK (`openai-agents`):**
- Risk: Early/beta SDK version. API may change. No major version stability guarantee.
- Impact: Agent definitions could break with minor version bump. Tool registration API may change.
- Migration plan: Monitor releases. Pin version in requirements.txt. Test upgrades in staging before production. Consider fallback to base OpenAI client if SDK deprecated.

**Docling for HTML→Markdown Conversion:**
- Risk: Heavy dependency with external model downloads. May fail without internet.
- Impact: Ingestion pipeline breaks if docling initialization fails or document conversion times out.
- Migration plan: Add timeout to converter. Cache downloaded models. Fallback to BeautifulSoup-based extraction if docling unavailable.

**gTTS (Google Text-to-Speech):**
- Risk: Unofficial library for Google TTS. API changes or blocks possible.
- Impact: Fallback TTS will fail silently if Google TTS endpoint changes. No fallback to fallback.
- Migration plan: Migrate primary TTS provider fully to OpenAI. Remove gTTS as fallback. Use commercial TTS (e.g., Elevenlabs) with SLA.

**Pinecone Python Client Version Mismatch:**
- Risk: `pinecone-client` major versions have breaking API changes. Current code uses `Index.fetch()` and `Index.delete()` which may be refactored.
- Impact: Codebase tied to specific Pinecone SDK version. Upgrading breaks version conflict handling and stale section cleanup.
- Migration plan: Use Pinecone SDK version constraints. Test major upgrades. Monitor Pinecone SDK changelog.

## Missing Critical Features

**No Circuit Breaker for External APIs:**
- Problem: If Confluence API goes down, meeting continues to queue requests indefinitely
- Blocks: Graceful degradation when Confluence is unavailable
- Fix: Implement circuit breaker that stops accepting edit requests and speaks error message after 2-3 consecutive failures

**No Request Timeout on Confluence Operations:**
- Problem: Confluence API calls hang indefinitely if network drops
- Blocks: Meeting can become unresponsive
- Fix: Add `timeout` parameter to all `requests` calls (currently missing in most places)

**No Async/Await Pattern for Tool Execution:**
- Problem: Tools block event loop. If edit takes 5 seconds, no other requests processed.
- Blocks: Concurrent meeting requests
- Fix: Convert tools to async functions or wrap in `asyncio.to_thread()`. Update FastAPI handlers to be truly async.

**No Request Cancellation Mechanism:**
- Problem: If user requests cancellation during a long edit, tool keeps running
- Blocks: User can't interrupt slow operations
- Fix: Add cancellation token support. Pass cancel token to all long-running tools. Implement task cancellation in EditorAgent.

## Test Coverage Gaps

**Version Conflict Handling Not Covered:**
- What's not tested: The retry logic in `commit_document_edit`, `commit_delete`, `update_page_title` when version conflicts occur
- Files: `confluence_logic/agents/tools.py` (lines 268-281, 322-339, 378-390)
- Risk: Retries may have infinite loops or fail silently. Edge case of rapid concurrent edits never validated.
- Priority: High - this is production failure path

**HTML Parser Edge Cases:**
- What's not tested: Malformed HTML, deeply nested sections, missing headings, empty pages, pages with duplicate heading names
- Files: `confluence_logic/utils/html_parser.py`
- Risk: Silent failures or incorrect section targeting
- Priority: High - blocks user edits

**Pinecone Search Dimension Mismatches:**
- What's not tested: Error path when embedding dimension doesn't match index dimension
- Files: `confluence_logic/db/vector_store.py` (lines 88-105)
- Risk: Search fails with cryptic error message
- Priority: Medium - would catch config errors early

**Integration Tests for Full Edit Workflow:**
- What's not tested: End-to-end flow from voice command → editor agent → Confluence push → Pinecone index
- Files: Entire `confluence_logic/agents/` + `tools.py`
- Risk: Agent may not route correctly to right specialist. Tools may fail in combination.
- Priority: Medium - currently relies on manual testing

**Meeting State Concurrency:**
- What's not tested: Multiple simultaneous transcript events with rapid wake-word triggering
- Files: `confluence_logic/jarvis_agentic.py` (WebSocket handler)
- Risk: Race conditions in meeting_state updates. One request clears state another is reading.
- Priority: Medium - happens in real meetings

---

*Concerns audit: 2026-04-10*
