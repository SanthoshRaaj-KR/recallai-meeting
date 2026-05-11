# Technical Concerns

**Analysis Date:** 2026-05-11

## High Priority

### Nonexistent OpenAI model `gpt-5-mini` used as default
- **Location:** `confluence_logic/jarvis_agentic.py:64`, `local_office_logic/jarvis_agentic.py:55`, `confluence_logic/review/api.py:62` (`JARVIS_REVIEW_MODEL`), all agent constructors including `confluence_logic/agents/proposed_changes_agent.py:28`
- **Issue:** `gpt-5-mini` does not exist on the OpenAI API. Any deployment without the `JARVIS_AGENT_MODEL` or `JARVIS_REVIEW_MODEL` env vars set will fail at runtime with an API error.
- **Risk:** Silent breakage in production — no startup-time validation.

### Monolithic orchestrator files
- **Location:** `confluence_logic/jarvis_agentic.py` (2,457 lines), `local_office_logic/jarvis_agentic.py` (1,860 lines)
- **Issue:** Both files mix event loop management, intent routing, agent delegation, TTS, WebSocket handling, and state management in a single file with no layer separation.
- **Risk:** High change risk, hard to test, merge conflicts inevitable as both modules diverge.

### Unauthenticated bot start endpoint
- **Location:** `confluence_logic/review/api.py` — `POST /bot/start`
- **Issue:** Auth is optional and only used for history association, not to gate the action. Any caller can trigger a bot start. The `review-ui` frontend sends no auth token at all.
- **Risk:** Unauthorized resource consumption, potential abuse vector.

### `asyncio.run()` called inside a running event loop
- **Location:** `confluence_logic/jarvis_agentic.py:868` — edge-tts TTS path
- **Issue:** `asyncio.run()` creates a new event loop and errors when called from within an already-running loop.
- **Risk:** Runtime crash on the TTS code path when called from the async event loop.

### Nonexistent PyPI package `neo4j==6.1.0`
- **Location:** `requirements.txt:17`
- **Issue:** `neo4j==6.1.0` does not exist on PyPI. Fresh installs (`pip install -r requirements.txt`) will fail.
- **Risk:** Breaks CI, new developer onboarding, and fresh deployments.

---

## Medium Priority

### Broad `except Exception` blocks swallowing errors silently
- **Location:** 60+ instances throughout both `confluence_logic/` and `local_office_logic/`
- **Issue:** Multiple bare `except Exception: pass` blocks hide failures. Errors are silently ignored.
- **Risk:** Debugging is extremely difficult; failures manifest as mysterious empty responses.

### Diverging `meeting_state` implementations
- **Location:** `local_office_logic/jarvis_agentic.py` (bare module-level dict) vs `confluence_logic/jarvis_agentic.py` (upgraded to `MeetingStateProxy` with `contextvars`)
- **Issue:** The two modules have diverged on session isolation strategy. The local office module is not safe for concurrent sessions.
- **Risk:** Session state bleed between concurrent meetings in local office mode.

### `asyncio.run()` used inside sync test functions
- **Location:** Multiple test files across both modules
- **Issue:** Tests call `asyncio.run()` inside sync functions despite `pytest-asyncio` being installed. This is a workaround anti-pattern.
- **Risk:** Event loop conflicts, flaky tests under certain pytest-asyncio modes.

### File-based bot_id state recovery
- **Location:** `confluence_logic/jarvis_agentic.py:2345` — `bot_id.txt` written/read from disk
- **Issue:** WebSocket handler uses a local file for bot state recovery. Not suitable for multi-instance or containerized deployment.
- **Risk:** State loss or incorrect recovery in any deployment with ephemeral filesystems.

### `time.sleep(2)` in startup
- **Location:** `jarvis_agentic.py:2427`, `local_office_logic/jarvis_agentic.py:1830`
- **Issue:** Hard-coded sleep during startup instead of polling a health condition.
- **Risk:** Fragile under slow startup conditions; fails fast on fast hardware and hangs on slow ones.

### `review-ui` has no authentication
- **Location:** `review-ui/src/lib/api.ts`, `review-ui/src/app/page.tsx`
- **Issue:** The review-ui frontend sends no `Authorization` header to the backend. All API calls are anonymous. The backend only optionally validates auth for history association, not to protect change execution (`POST /review/execute`).
- **Risk:** Any user with network access to the backend can read summaries, see proposed changes, and execute Confluence writes.

### `confluence_page_graph` TTL and refresh coupling
- **Location:** `confluence_logic/confluence_page_graph.py:25-27`
- **Issue:** `GRAPH_TTL_SECONDS` and `REFRESH_INTERVAL_SECONDS` both default to 7200s (2h), but the refresh logic is tied to in-process state and will not survive a server restart.
- **Risk:** Stale Confluence graph served after any restart within the TTL window; full graph re-index required on every cold start.

---

## Low Priority / Observations

### Module-level singleton clients
- **Location:** OpenAI, Pinecone, Neo4j clients instantiated at import time across both modules
- **Issue:** Credentials consumed at import time; no lazy initialization in some modules (`graph_rag.py`). Makes unit testing harder (requires patching before import).
- **Partially fixed:** `agents/tools.py`, `classifier.py`, `general_responder.py`, `review/api.py` use the lazy `_store = None; def get_store()` pattern.

### No input validation at API boundaries
- **Location:** `review/api.py` endpoints
- **Issue:** Request bodies rely on Pydantic schema validation but error responses are not consistently structured.

### `review-ui` has no tests
- **Location:** `review-ui/`
- **Issue:** No testing framework configured. No unit, component, or integration tests for the Next.js frontend.
- **Risk:** Regressions in UI logic (polling, change selection, error states) go undetected.

---

## Duplication

7 near-identical file pairs between `confluence_logic/` and `local_office_logic/`:

| File | Status |
|------|--------|
| `audio_cache.py` | Near-identical |
| `meeting_responder.py` | Near-identical |
| `general_responder.py` | **Diverging** — `confluence_logic` has Tavily web search + `_needs_web_search` router; `local_office_logic` does not |
| `classifier.py` | Near-identical |
| `db/vector_store.py` | Near-identical |
| `graph_rag.py` | **Diverging** — `confluence_logic` has `confluence_page_graph` integration and separate `_local_nodes`/`_local_edges` in-memory graph |
| `agents/reframer_agent.py` | Near-identical |

No shared base package exists — changes must be made twice. `confluence_logic` is now ahead of `local_office_logic` on several features (web search, MeetingStateProxy, proposed changes, page graph).

---

## Security

### Supabase service role key used for all DB writes
- **Location:** `confluence_logic/review/supabase_store.py:48-49` — `_db_key()` returns service role key when set
- **Issue:** The service role key bypasses row-level security. Used as the default for all write operations rather than a scoped key.
- **Risk:** Any compromise of this key grants full database access, bypassing RLS policies defined in `supabase_schema.sql`.

### `RECALL_API_KEY` not validated on startup
- **Location:** Across `jarvis_agentic.py` in both modules
- **Issue:** Key loaded from env without existence check; sends `Authorization: Token None` to Recall.ai API when key is missing.
- **Risk:** Silent auth failure, misleading error messages from the Recall API.

---

## TODOs & Incomplete Work

### Missing test coverage
The following critical paths have zero test coverage:

| File | Module |
|------|--------|
| `ingestion/doc_pipeline.py` | Both |
| `confluence_page_graph.py` | confluence_logic — partially covered (HTML parsing, user_id scoping) |
| `review/api.py` (most endpoints — propose_changes, execute, summary generation) | confluence_logic |
| `general_responder.py` (Tavily web search call path) | confluence_logic |
| TTS pipeline (all providers) | Both |
| `local_office_logic/general_responder.py` | local_office_logic |
| `local_office_logic/meeting_responder.py` | local_office_logic |
| `utils/sandbox.py`, `office_runtime.py` | local_office_logic |
| `review-ui/` (all frontend components and API client) | review-ui |

### `jarvis.py` at repo root
- **Issue:** The description mentions a `jarvis.py` top-level entry point but this file was not found in the repository. May have been removed or not yet committed.
- **Risk:** Documentation drift; unclear if this is a planned or abandoned entry point.
