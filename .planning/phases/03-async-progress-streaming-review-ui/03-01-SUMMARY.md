---
phase: 03-async-progress-streaming-review-ui
plan: 01
subsystem: api
tags: [sse, fastapi, asyncio, pipeline, streaming, supabase]

# Dependency graph
requires:
  - phase: 02-multi-agent-pipeline-core
    provides: _run_pipeline, _draft_verify_persist, upsert_proposal, update_pipeline_job, start_pipeline endpoint

provides:
  - _job_queues dict (module-level asyncio.Queue map keyed by job_id) in confluence_logic/review/api.py
  - _SENTINEL sentinel object for SSE stream termination
  - _emit() helper that puts events into the job's queue (no-op when no consumer)
  - GET /review/pipeline/{job_id}/stream SSE endpoint with Bearer auth + ownership check
  - upsert_proposal now returns Optional[str] (Supabase-assigned UUID)
  - get_pipeline_job() helper in supabase_store.py for ownership verification

affects: [03-02, 03-03, 04-confluence-apply]

# Tech tracking
tech-stack:
  added: [fastapi.responses.StreamingResponse]
  patterns:
    - asyncio.Queue per job_id as in-memory SSE push mechanism
    - _emit() no-op helper guards against missing consumer gracefully
    - asyncio.wait_for(queue.get(), timeout=30.0) with ping keepalive
    - asyncio.get_event_loop().call_later(300, ...) for TTL cleanup of unclaimed queues
    - Bearer token via Authorization header OR ?token= query param for EventSource compatibility

key-files:
  created: []
  modified:
    - confluence_logic/review/api.py
    - confluence_logic/review/supabase_store.py
    - confluence_logic/tests/test_pipeline.py

key-decisions:
  - "SSE stage names use UI-SPEC labels (fact_extraction, rag_retrieval, drafting, verification) — NOT internal Supabase stage labels (retrieval, complete)"
  - "verification stage_start emitted once after asyncio.gather completes (all drafts done), not per-card"
  - "_emit() is a silent no-op when no consumer is connected — early events are dropped gracefully"
  - "proposal_ready event emitted AFTER upsert_proposal returns to carry the Supabase-assigned UUID"
  - "Async tests converted from @pytest.mark.asyncio/@pytest.mark.anyio to asyncio.run() wrappers — avoids pytest-asyncio/trio incompatibility with asyncio.to_thread"

patterns-established:
  - "Pattern 1: SSE streaming — StreamingResponse + async generator + asyncio.Queue per job, with 30s timeout + ping keepalive"
  - "Pattern 2: EventSource auth — token accepted from Authorization header OR ?token= query param"
  - "Pattern 3: Queue TTL cleanup — call_later(300, pop) on both terminal paths (complete/error)"
  - "Pattern 4: emit-after-persist — _emit(proposal_ready) called after upsert_proposal returns to ensure id field is valid"

requirements-completed: [PIPE-05]

# Metrics
duration: 5min
completed: 2026-05-12
---

# Phase 3 Plan 01: SSE Backend Infrastructure for PIPE-05 Summary

**In-memory asyncio.Queue SSE endpoint at GET /review/pipeline/{job_id}/stream emitting stage_start, proposal_ready, pipeline_complete, and pipeline_error events with Bearer + ownership auth**

## Performance

- **Duration:** 5 min
- **Started:** 2026-05-12T16:09:44Z
- **Completed:** 2026-05-12T16:15:40Z
- **Tasks:** 3
- **Files modified:** 3

## Accomplishments
- Added `_job_queues: dict[str, asyncio.Queue]`, `_SENTINEL`, and `_emit()` helper to `api.py` — the foundation for server-push pipeline progress
- Implemented `GET /review/pipeline/{job_id}/stream` SSE endpoint with dual-token auth (Authorization header + `?token=` query param) and Supabase ownership verification
- Injected `_emit()` calls at all five hook points in `_run_pipeline` using UI-SPEC stage names; modified `_draft_verify_persist` to capture and forward the Supabase-assigned UUID
- Modified `upsert_proposal` to return `Optional[str]` UUID and added `get_pipeline_job()` helper for SSE ownership verification
- All four PIPE-05 tests pass (GREEN): `test_sse_stream_returns_events`, `test_stage_events_emitted`, `test_proposal_ready_includes_id`, `test_pipeline_error_event`

## Task Commits

Each task was committed atomically:

1. **Task 1: Add four PIPE-05 SSE tests + mock_pipeline_queue fixture (RED)** - `eff6390` (test)
2. **Task 2: upsert_proposal returns Optional[str] UUID + get_pipeline_job helper** - `679feb5` (feat)
3. **Task 3: SSE infrastructure, _emit injection, xfail removal (GREEN)** - `91acf65` (feat)

## Files Created/Modified
- `confluence_logic/review/api.py` — Added StreamingResponse import, _job_queues dict, _SENTINEL, _emit(), SSE endpoint, and _emit injections in _run_pipeline and _draft_verify_persist
- `confluence_logic/review/supabase_store.py` — upsert_proposal now returns Optional[str] UUID; added get_pipeline_job() helper
- `confluence_logic/tests/test_pipeline.py` — Added mock_pipeline_queue fixture and four PIPE-05 tests (all passing)

## Decisions Made
- Stage names in SSE events use UI-SPEC names (`rag_retrieval`, `verification`) while Supabase `update_pipeline_job` keeps internal names (`retrieval`, `complete`) — separation documented in api.py comment and RESEARCH.md Pitfall 5
- `verification` stage_start is emitted once after `asyncio.gather` completes, before the `pipeline_complete` event — no per-card verification events needed per D-02
- TTL cleanup via `asyncio.get_event_loop().call_later(300, _job_queues.pop, job_id, None)` on both success and error paths — prevents memory leak for unclaimed queues

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed incorrect patch target `_extract_facts` to `_run_fact_extraction` in test_pipeline_error_event**
- **Found during:** Task 3 (implementation + test verification)
- **Issue:** Plan's test template used `confluence_logic.review.api._extract_facts` but the function imported in api.py is `_run_fact_extraction`; patch would silently no-op leaving test non-deterministic
- **Fix:** Changed patch target to `confluence_logic.review.api._run_fact_extraction`; added `get_history_item` patch to prevent real Supabase HTTP calls
- **Files modified:** confluence_logic/tests/test_pipeline.py
- **Verification:** test_pipeline_error_event passes with error event and _SENTINEL in queue
- **Committed in:** 91acf65 (Task 3 commit)

**2. [Rule 1 - Bug] Converted async tests from @pytest.mark.asyncio/@pytest.mark.anyio to asyncio.run() wrappers**
- **Found during:** Task 3 (test run: "async def functions are not natively supported")
- **Issue:** `pytest-asyncio` is not installed; using `@pytest.mark.anyio` caused tests to run under both asyncio and trio backends — `asyncio.to_thread` inside `_draft_verify_persist` and `_run_pipeline` fails under trio ("no running event loop")
- **Fix:** Replaced `@pytest.mark.asyncio` / `@pytest.mark.anyio` decorators on PIPE-05 tests with `asyncio.run(...)` calls inside synchronous test functions
- **Files modified:** confluence_logic/tests/test_pipeline.py
- **Verification:** All 4 PIPE-05 tests pass; existing 6 xfail tests unaffected
- **Committed in:** 91acf65 (Task 3 commit)

---

**Total deviations:** 2 auto-fixed (2 Rule 1 bugs)
**Impact on plan:** Both fixes necessary for test correctness. No scope creep — implementation matches plan specification exactly.

## Issues Encountered
- `pytest-asyncio` not installed in the `ml` conda environment; anyio plugin triggers both asyncio and trio test variants — resolved by converting async tests to synchronous wrappers with `asyncio.run()`

## Threat Flags

| Flag | File | Description |
|------|------|-------------|
| threat_flag: new-endpoint | confluence_logic/review/api.py | GET /review/pipeline/{job_id}/stream — new authenticated SSE endpoint with job ownership verification (mitigated per T-3-01 through T-3-06 in plan threat model) |

## Next Phase Readiness
- SSE endpoint is live and tested — Plan 02 can add frontend API client (`startPipeline()`, `openPipelineStream()`) and pipeline route to sync-sage-bot
- The `proposal_ready` event carries the full ChangeItem payload including Supabase UUID — Plan 03 ProposalCard Accept button can reference this id for Phase 4 apply endpoint
- No blockers for Plan 02 execution

---
*Phase: 03-async-progress-streaming-review-ui*
*Completed: 2026-05-12*
