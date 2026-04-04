---
phase: 01-prerequisite-refactor
plan: 02
subsystem: infra
tags: [asyncio, meeting-state, concurrency, jarvis, refactor, integration-tests, pytest-asyncio]

# Dependency graph
requires:
  - 01-01 (MeetingState class with asyncio.Lock)
provides:
  - jarvis.py fully wired to MeetingState — zero direct dict mutations
  - Three concurrent integration tests proving asyncio.Lock prevents state corruption
  - _sync_set_state helper for calling async state from synchronous main()
affects:
  - All subsequent phases (jarvis.py is now safe for concurrent agent work)

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "asyncio.create_task used instead of Thread for async handler dispatch (handle_query, speak)"
    - "asyncio.to_thread used for synchronous speak() calls from async context"
    - "_sync_set_state helper pattern: new_event_loop + run_until_complete for sync-to-async bridge in main()"
    - "asyncio.gather used in integration tests to stress concurrent access"

key-files:
  created:
    - tests/test_jarvis_integration.py
  modified:
    - jarvis.py

key-decisions:
  - "_sync_set_state helper used for main() context — uvicorn loop not yet running when main() sets bot_id/active"
  - "Thread retained only for _start_server (uvicorn daemon) — all other Thread usages replaced with asyncio primitives"
  - "speak() called via asyncio.to_thread — keeps synchronous HTTP call off the event loop"
  - "handle_query dispatched with asyncio.create_task — non-blocking from websocket_endpoint"

# Metrics
duration: 5min
completed: 2026-04-04
---

# Phase 1 Plan 2: Prerequisite Refactor - MeetingState Wiring Summary

**jarvis.py fully refactored to use async MeetingState class — zero direct dict mutations, all state access behind asyncio.Lock, three concurrent integration tests confirm no corruption under simultaneous WebSocket writes and agent reads.**

## Performance

- **Duration:** 5 min
- **Started:** 2026-04-04
- **Completed:** 2026-04-04
- **Tasks:** 2 completed
- **Files modified:** 2

## Accomplishments

- Replaced `meeting_state` dict (lines 62-67) with `state = MeetingState()` — all 4 state fields now behind asyncio.Lock
- Refactored `_get_meeting_transcript`, `_run_tool`, and `handle_query` to async, chaining through `await state.get_transcript()`
- Replaced 6 direct dict accesses in `websocket_endpoint` with `await state.*` calls; replaced 2 `Thread()` dispatches with `asyncio.create_task` and `asyncio.to_thread`
- Refactored `/health` endpoint to `return await state.get_health_snapshot()`
- Added `_sync_set_state()` helper for sync-to-async bridge in `main()` (bot_id, active flag)
- Created 3 concurrent integration tests using `asyncio.gather` — all pass with 13/13 total tests green

## Task Commits

Each task was committed atomically:

1. **Task 1: Refactor jarvis.py to use MeetingState class** - `d6eb1aa` (feat)
2. **Task 2: Concurrent integration tests** - `684b51c` (feat)

## Files Created/Modified

- `jarvis.py` - Refactored: MeetingState wired, all dict access replaced, async chain updated, Thread replaced with asyncio primitives
- `tests/test_jarvis_integration.py` - 3 concurrent integration tests (50-write+read, bot_id/listening toggles, writes+health snapshots)

## Decisions Made

- **_sync_set_state pattern**: `main()` runs synchronously before uvicorn starts; cannot `await` directly. Used `asyncio.new_event_loop().run_until_complete()` as one-shot bridge
- **Thread retained for _start_server only**: uvicorn must run in a daemon thread from synchronous `main()`; all other Thread usages eliminated
- **asyncio.to_thread for speak()**: `speak()` makes blocking HTTP requests (Recall.ai audio API); offloaded to thread pool to avoid blocking the event loop
- **asyncio.create_task for handle_query**: Fire-and-forget dispatch from websocket_endpoint; task runs concurrently without blocking the WebSocket receive loop

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 2 - Missing critical functionality] speak() calls in handle_query also converted to asyncio.to_thread**
- **Found during:** Task 1
- **Issue:** Plan specified `asyncio.to_thread` for the wake-word speak("Yes?") path but did not explicitly mention the three `speak()` calls inside `handle_query()`. Since `handle_query` became async, those synchronous blocking HTTP calls needed to be offloaded too.
- **Fix:** All `speak()` calls in `handle_query` use `await asyncio.to_thread(speak, ...)` to avoid blocking the event loop.
- **Files modified:** jarvis.py
- **Commit:** d6eb1aa

## Known Stubs

None - jarvis.py is fully wired to MeetingState. All state reads and writes go through async methods protected by asyncio.Lock.

## Self-Check: PASSED

- `tests/test_jarvis_integration.py` exists and contains 3 async tests
- `jarvis.py` imports cleanly (`python -c "import jarvis"` exits 0)
- All 13 tests pass (`python -m pytest tests/ -v` → 13 passed)
- Zero direct dict access (`grep -c 'meeting_state\[' jarvis.py` → 0)
- Commits d6eb1aa and 684b51c confirmed in git log
