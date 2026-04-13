---
phase: 04-speaker-isolation-speech-debounce-and-general-question-filler-audio
plan: 01
subsystem: websocket-transcript-handler
tags: [speaker-isolation, debounce, async, meeting-state, SPEAKER-01, DEBOUNCE-01]
dependency_graph:
  requires: []
  provides: [SPEAKER-01, DEBOUNCE-01]
  affects: [websocket_endpoint, meeting_state, handle_spoken_request]
tech_stack:
  added: []
  patterns: [asyncio.create_task cancellable debounce, speaker isolation filter, accumulated text joining]
key_files:
  created: []
  modified:
    - confluence_logic/jarvis_agentic.py
    - confluence_logic/tests/test_jarvis_agentic.py
decisions:
  - "JARVIS_DEBOUNCE_SECONDS defaults to 1.0s — long enough to catch mid-sentence continuation, short enough to feel responsive"
  - "_debounced_dispatch uses asyncio.sleep + meeting_state mutation: cancel-and-restart pattern keeps accumulated text growing across segments"
  - "invoker_participant locked on first query detection (not wake word alone) — bare wake uses _handle_bare_wake which handles its own follow-up window"
  - "Non-invoker transcripts silently dropped with logger.debug only — no user-visible feedback to avoid interrupting the invoker"
metrics:
  duration: ~8 min
  completed_date: "2026-04-13"
  tasks: 2/2
  files: 2
---

# Phase 04 Plan 01: Speaker Isolation and Speech Debounce Summary

**One-liner:** Cancellable 1-second debounce with per-participant invoker lock prevents mic bleed and premature dispatch using asyncio.create_task cancel-and-restart.

## What Was Built

Implemented speaker isolation and speech-completion debounce in the WebSocket transcript handler (`websocket_endpoint`):

1. **JARVIS_DEBOUNCE_SECONDS env var** — configurable debounce window (default 1.0s), placed after the existing JARVIS_ env var block.

2. **Three new meeting_state fields:**
   - `invoker_participant`: set to the participant name when they trigger a wake-word query; cleared after dispatch
   - `_pending_debounce_task`: holds the cancellable asyncio.Task for the current debounce window
   - `_accumulated_query`: space-joined text from all invoker segments received during the window

3. **`_debounced_dispatch` coroutine** — waits `JARVIS_DEBOUNCE_SECONDS`, clears the three state fields, then awaits `handle_spoken_request`. Inserted between `_speak_filler` and `_generate_contextual_gap_filler`.

4. **Replaced dispatch block in `websocket_endpoint`** — old single-line `create_task(handle_spoken_request(...))` replaced with:
   - Speaker isolation check: if `invoker` is set and `participant != invoker`, log at DEBUG and skip
   - On query detection: lock invoker, accumulate text, cancel any pending debounce task, schedule new `_debounced_dispatch`
   - On bare wake: lock invoker, fire `_handle_bare_wake`

5. **Test updates** — `_reset_meeting_state` extended with three new field resets; five new unit tests added covering all SPEAKER-01/DEBOUNCE-01 behaviors.

## Commits

| Task | Commit | Description |
|------|--------|-------------|
| 1 + 2 | 7d38032 | feat(04-01): speaker isolation and speech debounce |

## Deviations from Plan

None — plan executed exactly as written.

## Known Stubs

None. All new state fields are wired into the websocket handler and cleared by `_debounced_dispatch`.

## Test Results

- 5 new tests: all PASS
- Previously passing tests: unchanged (8 pre-existing failures confirmed unrelated to this plan — same failures present before any changes in this plan)

## Self-Check: PASSED

Files confirmed present:
- `confluence_logic/jarvis_agentic.py` — modified with all 4 edits
- `confluence_logic/tests/test_jarvis_agentic.py` — modified with _reset_meeting_state extension and 5 new tests

Commit confirmed: `7d38032` present in git log.
