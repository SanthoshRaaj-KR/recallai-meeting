---
phase: "07"
plan: "03"
subsystem: jarvis.py — rolling pipeline wiring
tags: [rolling-pipeline, sentence-buffer, flush, history-manager, meeting-writer, websocket]
dependency_graph:
  requires: [07-01-PLAN (RollingSummarizerAgent, MeetingWriterAgent), 07-02-PLAN (HistoryManagerAgent)]
  provides: [end-to-end rolling .md pipeline, memory query routing via HistoryManagerAgent]
  affects: [jarvis.py websocket_endpoint, jarvis.py handle_query, jarvis.py disconnect handler]
tech_stack:
  added: [datetime, pathlib.Path, asyncio.Lock for sentence buffer]
  patterns: [module-level asyncio.Lock for shared state, fire-and-forget asyncio.create_task, force flush on wake word]
key_files:
  created: []
  modified: [jarvis.py, tests/test_jarvis_rolling_pipeline.py (new)]
decisions:
  - _disconnect_bot_id retrieved fresh from state.get_bot_id() in disconnect handler to avoid NameError when bot_id was only assigned inside the loop
  - Global declarations for all buffer state moved to top of websocket_endpoint function scope to avoid Python "used prior to global declaration" SyntaxError
  - speak_chunked used for pre-generated HistoryManagerAgent answers (not _stream_llm_and_speak which requires an unanswered LLM messages list)
  - Timestamp uses %I:%M %p + lstrip("0") rather than %-I (macOS %-I portability constraint)
metrics:
  duration: "~8 minutes"
  completed: "2026-04-05"
  tasks: 3
  files: 2
---

# Phase 7 Plan 3: Wire Rolling Pipeline into jarvis.py — Summary

**One-liner:** Real-time sentence buffer with dual-threshold flush, HistoryManagerAgent-first memory routing, and end-of-meeting .md flush wired into jarvis.py WebSocket pipeline.

## What Was Built

Plan 07-03 wired all agent classes from Plans 07-01 and 07-02 into `jarvis.py`, making the full rolling meeting pipeline operational end-to-end.

### Task 1: Imports, Configuration, and Singletons

Added to `jarvis.py`:
- `import datetime` and `from pathlib import Path`
- `from storage.models import MeetingIndexEntry`
- `from agents.rolling_summarizer import RollingSummarizerAgent`
- `from agents.meeting_writer import MeetingWriterAgent`
- `from agents.history_manager import HistoryManagerAgent`
- `SENTENCE_FLUSH_COUNT = int(os.getenv("SENTENCE_FLUSH_COUNT", "10"))` (D-01)
- `SENTENCE_FLUSH_SECONDS = int(os.getenv("SENTENCE_FLUSH_SECONDS", "120"))` (D-01)
- Singletons: `rolling_summarizer`, `meeting_writer`, `history_manager`
- Module-level buffer state: `_sentence_buffer`, `_sentence_buffer_lock`, `_last_flush_ts`, `_current_batch_num`, `_current_meeting_entry`, `_meeting_header_written`

### Task 2: `_flush_sentence_buffer` and `websocket_endpoint` modifications

New `_flush_sentence_buffer(bot_id, force=False)` async function:
- Acquires `_sentence_buffer_lock` to check thresholds and drain atomically
- Flushes when: `force=True` OR `sentence_count >= SENTENCE_FLUSH_COUNT` OR `time_elapsed >= SENTENCE_FLUSH_SECONDS` (D-01)
- Empty buffer returns immediately (no-op)
- Calls `RollingSummarizerAgent.run()` outside the lock
- Writes meeting header on first batch only (`_meeting_header_written` flag)
- Calls `MeetingWriterAgent.append_batch()` for each batch
- Upserts index after each batch with updated participants and overview
- Uses `%I:%M %p` + `lstrip("0")` for portable timestamp (macOS constraint)

`websocket_endpoint` changes:
- Initializes `_current_meeting_entry` on first transcript event (creates `MeetingIndexEntry` with channel/date/meeting_id metadata)
- Appends `"{participant}: {sentence}"` to `_sentence_buffer` under `_sentence_buffer_lock` after each transcript
- Fires `asyncio.create_task(_flush_sentence_buffer(bot_id))` for threshold checking
- D-02 interrupt flush: `await _flush_sentence_buffer(bot_id, force=True)` before `handle_query` dispatch in BOTH wake-word branches ("full query" and "bare wake word")
- D-10 disconnect: `asyncio.create_task(_flush_sentence_buffer(..., force=True))` alongside existing `run_ingestion_pipeline` task
- Resets all buffer state on disconnect for next meeting

### Task 3: Memory query routing and disconnect handler

`handle_query()` memory routing replaced:
- Removed: `if orchestrator is not None and _is_memory_query(query):`
- Added: `if _is_memory_query(query):` with `HistoryManagerAgent.run()` first, `OrchestratorAgent.run()` as fallback on exception
- Both result paths use `speak_chunked()` for voice output (pre-generated answer text)
- Disambiguation options spoken aloud when `result.needs_disambiguation=True`

### Tests: 7 passing

File: `tests/test_jarvis_rolling_pipeline.py`

| Test | Description |
|------|-------------|
| 1 | Sentence count threshold triggers flush |
| 2 | Force flush clears buffer and increments batch_num |
| 3 | Empty buffer flush is a no-op |
| 4 | Meeting header written exactly once across two flushes |
| 5 | handle_query routes memory query to HistoryManagerAgent first |
| 6 | handle_query falls back to OrchestratorAgent on HistoryManager exception |
| 7 | Disconnect reset sets entry=None, batch_num=0, buffer=[] |

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] global declaration SyntaxError in websocket_endpoint**
- **Found during:** Task 2 implementation
- **Issue:** Plan specified `global _current_meeting_entry, _meeting_header_written` inside the loop body and `global _current_meeting_entry, _meeting_header_written, _current_batch_num, _last_flush_ts` in the `except WebSocketDisconnect` block. Python raises `SyntaxError: name '_current_meeting_entry' is used prior to global declaration` when the same variable appears in two separate `global` statements in the same function.
- **Fix:** Moved single consolidated `global` declaration to the top of `websocket_endpoint` function before the `try:` block.
- **Files modified:** `jarvis.py`
- **Commit:** 6432729

**2. [Rule 2 - Correctness] bot_id scope in disconnect handler**
- **Found during:** Task 2 review
- **Issue:** `bot_id` is assigned inside the `while True:` loop; in a `WebSocketDisconnect` exception handler, `bot_id` may be undefined if the disconnect happened before a transcript event assigned it.
- **Fix:** Introduced `_disconnect_bot_id = await state.get_bot_id() or ""` in the disconnect handler.
- **Files modified:** `jarvis.py`
- **Commit:** 6432729

**3. [Rule 3 - Correctness] speak_chunked instead of _stream_llm_and_speak for pre-generated answers**
- **Found during:** Task 3 implementation
- **Issue:** Plan suggested using `_stream_llm_and_speak` for HistoryManagerAgent answers, but `_stream_llm_and_speak` calls the OpenAI API directly with a `messages` list — it cannot stream a pre-generated answer string from HistoryManagerAgent.
- **Fix:** Used `speak_chunked(result.answer, bot_id)` for pre-generated answers (consistent with Phase 6 pattern for non-streaming answer paths).
- **Files modified:** `jarvis.py`
- **Commit:** 45f50cf

## Success Criteria Verification

1. `SENTENCE_FLUSH_COUNT` and `SENTENCE_FLUSH_SECONDS` configurable via env vars — DONE
2. Each transcript event appends to `_sentence_buffer` under `_sentence_buffer_lock` — DONE
3. `_flush_sentence_buffer(force=False)` flushes on sentence count OR time threshold (D-01) — DONE
4. Wake-word detection awaits `_flush_sentence_buffer(force=True)` before `handle_query` (D-02) — DONE (both branches)
5. Meeting .md created with header on first batch (D-05) and subsequent batches append sections (D-03, D-04, D-06) — DONE (via MeetingWriterAgent, tested in Plan 07-01)
6. `meeting_index.json` upserted after each batch flush — DONE
7. `memory_query` routed to `HistoryManagerAgent` first; falls back to `OrchestratorAgent` on exception — DONE
8. `WebSocketDisconnect` triggers both `run_ingestion_pipeline` (Pinecone) AND `_flush_sentence_buffer(force=True)` (D-10) — DONE
9. Buffer state fully reset on disconnect — DONE
10. `pytest tests/test_jarvis_rolling_pipeline.py -v` — 7/7 PASS
11. All existing tests continue to pass — 208/208 pass (1 pre-existing network live-smoke test conditionally skipped — unrelated to these changes)

## Known Stubs

None — all data flows are wired with real implementations from Plans 07-01 and 07-02.

## Self-Check: PASSED

- jarvis.py: FOUND
- tests/test_jarvis_rolling_pipeline.py: FOUND
- 07-03-SUMMARY.md: FOUND
- Commits found: e5b9038, 6432729, 45f50cf, 2b1a6b7
