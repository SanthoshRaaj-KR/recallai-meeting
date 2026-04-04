---
phase: 07-fully-agentic-meeting-pipeline-redesign
plan: "01"
subsystem: agents
tags: [pydantic, aiofiles, asyncio, openai-agents, rolling-summarizer, meeting-writer]

# Dependency graph
requires:
  - phase: 06-budget-model-switch-and-streaming-llm-to-tts-pipeline
    provides: gpt-4o-mini model mandate, streaming TTS pipeline
  - phase: 02-storage-foundation
    provides: MeetingRecord Pydantic model, MetadataStore async I/O pattern

provides:
  - BatchSummaryOutput Pydantic model (summary_text, key_points, speakers)
  - MeetingIndexEntry Pydantic model (meeting_id, title, date, channel, overview, md_path, participants, start_ts)
  - RollingSummarizerAgent: stateless Agent(output_type=BatchSummaryOutput) for batch summarization
  - MeetingWriterAgent: async file writer for per-meeting .md files and JSON meeting index

affects:
  - 07-02 (HistoryManagerAgent reads meeting_index.json and .md files via MeetingWriterAgent)
  - 07-03 (jarvis.py sentence buffer wiring calls RollingSummarizerAgent and MeetingWriterAgent)

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "RollingSummarizerAgent follows Agent(output_type=PydanticModel) pattern from SummarizerAgent"
    - "MeetingWriterAgent is a plain Python class (no LLM) — same pattern as RetrieverAgent"
    - "asyncio.Lock protects all file writes in MeetingWriterAgent (Phase 1 mandate)"
    - "aiofiles for all async file I/O — no blocking I/O in event loop"
    - ".md file is append-only after header — safe to read concurrently (D-06)"

key-files:
  created:
    - agents/rolling_summarizer.py
    - agents/meeting_writer.py
    - tests/test_rolling_pipeline.py
  modified:
    - storage/models.py

key-decisions:
  - "MeetingWriterAgent is a plain Python class (no LLM) — consistent with RetrieverAgent pattern from Phase 4"
  - "append_batch opens .md in append mode ('a') — existing content never rewritten (D-06)"
  - "upsert_index stores entries as json.loads(entry.model_dump_json()) dicts in a JSON array"
  - "read_md has no lock — append-only target makes concurrent reads safe"
  - ".md file path convention: {base_dir}/{channel_id}/{date_str}_{meeting_id}.md"

patterns-established:
  - "Rolling summarizer: stateless per-batch Agent(output_type=...) — no cross-batch context"
  - "Meeting index: JSON array at meetings/meeting_index.json, upserted after every batch flush"
  - "asyncio.Lock acquired per-method in MeetingWriterAgent — fine-grained locking, no reentrance"

requirements-completed: []

# Metrics
duration: 2min
completed: 2026-04-05
---

# Phase 7 Plan 01: Rolling Summarizer + Meeting Writer Summary

**BatchSummaryOutput and MeetingIndexEntry Pydantic models, RollingSummarizerAgent using Agent(output_type=BatchSummaryOutput), and MeetingWriterAgent with append-only .md writes and JSON meeting index — all verified with 7 passing tests**

## Performance

- **Duration:** 2 min
- **Started:** 2026-04-04T18:46:41Z
- **Completed:** 2026-04-04T18:48:30Z
- **Tasks:** 3 implementation tasks + test file (4 commits)
- **Files modified:** 4

## Accomplishments

- Added `BatchSummaryOutput` and `MeetingIndexEntry` to `storage/models.py` (append-only, no existing classes modified)
- Created `RollingSummarizerAgent` following the `Agent(output_type=...)` pattern from `SummarizerAgent` — stateless, one batch per call
- Created `MeetingWriterAgent` (plain Python class, no LLM) with full async file I/O: header write, batch append, index upsert, index read, and .md read
- All 7 tests pass; full test suite 195 passed, 1 skipped — no regressions

## Task Commits

Each task was committed atomically:

1. **Task 1: BatchSummaryOutput and MeetingIndexEntry models** - `ba2734b` (feat)
2. **Task 2: RollingSummarizerAgent** - `bfd20bb` (feat)
3. **Task 3: MeetingWriterAgent** - `868b913` (feat)
4. **Tests: test_rolling_pipeline.py** - `f34651a` (test)

## Files Created/Modified

- `storage/models.py` - Added `BatchSummaryOutput` (summary_text, key_points, speakers) and `MeetingIndexEntry` (meeting fast-lookup index entry); existing `MeetingRecord`/`ActionItem` untouched
- `agents/rolling_summarizer.py` - `RollingSummarizerAgent` with `run(batch_transcript: str) -> BatchSummaryOutput`; stateless, delegates to `Runner.run()` with `output_type=BatchSummaryOutput`
- `agents/meeting_writer.py` - `MeetingWriterAgent` with `write_meeting_header`, `append_batch`, `upsert_index`, `read_index`, `read_md`; asyncio.Lock on all writes; aiofiles throughout
- `tests/test_rolling_pipeline.py` - 7 tests covering all new code (Pydantic validation, mocked agent, real filesystem via tmp_path)

## Decisions Made

- `MeetingWriterAgent` is a plain Python class (no LLM), consistent with the `RetrieverAgent` pattern established in Phase 4
- `read_md` has no lock — append-only .md target makes concurrent reads safe (D-06)
- `upsert_index` uses `json.loads(entry.model_dump_json())` to serialize entries as plain dicts before storing in the JSON array
- `.md` file path convention: `{base_dir}/{channel_id}/{date_str}_{meeting_id}.md` (per discretion decision from 07-CONTEXT.md)

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None — all imports worked correctly through the conftest.py SDK namespace extension pattern established in Phase 3.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- `BatchSummaryOutput`, `MeetingIndexEntry` importable from `storage.models`
- `RollingSummarizerAgent` ready to be called from jarvis.py flush handler (Plan 07-03)
- `MeetingWriterAgent` ready for History Manager (Plan 07-02) to call `read_index()` and `read_md()`
- No blockers for Plan 07-02 or 07-03

---
*Phase: 07-fully-agentic-meeting-pipeline-redesign*
*Completed: 2026-04-05*

## Self-Check: PASSED

- FOUND: storage/models.py
- FOUND: agents/rolling_summarizer.py
- FOUND: agents/meeting_writer.py
- FOUND: tests/test_rolling_pipeline.py
- FOUND: 07-01-SUMMARY.md
- FOUND: commit ba2734b (feat: models)
- FOUND: commit bfd20bb (feat: rolling_summarizer)
- FOUND: commit 868b913 (feat: meeting_writer)
- FOUND: commit f34651a (test: test_rolling_pipeline)
- All 7 plan tests pass; full suite: 195 passed, 1 skipped
