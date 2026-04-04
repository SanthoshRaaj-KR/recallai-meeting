---
phase: 07-fully-agentic-meeting-pipeline-redesign
plan: "02"
subsystem: agents
tags: [pydantic, asyncio, openai-agents, history-manager, disambiguation, pinecone-fallback]

# Dependency graph
requires:
  - phase: 07-fully-agentic-meeting-pipeline-redesign
    plan: "01"
    provides: MeetingIndexEntry, MeetingWriterAgent.read_index(), MeetingWriterAgent.read_md()
  - phase: 05-orchestration-answer-path
    provides: OrchestratorResult (reused for disambiguation), AnswerAgent, RetrievalResult

provides:
  - HistoryManagerAgent: async run(query, user_id, channel_id) -> OrchestratorResult
  - _md_to_retrieval_result(): wraps .md content as synthetic RetrievalResult (all 4 required Pydantic fields)
  - _select_meeting(): LLM-based meeting index selection with safe JSON parse fallback

affects:
  - 07-03 (jarvis.py wires HistoryManagerAgent into handle_query() for memory_query path)

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "HistoryManagerAgent is a plain Python class (not openai-agents Agent()) — consistent with RetrieverAgent/OrchestratorAgent pattern"
    - "LLM selection uses response_format=json_object for reliable structured output without Agent() wrapper"
    - "Disambiguation reuses OrchestratorResult with needs_disambiguation=True — identical to orchestrator.py lines 224-247"
    - "Pinecone fallback injected via optional constructor parameter (D-09 coexistence strategy)"
    - "RetrieverAgent imported inside run() body to avoid circular import at module level"

key-files:
  created:
    - agents/history_manager.py
    - tests/test_history_manager.py
  modified: []

key-decisions:
  - "HistoryManagerAgent is a plain Python class (no Agent() instance) — consistent with OrchestratorAgent/RetrieverAgent from Phases 4-5"
  - "_select_meeting() swallows all exceptions and returns safe fallback dict — no exception propagates to caller"
  - "_md_to_retrieval_result() populates all 4 required RetrievalResult Pydantic fields (query, results, total_candidates, returned_count)"
  - "RetrieverAgent imported inside run() method body to prevent circular import (agents.history_manager -> agents.retriever -> storage.pinecone_client)"
  - "Disambiguation options use same shape as OrchestratorAgent: {index, meeting_id, title, channel, date}"

patterns-established:
  - "Synthetic RetrievalResult wrapping: embed .md content in metadata.summary_text for AnswerAgent consumption"
  - "LLM-as-meeting-selector: json_object response_format with {selected, candidates} schema"
  - "Optional retriever fallback pattern: inject None by default, Pinecone path activated only when provided"

requirements-completed: []

# Metrics
duration: 5min
completed: 2026-04-05
---

# Phase 7 Plan 02: History Manager Agent Summary

**HistoryManagerAgent with LLM-based meeting index selection, disambiguation via OrchestratorResult, optional Pinecone fallback, and safe parse-error handling — verified with 6 passing unit tests**

## Performance

- **Duration:** ~5 min
- **Completed:** 2026-04-05
- **Tasks:** 2 implementation tasks (1 new file) + 1 test file (2 commits)
- **Files modified:** 2

## Accomplishments

- Created `agents/history_manager.py` with `HistoryManagerAgent` implementing:
  - `_select_meeting()`: GPT call with `json_object` response format, safe parse-error fallback
  - `_md_to_retrieval_result()`: wraps `.md` content as synthetic `RetrievalResult` with all 4 required Pydantic fields
  - `run()`: full pipeline — empty index check, LLM selection, single/multi/no-match branches, Pinecone fallback
- Created `tests/test_history_manager.py` with 6 unit tests covering all `run()` branches
- All 201 tests pass (6 new + 195 pre-existing), 1 skipped — no regressions

## Task Commits

Each task was committed atomically:

1. **Task 1+2: HistoryManagerAgent implementation** - `c1cf574` (feat)
2. **Tests: test_history_manager.py** - `d0e5142` (test)

## Files Created/Modified

- `agents/history_manager.py` — New file. `HistoryManagerAgent` plain Python class with `_select_meeting()`, `_md_to_retrieval_result()`, and `run()`. Optional `RetrieverAgent` Pinecone fallback parameter. `OrchestratorResult` imported inside `run()` to avoid circular imports.
- `tests/test_history_manager.py` — 6 unit tests: single match, multi-candidate disambiguation, empty index, Pinecone fallback, no-retriever graceful message, parse-error safe fallback. All `AsyncOpenAI` calls mocked.

## Decisions Made

- `HistoryManagerAgent` is a plain Python class (no `openai-agents Agent()` instance), consistent with the established `RetrieverAgent` and `OrchestratorAgent` pattern
- `_select_meeting()` uses `response_format={"type": "json_object"}` for reliable structured output without the `Agent(output_type=...)` wrapper — appropriate for a simple two-field selection response
- All 4 `RetrievalResult` Pydantic fields populated in `_md_to_retrieval_result()` (`query`, `results`, `total_candidates=1`, `returned_count=1`) — omitting any raises `ValidationError` at runtime
- `RetrieverAgent` imported inside `run()` body (not at module top) to prevent circular import through `storage.pinecone_client`
- Disambiguation options shape exactly mirrors `agents/orchestrator.py` lines 226-238: `{index, meeting_id, title, channel, date}`

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None — all imports resolved correctly through the conftest.py SDK namespace extension pattern from Phase 3.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- `HistoryManagerAgent` importable from `agents.history_manager`
- `run(query, user_id, channel_id) -> OrchestratorResult` interface ready for Plan 07-03 wiring into `jarvis.py handle_query()`
- Disambiguation and Pinecone fallback already tested — no additional integration required before wiring

---
*Phase: 07-fully-agentic-meeting-pipeline-redesign*
*Completed: 2026-04-05*

## Self-Check: PASSED

- FOUND: agents/history_manager.py
- FOUND: tests/test_history_manager.py
- FOUND: .planning/phases/07-fully-agentic-meeting-pipeline-redesign/07-02-SUMMARY.md
- FOUND: commit c1cf574 (feat: history_manager implementation)
- FOUND: commit d0e5142 (test: test_history_manager)
- All 6 plan tests pass; full suite: 201 passed, 1 skipped
