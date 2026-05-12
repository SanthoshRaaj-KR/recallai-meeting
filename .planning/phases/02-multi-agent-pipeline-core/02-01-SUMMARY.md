---
phase: 02-multi-agent-pipeline-core
plan: "01"
subsystem: testing
tags: [pytest, tdd, pipeline, rag, fact-extraction, supabase, pinecone, neo4j, openai-agents]

requires:
  - phase: 01-schema-and-blockers
    provides: pipeline_jobs DDL, ChangeItem schema, supabase_store helpers

provides:
  - pytest test stubs for all Phase 2 requirements (RETR-01 through PIPE-04)
  - 4 shared fixtures for Pinecone, Neo4j, Supabase, and OpenAI Runner mocking
  - 6 xfail test functions that define behavioral contracts for Wave 1 production code

affects:
  - 02-02-PLAN (fact_extraction_agent — tested by test_fact_extraction_output_schema)
  - 02-03-PLAN (pipeline_coordinator — tested by test_merged_rag_queries_both_sources, test_drafter_pool_parallel, test_verifier_enriches_not_drops, test_incremental_proposal_writes)
  - 02-04-PLAN (pipeline/start endpoint — tested by test_pipeline_start_returns_202)

tech-stack:
  added: []
  patterns:
    - "try/except ImportError guards on all Wave 1+ production module imports"
    - "@pytest.mark.xfail(strict=False) for tests targeting not-yet-implemented modules"
    - "monkeypatch + AsyncMock for async Neo4j and OpenAI Runner patching"

key-files:
  created:
    - confluence_logic/tests/test_pipeline.py
  modified: []

key-decisions:
  - "Use @pytest.mark.xfail(strict=False) + ImportError raise inside test body to achieve RED state without collection errors"
  - "Guard ALL Wave 1+ production imports at module level with try/except ImportError fallback to None"
  - "Use existing confluence_logic.review.api router (which exists) for test_pipeline_start_returns_202; 404 vs 202 is the RED indicator"
  - "pipeline_coordinator module name chosen for the merged-RAG + drafter + verifier + run_pipeline functions"

patterns-established:
  - "Pattern 1 (TDD RED): import guard at top of test file, raise ImportError inside test body as first assertion"
  - "Pattern 2 (fixture scope): all 4 fixtures use monkeypatch (function-scoped) to prevent state bleed between tests"
  - "Pattern 3 (async mock): AsyncMock for query_user_confluence_graph and Runner.run; MagicMock for synchronous Supabase calls"

requirements-completed:
  - RETR-01
  - RETR-02
  - RETR-03
  - PIPE-01
  - PIPE-02
  - PIPE-03
  - PIPE-04

duration: 15min
completed: 2026-05-12
---

# Phase 2 Plan 01: Multi-Agent Pipeline — Test Stubs Summary

**pytest test stubs with 4 shared mock fixtures defining behavioral contracts for RETR-01 through PIPE-04 before production code exists**

## Performance

- **Duration:** ~15 min
- **Started:** 2026-05-12T00:00:00Z
- **Completed:** 2026-05-12T00:15:00Z
- **Tasks:** 1 of 1
- **Files modified:** 1

## Accomplishments

- Created `confluence_logic/tests/test_pipeline.py` (367 lines) with 6 xfail test stubs covering every Phase 2 requirement
- Established 4 shared fixtures: `mock_pinecone_store`, `mock_neo4j_graph`, `mock_supabase`, `mock_openai_runner` using `monkeypatch` + `AsyncMock`
- All 6 tests collect without errors (`pytest --collect-only`: 6 items, 0 errors) and all run as `xfail` (RED state confirmed: `6 xfailed`)

## Task Commits

Each task was committed atomically:

1. **Task 1: Create test_pipeline.py with fixtures and 6 failing test stubs** - `487d403` (test)

**Plan metadata:** _(this commit, docs)_

## Files Created/Modified

- `confluence_logic/tests/test_pipeline.py` - 6 xfail test stubs + 4 fixtures for Phase 2 pipeline behavioral contracts

## Decisions Made

- Chose `pipeline_coordinator` as the module name for the merged RAG + drafter + verifier + `_run_pipeline` functions — consolidates coordination logic into one module rather than scattering across agent files
- Used `@pytest.mark.xfail(strict=False)` with an explicit `raise ImportError` as the first line in each test body — gives a clear RED indicator while still allowing pytest to collect normally
- Used the existing `confluence_logic.review.api` router for `test_pipeline_start_returns_202` — the `POST /review/pipeline/start` route does not exist yet so the test receives 404 vs expected 202, which is the correct RED state without needing a new app import
- All Supabase fixtures use empty dummy values (no real `SUPABASE_URL` or keys) — satisfies T-02-W0-02 threat mitigation

## Deviations from Plan

None — plan executed exactly as written.

## Issues Encountered

None.

## User Setup Required

None — no external service configuration required.

## Next Phase Readiness

- `confluence_logic/tests/test_pipeline.py` is the Wave 0 output; Wave 1 plans (02-02 through 02-04) implement against these contracts
- Wave 1 production code must: create `confluence_logic/agents/fact_extraction_agent.py` (satisfies `test_fact_extraction_output_schema`), create `confluence_logic/agents/pipeline_coordinator.py` (satisfies 4 coordinator tests), add `POST /review/pipeline/start` route to `confluence_logic/review/api.py` (satisfies `test_pipeline_start_returns_202`)
- The `pytest.mark.asyncio` warnings are expected — `pytest-asyncio` is installed but `asyncio_mode = "auto"` is not configured in `pytest.ini`; this is a pre-existing project state, not introduced by this plan

## Self-Check: PASSED

- `confluence_logic/tests/test_pipeline.py` exists and has 367 lines (min_lines: 120 — met)
- Commit `487d403` exists in git log
- All 6 test names match plan spec exactly: `test_merged_rag_queries_both_sources`, `test_fact_extraction_output_schema`, `test_pipeline_start_returns_202`, `test_drafter_pool_parallel`, `test_verifier_enriches_not_drops`, `test_incremental_proposal_writes`
- pytest collection: 6 items collected, 0 errors
- pytest run: 6 xfailed, 0 passed — RED state confirmed

---
*Phase: 02-multi-agent-pipeline-core*
*Completed: 2026-05-12*
