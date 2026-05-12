---
phase: 02-multi-agent-pipeline-core
plan: "02"
subsystem: database
tags: [supabase, postgresql, rls, rest-api, pipeline-persistence]

# Dependency graph
requires:
  - phase: 01-schema-blockers
    provides: pipeline_jobs table (proposals has FK dependency on pipeline_jobs)
provides:
  - proposals table DDL with FK to pipeline_jobs, RLS policies, and index
  - create_pipeline_job, update_pipeline_job, upsert_proposal synchronous helpers in supabase_store.py
affects:
  - 02-04-pipeline-orchestrator
  - 03-sse-streaming

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "requests.post/patch with Supabase REST API — service role key via _rest_headers()"
    - "is_configured() guard before any HTTP call — silent no-op when env not set"
    - "Strip None values from payload before upsert — matches existing upsert_history pattern"

key-files:
  created: []
  modified:
    - confluence_logic/review/supabase_schema.sql
    - confluence_logic/review/supabase_store.py

key-decisions:
  - "FK from proposals to pipeline_jobs with ON DELETE CASCADE — job deletion cleans up all cards"
  - "Index on (job_id, created_at desc) chosen for Phase 3 SSE streaming query pattern"
  - "All three helpers are synchronous — callers use asyncio.to_thread when needed"
  - "User applies proposals DDL manually via Supabase Dashboard SQL Editor (same gate as pipeline_jobs in Phase 1)"

patterns-established:
  - "Supabase persistence helpers: guard with is_configured(), strip None values, catch Exception + logger.warning"
  - "RLS policy pattern: auth.uid() = user_id for select/insert/update — server writes via service role key"

requirements-completed:
  - PIPE-01
  - PIPE-04

# Metrics
duration: ~10min
completed: 2026-05-12
---

# Phase 2 Plan 02: Supabase Proposals Table + Store Helpers Summary

**proposals DDL block appended to supabase_schema.sql with FK to pipeline_jobs and RLS; three synchronous Supabase helpers (create_pipeline_job, update_pipeline_job, upsert_proposal) added to supabase_store.py**

## Performance

- **Duration:** ~10 min
- **Started:** 2026-05-12
- **Completed:** 2026-05-12
- **Tasks:** 2 (+ 1 human checkpoint approved)
- **Files modified:** 2

## Accomplishments

- Appended proposals DDL block to supabase_schema.sql: FK to pipeline_jobs (ON DELETE CASCADE), 3 RLS policies (select/insert/update via auth.uid() = user_id), index on (job_id, created_at desc)
- Added create_pipeline_job helper: POSTs to pipeline_jobs, returns generated job_id string or None
- Added update_pipeline_job helper: PATCHes pipeline_jobs row with stage/status/error/completed_at; no-ops on empty payload
- Added upsert_proposal helper: POSTs to proposals table; strips None values; guards on job_id + user_id presence
- Human checkpoint approved: Supabase DDL application deferred to user (code-only approval)

## Task Commits

Each task was committed atomically:

1. **Task 1: Append proposals DDL to supabase_schema.sql** - `527d840` (feat)
2. **Task 2: Add create_pipeline_job, update_pipeline_job, upsert_proposal to supabase_store.py** - `7437e1e` (feat)

## Files Created/Modified

- `confluence_logic/review/supabase_schema.sql` - proposals table DDL block appended after pipeline_jobs block; FK to pipeline_jobs, RLS enabled with 3 policies, index on (job_id, created_at desc)
- `confluence_logic/review/supabase_store.py` - three new synchronous helpers appended after existing functions; no existing functions modified

## Decisions Made

- FK from proposals to pipeline_jobs uses ON DELETE CASCADE so that deleting a job automatically removes all its proposal cards
- Index on (job_id, created_at desc) was chosen specifically to support the Phase 3 SSE streaming query pattern (ordered per-job card delivery)
- All three helpers are synchronous (not async) to match the existing pattern in supabase_store.py; callers that need async use asyncio.to_thread
- User applies proposals DDL manually via Supabase Dashboard SQL Editor — same human gate pattern as pipeline_jobs in Phase 1 checkpoint

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None.

## User Setup Required

The proposals DDL block must be applied in Supabase before Plan 02-04 (pipeline orchestrator) can write proposal cards.

Steps:
1. Open Supabase Dashboard → SQL Editor
2. Open `confluence_logic/review/supabase_schema.sql`
3. Copy ONLY the proposals block (from `-- proposals: incremental per-card proposal storage` to end of file)
4. Paste and click Run
5. Verify proposals table appears in Table Editor with correct columns and RLS enabled

**Prerequisite:** pipeline_jobs table (Phase 1 checkpoint) must exist first — proposals has a FK dependency on it.

## Next Phase Readiness

- Plan 02-03 (FactExtractionAgent) already complete
- Plan 02-04 (pipeline orchestrator) can now import create_pipeline_job, update_pipeline_job, upsert_proposal from supabase_store
- Supabase table must be created by user before 02-04 pipeline calls will persist data (non-blocking for code; blocking for runtime)

---
*Phase: 02-multi-agent-pipeline-core*
*Completed: 2026-05-12*
