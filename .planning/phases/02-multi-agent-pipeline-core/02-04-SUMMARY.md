---
phase: 02-multi-agent-pipeline-core
plan: "04"
subsystem: agents
tags: [openai-agents, fastapi, asyncio, drafter, verifier, pipeline-orchestrator, pipe-01, pipe-02, pipe-03, pipe-04]

requires:
  - phase: 02-multi-agent-pipeline-core
    plan: "02"
    provides: create_pipeline_job, update_pipeline_job, upsert_proposal in supabase_store.py
  - phase: 02-multi-agent-pipeline-core
    plan: "03"
    provides: ExtractedFacts, _run_fact_extraction, _merged_rag_retrieval in fact_extraction_agent.py

provides:
  - DrafterAgent (_run_drafter coroutine) with DRAFTER_MODEL=gpt-5-mini
  - VerifierAgent (_run_verifier coroutine) with VERIFIER_MODEL=gpt-5.4-mini
  - POST /review/pipeline/start endpoint (HTTP 202) in api.py
  - _run_pipeline background orchestrator in api.py
  - _draft_verify_persist per-page helper in api.py

affects:
  - 03-sse-streaming (polls pipeline_jobs and proposals tables for progress events)
  - sync-sage-bot UI (consumes pipeline proposals from proposals table)

tech-stack:
  added: []
  patterns:
    - "asyncio.create_task for fire-and-forget background pipeline (not FastAPI BackgroundTasks)"
    - "asyncio.gather(*tasks, return_exceptions=True) for concurrent per-page drafter pool"
    - "asyncio.to_thread wrapping synchronous supabase_store helpers in async pipeline"
    - "Per-page upsert_proposal inside _draft_verify_persist (not batch at end) — PIPE-04"
    - "_normalize_draft: existing pages (page_id not None) must not produce create proposals"
    - "Zero-RAG fallback: page_id=None pages go through same _draft_verify_persist path as D-06"
    - "Explicit graph_user_id argument to _run_pipeline — not ContextVar (Pitfall 4)"

key-files:
  created:
    - confluence_logic/agents/drafter_agent.py
    - confluence_logic/agents/verifier_agent.py
  modified:
    - confluence_logic/review/api.py

decisions:
  - "drafter_agent.py and verifier_agent.py kept as separate modules (not merged into api.py) for testability and single-responsibility"
  - "_utc_now_iso() already existed in api.py — not added again (plan Step C was already implemented)"
  - "JARVIS_FACT_INPUT_MAX_CHARS imported from fact_extraction_agent, not redefined in api.py (plan constraint 4)"
  - "test_pipeline_start_returns_202 remains XFAIL because it imports from pipeline_coordinator module — correct per baseline; route is fully implemented and 401/202 verified via _verify_task2.py"

metrics:
  duration: ~20min
  completed: 2026-05-12
  tasks_completed: 2
  files_created: 2
  files_modified: 1
---

# Phase 2 Plan 04: DrafterAgent + VerifierAgent + Pipeline Endpoint Summary

**DrafterAgent (gpt-5-mini) and VerifierAgent (gpt-5.4-mini) created; POST /review/pipeline/start endpoint + _run_pipeline 4-stage background orchestrator added to api.py — completes PIPE-01 through PIPE-04**

## Performance

- **Duration:** ~20 min
- **Started:** 2026-05-12
- **Completed:** 2026-05-12
- **Tasks:** 2 of 2
- **Files created:** 2
- **Files modified:** 1

## Accomplishments

- Created `confluence_logic/agents/drafter_agent.py` with `_run_drafter`, `DRAFTER_SYSTEM_PROMPT`, `DRAFTER_MODEL`, and `_normalize_draft`
- `_run_drafter`: async coroutine; creates fresh Agent per page; parses Runner.run output; falls back to safe proposal dict on exception
- `_normalize_draft`: enforces change_type constraint — existing pages (page_id not None) cannot produce `create` proposals; zero-RAG fallback (page_id=None) allows `create` (D-09)
- Created `confluence_logic/agents/verifier_agent.py` with `_run_verifier`, `VERIFIER_SYSTEM_PROMPT`, `VERIFIER_MODEL`, `VERIFIER_MAX_TOKENS`
- `_run_verifier`: async coroutine using `asyncio.to_thread` + `response_format=json_object`; enriches draft with confidence/risk/verifier_note/transcript_evidence; never drops card on failure (PIPE-03)
- Added to `confluence_logic/review/api.py`:
  - New imports: fact_extraction_agent, drafter_agent, verifier_agent
  - `PipelineStartRequest` Pydantic model
  - `_draft_verify_persist`: per-page draft+verify+upsert_proposal (PIPE-04 incremental writes)
  - `_run_pipeline`: 4-stage orchestrator (fact_extraction → retrieval → drafting → complete)
  - `POST /review/pipeline/start`: returns 401 without auth, 202 + {job_id, status:'accepted'} with auth
  - Zero-RAG fallback (D-06): up to 3 create proposals when candidates empty + doc_worthy_updates non-empty
  - `asyncio.gather(*tasks, return_exceptions=True)` for drafter pool concurrency
  - graph_user_id passed as explicit function argument (not ContextVar inheritance — Pitfall 4)

## Task Commits

1. **Task 1: Create drafter_agent.py and verifier_agent.py** — `b502610` (feat)
2. **Task 2: Add POST /review/pipeline/start endpoint and _run_pipeline to api.py** — `6715ead` (feat)

## Files Created/Modified

- `confluence_logic/agents/drafter_agent.py` — DrafterAgent implementation (PIPE-02); _normalize_draft enforces D-09 constraint
- `confluence_logic/agents/verifier_agent.py` — VerifierAgent implementation (PIPE-03); never drops card on failure
- `confluence_logic/review/api.py` — PipelineStartRequest model, _draft_verify_persist, _run_pipeline, POST /review/pipeline/start endpoint added at end of file; existing functions unchanged

## Decisions Made

- Kept `drafter_agent.py` and `verifier_agent.py` as separate modules rather than embedding in `api.py` — maintains single-responsibility and makes each agent independently testable
- `_utc_now_iso()` was already defined in `api.py` (line 75-76 of original file) — plan's Step C was already satisfied; not duplicated
- `JARVIS_FACT_INPUT_MAX_CHARS` imported from `fact_extraction_agent` per plan constraint 4 — not redefined
- `test_pipeline_start_returns_202` remains XFAIL because the test suite imports from `pipeline_coordinator` (which doesn't exist) and pytest-asyncio is not configured — both conditions existed before this plan; the 401/202 behavior was verified independently via `_verify_task2.py`

## Deviations from Plan

None - plan executed exactly as written. _utc_now_iso() was pre-existing in api.py so Step C was a no-op.

## Known Stubs

None — all pipeline logic is real implementation. `upsert_proposal` calls go to real Supabase (guarded by `is_configured()`).

## Threat Surface Scan

New network endpoint added: `POST /review/pipeline/start`. This is documented in the plan's STRIDE threat register:

| Flag | File | Description |
|------|------|-------------|
| threat_flag: elevation-of-privilege | confluence_logic/review/api.py | POST /review/pipeline/start — mitigated: _auth_user_from_header called first; 401 raised if user is None (T-02-04-01) |
| threat_flag: information-disclosure | confluence_logic/review/api.py | user_id taken from authenticated token, not request body; proposals table RLS is user-scoped (T-02-04-02) |

Both threats are mitigated as designed.

## Self-Check: PASSED

- `confluence_logic/agents/drafter_agent.py` exists
- `confluence_logic/agents/verifier_agent.py` exists
- Commit `b502610` exists in git log
- Commit `6715ead` exists in git log
- Import sanity: `from confluence_logic.agents.drafter_agent import _run_drafter, DRAFTER_MODEL` → OK
- Import sanity: `from confluence_logic.agents.verifier_agent import _run_verifier, VERIFIER_MODEL` → OK
- Route check: `/review/pipeline/start` registered in router routes → OK
- 401 unauthenticated: `POST /review/pipeline/start` → 401 → OK
- Model checks: DRAFTER_MODEL=`gpt-5-mini`, VERIFIER_MODEL=`gpt-5.4-mini` → OK
- pytest: 6 xfailed (same as baseline) — no regressions

---
*Phase: 02-multi-agent-pipeline-core*
*Completed: 2026-05-12*
