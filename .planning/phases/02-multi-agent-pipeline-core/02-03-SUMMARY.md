---
phase: 02-multi-agent-pipeline-core
plan: "03"
subsystem: agents
tags: [openai-agents, pydantic, fact-extraction, rag, pinecone, neo4j, asyncio]

requires:
  - phase: 02-multi-agent-pipeline-core
    plan: "01"
    provides: test stubs defining behavioral contracts (RETR-01, RETR-02, RETR-03)

provides:
  - ExtractedFacts Pydantic model with 7 fields
  - _run_fact_extraction async coroutine (RETR-02)
  - _merged_rag_retrieval async coroutine (RETR-01, RETR-03)

affects:
  - 02-04-PLAN (pipeline_coordinator._run_pipeline imports _run_fact_extraction and _merged_rag_retrieval)

tech-stack:
  added: []
  patterns:
    - "OpenAI Agents SDK Agent + Runner.run with output_type=PydanticModel (mirrors editor_agent.py)"
    - "Dual-check: isinstance(result.final_output, Model) else Model.model_validate(result.final_output)"
    - "asyncio.to_thread wrapping synchronous PineconeStore.search"
    - "asyncio.gather with return_exceptions=True for parallel fan-out to both RAG sources"
    - "Module-level singleton pattern: _fact_agent and _store constructed once at import time"
    - "Flexible function signatures supporting both plan-spec names (transcript_text, graph_user_id) and test contract names (transcript, user_id)"

key-files:
  created:
    - confluence_logic/agents/fact_extraction_agent.py
  modified: []

decisions:
  - "Used flexible parameter signatures for _run_fact_extraction and _merged_rag_retrieval to bridge gap between plan spec (transcript_text, graph_user_id) and test contract (transcript, user_id); positional and keyword forms both work"
  - "Imported confluence_page_graph with deferred import inside _merged_rag_retrieval to avoid potential circular import issues (matching pattern from existing codebase)"
  - "Capped query_terms at 3 for concurrency fan-out control: 3 Neo4j + 3 Pinecone = 6 concurrent coroutines max (T-02-03-02 mitigation)"
  - "_merged_rag_retrieval placed in fact_extraction_agent.py per plan spec; test imports it from pipeline_coordinator (plan 02-02) — test remains XFAIL until plan 02-02 creates pipeline_coordinator; this is correct per success criteria (XFAIL not ERROR)"

metrics:
  duration: 2min
  completed: 2026-05-12
  tasks_completed: 2
  files_created: 1
---

# Phase 2 Plan 03: FactExtractionAgent — Transcript Facts + Merged RAG Retrieval Summary

**ExtractedFacts Pydantic model + _run_fact_extraction (OpenAI Agents SDK) + _merged_rag_retrieval (parallel Neo4j + Pinecone fan-out with page_id deduplication)**

## Performance

- **Duration:** ~2 min
- **Started:** 2026-05-12T01:15:53Z
- **Completed:** 2026-05-12T01:17:53Z
- **Tasks:** 2 of 2
- **Files created:** 1

## Accomplishments

- Created `confluence_logic/agents/fact_extraction_agent.py` (218 lines) implementing both RETR-02 and RETR-01/RETR-03
- `ExtractedFacts` Pydantic BaseModel with 7 empty-list-default fields: `decisions`, `action_items`, `new_requirements`, `owners`, `deadlines`, `doc_worthy_updates`, `query_terms`
- `_run_fact_extraction`: async, uses OpenAI Agents SDK `Runner.run` with `output_type=ExtractedFacts`; caps input at `JARVIS_FACT_INPUT_MAX_CHARS=30000`; falls back to empty `ExtractedFacts()` on any exception (never raises to caller)
- `_merged_rag_retrieval`: async, fans out to both Neo4j (`query_user_confluence_graph`, async-native) and Pinecone (`asyncio.to_thread(PineconeStore.search)`) in parallel; deduplicates by `page_id`; returns sorted-by-score, capped at `JARVIS_PIPELINE_MAX_PAGES=8`
- Import sanity check: `OK` — no network calls at import time
- pytest: `test_merged_rag_queries_both_sources` and `test_fact_extraction_output_schema` both `XFAIL` (not `ERROR`) — correct per success criteria

## Task Commits

Each task was committed atomically:

1. **Task 1+2: Create fact_extraction_agent.py with ExtractedFacts, _run_fact_extraction, _merged_rag_retrieval** - `be820fe` (feat)

**Plan metadata:** _(this commit, docs)_

## Files Created/Modified

- `confluence_logic/agents/fact_extraction_agent.py` — ExtractedFacts model, FACT_EXTRACTION_PROMPT, _fact_agent singleton, _run_fact_extraction, _get_store singleton, _merged_rag_retrieval

## Decisions Made

- Flexible parameter signatures on both functions: `_run_fact_extraction(transcript_text=None, *, transcript=None)` and `_merged_rag_retrieval(graph_user_id=None, query_terms=None, *, user_id=None)`. The plan spec uses `transcript_text`/`graph_user_id` but the test contracts use `transcript`/`user_id`. Both forms now work.
- Deferred import of `confluence_page_graph` inside `_merged_rag_retrieval` to match codebase circular-import avoidance pattern (see `tools.py`).
- `_merged_rag_retrieval` placed in `fact_extraction_agent.py` as specified by plan 02-03. The test file imports it from `pipeline_coordinator` — that test stays XFAIL until plan 02-02 (`pipeline_coordinator.py`) is implemented, which is correct and expected.
- Query terms capped at 3 inside `_merged_rag_retrieval` to limit concurrency fan-out (T-02-03-02 mitigation from threat model).

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Test contract signature mismatch**
- **Found during:** Task 1 analysis
- **Issue:** Plan spec defined `_run_fact_extraction(transcript_text: str)` but the test in `test_pipeline.py` calls `_run_fact_extraction(transcript=short_transcript)` with a list argument. Similarly `_merged_rag_retrieval(graph_user_id, query_terms)` but test calls it with `user_id=` keyword.
- **Fix:** Used flexible keyword-based signatures with `Optional` positional args and starred keyword-only params, plus a `_transcript_to_text` helper that handles both list-of-dicts and string inputs.
- **Files modified:** `confluence_logic/agents/fact_extraction_agent.py`
- **Commit:** `be820fe`

## Known Stubs

None — `fact_extraction_agent.py` contains no hardcoded placeholder data; all logic is real implementation.

## Threat Surface Scan

No new network endpoints or auth paths introduced. `fact_extraction_agent.py` is a pure library module (no HTTP server routes). The threat mitigations from the plan's STRIDE register are applied:

- **T-02-03-01** (Tampering): Transcript passed as user-role content; `FACT_EXTRACTION_PROMPT` is hardcoded and not user-influenced.
- **T-02-03-02** (DoS): `query_terms[:3]` cap enforced; `JARVIS_PIPELINE_MAX_PAGES=8` cap enforced.
- **T-02-03-04** (Tampering): `JARVIS_FACT_INPUT_MAX_CHARS=30000` applied before LLM call.

## Self-Check: PASSED

- `confluence_logic/agents/fact_extraction_agent.py` exists (created via Write tool — 218 lines)
- Commit `be820fe` exists in git log
- Import sanity: `from confluence_logic.agents.fact_extraction_agent import ExtractedFacts, _run_fact_extraction, _merged_rag_retrieval, JARVIS_FACT_INPUT_MAX_CHARS, JARVIS_PIPELINE_MAX_PAGES` → `OK`
- pytest: `2 xfailed` (not errors) — correct RED/XFAIL state

---
*Phase: 02-multi-agent-pipeline-core*
*Completed: 2026-05-12*
