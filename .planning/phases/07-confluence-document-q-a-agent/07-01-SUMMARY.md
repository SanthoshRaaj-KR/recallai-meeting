---
phase: 07-confluence-document-q-a-agent
plan: "01"
subsystem: testing
tags: [pytest, pytest-asyncio, openai-agents, pinecone, neo4j, confluence, tdd]

requires:
  - phase: 06-safe-apply-hardening
    provides: "test infrastructure, patch patterns, pytest-asyncio usage"

provides:
  - "Failing test suite (4 tests) for ConfluenceQAAgent covering QA-01 through QA-04"
  - "RED baseline: ModuleNotFoundError on confluence_qa_agent until Plan 07-02 ships"
  - "Behavioral contracts for Pinecone-first retrieval, REST fallback, model split, and read gate"

affects:
  - "07-02-PLAN.md — implementation plan that turns these tests GREEN"

tech-stack:
  added: []
  patterns:
    - "TDD RED phase: test file imports non-existent module to establish behavioral contracts before implementation"
    - "_call() FunctionTool helper copied verbatim from test_flow.py for tool invocation in sync contexts"
    - "Patch targets use full confluence_logic.agents.confluence_qa_agent.* namespace to avoid singleton cross-contamination"

key-files:
  created:
    - confluence_logic/tests/test_confluence_qa_agent.py
  modified: []

key-decisions:
  - "test_qa_read_gate_blocks_mutations imports jarvis_agentic directly (no mocking) since _is_confluence_read_query is a pure synchronous function — no isolation needed"
  - "QA-04 covered in this file as an additional direct test rather than delegating entirely to test_jarvis_agentic.py — provides clear documentation and avoids relying on test discovery across files"
  - "Synthesis model assertion checks call_args.kwargs.get('model') == 'gpt-4o-mini' — tests the runtime contract, not a hardcoded string in the production module"

patterns-established:
  - "Pattern: assert agent.agent.model == 'gpt-5-mini' as structural check before behavioral mocking — validates config before behavior"
  - "Pattern: patch _get_openai_client (module-local lazy singleton) not openai.OpenAI directly — avoids patching the wrong module reference"

requirements-completed:
  - QA-01
  - QA-02
  - QA-03
  - QA-04

duration: 2min
completed: "2026-05-17"
---

# Phase 7 Plan 01: Confluence QA Agent - Failing Test Suite Summary

**4 RED pytest tests establish behavioral contracts for ConfluenceQAAgent: Pinecone-first retrieval, REST fallback when index is empty, gpt-5-mini/gpt-4o-mini model split, and mutation-query read gate**

## Performance

- **Duration:** 2 min
- **Started:** 2026-05-17T03:44:20Z
- **Completed:** 2026-05-17T03:45:55Z
- **Tasks:** 1
- **Files modified:** 1

## Accomplishments

- Created `confluence_logic/tests/test_confluence_qa_agent.py` with 4 test functions covering all four phase requirements (QA-01 through QA-04)
- Tests are correctly RED — `ModuleNotFoundError: No module named 'confluence_logic.agents.confluence_qa_agent'` on collection
- Syntax is valid (`ast.parse` passes); all patch targets use the correct full module path per threat mitigation T-07-01-02
- `_call()` FunctionTool helper copied verbatim from `test_flow.py` for use in tool-level testing

## Task Commits

Each task was committed atomically:

1. **Task 1: Write failing tests for QA-01, QA-02, QA-03 in test_confluence_qa_agent.py** - `191533c` (test)

**Plan metadata:** _(docs commit to follow)_

## Files Created/Modified

- `confluence_logic/tests/test_confluence_qa_agent.py` — 4 failing test functions covering all QA requirements; RED until Plan 07-02 implementation

## Decisions Made

- `test_qa_read_gate_blocks_mutations` imports `jarvis_agentic` directly without mocking — `_is_confluence_read_query` is a pure synchronous function that requires no isolation
- QA-04 is given its own test function in this file even though `test_jarvis_agentic.py` already covers `test_confluence_read_query_detection_excludes_mutations` — the plan explicitly documents this as a "reference-only stub" that adds extra assertions for completeness
- `_get_openai_client` is patched as a module-local function in `confluence_qa_agent` (not imported from `jarvis_agentic`) — enforces the no-circular-import constraint from RESEARCH.md Pitfall 6

## Deviations from Plan

None — plan executed exactly as written.

## Issues Encountered

None.

## Threat Surface Scan

No new network endpoints, auth paths, file access patterns, or schema changes introduced. Test file only imports and patches production code; no real credentials or Pinecone/Neo4j connections used. Consistent with T-07-01-01 (accept disposition).

## Known Stubs

None — this is a test-only file. No UI rendering or data flow stubs present.

## Self-Check

- `confluence_logic/tests/test_confluence_qa_agent.py` — FOUND
- Commit `191533c` — FOUND (`git log --oneline | head -1` confirms)

## Self-Check: PASSED

All created files exist. Commit recorded.

## Next Phase Readiness

- Plan 07-02 (implementation) can proceed — test contracts are locked in and will not be modified
- Tests will turn GREEN when `confluence_logic/agents/confluence_qa_agent.py` is created with `ConfluenceQAAgent`, `search_confluence_pages`, `get_full_page_content`, and `list_confluence_pages`
- No blockers

---
*Phase: 07-confluence-document-q-a-agent*
*Completed: 2026-05-17*
