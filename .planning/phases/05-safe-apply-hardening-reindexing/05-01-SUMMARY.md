---
phase: "05"
plan: "01"
subsystem: confluence_logic/tests
tags: [tdd, testing, apply-hardening, red-phase]
dependency_graph:
  requires: []
  provides: [test_apply_hardening.py]
  affects: [confluence_logic/review/api.py]
tech_stack:
  added: [pytest-asyncio]
  patterns: [unittest.mock.patch, AsyncMock, patch.object]
key_files:
  created:
    - confluence_logic/tests/test_apply_hardening.py
  modified: []
decisions:
  - Installed pytest-asyncio (was in requirements.txt but not in ml conda env) to allow async tests to run properly
  - Tests use patch.object(api, "_resolve_page_id") to isolate _direct_apply_change from page-lookup I/O
  - APPLY-02 tests reference api._version_cache directly — this confirms the attribute is absent and flags the missing implementation
metrics:
  duration: "5 minutes"
  completed_date: "2026-05-16"
  tasks_completed: 3
  files_created: 1
---

# Phase 05 Plan 01: Tests for Safe Apply Hardening (APPLY-01, APPLY-02, APPLY-03) Summary

**One-liner:** Failing test suite for `_direct_apply_change` covering pre-flight heading check, version-chain cache, and fire-and-forget re-index (TDD RED phase for Wave 1)

## What Was Built

Created `confluence_logic/tests/test_apply_hardening.py` with 11 `@pytest.mark.asyncio` tests that drive Wave 1 implementation:

- **Task 1 (Scaffold):** Module docstring, imports, `mock_connector` fixture (fetch_page_html, get_page_metadata, push_update stubs), `base_edit_proposal` and `base_delete_proposal` fixtures.
- **Task 2 (APPLY-01 tests):** 6 tests verifying pre-flight heading check behavior — missing headings should return `{success: False, error: "heading_not_found"}` for edit/delete cards; create and title cards bypass pre-flight.
- **Task 3 (APPLY-02/03 tests):** 5 tests verifying version chaining (`_version_cache` population, stale cache invalidation on conflict) and fire-and-forget re-index (`asyncio.create_task` scheduling, failure swallowing).

## Test Results (RED phase)

```
11 collected, 6 failed, 5 passed
```

**Failing tests (expected — features not yet implemented):**
- `test_edit_card_missing_heading_returns_heading_not_found` — APPLY-01: currently returns `success=True` (falls back to FULL_PAGE)
- `test_delete_card_missing_heading_returns_heading_not_found` — APPLY-01: returns generic delete error, not `heading_not_found`
- `test_version_cache_populated_after_successful_commit` — APPLY-02: `AttributeError: module has no attribute '_version_cache'`
- `test_second_accept_uses_cached_version_as_expected_version` — APPLY-02: same
- `test_version_conflict_invalidates_cache_and_returns_error` — APPLY-02: same
- `test_successful_commit_fires_reindex_create_task` — APPLY-03: `create_task` not called

**Passing tests (existing behavior already correct):**
- `test_edit_card_present_heading_proceeds_to_push_update` — heading present → proceeds
- `test_delete_card_present_heading_proceeds` — heading present → proceeds
- `test_create_card_skips_preflight_heading_check` — create bypasses pre-flight
- `test_title_card_skips_preflight_heading_check` — title bypasses pre-flight
- `test_reindex_failure_does_not_affect_success_response` — no create_task call = no exception

## Commits

| Hash | Message |
|------|---------|
| 657d3cd | test(05-01): add failing tests for APPLY-01/02/03 safe-apply hardening |

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] pytest-asyncio not installed in ml conda environment**
- **Found during:** Task 2 verification
- **Issue:** `pytest-asyncio` was listed in `requirements.txt` but not installed in the `ml` conda environment; tests failed with "async def functions are not natively supported" instead of `AssertionError`
- **Fix:** Ran `pip install pytest-asyncio` in the `ml` conda env; installed version 1.3.0
- **Files modified:** None (environment change only)

## Known Stubs

None — this is a test-only plan; no production code was modified.

## Threat Flags

None — test file only; no new network endpoints or auth paths introduced.

## Self-Check: PASSED

- `confluence_logic/tests/test_apply_hardening.py` — FOUND (657d3cd)
- 11 tests collected, 0 ImportErrors
- 6 fail with AssertionError/AttributeError (confirming APPLY-01/02/03 not implemented)
- 5 pass (confirming existing code handles those paths correctly)
