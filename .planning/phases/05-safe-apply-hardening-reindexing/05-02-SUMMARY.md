---
phase: "05"
plan: "02"
subsystem: confluence_logic/review
tags: [apply-hardening, tdd-green, version-chain, pre-flight, reindex, direct-apply]
dependency_graph:
  requires: [test_apply_hardening.py (05-01)]
  provides: [APPLY-01, APPLY-02, APPLY-03 implementations]
  affects: [confluence_logic/review/api.py, confluence_logic/confluence_page_graph.py]
tech_stack:
  added: []
  patterns: [asyncio.create_task fire-and-forget, module-level in-memory version cache, session_id fallback from proposal dict]
key_files:
  created: []
  modified:
    - confluence_logic/review/api.py
    - confluence_logic/confluence_page_graph.py
decisions:
  - session_id falls back to proposal.get('session_id') when not passed as kwarg — allows test callers and legacy pipeline paths to store it in the proposal dict without breaking the cache key
  - _fire_reindex placed between _version_cache declaration and _direct_apply_change so the hot path stays readable and re-index logic is self-contained
  - _execute_pipeline_proposal rewired to _direct_apply_change directly; EditorAgent path preserved in _execute_single_change and _execute_pipeline_proposals_batched (unaffected)
metrics:
  duration: "20 minutes"
  completed_date: "2026-05-16"
  tasks_completed: 3
  files_created: 0
---

# Phase 05 Plan 02: Heading Pre-flight, Version Chain, Re-index + Wiring Summary

**One-liner:** Three surgical patches to `_direct_apply_change` — pre-flight heading gating, version-chain cache across sequential same-page accepts, and fire-and-forget post-commit re-index — plus direct wiring of `_execute_pipeline_proposal` so all patches actually fire.

## What Was Built

### Task 1: `_version_cache` + APPLY-01 pre-flight + APPLY-02 version chain

**`confluence_logic/review/api.py`**

- Added module-level `_version_cache: dict[tuple[str, str], int] = {}` (APPLY-02).
- Added `session_id: Optional[str] = None` parameter to `_direct_apply_change`.
- Added fallback: `session_id = proposal.get("session_id") or None` when the parameter is `None` — allows tests and legacy callers to store session_id in the proposal dict.
- **DELETE branch (APPLY-01):** Pre-flight heading check immediately after `extract_headings(live_html)`. If the target heading is absent (case-insensitive substring), returns `{success: False, error: "heading_not_found", message: "..."}` before calling `push_update`. On success, stores `version + 1` in cache and fires re-index.
- **EDIT branch (APPLY-01):** Pre-flight heading check at the top of each retry iteration, after fetching `live_html` and `meta`. Same return contract as delete branch.
- **EDIT branch (APPLY-02):** Replaced bare `push_update` call with a try/except block that uses the cached version as `expected_version` when available. On `ValueError("Version Conflict ...")`, invalidates the cache entry and returns `{success: False, error: "version_conflict"}` without re-raising. On success, stores `version + 1` in cache.

### Task 2: `_fire_reindex` helper + `refresh_page_in_graph`

**`confluence_logic/review/api.py`**

- Added `_fire_reindex(resolved_id, proposal, session_id)` immediately above `_direct_apply_change`. Derives `graph_user_id` from the proposal's `user_id`. Schedules an async inner coroutine via `asyncio.create_task()` that:
  - Runs `IngestionPipeline().process_page(resolved_id)` in a thread (Pinecone).
  - Calls `confluence_page_graph.refresh_page_in_graph(graph_user_id, resolved_id)` (Neo4j).
  - Each step catches all exceptions and logs at WARNING — re-index failure is non-fatal.
  - `asyncio.create_task()` itself is wrapped in a try/except — scheduling errors are also swallowed.

**`confluence_logic/confluence_page_graph.py`**

- Appended `async def refresh_page_in_graph(user_id, page_id)` after `refresh_known_user_graphs_forever`. The function:
  1. Guards on `user_id` and `page_id` being non-empty and the Neo4j driver being available.
  2. Runs a `DETACH DELETE` Cypher query to remove the stale `CfPage` node (cascades to `CfSection` children).
  3. Fetches fresh page metadata via `ConfluenceConnector.get_page_metadata`.
  4. Calls `_write_single_page(driver, user_id, connector, page_dict)` to re-create the node.
  5. Swallows all exceptions with `logger.warning` and returns `False` on failure, `True` on success.

### Task 3: Rewire `_execute_pipeline_proposal`

**`confluence_logic/review/api.py`**

- Replaced the EditorAgent-based body of `_execute_pipeline_proposal` with a direct call to `_direct_apply_change(proposal, session_id=session_id)` inside the existing per-page lock.
- Kept the top guard logic (proposal lookup, status=executed/rejected early returns).
- Maps `result["success"]` → `update_proposal_status("executed"/"failed")` → response dict.
- Removed: `_format_approved_change_request`, `_get_editor_agent`, `confluence_page_graph.set_current_graph_user_id` calls from this function (they remain in `_execute_single_change` and `_execute_pipeline_proposals_batched` which are unaffected).

## Test Results (GREEN phase)

```
11 collected, 11 passed
```

All 11 tests in `confluence_logic/tests/test_apply_hardening.py` pass.

Pre-existing `test_execute_changes_uses_existing_editor_agent_logic` failure in `test_review_api.py` confirmed pre-existing (verified by stash test before changes). All other 15 tests in `test_review_api.py` pass.

## Commits

| Hash | Message |
|------|---------|
| 40d995e | feat(05-02): add heading pre-flight, version chain, post-commit re-index, and direct-apply wiring |

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] session_id not propagated from proposal dict to version cache**

- **Found during:** Test run (tests 8 and 9 of 11 failing)
- **Issue:** Tests passed `session_id` in the proposal dict (`"session_id": "sess-seq"`) not as the `session_id` kwarg. The cache key `(session_id, resolved_id)` was always `None` because `session_id` parameter was `None`.
- **Fix:** Added `if session_id is None: session_id = proposal.get("session_id") or None` immediately after extracting proposal fields. This also correctly handles legacy pipeline callers that embed session_id in the proposal payload.
- **Files modified:** `confluence_logic/review/api.py`
- **Commit:** 40d995e

## Known Stubs

None — all three patches are fully implemented and verified by tests.

## Threat Flags

None — no new network endpoints, auth paths, or schema changes. Changes are confined to internal apply-path logic.

## Self-Check: PASSED

- `confluence_logic/review/api.py` — FOUND (modified, 40d995e)
- `confluence_logic/confluence_page_graph.py` — FOUND (modified, 40d995e)
- 11/11 tests pass in `test_apply_hardening.py`
- `from confluence_logic.review import api` — import ok
- `from confluence_logic.confluence_page_graph import refresh_page_in_graph` — import ok
- `grep _version_cache api.py` — declaration at line 1395, usages at lines 1564, 1911, 1919, 1924
- `grep heading_not_found api.py` — appears at lines 1547 (delete branch) and 1669 (edit branch)
