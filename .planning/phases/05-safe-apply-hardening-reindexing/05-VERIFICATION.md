---
phase: 05-safe-apply-hardening-reindexing
verified: 2026-05-16T12:00:00Z
status: passed
score: 3/3 must-haves verified
overrides_applied: 0
re_verification:
  previous_status: gaps_found
  previous_score: 2/3
  gaps_closed:
    - "CR-01: refresh_page_in_graph DETACH DELETE no longer includes graph_kind filter on CfPage nodes — stale page node is now properly deleted before re-creation"
    - "CR-03: _fire_reindex in DELETE branch now fires unconditionally on success — session_id guard only protects _version_cache update, not re-index scheduling"
  gaps_remaining: []
  regressions: []
---

# Phase 5: Safe Apply Hardening + Re-indexing Verification Report

**Phase Goal:** Accepted changes are applied to Confluence safely — section anchors are verified before any edit, multi-card sequences on the same page never use stale version numbers, and every committed page is immediately re-indexed in both Pinecone and the Neo4j confluence_page_graph so the RAG layer stays current

**Verified:** 2026-05-16T12:00:00Z
**Status:** passed
**Re-verification:** Yes — after gap closure (CR-01 and CR-03 fixed)

---

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | Accepting an edit card whose section_heading no longer exists in the live Confluence page returns a clear error to the UI rather than silently creating a misplaced edit (APPLY-01) | VERIFIED | Pre-flight check at api.py:1540-1552 (DELETE branch) and api.py:1661-1674 (EDIT branch). Returns `{success: False, error: "heading_not_found", message: "Section '...' no longer exists..."}`. Create and title cards bypass check. Tests 1-6 pass. |
| 2 | When two accepted cards target the same page, the second card's commit uses the page version returned by the first card's successful commit — a stale-version conflict error never occurs for sequential same-page accepts (APPLY-02) | VERIFIED | Module-level `_version_cache` at api.py:1395, populated at api.py:1924 (edit) and api.py:1565 (delete). Cache used as expected_version at api.py:1911-1914. Version conflict evicts cache at api.py:1919. Tests 7-9 pass. |
| 3 | After a page is committed, querying the RAG pipeline with a topic from that page's new content surfaces the updated page within the same session — stale graph nodes are not returned (APPLY-03) | VERIFIED | `_fire_reindex` fires unconditionally on success (edit: api.py:1926, delete: api.py:1566 — outside session_id guard). Pinecone re-index via `IngestionPipeline().process_page` wired at api.py:1413. Neo4j re-index via `refresh_page_in_graph` wired at api.py:1420. CR-01 fixed: DETACH DELETE at confluence_page_graph.py:477-480 now uses only `user_id` and `page_id` — no spurious `graph_kind` filter — stale CfPage node is properly deleted before _write_single_page re-creates it. Tests 10-11 pass. |

**Score:** 3/3 truths verified

---

### Deferred Items

None.

---

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `confluence_logic/tests/test_apply_hardening.py` | 11-test suite covering APPLY-01/02/03 | VERIFIED | Exists, 258 lines, 11 tests, all pass per reported test run. Full behavioral coverage of heading pre-flight (tests 1-6), version chain (tests 7-9), and fire-and-forget re-index (tests 10-11). |
| `confluence_logic/review/api.py` | `_version_cache`, `_fire_reindex`, APPLY-01 pre-flight in DELETE+EDIT branches, APPLY-02 version chain in EDIT branch, rewired `_execute_pipeline_proposal` | VERIFIED | `_version_cache` at line 1395. `_fire_reindex` at lines 1398-1428. `heading_not_found` in DELETE branch (line 1547) and EDIT branch (line 1669). CR-03 fixed: `_fire_reindex` at line 1566 is inside `if success:` but outside `if session_id:` — fires unconditionally on success. Version chain at lines 1911-1924. `_execute_pipeline_proposal` calls `_direct_apply_change` at line 2062. |
| `confluence_logic/confluence_page_graph.py` | `refresh_page_in_graph` function with correct DETACH DELETE | VERIFIED | Function at lines 458-507. CR-01 fixed: DETACH DELETE Cypher at line 477 is `MATCH (p:CfPage {user_id: $user_id, page_id: $page_id}) DETACH DELETE p` with parameters dict `{"user_id": user_id, "page_id": page_id}` — no `graph_kind` key. Stale CfPage node (and its CfSection children via DETACH DELETE cascade) is properly removed before `_write_single_page` re-creates it at line 500. |

---

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `_direct_apply_change` | `heading_not_found` return | Pre-flight check in DELETE branch | WIRED | api.py:1540-1552 — `extract_headings` → substring match → early return if absent |
| `_direct_apply_change` | `heading_not_found` return | Pre-flight check in EDIT branch | WIRED | api.py:1661-1674 — inside retry loop, `extract_headings` → substring match → early return |
| `_direct_apply_change` DELETE branch | `_fire_reindex` | Unconditional on success | WIRED | api.py:1563-1566 — `if success:` outer guard, `_fire_reindex` at line 1566 outside `if session_id:` at line 1564. CR-03 fixed. |
| `_direct_apply_change` EDIT branch | `_fire_reindex` | Unconditional on success | WIRED | api.py:1926 — called after `if success:` block, no session_id gating on re-index. |
| `_direct_apply_change` | `_version_cache` | APPLY-02 version chain | WIRED | Read at 1911, written at 1924 (edit), 1565 (delete). |
| `_fire_reindex` | `refresh_page_in_graph` | Neo4j re-index | WIRED | api.py:1420 calls `confluence_page_graph.refresh_page_in_graph`. CR-01 fixed: DETACH DELETE correctly removes CfPage node. |
| `_fire_reindex` | `IngestionPipeline().process_page` | Pinecone re-index | WIRED | api.py:1413 — lazy import + `asyncio.to_thread` call. |
| `_execute_pipeline_proposal` | `_direct_apply_change` | Rewiring | WIRED | api.py:2062. All three patches fire via this path. |

---

### Data-Flow Trace (Level 4)

| Artifact | Data Variable | Source | Produces Real Data | Status |
|----------|---------------|--------|-------------------|--------|
| `_direct_apply_change` EDIT branch | `live_html`, `version` | `connector.fetch_page_html`, `connector.get_page_metadata` | Yes — live Confluence API calls in asyncio thread | FLOWING |
| `_direct_apply_change` DELETE branch | `live_html`, `version` | Same as above | Yes | FLOWING |
| `_version_cache` | `(session_id, resolved_id) → int` | Written on successful commit, read on next commit | Real version numbers from Confluence | FLOWING |
| `refresh_page_in_graph` | `CfPage`, `CfSection` nodes | `connector.get_page_metadata`, `_write_single_page` | Yes — CR-01 fixed: DETACH DELETE removes stale CfPage + children; `_write_single_page` re-creates from live Confluence data | FLOWING |

---

### Behavioral Spot-Checks

| Behavior | Evidence | Status |
|----------|----------|--------|
| All 11 APPLY tests pass | Reported: 11/11 pass post-fix | PASS |
| CR-01 fix: DETACH DELETE uses only user_id + page_id | confluence_page_graph.py:477-480 — Cypher string and params dict verified by code read | PASS |
| CR-03 fix: _fire_reindex outside session_id guard in DELETE branch | api.py:1563-1566 — _fire_reindex at line 1566 is inside `if success:` but NOT inside `if session_id:` at line 1564 | PASS |
| `_version_cache` module attribute exists | api.py:1395 — `_version_cache: dict[tuple[str, str], int] = {}` | PASS |
| `heading_not_found` in both branches | api.py:1547 (DELETE), api.py:1669 (EDIT) | PASS |
| `_fire_reindex` callable in module | api.py:1398-1428 — function definition confirmed | PASS |
| 21 pre-existing failures in other test files unchanged | Reported: same count before and after Phase 5 — zero new failures introduced | PASS |

---

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|-------------|--------|----------|
| APPLY-01 | 05-01-PLAN.md, 05-02-PLAN.md | Before applying an edit card, section anchor pre-flight check confirms section_heading still exists | SATISFIED | Pre-flight in DELETE (api.py:1540-1552) and EDIT (api.py:1661-1674) branches. 6 tests verify. |
| APPLY-02 | 05-01-PLAN.md, 05-02-PLAN.md | When multiple accepted cards target the same page_id, each card uses committed version from prior accept — no stale version chain | SATISFIED | `_version_cache` dict wired through edit and delete paths. 3 tests verify. |
| APPLY-03 | 05-01-PLAN.md, 05-02-PLAN.md | After a page is committed, the Neo4j confluence_page_graph node is invalidated/refreshed AND Pinecone is re-indexed | SATISFIED | `_fire_reindex` fires unconditionally on success in both edit and delete branches. `refresh_page_in_graph` DETACH DELETE fixed (no spurious graph_kind filter). Both Pinecone and Neo4j re-index paths wired. 2 tests verify scheduling. |

---

### Anti-Patterns Found

None blocking. Previous blockers (CR-01, CR-03) are resolved. The CR-02 edge-case noted in the initial verification (wrong version stored under a race condition with an external edit between sequential accepts) remains, but does not affect the sequential same-page accept scenario described by APPLY-02.

---

### Human Verification Required

None — all behavioral checks are programmatically verifiable.

---

## Gaps Summary

No gaps. All three must-have truths are verified.

**CR-01 closed:** `refresh_page_in_graph` DETACH DELETE (confluence_page_graph.py:477-480) now uses `MATCH (p:CfPage {user_id: $user_id, page_id: $page_id}) DETACH DELETE p` — the `graph_kind` filter has been removed. Stale CfPage nodes are correctly deleted before re-creation.

**CR-03 closed:** `_fire_reindex` in the DELETE branch (api.py:1566) is now called unconditionally when `success` is True. Only the `_version_cache` update at line 1564-1565 is gated on `session_id`. DELETE-section commits without a session_id now correctly trigger Pinecone and Neo4j re-indexing.

**Requirements traceability:** APPLY-01, APPLY-02, APPLY-03 mapped to Phase 5 in REQUIREMENTS.md (lines 119-121). All three now satisfied.

---

_Verified: 2026-05-16T12:00:00Z_
_Verifier: Claude (gsd-verifier)_
