---
phase: 05-safe-apply-hardening-reindexing
reviewed: 2026-05-16T12:12:57Z
depth: standard
files_reviewed: 3
files_reviewed_list:
  - confluence_logic/review/api.py
  - confluence_logic/confluence_page_graph.py
  - confluence_logic/tests/test_apply_hardening.py
findings:
  critical: 3
  warning: 5
  info: 3
  total: 11
status: issues_found
---

# Phase 5: Code Review Report

**Reviewed:** 2026-05-16T12:12:57Z
**Depth:** standard
**Files Reviewed:** 3
**Status:** issues_found

## Summary

Phase 5 introduces three surgical patches to `_direct_apply_change`: APPLY-01 (pre-flight heading check), APPLY-02 (version chain cache), and APPLY-03 (fire-and-forget re-index via `asyncio.create_task`). It also rewires `_execute_pipeline_proposal` to bypass EditorAgent in favour of the direct REST path, and adds `refresh_page_in_graph` as the lightweight Neo4j re-index target.

The overall direction is sound, but three blockers were found: a Cypher schema mismatch that silently makes `refresh_page_in_graph` a no-op, incorrect version cache behaviour when `session_id` is `None` paired with a stale cache entry that causes the wrong version to be submitted to Confluence, and a missing `_fire_reindex` call on successful DELETE-section commits when `session_id` is falsy. Five warnings cover logic gaps and test deficiencies that degrade correctness or observability.

---

## Critical Issues

### CR-01: `refresh_page_in_graph` DETACH DELETE Cypher matches no nodes — silently skips re-index

**File:** `confluence_logic/confluence_page_graph.py:477`

**Issue:** The Cypher query used to delete the stale `CfPage` node before re-creation includes the property `graph_kind: $graph_kind` as a filter:

```cypher
MATCH (p:CfPage {user_id: $user_id, page_id: $page_id, graph_kind: $graph_kind})
DETACH DELETE p
```

But `CfPage` nodes are never written with a `graph_kind` property. Looking at `_write_single_page` (line 283–303), the properties stored on a `CfPage` node are: `user_id`, `page_id`, `title`, `space_key`, `version`, `excerpt`, `terms`, `updated_at`. The `graph_kind` property is stored only on the `CfGraphUser` node. Because the filter never matches any real node, the DETACH DELETE is a silent no-op: the stale `CfPage` survives, and `_write_single_page` then calls `MERGE (p:CfPage ...)` which finds the old node and resets properties — BUT also runs `OPTIONAL MATCH (p)-[:HAS_SECTION]->(old:CfSection) DETACH DELETE old`, which does clear the old sections. The end result is that section data is refreshed but the page node itself may carry stale `version`/`terms` values from before the delete step was intended to reset it. More critically, the false reassurance that "re-index worked" is logged while the DETACH DELETE did nothing, making debugging extremely difficult.

**Fix:** Remove `graph_kind` from the `CfPage` MATCH in `refresh_page_in_graph`:

```python
await driver.execute_query(
    "MATCH (p:CfPage {user_id: $user_id, page_id: $page_id}) "
    "DETACH DELETE p",
    {"user_id": user_id, "page_id": page_id},
    routing_=neo4j.RoutingControl.WRITE,
)
```

---

### CR-02: Version cache stores `version + 1` using live metadata version, but live metadata may already be ahead of the cached version — producing a wrong expected version on the next commit

**File:** `confluence_logic/review/api.py:1909–1924`

**Issue:** When a cached version (`_expected_version`) exists and is used as the `expected_version` argument to `push_update`, Confluence accepts the commit and bumps the page to `_expected_version + 1`. However, `_version_cache` is then updated to `version + 1` (line 1924), where `version` is the version returned by `get_page_metadata` at the top of the retry loop (line 1659). If the cached version was stale and the live metadata version was already higher than `_expected_version`, Confluence's internal version after the commit equals `_expected_version + 1`, not `version + 1`. The cache now stores the wrong expected version for the next card in the same session, causing a spurious version conflict on the subsequent call.

Concrete scenario:
1. Card A commits. Cache stores key → 4 (live was 3, cached=None so live was used, +1=4).
2. Between cards A and B an external edit bumps the page to version 5.
3. Card B reads live metadata: `version = 5`. Cache has 4. `_expected_version = 4`.
4. `push_update(expected=4)` is called. Confluence rejects with version conflict (live is 5, expected is 4).
5. ValueError is raised, cache entry is popped — correct.

That scenario is actually handled. But the inverse is problematic:
1. Cache has key → 6 (written after Card A succeeded at version 5).
2. Card B reads live metadata: `version = 5` (same version, no external edit). `_expected_version = 6`.
3. `push_update(expected=6)` is sent to Confluence while live is still 5 → version conflict.
4. Cache is popped (correct), but the user sees a spurious conflict.

Root cause: On a successful push, the cache must store `_expected_version + 1` (the version Confluence will assign), not `version + 1`. When `_expected_version is None`, `version + 1` is correct. When `_expected_version is not None`, the cache should store `_expected_version + 1`.

**Fix:**

```python
if success:
    if _cache_key:
        committed_version = _expected_version if _expected_version is not None else version
        _version_cache[_cache_key] = committed_version + 1
    _fire_reindex(resolved_id, proposal, session_id)
    return {"success": True}
```

---

### CR-03: `_fire_reindex` is not called when a DELETE-section succeeds but `session_id` is `None`

**File:** `confluence_logic/review/api.py:1563–1565`

**Issue:** In the DELETE branch (section delete path), `_fire_reindex` is gated behind `if success and session_id:` (line 1563). When `session_id` is `None` (e.g., legacy callers that do not pass a session, or `_execute_pipeline_proposal` where the session might resolve to an empty string), a successful section deletion never triggers the re-index. The Pinecone and Neo4j indexes retain the deleted section's content and will surface it to future RAG queries.

By contrast, the EDIT branch (line 1925) calls `_fire_reindex` unconditionally on success (the version cache update is guarded separately). The DELETE branch is inconsistent.

**Fix:**

```python
new_html = delete_content_in_section(live_html, matched_heading, "", delete_entire_section=True)
success = await asyncio.to_thread(connector.push_update, resolved_id, new_html, version)
if success:
    if session_id:
        _version_cache[(session_id, resolved_id)] = version + 1
    _fire_reindex(resolved_id, proposal, session_id)  # always fire, regardless of session_id
return {"success": success, "error": None if success else "push_update returned false"}
```

---

## Warnings

### WR-01: `_version_cache` is a module-level dict with no maximum size — unbounded growth under long-running server

**File:** `confluence_logic/review/api.py:1395`

**Issue:** `_version_cache: dict[tuple[str, str], int] = {}` grows indefinitely. Every unique `(session_id, page_id)` pair adds an entry that is never evicted except on a version-conflict pop. In a long-running server with many distinct sessions, this is a slow memory leak. While each entry is tiny (two strings + one int), the design comment "lost on restart" acknowledges restart-based cleanup, which does not occur in production without restarts.

**Fix:** Use a bounded structure, e.g. `collections.OrderedDict` with a max-size eviction, or limit entries to a configurable cap:

```python
_VERSION_CACHE_MAX = int(os.getenv("JARVIS_VERSION_CACHE_MAX", "1000"))
_version_cache: dict[tuple[str, str], int] = {}

def _version_cache_set(key: tuple, value: int) -> None:
    if len(_version_cache) >= _VERSION_CACHE_MAX:
        # evict oldest (FIFO approximation)
        _version_cache.pop(next(iter(_version_cache)), None)
    _version_cache[key] = value
```

---

### WR-02: Pre-flight heading check in EDIT branch is inside the retry loop — exits entire function on first attempt even if the heading appears after a retry

**File:** `confluence_logic/review/api.py:1662–1674`

**Issue:** The APPLY-01 pre-flight heading check is positioned inside the `for _attempt in range(_MAX_EDIT_RETRIES):` loop. On any attempt, if the heading is not found, the function immediately returns `heading_not_found`. This is correct on semantics but interacts badly with the retry loop: if the first call to `fetch_page_html` fails silently (e.g., stale CDN response) and the heading actually exists, the caller gets a permanent `heading_not_found` error rather than a retry. More importantly, `extract_headings` may return an empty list on malformed HTML without raising, causing a false-negative pre-flight failure. The check should happen once before the retry loop to avoid wasted subsequent fetches, or the empty-headings case should be guarded.

**Fix:** Add a guard for empty heading list to avoid false rejection:

```python
_pf_available = extract_headings(live_html)
_pf_lower = heading.strip().lower()
# Only reject if headings were extractable but the target is absent.
# If the page has no headings at all, skip the check (malformed HTML).
if _pf_available and not any(_pf_lower in h.strip().lower() for h in _pf_available):
    return {
        "success": False,
        "error": "heading_not_found",
        "message": ...,
    }
```

Also consider moving the check before the retry loop.

---

### WR-03: `_fire_reindex` resolves `user_id` from `proposal.get("user_id")` which is not a standard proposal field — always falls back to session-scoped ID

**File:** `confluence_logic/review/api.py:1403`

**Issue:** `proposal.get("user_id")` is used inside `_fire_reindex` to derive the `graph_user_id`. However, looking at how proposals are built — both in `_append_agent_generated_changes` (line 1003–1024) and the Supabase proposal schema — `user_id` is not a field stored in the proposal dict. It is stored separately in the Supabase `proposals` table as a column but not included in the in-memory or pipeline proposal dict that `_direct_apply_change` receives. This means `proposal.get("user_id")` always returns `None`, and the re-index always runs with a session-scoped graph user ID (`session:<session_id>`), not the authenticated Supabase user ID (`supabase:<user_id>`). If the graph was indexed under the Supabase user ID (the normal path for authenticated users), the re-index will write under the wrong user scope and the updated data will never appear in subsequent RAG queries for that user.

**Fix:** Pass `user_id` explicitly to `_fire_reindex` from `_direct_apply_change` (the caller already has the proposal, and `_execute_pipeline_proposal` already has access to the session user). Alternatively, `supabase_store.get_proposal_by_id` returns the full row including `user_id` — that value should be forwarded.

---

### WR-04: DELETE of entire page (non-section) does not update version cache or call `_fire_reindex`

**File:** `confluence_logic/review/api.py:1568–1570`

**Issue:** The whole-page delete path at line 1569 calls `connector.delete_page(resolved_id)` and returns immediately without clearing the version cache entry for `(session_id, resolved_id)` or calling `_fire_reindex`. After a successful page deletion, any cached version for that page is stale and will trigger a version conflict if a subsequent proposal tries to edit the now-nonexistent page in the same session. More importantly, the page's Pinecone and Neo4j nodes are not cleaned up — orphaned vectors and graph nodes persist indefinitely for a deleted page.

**Fix:**

```python
success = await asyncio.to_thread(connector.delete_page, resolved_id)
if success:
    if session_id:
        _version_cache.pop((session_id, resolved_id), None)
    _fire_reindex(resolved_id, proposal, session_id)  # will 404 on Confluence but clears indexes
return {"success": success}
```

Note: `refresh_page_in_graph` already handles the case where the page no longer exists (it returns False if metadata fetch fails). A similar guard should be added to the Pinecone pipeline.

---

### WR-05: Test `test_version_cache_populated_after_successful_commit` does not verify the cache key matches the `session_id` extracted from proposal vs. passed kwarg

**File:** `confluence_logic/tests/test_apply_hardening.py:156–175`

**Issue:** The test calls `api._direct_apply_change(proposal, session_id="sess-version")` where `proposal["session_id"]` is also `"sess-version"`. Since both match, the test does not exercise the APPLY-02 code path where `session_id` kwarg is `None` and the fallback reads from `proposal["session_id"]` (line 1455–1456). A test with `session_id=None` and `proposal["session_id"] = "sess-fallback"` would verify that the fallback is invoked and that the correct key `("sess-fallback", page_id)` is written to the cache. Without this test, a regression in the fallback logic would go undetected.

**Fix:** Add a test case:

```python
@pytest.mark.asyncio
async def test_version_cache_uses_proposal_session_id_when_kwarg_is_none(mock_connector, base_edit_proposal):
    api._version_cache.clear()
    proposal = {**base_edit_proposal, "session_id": "sess-fallback", "page_id": "pg-fb"}
    mock_connector.get_page_metadata.return_value = {"version": {"number": 2}, "title": "T"}
    with patch.object(api, "_get_connector", return_value=mock_connector), \
         patch.object(api, "_resolve_page_id", new=AsyncMock(return_value="pg-fb")):
        result = await api._direct_apply_change(proposal, session_id=None)
    assert result["success"] is True
    assert api._version_cache.get(("sess-fallback", "pg-fb")) == 3
```

---

## Info

### IN-01: `_write_single_page` re-assigns `page_id` inside the loop body — variable shadowing

**File:** `confluence_logic/confluence_page_graph.py:271`

**Issue:** Inside the `for index, section in enumerate(sections):` loop, `page_id = page.get("page_id")` (line 271) redundantly re-assigns the variable that was already set at line 258. While the value is always the same (it comes from the same `page` dict), this is a code smell: a reader must verify that the loop body cannot change `page`, and it creates a shadowed variable situation that obfuscates intent. If the loop were ever refactored to iterate over multiple pages, this would silently produce the last page's ID for all sections.

**Fix:** Remove the redundant assignment inside the loop:

```python
# Before the loop:
page_id = page.get("page_id")
if not page_id:
    return
# ... rest of setup ...
section_payload = []
for index, section in enumerate(sections):
    section_id = f"{page_id}::{index}"   # page_id already set above
    ...
```

---

### IN-02: `_execute_pipeline_proposals_batched` still routes through EditorAgent but `_execute_pipeline_proposal` now routes through `_direct_apply_change` — divergent paths with no comment

**File:** `confluence_logic/review/api.py:1958–2036`

**Issue:** Phase 5 rewires single-proposal execution (`_execute_pipeline_proposal`, line 2039) to use `_direct_apply_change`, gaining APPLY-01/02/03 protections. However, the batched path (`_execute_pipeline_proposals_batched`, line 1958) still uses the EditorAgent via `handle_prepared_query`. The two paths are now asymmetric in safety guarantees: single accepts get heading pre-flight and version chain, batch accepts do not. This divergence is not documented in either function's docstring. A future developer accepting a batch of proposals will silently lose the hardening benefits.

**Fix:** Add a docstring note to `_execute_pipeline_proposals_batched` clearly stating it does not use the APPLY-01/02/03 path, and file a follow-up task to migrate it. Alternatively, refactor the batch path to call `_direct_apply_change` per proposal within each page group.

---

### IN-03: Test file is missing `conftest.py` or `pytest.ini` `asyncio_mode` setting — tests may fail with pytest-asyncio 0.21+ strict mode

**File:** `confluence_logic/tests/test_apply_hardening.py:1`

**Issue:** All test functions use `@pytest.mark.asyncio` without a module-level `pytestmark` or `asyncio_mode = "auto"` in `pytest.ini`/`pyproject.toml`. With `pytest-asyncio >= 0.21`, the default mode is `strict`, which requires explicit markers. This is present, but also requires that the event loop scope be configured. If `asyncio_mode` is not set to `"auto"` in project config, each test creates its own event loop — which means `asyncio.create_task` in `_fire_reindex` (which targets the running loop) will use a different loop than the test coroutine in certain scenarios, causing `RuntimeError: no current event loop`. The existing test `test_successful_commit_fires_reindex_create_task` mocks `asyncio.create_task` so this is not triggered by the test itself, but it means the real behavior under test is different from production behavior.

**Fix:** Add to `pytest.ini` or `pyproject.toml`:

```ini
[pytest]
asyncio_mode = auto
```

Or add a module-level mark to the test file:

```python
pytestmark = pytest.mark.asyncio
```

---

_Reviewed: 2026-05-16T12:12:57Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
