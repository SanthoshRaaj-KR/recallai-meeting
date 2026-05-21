# Phase 5: Safe Apply Hardening + Re-indexing - Context

**Gathered:** 2026-05-15
**Status:** Ready for planning

<domain>
## Phase Boundary

Three independent patches to `_direct_apply_change` in `confluence_logic/review/api.py`:

1. **Pre-flight heading check** — verify the target section still exists in the live page before calling `push_update`. Fail fast with a clear error instead of writing to the wrong place.
2. **Version chain** — thread post-commit version numbers across sequential accepts of the same page so version conflicts don't occur for rapid same-page accepts.
3. **Post-commit re-index** — after a successful commit, trigger async re-ingestion of the changed page into both Pinecone and Neo4j so RAG results stay current within the session.

No new endpoints, no UI changes, no new agents. All changes are inside the apply path.

</domain>

<decisions>
## Implementation Decisions

### Anchor Pre-flight (APPLY-01)

- **D-01:** Pre-flight runs for **edit and delete** cards only. Create cards add new content (no heading to verify). Title-rename cards target the page title, not a section.
- **D-02:** Heading match uses **case-insensitive substring** (`heading.strip().lower() in line.strip().lower()`) — consistent with the existing `_fetch_live_page_content` behavior. No fuzzy difflib matching needed.
- **D-03:** On failure: return `{success: False, error: "heading_not_found", message: "Section '{heading}' no longer exists in the live page. The page may have been edited since this proposal was generated."}`. Card remains visible and actionable in sync-sage-bot — user can retry or ignore.
- **D-04:** Pre-flight fetches the live page HTML (same call as the apply itself) — no extra round-trip. Extract headings from the HTML, then check presence before proceeding to the write.

### Version Chain (APPLY-02)

- **D-05:** Post-commit version stored in a **module-level in-memory dict** keyed by `(session_id, page_id)`. Follows the existing `meeting_state` in-memory singleton pattern. Lost on server restart — acceptable since the review flow is always per-session.
- **D-06:** Cache is populated after every successful `push_update` call with `new_version = current_version + 1`. On the next accept for the same `(session_id, page_id)`, pass the cached version as `expected_version` to `push_update` instead of re-fetching.
- **D-07:** On version conflict error from Confluence (despite the cache, e.g. external edit): return `{success: False, error: "version_conflict"}`, **invalidate the cache entry** so the next retry re-fetches fresh. No auto-retry — surface to user, card stays actionable.

### Re-index After Commit (APPLY-03)

- **D-08:** Re-index is **fire-and-forget** via `asyncio.create_task()` immediately after a successful commit. The accept response returns to the UI without waiting for re-index to complete.
- **D-09:** Re-index both **Pinecone and Neo4j** in one call via the existing `IngestionPipeline.ingest_page(page_id)` path (`doc_pipeline.py` — `upsert_chunks` + `clear_stale_sections`). No partial re-index.
- **D-10:** Re-index failure is **logged at WARNING level and silently dropped** — does not affect the accept response. Stale RAG within the same session is acceptable; the next full ingestion pass will catch it.

### Claude's Discretion

The user deferred all remaining implementation choices to Claude. Apply pragmatic defaults throughout: reuse existing patterns, minimize new dependencies, keep it simple and effective. If a choice comes up during planning that isn't covered above, pick the lower-complexity option.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Apply path (primary target)
- `confluence_logic/review/api.py` — `_direct_apply_change` function and `_fetch_live_page_content` helper; all three patches land here
- `confluence_logic/connectors/confluence.py` — `push_update(page_id, content, expected_version)` and `get_page_metadata(page_id)`; version chain uses these directly

### Re-indexing
- `confluence_logic/ingestion/doc_pipeline.py` — `IngestionPipeline.ingest_page` / `upsert_chunks` / `clear_stale_sections`; post-commit re-index calls this

### Requirements
- `.planning/REQUIREMENTS.md` §APPLY-01, APPLY-02, APPLY-03 — the three success criteria this phase must satisfy

### Project context
- `.planning/PROJECT.md` — core value and constraints (no Confluence edits without user approval; RAG-first)
- `./CLAUDE.md` — project conventions and architecture overview

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `connector.push_update(page_id, content, expected_version)` — already implemented with version check; just needs `expected_version` wired in from the cache
- `connector.get_page_metadata(page_id)` — returns `{"version": {"number": N}, ...}`; use to extract headings list for pre-flight
- `connector.fetch_page_html(page_id)` — already called in `_direct_apply_change`; pre-flight can reuse the same fetch result (no extra round-trip)
- `IngestionPipeline.ingest_page(page_id)` — handles full re-ingest of a page into both Pinecone and Neo4j; already exists in `doc_pipeline.py`

### Established Patterns
- **In-memory dict state**: module-level `_version_cache: dict[tuple[str, str], int] = {}` follows the `meeting_state`/`_store` singleton pattern used throughout
- **asyncio.create_task()**: already used for background pipeline jobs in `review/api.py`; same pattern for re-index task
- **Broad except + logger.warning**: standard error handling pattern for non-critical background work

### Integration Points
- `_direct_apply_change` is the single apply entry point for all card types — all three patches go here
- Version cache is keyed by `(session_id, page_id)`; `session_id` is already available from the proposal payload
- Post-commit re-index needs the `page_id` that was actually written (use `resolved_id` from inside `_direct_apply_change`, not the proposal's `page_id` which may be None)

</code_context>

<specifics>
## Specific Ideas

- User preference: pragmatic, low-cost, effective — trust Claude to make implementation calls
- No new UI changes required — sync-sage-bot already handles `success: false` responses as error state on cards
- Pre-flight heading extraction should reuse the HTML already fetched for the apply — avoid an extra Confluence API round-trip

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope.

</deferred>

---

*Phase: 5-safe-apply-hardening-reindexing*
*Context gathered: 2026-05-15*
