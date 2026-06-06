# Phase 5: Safe Apply Hardening + Re-indexing - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-05-15
**Phase:** 5-safe-apply-hardening-reindexing
**Areas discussed:** Anchor pre-flight failure UX, Version state location, Re-index scope and trigger

---

## Anchor Pre-flight Failure UX

| Option | Description | Selected |
|--------|-------------|----------|
| Hard error, card stays actionable | HTTP 409 with heading_not_found; card shows error badge; user retries or ignores | ✓ |
| Suggest closest match | difflib.get_close_matches on live headings; return suggestion alongside error | |
| You decide | Pick whichever is simpler to implement safely | |

**User's choice:** Hard error, card stays actionable

---

### Heading match strictness

| Option | Description | Selected |
|--------|-------------|----------|
| Case-insensitive substring (current behavior) | `heading.lower() in line.lower()` — consistent with existing code | ✓ |
| Exact normalized match | Must match exactly after strip + lower | |
| Fuzzy threshold (difflib >= 0.8) | Most resilient, introduces similarity threshold to maintain | |

**User's choice:** Case-insensitive substring (keep current behavior)

---

### Pre-flight scope

| Option | Description | Selected |
|--------|-------------|----------|
| Edit and delete only | Pre-flight where target section must exist; create adds new content, no heading to verify | ✓ |
| All card types | Run for every card with a section_heading set | |

**User's choice:** Edit and delete only

---

## Version State Location

| Option | Description | Selected |
|--------|-------------|----------|
| In-memory dict on server | Module-level dict keyed by (session_id, page_id); simple, follows existing patterns | ✓ |
| Supabase proposal row | Write new version back after commit; survives restarts; extra round-trip + schema change | |
| Computed fresh (status quo) | Always re-fetch; relies on Confluence serializing concurrent accepts — broken on rapid clicks | |

**User's choice:** In-memory dict on server

---

### Version cache scoping

| Option | Description | Selected |
|--------|-------------|----------|
| Per (session_id, page_id) | Isolated per user session — no cross-user interference | ✓ |
| Per page_id only (global) | Simpler key; risks interference if two users review the same workspace concurrently | |

**User's choice:** Per (session_id, page_id)

---

### Version conflict handling

| Option | Description | Selected |
|--------|-------------|----------|
| Error + invalidate cache + card stays actionable | Surface conflict, clear cache so retry re-fetches fresh | ✓ |
| Auto-retry once with fresh version | Smoother UX; risks applying to a page state the user didn't see | |

**User's choice:** Error + invalidate cache + card stays actionable

---

## Re-index Scope and Trigger

**User deferred to Claude:** "you only research and do — I am fine with anything as long it properly works and is low cost and effective."

Claude selected:
- Background `asyncio.create_task()` (fire-and-forget) — non-blocking, consistent with existing background job pattern
- Both Pinecone and Neo4j via existing `IngestionPipeline.ingest_page(page_id)` — one call, complete coverage
- Silent failure — re-index errors logged at WARNING, never surface to user accept response

---

## Claude's Discretion

Re-index scope and trigger decisions — user explicitly deferred all implementation choices on this area to Claude.

General preference stated: pragmatic, low-cost, effective. Trust Claude to make calls for anything not explicitly decided above.

## Deferred Ideas

None.
