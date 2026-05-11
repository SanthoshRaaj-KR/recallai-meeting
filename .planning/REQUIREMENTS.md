# Requirements: Jarvis — Post-Meeting Confluence Change Proposal

**Defined:** 2026-05-11
**Core Value:** Users never manually update Confluence after a meeting — the system proposes the right changes to the right pages and the user just approves or rejects.

## v1 Requirements

### Fixes & Foundation

- [ ] **FIX-01**: `neo4j` dependency updated from non-existent `6.1.0` to `>=5.14,<6` in `requirements.txt`
- [ ] **FIX-02**: Default model env vars (`JARVIS_AGENT_MODEL`, `JARVIS_REVIEW_MODEL`) updated to valid model IDs (`gpt-5-mini`, `gpt-5.4-mini`)

### Pipeline — Retrieval

- [ ] **RETR-01**: RAG retrieval always runs both Neo4j graph and Pinecone in parallel and merges results — neither source is skipped when the other returns hits
- [ ] **RETR-02**: `FactExtractionAgent` extracts structured facts from transcript: decisions, action items, new/changed requirements, owners, deadlines, doc-worthy updates
- [ ] **RETR-03**: Candidate Confluence pages are identified automatically via merged RAG retrieval — no user input required for page selection

### Pipeline — Proposal Generation

- [ ] **PIPE-01**: `POST /review/pipeline/start` endpoint accepts `{session_id}` and returns `{job_id}` immediately (HTTP 202) — pipeline runs in background
- [ ] **PIPE-02**: One `DrafterAgent` per candidate page runs in parallel via `asyncio.gather`, each with its own bounded context (not one mega-prompt for all pages)
- [ ] **PIPE-03**: `VerifierAgent` (using `gpt-5.4-mini`) reviews each draft card against transcript evidence and current Confluence content — adds `confidence`, `risk`, `verifier_note` but never drops a card
- [ ] **PIPE-04**: Proposals are written to Supabase incrementally as each page's draft+verify completes — pipeline is resumable if server restarts
- [ ] **PIPE-05**: `GET /review/pipeline/{job_id}/stream` SSE endpoint emits typed progress events: `stage_start`, `proposal_ready`, `verification_complete`, `pipeline_complete`

### Schema

- [ ] **SCHEMA-01**: `ChangeItem` extended with `transcript_evidence: list[str]`, `confidence: "high"|"medium"|"low"`, `risk: "safe"|"review"|"risky"`, `verifier_note: str | null`
- [ ] **SCHEMA-02**: Supabase `pipeline_jobs` table added: `job_id`, `session_id`, `status`, `stage`, `created_at`, `completed_at`
- [ ] **SCHEMA-03**: Pydantic models in `confluence_logic/review/` updated to match new `ChangeItem` schema
- [ ] **SCHEMA-04**: TypeScript `ChangeItem` interface in `sync-sage-bot/src/types.ts` (or equivalent) updated to match backend schema

### Apply — Safe Execution

- [ ] **APPLY-01**: Before applying an edit card, section anchor pre-flight check confirms `section_heading` still exists in the live page (using `extract_headings` + `difflib.get_close_matches`)
- [ ] **APPLY-02**: When multiple accepted cards target the same `page_id`, each card re-fetches the live page version after the previous card's successful commit — no stale version chain
- [ ] **APPLY-03**: After a page is committed, the Neo4j `confluence_page_graph` node for that page is invalidated/refreshed — not just Pinecone re-indexing

### UI — Review Experience (sync-sage-bot)

- [ ] **UI-01**: "Generate Confluence Changes" button visible on `MeetingSummary` page after meeting ends; triggers `POST /review/pipeline/start`
- [ ] **UI-02**: Pipeline progress component shows named stages with check/active/pending states (not just a spinner) — driven by SSE stream
- [ ] **UI-03**: Each proposal card displays: change type badge, target page + section, before/after diff, rationale, transcript evidence blockquotes, confidence badge, risk badge, verifier note
- [ ] **UI-04**: Each card has an explicit "Accept" and "Reject" button — reject is a distinct action, not just unchecking a checkbox
- [ ] **UI-05**: Delete-type cards have distinct visual treatment (red border, warning text) and require a separate confirmation step before the accept action is enabled
- [ ] **UI-06**: Proposal cards are grouped by target Confluence page — per-card accept/reject control is preserved within each group

## v2 Requirements

### Pipeline Enhancements

- **PIPE-V2-01**: Inline edit of `after_content` in proposal cards before accepting
- **PIPE-V2-02**: Bulk-by-filter select (e.g., "select all HIGH confidence cards")
- **PIPE-V2-03**: Pipeline retry / resume from last completed card on server restart
- **PIPE-V2-04**: Append change type (add content to section without replacing existing content)

### UI Enhancements

- **UI-V2-01**: Side-by-side diff view (not just before/after text)
- **UI-V2-02**: Page-level accept/reject (accept all cards for a page in one click)
- **UI-V2-03**: Card feedback — user can annotate why they rejected a proposal (for model improvement)

## Out of Scope

| Feature | Reason |
|---------|--------|
| Auto-approve or bulk approve all | Safety requirement — every card must be individually reviewed |
| User specifying page names or candidates | System must infer from RAG; user input breaks the automation value |
| Live full-workspace scan after button click | RAG + pre-indexed graph only; live scan is too slow |
| local_office_logic changes | Focus scope is Confluence connector only |
| Models above `gpt-5.4-mini` | Cost constraint; gpt-5.4-mini is the ceiling |
| Auth system changes | Auth already implemented in sync-sage-bot |
| review-ui (Next.js) changes | sync-sage-bot is the primary UI |

## Model Assignment

| Role | Model |
|------|-------|
| Routing / classification | `gpt-5.4-nano` |
| Fact extraction from transcript | `gpt-5-mini` |
| Drafter workers (per page) | `gpt-5-mini` |
| Orchestrator decisions | `gpt-5.4-mini` |
| Verifier / critic | `gpt-5.4-mini` |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| FIX-01 | Phase 1 | Pending |
| FIX-02 | Phase 1 | Pending |
| SCHEMA-01 | Phase 1 | Pending |
| SCHEMA-02 | Phase 1 | Pending |
| SCHEMA-03 | Phase 1 | Pending |
| SCHEMA-04 | Phase 1 | Pending |
| RETR-01 | Phase 2 | Pending |
| RETR-02 | Phase 2 | Pending |
| RETR-03 | Phase 2 | Pending |
| PIPE-01 | Phase 2 | Pending |
| PIPE-02 | Phase 2 | Pending |
| PIPE-03 | Phase 2 | Pending |
| PIPE-04 | Phase 2 | Pending |
| PIPE-05 | Phase 3 | Pending |
| UI-01 | Phase 3 | Pending |
| UI-02 | Phase 3 | Pending |
| UI-03 | Phase 3 | Pending |
| UI-04 | Phase 3 | Pending |
| UI-05 | Phase 3 | Pending |
| UI-06 | Phase 3 | Pending |
| APPLY-01 | Phase 4 | Pending |
| APPLY-02 | Phase 4 | Pending |
| APPLY-03 | Phase 4 | Pending |

**Coverage:**
- v1 requirements: 21 total
- Mapped to phases: 21
- Unmapped: 0 ✓

---
*Requirements defined: 2026-05-11*
*Last updated: 2026-05-11 — traceability confirmed after roadmap creation*
