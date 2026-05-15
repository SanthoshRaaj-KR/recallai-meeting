# Roadmap: Jarvis — Post-Meeting Confluence Change Proposal Pipeline

## Overview

This milestone builds the full end-to-end pipeline from "meeting ends" to "accepted changes applied to Confluence." Phase 1 fixes broken dependencies and extends the data schema so every downstream agent has a solid foundation. Phase 2 wires up the multi-agent core: FactExtractionAgent, merged RAG retrieval, parallel DrafterAgent pool, and VerifierAgent. Phase 3 adds the SSE progress stream and the complete review UI in sync-sage-bot so users can watch the pipeline run and act on proposal cards. Phase 4 hardens proposal quality — no contradictions, duplicates, or hallucinated content. Phase 5 hardens the safe apply layer and closes the re-indexing loop so accepted changes immediately reflect in the RAG graph.

## Phases

- [ ] **Phase 1: Schema & Blockers** - Fix broken dependencies, extend ChangeItem schema, add pipeline_jobs table
- [x] **Phase 2: Multi-Agent Pipeline Core** - FactExtractionAgent, merged RAG retrieval, parallel DrafterAgent pool, VerifierAgent, background job endpoints (completed 2026-05-12)
- [x] **Phase 3: Async Progress Streaming + Review UI** - SSE stream endpoint, sync-sage-bot pipeline progress component, full proposal card review experience (completed 2026-05-12)
- [x] **Phase 4: Pipeline Proposal Quality Fixes** - Final-state dedup in FactExtractionAgent, verbatim grounding in DrafterAgent, no contradictions/duplicates/hallucinations (completed 2026-05-15)
- [ ] **Phase 5: Safe Apply Hardening + Re-indexing** - Section anchor pre-flight, stale-version chain prevention, Neo4j + Pinecone re-index after commit

## Phase Details

### Phase 1: Schema & Blockers
**Goal**: The codebase compiles and runs with valid dependencies, and all data contracts (Pydantic models, Supabase tables, TypeScript interfaces) reflect the extended proposal schema before any agent code is written
**Depends on**: Nothing (first phase)
**Requirements**: FIX-01, FIX-02, SCHEMA-01, SCHEMA-02, SCHEMA-03, SCHEMA-04
**Success Criteria** (what must be TRUE):
  1. `pip install -r requirements.txt` completes without error; Neo4j driver imports successfully at runtime
  2. `JARVIS_AGENT_MODEL` and `JARVIS_REVIEW_MODEL` default to valid model IDs so the server starts without an "unknown model" rejection from the OpenAI API
  3. A proposal card persisted to Supabase carries `transcript_evidence`, `confidence`, `risk`, and `verifier_note` fields; a card missing these fields fails Pydantic validation
  4. The TypeScript `ChangeItem` interface in sync-sage-bot compiles with the extended fields so the UI layer cannot reference a stale schema
**Plans**: 3 plans

Plans:
- [ ] 01-01-PLAN.md — Fix neo4j version constraint and invalid JARVIS_REVIEW_MODEL default
- [ ] 01-02-PLAN.md — Add ChangeItem Pydantic model with verifier fields and update proposal builders
- [ ] 01-03-PLAN.md — Add pipeline_jobs Supabase DDL and extend TypeScript ChangeItem interface

### Phase 2: Multi-Agent Pipeline Core
**Goal**: Calling `POST /review/pipeline/start` with a session ID launches a background job that extracts facts from the meeting transcript, retrieves candidate Confluence pages via merged RAG, drafts proposals in parallel, runs each through the VerifierAgent, and incrementally persists results to Supabase
**Depends on**: Phase 1
**Requirements**: RETR-01, RETR-02, RETR-03, PIPE-01, PIPE-02, PIPE-03, PIPE-04
**Success Criteria** (what must be TRUE):
  1. `POST /review/pipeline/start` returns HTTP 202 with a `job_id` within one second; the pipeline continues running in the background after the response is sent
  2. For a transcript mentioning two distinct topics, the merged RAG step surfaces candidate pages from both Pinecone and Neo4j — neither source is silently skipped when the other returns results
  3. Each candidate page is drafted by its own DrafterAgent instance running concurrently; the Supabase `proposals` table receives rows as each page completes, not all at once at the end
  4. Every persisted proposal card has a `verifier_note`, `confidence` level, and `risk` level populated by the VerifierAgent; no card reaches Supabase with those fields null
**Plans**: 4 plans

Plans:
- [x] 02-01-PLAN.md — Test infrastructure: test_pipeline.py with stubs and fixtures for RETR-01 through PIPE-04 (Wave 0)
- [x] 02-02-PLAN.md — proposals table DDL + supabase_store helpers (create_pipeline_job, update_pipeline_job, upsert_proposal) + human checkpoint
- [x] 02-03-PLAN.md — FactExtractionAgent + _merged_rag_retrieval (parallel Pinecone + Neo4j)
- [x] 02-04-PLAN.md — DrafterAgent + VerifierAgent + POST /review/pipeline/start endpoint + _run_pipeline orchestrator

### Phase 3: Async Progress Streaming + Review UI
**Goal**: Users can click "Generate Confluence Changes" on the MeetingSummary page, watch named pipeline stages advance in real time, and then review, accept, or reject each proposal card with full evidence context before anything touches Confluence
**Depends on**: Phase 2
**Requirements**: PIPE-05, UI-01, UI-02, UI-03, UI-04, UI-05, UI-06
**Success Criteria** (what must be TRUE):
  1. After clicking "Generate Confluence Changes," the UI transitions to a progress view that shows named stages (fact extraction, retrieval, drafting, verification) with check/active/pending states driven by the SSE stream — no polling, no plain spinner
  2. When the pipeline completes, proposal cards appear grouped by target Confluence page; each card shows change-type badge, before/after diff, rationale, transcript evidence blockquotes, confidence badge, risk badge, and verifier note
  3. Each card has distinct "Accept" and "Reject" buttons; a delete-type card additionally requires a confirmation step before Accept is enabled, and is visually distinct (red border, warning text)
  4. Rejecting a card does not affect other cards; accepting a card triggers the apply flow for that card only
**Plans**: 3 plans
**UI hint**: yes

Plans:
- [x] 03-01-PLAN.md — Backend SSE endpoint + queue injection + upsert_proposal returns UUID (PIPE-05)
- [x] 03-02-PLAN.md — Frontend types + api client + routing + MeetingSummary button & banner (UI-01)
- [x] 03-03-PLAN.md — StageIndicator + ProposalCard + ProposalCardGroup + PipelinePage with SSE wiring (UI-02 through UI-06)

### Phase 4: Pipeline Proposal Quality Fixes
**Goal**: The auto-propose-changes pipeline produces exactly one correct proposal per distinct decision — no contradictions (reverted discussions don't generate two opposing cards), no duplicates (same change mentioned twice generates one card), and no hallucinated content (drafter uses verbatim meeting text for additive changes)
**Depends on**: Phase 3
**Requirements**: QUAL-01, QUAL-02, QUAL-03
**Success Criteria** (what must be TRUE):
  1. A meeting that says "change Q3 to Q1, actually Q3 is fine" produces ZERO proposals for that topic (final state = no change)
  2. The same change mentioned twice at different points in the meeting produces exactly ONE proposal
  3. "Add these X concerns to the page" produces after_content that contains only those exact X concerns, not invented ones
**Plans**: 2 plans

Plans:
- [x] 04-01-PLAN.md — Fix FactExtractionAgent: final-state extraction, normalized dedup, verbatim_content field (Wave 1)
- [x] 04-02-PLAN.md — Fix DrafterAgent: verbatim grounding rule + relevant transcript window (Wave 1)

### Phase 5: Safe Apply Hardening + Re-indexing
**Goal**: Accepted changes are applied to Confluence safely — section anchors are verified before any edit, multi-card sequences on the same page never use stale version numbers, and every committed page is immediately re-indexed in both Pinecone and the Neo4j confluence_page_graph so the RAG layer stays current
**Depends on**: Phase 3
**Requirements**: APPLY-01, APPLY-02, APPLY-03
**Success Criteria** (what must be TRUE):
  1. Accepting an edit card whose `section_heading` no longer exists in the live Confluence page returns a clear error to the UI rather than silently creating a misplaced edit
  2. When two accepted cards target the same page, the second card's commit uses the page version returned by the first card's successful commit — a stale-version conflict error never occurs for sequential same-page accepts
  3. After a page is committed, querying the RAG pipeline with a topic from that page's new content surfaces the updated page within the same session — stale graph nodes are not returned
**Plans**: TBD

## Progress

**Execution Order:** 1 → 2 → 3 → 4 → 5

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Schema & Blockers | 0/3 | Ready to execute | - |
| 2. Multi-Agent Pipeline Core | 4/4 | Complete | 2026-05-12 |
| 3. Async Progress Streaming + Review UI | 3/3 | Complete | 2026-05-12 |
| 4. Pipeline Proposal Quality Fixes | 2/2 | Complete | 2026-05-15 |
| 5. Safe Apply Hardening + Re-indexing | 0/TBD | Not started | - |
