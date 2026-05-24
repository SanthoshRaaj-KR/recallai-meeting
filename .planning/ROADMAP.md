# Roadmap: Jarvis — Post-Meeting Confluence Change Proposal Pipeline

## Overview

This milestone builds the full end-to-end pipeline from "meeting ends" to "accepted changes applied to Confluence." Phase 1 fixes broken dependencies and extends the data schema so every downstream agent has a solid foundation. Phase 2 wires up the multi-agent core: FactExtractionAgent, merged RAG retrieval, parallel DrafterAgent pool, and VerifierAgent. Phase 3 adds the SSE progress stream and the complete review UI in sync-sage-bot so users can watch the pipeline run and act on proposal cards. Phase 4 hardens proposal quality — no contradictions, duplicates, or hallucinated content. Phase 5 hardens the safe apply layer and closes the re-indexing loop so accepted changes immediately reflect in the RAG graph.

## Phases

- [ ] **Phase 1: Schema & Blockers** - Fix broken dependencies, extend ChangeItem schema, add pipeline_jobs table
- [x] **Phase 2: Multi-Agent Pipeline Core** - FactExtractionAgent, merged RAG retrieval, parallel DrafterAgent pool, VerifierAgent, background job endpoints (completed 2026-05-12)
- [x] **Phase 3: Async Progress Streaming + Review UI** - SSE stream endpoint, sync-sage-bot pipeline progress component, full proposal card review experience (completed 2026-05-12)
- [x] **Phase 4: Pipeline Proposal Quality Fixes** - Final-state dedup in FactExtractionAgent, verbatim grounding in DrafterAgent, no contradictions/duplicates/hallucinations (completed 2026-05-15)
- [x] **Phase 5: Safe Apply Hardening + Re-indexing** - Section anchor pre-flight, stale-version chain prevention, Neo4j + Pinecone re-index after commit (completed 2026-05-16)
- [x] **Phase 7: Confluence Document Q&A Agent** - ConfluenceQAAgent (OpenAI Agents SDK) with Pinecone-first retrieval, live REST fallback, and gpt-5-mini/gpt-4o-mini model split for fast accurate spoken answers (completed 2026-05-17)
- [ ] **Phase 8: Auto-Generated Confluence Proposals — Quality, Accept Reliability, UI Clarity, Tests** - Honor user-spoken verbatim_content in the post-meeting create-fallback; sharpen explicit-create detection; tighten page qualifier + verifier; make Accept resilient with real error messages and a regenerate-from-current-page recovery; redesign the proposal card to show page + one-line change_summary + default-visible diff; ship a 3-layer test plan (unit + e2e quality eval + manual UAT). In-meeting "Hey Jarvis" voice path and the editor agent are LOCKED out of scope.
- [ ] **Phase 9: Merge LiveKit Voice Path + Confluence Q&A Pipeline into Master** - Consolidate the `master` branch (LiveKit-based fast voice answering via `agent_worker.py` / `agent_bridge.py` / `static/bot.html` / AssemblyAI + sonic-turbo) with the `confluence` branch (ConfluenceQAAgent, auto-proposal pipeline, Phase 7 + 8 work, Phase 8 UI redesign in `sync-sage-bot`) on a new `omg_merged` branch off master. No underlying logic in either feature set is rewritten — both are preserved and integrated.
- [ ] **Phase 10: Auto-Propose Pipeline Quality Redesign v2** - Redesign the post-meeting Confluence proposal pipeline end-to-end to eliminate four observed failure modes: (1) hallucinated changes referencing pages or words that don't exist, (2) flow-destroying edits that scramble ordered procedures, (3) ProposalCards that don't communicate what's changing, (4) missed obvious edits and edits routed to wrong pages. EditorAgent is LOCKED as the apply layer — it works correctly when given `do X in Y page`. The redesign builds the upstream pipeline that produces those well-formed `(X, Y)` pairs: a stricter page-router with verbatim grounding, a structure-aware editor that operates on a parsed page tree (heading/list/step nodes) instead of regenerating prose, a hard token-grounding gate, and a redesigned ProposalCard with inline word-level diff, plain-English change summary, page-context header, and one-click Confluence preview.

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
**Plans**: 2 plans

Plans:
- [x] 05-01-PLAN.md — Failing tests for pre-flight heading check, version chain, and post-commit re-index (Wave 0)
- [x] 05-02-PLAN.md — Implementation: _version_cache, heading pre-flight, version chain, _fire_reindex, refresh_page_in_graph, _execute_pipeline_proposal rewire (Wave 1)

## Progress

**Execution Order:** 1 → 2 → 3 → 4 → 5 → 7 → 8 → 9 → 10

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Schema & Blockers | 0/3 | Ready to execute | - |
| 2. Multi-Agent Pipeline Core | 4/4 | Complete | 2026-05-12 |
| 3. Async Progress Streaming + Review UI | 3/3 | Complete | 2026-05-12 |
| 4. Pipeline Proposal Quality Fixes | 2/2 | Complete | 2026-05-15 |
| 5. Safe Apply Hardening + Re-indexing | 2/2 | Complete | 2026-05-16 |
| 7. Confluence Document Q&A Agent | 2/2 | Complete | 2026-05-17 |
| 8. Auto-Generated Proposals — Quality, Accept, UI, Tests | 0/5 | Superseded by Phase 10 | - |
| 9. Merge LiveKit + Confluence Branches | 0/0 | Awaiting plans | - |
| 10. Auto-Propose Pipeline Quality Redesign v2 | 1/9 | Executing (Wave 0 RED scaffolds landed) | - |

### Phase 7: Confluence Document Q&A Agent
**Goal**: A voice query like "hey Jarvis, when is SOC2 coming?" is correctly classified as a Confluence read question, routed to a new `ConfluenceQAAgent` (OpenAI Agents SDK), answered via Pinecone-first semantic retrieval with live Confluence REST fallback, and spoken back — without touching the edit/proposal pipeline
**Depends on**: Phase 5
**Requirements**: QA-01, QA-02, QA-03, QA-04
**Success Criteria** (what must be TRUE):
  1. A factual Confluence question answered by the agent returns a correct spoken response within 3 seconds (including retrieval + synthesis); answer matches the relevant page/section content
  2. When the Pinecone index has no relevant chunks, the agent falls back to a live Confluence REST search and still returns an answer rather than "I don't know"
  3. The agent uses `gpt-5-mini` for tool orchestration and `gpt-4o-mini` for final answer synthesis; tool-use step never uses `gpt-4o-mini`
  4. Edit/mutation queries ("update the SOC2 page", "add a section") are NOT routed to the Q&A agent — `_is_confluence_read_query()` gate holds
**Plans**: 2 plans

Plans:
- [x] 07-01-PLAN.md — Failing test suite for ConfluenceQAAgent (QA-01 through QA-04, Wave 0 RED)
- [x] 07-02-PLAN.md — ConfluenceQAAgent implementation: Pinecone-first retrieval, REST fallback, model split, jarvis_agentic.py wiring (Wave 1)

### Phase 8: Auto-Generated Confluence Proposals — Quality, Accept Reliability, UI Clarity, Tests
**Goal**: After a meeting ends, the auto-generated proposal pipeline produces clear, faithful, accept-able change cards: user-spoken bullet lists survive verbatim through `create` and `add` actions, useless / off-topic proposals are filtered out before reaching the UI, Accept either succeeds or shows the real backend reason with a one-click "Regenerate from current page" recovery path, and every card answers at-a-glance "which page, what kind of change, what's the one-line summary, before vs after" without expanding anything. The in-meeting "Hey Jarvis" voice trigger flow and the editor agent are out of scope and remain bit-for-bit unchanged.
**Depends on**: Phase 5 (Safe Apply Hardening), Phase 4 (Pipeline Quality)
**Requirements**: AUTOPROP-01, AUTOPROP-02, AUTOPROP-03, AUTOPROP-04, AUTOPROP-05
**Scope Lock** (MUST NOT be modified by any plan in this phase):
  - `confluence_logic/jarvis_agentic.py` — wake-word loop, `_queue_confluence_proposal`, voice routing
  - `confluence_logic/agents/editor_agent.py` — works well, do not touch
  - `confluence_logic/agents/proposed_changes_agent.py` — used as fallback by both in-meeting (out of scope) and post-meeting; leave the class as-is, fix the create-fallback in `review/api.py` instead
**Success Criteria** (what must be TRUE):
  1. A post-meeting transcript that says "create a pros and cons page for bots: A is fast, B is cheap, C is expensive, D is slow" produces exactly one `create` proposal whose `after_content` contains the literal strings "A is fast", "B is cheap", "C is expensive", "D is slow" — no paraphrase, no extra invented bullets, no stub
  2. A post-meeting transcript that says "create a page about X" with X clearly named produces exactly one `create` proposal even when retrieval surfaces a tangentially related existing page — the explicit instruction overrides retrieval
  3. Page qualifier rejects every (intent, page) pair where `intent.old_value` is non-empty AND missing verbatim from the page AND `intent.subject` shares zero non-stopword tokens with `page.title` AND no heading on the page contains a subject token — without calling the LLM
  4. When Accept fails because the live page changed since the proposal was generated, the UI toast displays the real backend `message` field (not a generic "Failed to apply") AND a "Regenerate from current page" button appears on the card; clicking it re-drafts against the current page and lets the user re-accept
  5. Every ProposalCard renders, before any user click: page title, change-type badge, a one-line `change_summary` (≤120 chars), and a default-visible compact before/after preview
**Plans**: 5 plans

Plans:
- [ ] 08-01-PLAN.md — Verbatim_content end-to-end: create-fallback in `_run_pipeline` honors `intent.verbatim_content`; fact-extraction prompt sharpens explicit-create detection; drafter prompt + post-LLM code guard enforce verbatim adherence for create/add (AUTOPROP-01, AUTOPROP-02)
- [ ] 08-02-PLAN.md — Quality filter: page qualifier hard pre-filter (no LLM call for clearly-irrelevant pairs); verifier drops <20-char after_content for non-deletes, synthesizes missing change_summary, downgrades hallucinated before_content to append (AUTOPROP-03)
- [ ] 08-03-PLAN.md — Accept reliability: `sync-sage-bot/src/lib/api.ts` executeProposal surfaces `json.message`; ProposalCard toast shows real error; new `POST /sessions/{session_id}/review/regenerate/{proposal_id}` endpoint re-drafts against current page; pre-flight heading downgrade to `create_section` at proposal time (AUTOPROP-04)
- [ ] 08-04-PLAN.md — Card UI clarity: per-card headline (page_title + change_summary always visible); default-visible compact before/after preview (3 lines each) with "Show full" expansion; per-card change-type pill; optional inline diff highlighting via tiny LCS util (AUTOPROP-05)
- [ ] 08-05-PLAN.md — Test suite: 10 transcript fixtures in `tests/fixtures/transcripts/`; pytest unit tests (`test_proposal_quality.py`, `test_fact_extraction_explicit_create.py`, `test_apply_failure_paths.py`); vitest UI tests (`ProposalCard.test.tsx`); e2e quality scorecard (`tests/e2e_proposal_quality_eval.py`); documented `tests/MANUAL_TEST_PLAN.md`

### Phase 9: Merge LiveKit Voice Path + Confluence Q&A Pipeline into Master
**Goal**: A single `omg_merged` branch off `master` contains BOTH (a) master's LiveKit-based fast voice answering path (agent_worker.py, agent_bridge.py, static/bot.html browser publisher, AssemblyAI STT, sonic-turbo, tighter endpointing, audio_cache.py) AND (b) confluence branch's Phase 7 ConfluenceQAAgent + Phase 8 auto-proposal pipeline (FactExtraction → merged RAG → Drafter pool → Verifier → SSE stream → review UI) + Phase 8 sync-sage-bot UI redesign — with no underlying logic of either feature set rewritten. The merged tree must build, the LiveKit voice path must still answer in <3s, the auto-proposal pipeline must still produce verbatim-faithful cards, and the UI submodule must point at the confluence-branch tip.
**Depends on**: Phase 8 (completion of confluence-branch work is the source of truth being merged in)
**Requirements**: MERGE-01, MERGE-02, MERGE-03, MERGE-04, MERGE-05
**Branch Strategy** (LOCKED):
  - Target: new `omg_merged` branch off `master` — neither `master` nor `confluence` is rewritten
  - Direction: merge `confluence` INTO the integration branch (so master's LiveKit code is the base; confluence features are layered on)
  - UI submodule: advance `sync-sage-bot` from `72f2541` to `be285fc` (5 newer commits, Phase 8 UI)
**Conflict Surface** (known incompatibilities — must be reconciled, not auto-resolved):
  - `confluence_logic/jarvis_agentic.py` — ~927-line diff; both branches rewrote it independently
  - `confluence_logic/review/api.py` — ~1022-line diff; confluence added pipeline endpoints, master removed IPC relay code
  - `confluence_logic/tests/test_jarvis_agentic.py` — heavily refactored on both sides
  - `confluence_logic/classifier.py`, `agents/drafter_agent.py`, `agents/fact_extraction_agent.py`, `agents/page_qualifier.py`, `db/vector_store.py`, `confluence_page_graph.py` — moderate conflicts
**Scope Lock** (MUST NOT be modified during the merge):
  - The behaviour of `agent_worker.py` / `agent_bridge.py` / `static/bot.html` — preserved verbatim from master
  - The behaviour of `agents/confluence_qa_agent.py` / `agents/drafter_agent.py` / `agents/fact_extraction_agent.py` / Phase 8 review pipeline endpoints — preserved verbatim from confluence
  - ROADMAP.md, REQUIREMENTS.md, prior phase CONTEXT.md / PLAN.md / VERIFICATION.md files — read-only history
**Success Criteria** (what must be TRUE):
  1. `git checkout omg_merged && pip install -r requirements.txt && pytest` runs to completion: no import errors, no missing-symbol errors, no tests removed silently from either side
  2. `agent_worker.py` and `agent_bridge.py` (from master) exist on the integration branch with byte-identical logic to master's tip; their tests (`test_agent_worker.py`, `test_agent_bridge.py`) pass
  3. `agents/confluence_qa_agent.py` (from confluence) exists with byte-identical logic to confluence's tip; `test_confluence_qa_agent.py` and `test_qa_latency.py` pass
  4. `POST /review/pipeline/start` and the SSE stream endpoint (from confluence's `review/api.py`) function end-to-end; the 10 transcript fixtures + e2e quality scorecard still score the same as on the confluence branch
  5. `sync-sage-bot` submodule pointer is `be285fc`; the UI builds (`npm run build`) and Phase 8 ProposalCard renders with headline + diff + accept-error toast
  6. The LiveKit voice path answers a Confluence read query in under 3s (Phase 7 latency target), with the LiveKit agent now routing read queries to `ConfluenceQAAgent` as a tool (the only intentional integration point — not a rewrite of either side)
**Plans**: TBD (created by /gsd:plan-phase after discuss-phase locks the integration-point decisions)

### Phase 10: Auto-Propose Pipeline Quality Redesign v2
**Goal**: After any meeting, the auto-propose-changes pipeline produces a small set of proposal cards that the user can read in seconds and accept with confidence. Every card targets a real Confluence page with content the meeting actually said, places the edit in the structurally correct location on that page, preserves the page's existing flow (ordered procedures, headings, cross-references), and shows the user — at a glance, without expanding anything — which page is affected, where in the page, exactly what text is being added/removed/replaced, and why. EditorAgent remains the apply layer (it works correctly given a clean `do X in Y page` instruction); the redesign produces those clean instructions and the UI that lets the user trust them.
**Depends on**: Phase 5 (safe apply), Phase 9 (merged branch — code being modified lives here)
**Requirements**: PROP-V2-01, PROP-V2-02, PROP-V2-03, PROP-V2-04, PROP-V2-05, PROP-V2-06, PROP-V2-07
**Scope Lock** (MUST NOT be modified):
  - `confluence_logic/agents/editor_agent.py` — the apply layer. Works correctly given a structured instruction; Phase 10 produces better instructions, it does not rewrite the editor.
  - `confluence_logic/agents/confluence_qa_agent.py` — Phase 7 read-query agent, unrelated.
  - `agent_worker.py` / `agent_bridge.py` / `static/bot.html` — LiveKit voice path from Phase 9.
  - In-meeting "Hey Jarvis" voice path in `jarvis_agentic.py`.
**Failure Modes Being Eliminated** (observed in production, 2026-05-22):
  1. **Hallucinated page or content** — pipeline proposes edits to pages that don't exist in the workspace, or `after_content` contains words/phrases that were never in the transcript and never in the current page.
  2. **Flow disruption** — meeting says "login should come before payment in the onboarding flow"; pipeline writes after_content that destroys the existing prose ("enter username password click mouse button") instead of swapping ordered steps 2 and 3.
  3. **Card is illegible** — user opens the review UI and cannot tell what's being changed: no page-level context, no inline word-level diff, no plain-English summary; reads like a raw LLM dump.
  4. **Targeting failure** — meeting says "we're moving to Docker for deployment"; pipeline ignores the page that has the deployment section and instead writes "use Docker for deployment" into the Introduction of an unrelated page.
**Success Criteria** (what must be TRUE):
  1. Hard hallucination gate — zero cards persisted whose `page_id` is not a node in the user's confluence_page_graph (Neo4j) AND whose page title does not match a Confluence page returned by REST. No card whose `after_content` contains a token (after lowercasing + stopword removal) that appears in neither `{transcript ∪ current page content}` for additive operations, or that does not appear in `current page content` for replace-target text.
  2. Structure-aware editing — when meeting discusses reordering ordered-list steps, the proposal operates at node level (move step[i] before step[j]) and the rendered after_content preserves all other steps verbatim. The drafter never returns prose like "click mouse button" for an ordered procedure when the original used "Click 'Sign in'".
  3. Card UX clarity — every card, before any user interaction, displays: (a) page title + URL + breadcrumb (Space › Parent › Page), (b) section heading where the edit lands, (c) change-type pill (Replace/Insert/Reorder/Delete/Create), (d) a ≤120-char plain-English `change_summary`, (e) inline word-level red/green diff for replace/insert/delete, (f) for reorders, an explicit ordered-list before/after with moved items highlighted. No essential information is hidden behind a "Show more".
  4. Targeting recall — for the e2e scorecard's golden transcript set (≥20 fixtures), retrieval+routing surfaces the structurally correct target page for ≥90% of explicit subject mentions (frameworks discussion → frameworks page; deployment-arch discussion → deployment page). Wrong-page proposals (qualifier `page_relevance < 6`) never reach the UI.
  5. Targeting precision — for the same fixture set, no proposal is routed to a page whose title is unrelated to the change subject (zero false positives where the qualifier scores ≥6 for a page that has no semantic relationship to the change).
  6. Regenerate-from-current-page works — if a card was generated against a stale page version, clicking "Regenerate" re-runs the drafter against the live page content; the new card respects every Phase 10 grounding rule.
  7. EditorAgent boundary preserved — given any Phase 10 proposal card that passes the verifier, EditorAgent applies it without modification to its own code; all instructions to EditorAgent take the form "in section S of page P, replace exact text X with Y" or "in section S of page P, append node N", and EditorAgent's existing precise-edit semantics handle the rest.
**Plans**: 9 plans across 5 waves

Plans:
- [x] 10-01-PLAN.md — Wave 0 RED: 8 test scaffolds + 20-fixture stub + manual UAT doc (PROP-V2-01..07) — completed 2026-05-22
- [ ] 10-02-PLAN.md — PageParser: Confluence storage HTML → typed AST (Wave 1, PROP-V2-02)
- [ ] 10-03-PLAN.md — GroundingGate: hard hallucination + token-grounding gate (Wave 1, PROP-V2-01)
- [ ] 10-04-PLAN.md — EditorDispatcher: maps 6 D-02 op shapes onto tools.py primitives without touching editor_agent.py (Wave 1, PROP-V2-06)
- [ ] 10-05-PLAN.md — PageRouter: three-signal merge (Pinecone + Neo4j heading-aware + explicit-token gate) (Wave 1, PROP-V2-03)
- [ ] 10-06-PLAN.md — StructureAwareDrafter: emits structured ops, not prose; replaces drafter for Phase 10 path (Wave 2, PROP-V2-02 + 06)
- [ ] 10-07-PLAN.md — Orchestrator rewire in _run_pipeline + Regenerate endpoint + verifier slimming + connector ancestors/space expand (Wave 3, PROP-V2-01..03, 05, 06)
- [ ] 10-08-PLAN.md — ProposalCardV2 + wordDiff util + Regenerate UI wire (Wave 4, PROP-V2-04 + 05)
- [ ] 10-09-PLAN.md — e2e quality scorecard + 20 golden transcript fixtures (Wave 4, PROP-V2-07)


