# Requirements: Jarvis — Post-Meeting Confluence Change Proposal

**Defined:** 2026-05-11
**Core Value:** Users never manually update Confluence after a meeting — the system proposes the right changes to the right pages and the user just approves or rejects.

## v1 Requirements

### Fixes & Foundation

- [ ] **FIX-01**: `neo4j` dependency updated from non-existent `6.1.0` to `>=5.14,<6` in `requirements.txt`
- [ ] **FIX-02**: Default model env vars (`JARVIS_AGENT_MODEL`, `JARVIS_REVIEW_MODEL`) updated to valid model IDs (`gpt-5-mini`, `gpt-5.4-mini`)

### Pipeline — Retrieval

- [x] **RETR-01**: RAG retrieval always runs both Neo4j graph and Pinecone in parallel and merges results — neither source is skipped when the other returns hits
- [x] **RETR-02**: `FactExtractionAgent` extracts structured facts from transcript: decisions, action items, new/changed requirements, owners, deadlines, doc-worthy updates
- [x] **RETR-03**: Candidate Confluence pages are identified automatically via merged RAG retrieval — no user input required for page selection

### Pipeline — Proposal Generation

- [x] **PIPE-01**: `POST /review/pipeline/start` endpoint accepts `{session_id}` and returns `{job_id}` immediately (HTTP 202) — pipeline runs in background
- [x] **PIPE-02**: One `DrafterAgent` per candidate page runs in parallel via `asyncio.gather`, each with its own bounded context (not one mega-prompt for all pages)
- [x] **PIPE-03**: `VerifierAgent` (using `gpt-5.4-mini`) reviews each draft card against transcript evidence and current Confluence content — adds `confidence`, `risk`, `verifier_note` but never drops a card
- [x] **PIPE-04**: Proposals are written to Supabase incrementally as each page's draft+verify completes — pipeline is resumable if server restarts
- [x] **PIPE-05**: `GET /review/pipeline/{job_id}/stream` SSE endpoint emits typed progress events: `stage_start`, `proposal_ready`, `verification_complete`, `pipeline_complete`

### Schema

- [ ] **SCHEMA-01**: `ChangeItem` extended with `transcript_evidence: list[str]`, `confidence: "high"|"medium"|"low"`, `risk: "safe"|"review"|"risky"`, `verifier_note: str | null`
- [ ] **SCHEMA-02**: Supabase `pipeline_jobs` table added: `job_id`, `session_id`, `status`, `stage`, `created_at`, `completed_at`
- [ ] **SCHEMA-03**: Pydantic models in `confluence_logic/review/` updated to match new `ChangeItem` schema
- [ ] **SCHEMA-04**: TypeScript `ChangeItem` interface in `sync-sage-bot/src/types.ts` (or equivalent) updated to match backend schema

### Pipeline — Proposal Quality

- [x] **QUAL-01**: A meeting discussion that first proposes a change and then reverts it ("change Q3 to Q1, actually Q3 is fine") produces ZERO proposals for that topic — final agreed state wins via last-wins dedup in `_merge_facts`
- [x] **QUAL-02**: The same change mentioned at multiple points in a meeting produces exactly ONE proposal — duplicates are collapsed via `(normalized_subject, action)` key in `_merge_facts`; `DrafterAgent` receives `relevant_transcript_window` so discussions in the middle of long meetings are not lost
- [x] **QUAL-03**: For add/create proposals, `after_content` contains only the exact items named in the meeting transcript — `verbatim_content` field on `ChangeIntent` captures the quoted text; RULE 0 in `INTENT_DRAFTER_PROMPT` enforces verbatim use with no additions

### Apply — Safe Execution

- [ ] **APPLY-01**: Before applying an edit card, section anchor pre-flight check confirms `section_heading` still exists in the live page (using `extract_headings` + `difflib.get_close_matches`)
- [ ] **APPLY-02**: When multiple accepted cards target the same `page_id`, each card re-fetches the live page version after the previous card's successful commit — no stale version chain
- [ ] **APPLY-03**: After a page is committed, the Neo4j `confluence_page_graph` node for that page is invalidated/refreshed — not just Pinecone re-indexing

### UI — Review Experience (sync-sage-bot)

- [x] **UI-01**: "Generate Confluence Changes" button visible on `MeetingSummary` page after meeting ends; triggers `POST /review/pipeline/start`
- [x] **UI-02**: Pipeline progress component shows named stages with check/active/pending states (not just a spinner) — driven by SSE stream
- [x] **UI-03**: Each proposal card displays: change type badge, target page + section, before/after diff, rationale, transcript evidence blockquotes, confidence badge, risk badge, verifier note
- [x] **UI-04**: Each card has an explicit "Accept" and "Reject" button — reject is a distinct action, not just unchecking a checkbox
- [x] **UI-05**: Delete-type cards have distinct visual treatment (red border, warning text) and require a separate confirmation step before the accept action is enabled
- [x] **UI-06**: Proposal cards are grouped by target Confluence page — per-card accept/reject control is preserved within each group

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

### Auto-Propose Pipeline Quality Redesign v2 (Phase 10)

- **PROP-V2-01**: Hard hallucination gate — every card persisted to Supabase must (a) carry a `page_id` that is a node in the user's `confluence_page_graph` (Neo4j) OR a verified-existing Confluence REST hit, AND (b) have `after_content` whose tokens, after lowercasing + stopword stripping, are all present in `{transcript ∪ current page content}` for additive ops, or in `current page content` for the old-value portion of replace ops.
- **PROP-V2-02**: Structure-aware editing — the drafter operates on a parsed page tree (heading / paragraph / ordered-list / unordered-list / step nodes). Reordering or inserting steps in an ordered procedure produces node-level operations (`move`, `insert_after`, `replace_inline`), never a regenerated prose block. All sibling nodes that the meeting didn't reference are emitted byte-identical in the after-content.
- **PROP-V2-03**: Page targeting — a new PageRouter stage selects the structurally correct target page for each ChangeIntent BEFORE the drafter runs. Routing must combine (a) semantic search (Pinecone), (b) heading-aware graph lookup (Neo4j confluence_page_graph), (c) explicit subject-token presence in page title or headings. Page routing accuracy on the golden fixture set ≥90%; wrong-page proposals (qualifier `page_relevance < 6`) never reach the UI.
- **PROP-V2-04**: Card UX clarity — every ProposalCard renders, before any user click: page title + URL + breadcrumb (Space › Parent › Page), section heading where the edit lands, change-type pill, ≤120-char plain-English change_summary, inline word-level red/green diff for replace/insert/delete, ordered-list before/after with moved-item highlighting for reorders. No essential info hidden behind expanders.
- **PROP-V2-05**: Regenerate-from-current-page — when a card was generated against a stale page version (e.g., page was edited between draft and accept), a "Regenerate" action re-runs the drafter against the live page; the regenerated card respects every Phase 10 grounding rule.
- **PROP-V2-06**: EditorAgent boundary preserved — Phase 10 must NOT modify `editor_agent.py`. Every proposal card passes EditorAgent a structured instruction of the form `{action, page_id, section_heading, old_text, new_text}` (or `{action: "reorder", page_id, section_heading, from_index, to_index}` for moves), and EditorAgent's existing apply semantics handle the rest.
- **PROP-V2-07**: Quality scorecard — a `tests/e2e_proposal_quality_v2_eval.py` runs the 20+ golden transcript fixtures through the full pipeline and scores hallucination rate, targeting recall/precision, structure preservation, and card-clarity heuristics. Scorecard must hit hallucination = 0%, targeting recall ≥ 90%, structure preservation = 100% on ordered procedures.

### Meeting → Confluence Maintenance Pipeline — Production Redesign v3 (Phase 11)

First-principles rebuild. Phase 10 components are free to be replaced; `editor_agent.py` is reused unchanged as the apply layer. Model ceiling for this phase is **GPT-5-mini** (no larger), per user 2026-05-24.

- **EXT-V3-01**: Structured extraction — the transcript is parsed into typed `ChangeIntent` records (decision / fact-update / action-item / new-workstream / deprecation), each carrying the **final resolved state** (reversals and re-decisions collapse to the last-agreed value), a normalized dedup key, and **verbatim evidence spans with character offsets** into the transcript. No intent is emitted without at least one evidence span.
- **RETR-V3-01**: Hybrid retrieval — per intent, candidate pages/sections are retrieved by **both** dense semantic search (embeddings) **and** lexical/keyword search (BM25 or equivalent), then fused (e.g., Reciprocal Rank Fusion). Neither signal is silently skipped; the fusion is deterministic and logged.
- **RETR-V3-02**: Hierarchical / section-level targeting — retrieval resolves to page→section nodes, not whole pages. The pipeline knows *which heading/section* an edit lands in before drafting, and can target the correct section on a multi-section page.
- **RETR-V3-03**: Reranking — fused candidates pass through a reranking stage (cross-encoder or LLM-based relevance scoring) that reorders by true relevance to the intent before any drafting/operation planning; only top-k reranked candidates proceed.
- **RETR-V3-04**: Agentic iterative retrieval — for low-confidence intents (no strong candidate after fusion+rerank), a **bounded** retrieval loop reformulates the query and retries up to a capped number of iterations, and can conclude "no existing target" (→ create-page path) rather than forcing a wrong-page edit. The loop is bounded to protect latency/cost.
- **CON-V3-01**: Contradiction & stale-information detection — the pipeline detects, for each factual decision, **every** location in the workspace that states the fact the old way (transcript↔page and page↔page), and groups them as one logical decision. The canonical "SOC2 Q3 on two pages, meeting says Q2" case yields a proposal for both pages, surfaced together. Stale/outdated pages no longer referenced by current work are flagged for archive/deprecate review.
- **GND-V3-01**: Hard grounding gate + calibrated confidence — zero cards persist whose `page_id` is unverifiable (absent from `confluence_page_graph` and REST) or whose content contains tokens unsupported by `{transcript ∪ current page content}` (additive) / `current page content` (replace target). Every card carries a calibrated confidence; sub-threshold cards are suppressed or flagged with the reason, never silently shipped.
- **OPS-V3-01**: Operation planning — each surviving (intent, target) resolves to exactly one exact operation: `edit_section` (replace exact old text with new), `append` (add node to a section), `create_page` (structured new page for a new project/framework/workstream), or `archive_deprecate` (label/archive, default over hard-delete). The operation specifies the resolved page, section, and exact content. Ambiguous operations are not emitted.
- **EDIT-V3-01**: Editor handoff — `EditorAgent` is the **sole apply mechanism** and is **not modified**. On accept, the pipeline hands EditorAgent an unambiguous instruction encoding `{operation, page (title + id), section, exact content}` such that EditorAgent's existing search→fetch→preview→commit specialists execute the change without asking questions. Success is reported only when the underlying commit/create/delete tool returns `success=true`.
- **SAFE-V3-01**: Apply-time safety — per-card HITL approval (no edit without explicit accept); section-anchor + page-version preflight before commit; **regenerate-against-live** when the page changed since drafting; deprecation defaults to archive/label (hard-delete requires an explicit second confirmation); accepted pages are reindexed in Pinecone + Neo4j within the same session.
- **UI-V3-01**: Review experience — proposal cards (evolving `ProposalCardV2`) render, before any click: page title + URL + breadcrumb, target section, change-type pill, ≤120-char plain-English summary, word-level or structure-level diff, confidence + evidence, and **contradiction grouping** (related cards for the same decision shown together). No essential info hidden behind expanders.
- **OBS-V3-01**: Observability + evaluation — every pipeline run emits per-stage structured traces (latency, candidate counts in/out, drop reason + which gate dropped each, confidence) over the SSE stream and logs; an offline eval harness over ≥20 golden transcript fixtures runs in CI and prints a scorecard: extraction quality, targeting recall/precision, hallucination rate, contradiction recall, end-to-end card clarity. Phase 10's e2e scorecard is the baseline to beat (no regression).
- **ARCH-V3-01**: Modularization — pipeline-orchestration logic moves out of `review/api.py` into a dedicated `confluence_logic/pipeline/` package with typed stage contracts (Pydantic input/output models per stage). `review/api.py` is reduced to HTTP/SSE wiring. Each stage is unit-testable in isolation; killswitch env flags allow falling back to the Phase 10 path during rollout.
- **SPK-V3-01**: Speaker-attributed transcript source — the post-meeting pipeline sources a transcript whose utterances carry **real participant names** (e.g., `JohnDoe: …`), not the generic `Meeting:` label produced by the Phase-7 LiveKit mixed-audio STT path. A new pipeline transcript-source stage fetches Recall.ai's own diarized transcript (`participant.name` per utterance) when enabled (killswitch `JARVIS_RECALL_TRANSCRIPT_ENABLED`, default off to preserve the Phase-7 cost posture), normalizes it into `participant`-attributed entries, and **falls back** to the existing `transcript_log` (speaker=`Meeting`) when the Recall transcript is unavailable or the killswitch is off. Extracted decisions/action-items attribute owners to the real speaker when attribution is present. **Scope:** does NOT modify `agent_worker.py` / the live LiveKit voice path; only bot-provisioning config (`recording_config`, killswitch-gated) and the new in-scope pipeline stage. *(Added 2026-05-25 from a user-reported bug: the agent receives "Meeting: <dialogue>" instead of per-person attribution.)*

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
| RETR-01 | Phase 2 | Complete |
| RETR-02 | Phase 2 | Complete |
| RETR-03 | Phase 2 | Complete |
| PIPE-01 | Phase 2 | Complete |
| PIPE-02 | Phase 2 | Complete |
| PIPE-03 | Phase 2 | Complete |
| PIPE-04 | Phase 2 | Complete |
| PIPE-05 | Phase 3 | Complete |
| UI-01 | Phase 3 | Complete |
| UI-02 | Phase 3 | Complete |
| UI-03 | Phase 3 | Complete |
| UI-04 | Phase 3 | Complete |
| UI-05 | Phase 3 | Complete |
| UI-06 | Phase 3 | Complete |
| QUAL-01 | Phase 4 | Complete |
| QUAL-02 | Phase 4 | Complete |
| QUAL-03 | Phase 4 | Complete |
| APPLY-01 | Phase 5 | Pending |
| APPLY-02 | Phase 5 | Pending |
| APPLY-03 | Phase 5 | Pending |
| EXT-V3-01 | Phase 11 | Planning |
| RETR-V3-01 | Phase 11 | Planning |
| RETR-V3-02 | Phase 11 | Planning |
| RETR-V3-03 | Phase 11 | Planning |
| RETR-V3-04 | Phase 11 | Planning |
| CON-V3-01 | Phase 11 | Planning |
| GND-V3-01 | Phase 11 | Planning |
| OPS-V3-01 | Phase 11 | Planning |
| EDIT-V3-01 | Phase 11 | Planning |
| SAFE-V3-01 | Phase 11 | Planning |
| UI-V3-01 | Phase 11 | Planning |
| OBS-V3-01 | Phase 11 | TraceBus shell complete (Plan 11-02); full stage wiring Wave 3 |
| ARCH-V3-01 | Phase 11 | Pipeline shell complete (Plan 11-02); stages Wave 3, orchestrator Wave 4 |
| SPK-V3-01 | Phase 11 | Planning |

**Coverage:**
- v1 requirements: 24 total
- Mapped to phases: 24
- Unmapped: 0 ✓
- v3 (Phase 11) requirements: 14 total — all mapped to Phase 11

---
*Requirements defined: 2026-05-11*
*Last updated: 2026-05-11 — traceability confirmed after roadmap creation*
