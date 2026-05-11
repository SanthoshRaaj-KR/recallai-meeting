# Architecture Patterns: Multi-Agent Post-Meeting Confluence Pipeline

**Domain:** Multi-agent LLM pipeline with human-in-the-loop review
**Researched:** 2026-05-11
**Overall confidence:** HIGH (based on direct codebase analysis + established async Python patterns)

---

## Recommended Agent Topology

### Sequential-with-Parallel-Workers (Not a Full DAG)

**Recommendation:** A hybrid topology — sequential at the macro level between the four distinct stages (extract → retrieve → draft → verify), but parallel at the micro level inside the draft stage.

A pure DAG adds orchestration complexity (dependency resolution, topological sort, cycle detection) that provides no benefit here because:
- The stages are strictly ordered (you cannot draft without retrieval results, cannot verify without drafts)
- The only real parallelism opportunity is within the drafting stage, where multiple independent page proposals can be generated concurrently
- The OpenAI Agents SDK's `Runner` model already supports parallel tool calls within a single agent turn; you do not need a separate DAG runtime

**Macro topology (sequential):**

```
[Extractor Agent]
      ↓  structured FactBundle
[Retriever — RAG over Neo4j + Pinecone graph]
      ↓  List[CandidatePage with full section HTML]
[Drafter Pool — asyncio.gather over N worker agents, one per candidate]
      ↓  List[ProposalDraft]
[Verifier/Critic Agent — single pass, evaluates all drafts against transcript evidence]
      ↓  List[VerifiedProposal] (some may be rejected/flagged)
[API Response — proposal cards returned to frontend]
```

**Why this order is forced by data dependencies:**
1. Extractor must run first — its FactBundle is the query input for RAG retrieval
2. Retriever must run before drafters — drafters need section-level HTML context to produce accurate before/after diffs
3. Drafters must finish before the verifier — the verifier's job is to cross-check drafts against transcript evidence; it cannot run before drafts exist
4. Verifier must run before frontend delivery — users should never see unvalidated proposals

**Drafter parallelism pattern:**

```python
# Each candidate page gets its own independent drafter task
drafter_tasks = [
    draft_one_page(candidate, fact_bundle, transcript_text)
    for candidate in retrieved_candidates
]
raw_drafts = await asyncio.gather(*drafter_tasks, return_exceptions=True)
# Filter out exceptions before passing to verifier
valid_drafts = [d for d in raw_drafts if not isinstance(d, Exception)]
```

This replaces the current single-call `ProposedChangesAgent.propose()` which sends all candidates in one prompt. Per-page workers produce more focused, accurate diffs at the cost of more LLM calls (which is acceptable given the 20-minute budget).

**Model assignment per stage:**

| Stage | Model | Rationale |
|-------|-------|-----------|
| Extractor | GPT-4o mini / GPT-5 mini | Structured extraction with JSON schema; cheap, fast |
| Retriever | No LLM call — graph/vector lookup only | Pure RAG, deterministic |
| Drafter (each worker) | GPT-4o mini / GPT-5 mini | One page + one section; bounded context, low-stakes |
| Verifier | GPT-4o / GPT-5 | Cross-references multiple drafts against full transcript evidence; highest stakes |
| Orchestrator (Python) | No LLM — pure Python async coordinator | Manages task lifecycle, error handling, progress events |

---

## Extractor Agent Design

The extractor is a new stage that does not exist in the current codebase. The current `_generate_review_insights()` in `review/api.py` already runs four parallel sub-agents (summary, topics, decisions, action items) using `asyncio.gather`. The extractor should be a fifth parallel agent within that existing `_generate_review_insights` call, or a separate dedicated call that produces a `FactBundle`.

**FactBundle schema:**

```python
@dataclass
class FactBundle:
    decisions: List[str]         # explicit meeting decisions
    action_items: List[ActionItem]  # owner, description, due
    requirements_changed: List[str] # feature/spec changes mentioned
    doc_worthy_updates: List[str]   # items explicitly flagged for documentation
    participants: List[str]
    key_topics: List[str]
    # Derived from graph_rag._local_nodes — already available in current state
```

The extractor does not replace `_generate_review_insights`. It augments it by extracting doc-update-specific signals that the existing summary/topics/decisions agents do not specifically target. The proposal pipeline feeds on `FactBundle`, not on the full summary JSON.

---

## Data Flow for the Full Pipeline

```
POST /sessions/{session_id}/review/changes/propose
  ↓
[1] Load transcript + existing summary_json from state / Supabase
  ↓
[2] Extractor Agent → FactBundle (parallel with summary if not already cached)
  ↓
[3] RAG Retriever: FactBundle.decisions + action_items + doc_worthy_updates
    → multi-query Neo4j Confluence graph (existing: confluence_page_graph.query_user_confluence_graph)
    → fallback: Pinecone + live CQL search (existing: search_workspace_knowledge)
    → top-N CandidatePages with section HTML (fetch_live_page per candidate)
  ↓
[4] Drafter Pool: asyncio.gather over CandidatePages
    Each drafter: FactBundle + candidate section HTML → ProposalDraft
    (Model: GPT-4o mini per drafter)
  ↓
[5] Verifier Agent: receives all ProposalDrafts + transcript + FactBundle
    Output: VerifiedProposal[] — each with verdict (approve/flag/reject), confidence, evidence snippets
    (Model: GPT-4o / GPT-5)
  ↓
[6] Approved + flagged cards → stored in state["pending_changes"] + Supabase snapshot
  ↓
[7] HTTP response: {changes: VerifiedProposal[], generated_count, rejected_count, pipeline_ms}
```

**Key difference from current architecture:** The current `ProposedChangesAgent` collapses steps 2–5 into a single LLM call. This is fast but produces lower-quality results because (a) the model sees all pages in one context window and blends them, and (b) there is no verification step. The expanded topology separates concerns so each agent has a focused, bounded job.

---

## Proposal Card Schema

The current `ChangeItem` TypeScript type needs four new fields. The backend Python dict needs a matching expansion. These additions are backward-compatible — existing `status: "pending"` cards without the new fields will still render correctly in the UI.

**Extended ChangeItem (TypeScript, additive):**

```typescript
export interface ChangeItem {
  // Existing fields (unchanged)
  id: number;
  change_type: "create" | "edit" | "delete" | "title";
  page_id: string | null;
  page_title: string;
  section_heading: string | null;
  before_content: string | null;
  after_content: string | null;
  timestamp: string;
  session_id: string;
  status: "pending" | "approved" | "rejected" | "executed" | "failed";
  source?: string | null;
  rationale?: string | null;
  generation_query?: string | null;

  // New fields added by multi-agent pipeline
  transcript_evidence?: string[] | null;  // 1-3 verbatim transcript snippets that support this change
  confidence?: "high" | "medium" | "low" | null;  // verifier's confidence
  risk?: "safe" | "review" | "risky" | null;  // verifier's risk assessment
  verifier_note?: string | null;  // verifier's explanation when confidence is low or risk is review/risky
}
```

**Backend Python dict (additive to existing _append_agent_generated_changes):**

```python
{
    # Existing fields
    "id": next_id,
    "change_type": ...,
    "page_id": ...,
    "page_title": ...,
    "section_heading": ...,
    "before_content": ...,
    "after_content": ...,
    "timestamp": timestamp,
    "session_id": ...,
    "status": "pending",
    "source": "multi_agent_pipeline",
    "rationale": ...,
    "generation_query": ...,

    # New fields
    "transcript_evidence": [...],   # list of str
    "confidence": "high" | "medium" | "low",
    "risk": "safe" | "review" | "risky",
    "verifier_note": "...",          # None when confidence=high and risk=safe
}
```

**Rationale for this schema:**
- `transcript_evidence` lets the UI show the user _why_ the proposal was generated; increases trust
- `confidence` + `risk` let the UI visually differentiate high-confidence safe changes from uncertain ones without requiring the user to read the full rationale
- `verifier_note` is only populated when the verifier flagged something; avoids cluttering the card for routine proposals
- All new fields are nullable so existing card construction code does not break

---

## Partial Failure Handling

The pipeline must handle three distinct failure modes without blocking the user from reviewing the cards that did succeed.

**Drafter-level failure (one page fails, others succeed):**

```python
raw_drafts = await asyncio.gather(*drafter_tasks, return_exceptions=True)
# Exceptions are collected, not raised
valid_drafts = [d for d in raw_drafts if isinstance(d, ProposalDraft)]
failed_pages = [d for d in raw_drafts if isinstance(d, Exception)]
# Log failed_pages; continue to verifier with valid_drafts
```

**Verifier rejects some, approves others:**

The verifier returns a verdict per proposal. Rejected proposals are not discarded — they are stored with `status: "pending"` and `risk: "risky"` or a low confidence score so the user can still review them manually. The user sees all cards; the UI can visually dim or sort rejected-by-verifier cards to the bottom. The user retains final authority.

**Whole pipeline failure (extractor or verifier crashes):**

Fall back to the existing single-call `ProposedChangesAgent.propose()` path. This means the endpoint should have a try/except at the macro orchestration level:

```python
try:
    proposals = await run_multi_agent_pipeline(...)
except Exception:
    logger.warning("Multi-agent pipeline failed, falling back to single-agent proposal")
    proposals = await ProposedChangesAgent(...).propose(...)
```

This preserves the existing behavior as a safety net.

**No candidates found by RAG:**

If the retriever returns zero candidates, return `{changes: [], generated_count: 0}` with a `detail` field explaining no relevant pages were found in the indexed graph. Do not call drafters or the verifier. The user should be prompted to trigger re-indexing if this happens frequently.

---

## Async Job Pattern: Long-Running Pipeline

The pipeline can take up to 20 minutes. The correct pattern for this constraint is **server-sent events (SSE) streaming progress**, not WebSocket, not simple polling, and not a background task queue.

**Why SSE over the alternatives:**

| Pattern | Verdict | Reason |
|---------|---------|--------|
| Simple HTTP (wait for response) | Rejected | Times out at proxy/load-balancer (typically 30–60s); client has no feedback |
| Background task + polling | Workable but worse DX | Requires a job store, status endpoint, client polling loop; adds latency between stage completions and UI update |
| WebSocket | Overkill | Full bidirectional protocol; reconnect logic needed; adds complexity for a one-direction progress stream |
| SSE (EventSource) | Recommended | Unidirectional, HTTP/1.1 compatible, automatic reconnect built into browser EventSource, works through ngrok, no extra infrastructure |

**Backend SSE pattern (FastAPI):**

```python
from fastapi.responses import StreamingResponse
import asyncio, json

@router.post("/sessions/{session_id}/review/changes/propose/stream")
async def propose_changes_stream(session_id: str, body: ProposeChangesRequest):
    async def event_generator():
        try:
            yield f"data: {json.dumps({'stage': 'extracting', 'pct': 5})}\n\n"
            fact_bundle = await run_extractor(...)

            yield f"data: {json.dumps({'stage': 'retrieving', 'pct': 20})}\n\n"
            candidates = await run_retriever(fact_bundle)

            yield f"data: {json.dumps({'stage': 'drafting', 'pct': 30, 'total_pages': len(candidates)})}\n\n"
            drafts = await run_drafter_pool(candidates, fact_bundle, progress_cb=lambda i: ...)

            yield f"data: {json.dumps({'stage': 'verifying', 'pct': 80})}\n\n"
            proposals = await run_verifier(drafts, fact_bundle)

            # Store results
            _replace_agent_generated_changes(state, proposals, session_id, ...)
            _persist_history_snapshot(state, user, summary)

            yield f"data: {json.dumps({'stage': 'done', 'pct': 100, 'count': len(proposals)})}\n\n"
        except Exception as exc:
            yield f"data: {json.dumps({'stage': 'error', 'error': str(exc)})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
```

**Frontend SSE pattern (TanStack Query compatible):**

```typescript
// In MeetingSummary.tsx — replaces the simple proposeChanges() fetch
function useProposalStream(sessionId: string | null) {
  const [stage, setStage] = useState<string>("idle");
  const [pct, setPct] = useState(0);
  const queryClient = useQueryClient();

  const start = useCallback((query?: string) => {
    const url = `/api/sessions/${sessionId}/review/changes/propose/stream`;
    const es = new EventSource(url);  // POST with body needs fetch + ReadableStream instead
    es.onmessage = (e) => {
      const data = JSON.parse(e.data);
      setStage(data.stage);
      setPct(data.pct ?? pct);
      if (data.stage === "done") {
        es.close();
        // Invalidate the changes query to refresh the card list
        queryClient.invalidateQueries({ queryKey: ["changes", sessionId] });
      }
      if (data.stage === "error") { es.close(); }
    };
    return () => es.close();
  }, [sessionId]);

  return { stage, pct, start };
}
```

Note: `EventSource` only supports GET. Since `propose` uses POST (for the query body), use `fetch` with `ReadableStream` and `response.body.getReader()` instead, or move the query to a URL param and use GET. The latter is simpler for this use case since the query string is short.

**Fallback (if SSE proves incompatible with ngrok/reverse proxy):** Use the existing non-streaming POST endpoint plus TanStack Query polling at 3-second intervals on the `/review/changes` endpoint. Set `refetchInterval: 3000` while `proposingChanges === true`, and clear the interval when the `generated_count` in the response is non-zero. This is the backup path, not the primary.

---

## Safe Confluence Apply Pattern

The existing `_commit_with_retry()` in `tools.py` already implements the core of the safe apply pattern correctly: fetch latest version → apply transform → push with optimistic locking → retry up to 3 times on version conflict with exponential backoff.

What it lacks for the multi-agent pipeline is a **section anchor verification step** before the write.

**Enhanced safe apply sequence:**

```
1. fetch_live_page(page_id)  →  current HTML + version N + available_headings[]
2. VERIFY: section_heading from proposal card is in available_headings[]
   - If NOT found: surface conflict to user before attempting write
     → status: "conflict", conflict_detail: "Section '{heading}' no longer exists. Page may have been restructured."
3. VERIFY: before_content semantic match against current section HTML
   - If section content has drifted significantly: flag as "review" risk
   - Use difflib.SequenceMatcher ratio > 0.6 as the threshold
4. Apply: commit_document_edit / commit_delete / create_confluence_page
5. On version conflict (409): re-fetch, re-verify anchors, retry (existing _commit_with_retry handles this)
6. On success: _reindex_in_background(page_id)  [already exists in tools.py]
```

**Section anchor verification (new code, small):**

```python
def verify_section_still_exists(live_html: str, heading: str | None) -> tuple[bool, str]:
    """Returns (ok, reason). ok=True means safe to proceed."""
    if not heading:
        return True, ""  # root-level change, no heading to verify
    headings = extract_headings(live_html)  # already exists in utils/html_parser.py
    if heading in headings:
        return True, ""
    # Fuzzy match — heading may have been slightly renamed
    from difflib import get_close_matches
    close = get_close_matches(heading, headings, n=1, cutoff=0.75)
    if close:
        return False, f"Section '{heading}' not found; closest match: '{close[0]}'"
    return False, f"Section '{heading}' no longer exists in page. Available: {headings[:5]}"
```

This function plugs into `_execute_single_change()` in `review/api.py` before calling the editor agent:

```python
async def _execute_single_change(state, change):
    # New: pre-flight anchor check
    if change.get("page_id") and change.get("section_heading"):
        live = fetch_live_page(change["page_id"], change.get("section_heading"))
        ok, reason = verify_section_still_exists(live.section_html or "", change["section_heading"])
        if not ok:
            change["status"] = "conflict"
            change["execution_error"] = reason
            return {"id": change.get("id"), "success": False, "error": reason, "conflict": True}
    # ... existing editor_agent.handle_prepared_query path
```

**Version conflict surface to user:** When a conflict occurs during execution, return `{"success": false, "conflict": true, "error": "..."}` in the `ChangeResult`. The frontend should display a "Refresh and re-apply" prompt for conflict failures rather than a generic error, so users understand the page changed between proposal generation and apply time.

---

## Re-indexing Strategy

**Requirement:** Re-index changed Confluence pages into Pinecone + Neo4j after accepted changes are applied, without blocking the user's review experience.

**Current approach:** `_reindex_in_background(page_id)` in `tools.py` already fires a daemon thread for Pinecone re-indexing via `IngestionPipeline().process_page(page_id)`. This is the correct pattern — fire-and-forget, does not block the HTTP response.

**What is missing:** The Neo4j Confluence page graph (`confluence_page_graph`) is not re-indexed after a commit. Only Pinecone gets updated. After a change is applied, the graph nodes for that page become stale until the TTL expires (default 2 hours).

**Recommended enhancement to `_reindex_in_background`:**

```python
def _reindex_in_background(page_id: str, graph_user_id: str = "") -> None:
    """Fire-and-forget: re-index page in Pinecone + Neo4j graph."""
    def _run():
        # Step 1: Pinecone re-index (existing)
        try:
            from ..ingestion.doc_pipeline import IngestionPipeline
            IngestionPipeline().process_page(page_id)
        except Exception as exc:
            logger.error("Background Pinecone re-indexing failed for %s: %s", page_id, exc)

        # Step 2: Neo4j page graph node update (new)
        if graph_user_id:
            try:
                asyncio.run(_reindex_page_in_graph(page_id, graph_user_id))
            except Exception as exc:
                logger.error("Background Neo4j re-indexing failed for %s: %s", page_id, exc)

    threading.Thread(target=_run, daemon=True).start()
```

**`_reindex_page_in_graph` (new function in confluence_page_graph.py):**

Fetch fresh HTML for `page_id` via `ConfluenceConnector`, re-parse sections via `_sections_from_html`, and MERGE/UPDATE the existing `CfPage` + `CfSection` + `CfTerm` nodes in Neo4j using the existing schema. Do not rebuild the entire user graph — only update nodes for the specific `page_id`. This is the targeted invalidation pattern vs full rebuild.

**When to trigger re-index:** Call `_reindex_in_background(page_id, graph_user_id)` from `_execute_single_change()` when `change["status"] == "executed"`. Pass `graph_user_id` from `_confluence_graph_user_id(...)` which is already computed in `_execute_single_change`. This ensures both stores stay consistent after every successful apply.

**Re-indexing and user review independence:** Because re-indexing is fire-and-forget on a daemon thread, it runs concurrently with the user continuing to review remaining cards. The HTTP response for the execute call returns immediately after setting `status: "executed"`. The user does not wait for indexing.

---

## Long-Running Job: Timeout and Cancellation

The FastAPI server runs on a single asyncio event loop. A 20-minute SSE generator will hold the connection open. This is safe for FastAPI with `uvicorn --timeout-keep-alive 1200`. The two risks are:

1. **ngrok connection timeout:** ngrok default idle timeout is 30s. For active SSE streams, the connection is not idle, but confirm ngrok plan allows long-lived connections.
2. **Client navigation away:** If the user closes the tab mid-pipeline, the SSE generator should detect the disconnected client and cancel the remaining LLM calls. In FastAPI, check `await request.is_disconnected()` inside the SSE generator loop.

```python
async def event_generator():
    for task in pipeline_stages:
        if await request.is_disconnected():
            logger.info("Client disconnected, cancelling pipeline for %s", session_id)
            return
        result = await task
        yield f"data: {json.dumps(result)}\n\n"
```

---

## Build Order Implications

The components have strict build dependencies. Build in this order:

**Phase A — Schema and extractor (no UI changes needed yet):**
1. Expand `ChangeItem` TypeScript type with `transcript_evidence`, `confidence`, `risk`, `verifier_note` (additive, backward-compatible)
2. Build `FactBundle` dataclass and `ExtractorAgent` in `confluence_logic/agents/extractor_agent.py`
3. Add extractor call to the existing `_generate_review_insights` parallel gather OR as a separate step in the propose pipeline

**Phase B — Drafter pool and verifier:**
4. Build `DrafterAgent` in `confluence_logic/agents/drafter_agent.py` (one page, focused prompt)
5. Build `VerifierAgent` in `confluence_logic/agents/verifier_agent.py` (receives all drafts + transcript)
6. Build `multi_agent_pipeline()` orchestration function — wraps extractor → retriever → drafter pool → verifier
7. Wire `multi_agent_pipeline()` into `_propose_changes_for_state()` in `review/api.py` with fallback to existing `ProposedChangesAgent`

**Phase C — Async progress and UI:**
8. Add SSE streaming endpoint `/sessions/{session_id}/review/changes/propose/stream`
9. Update `MeetingSummary.tsx` to use streaming progress bar when calling propose (or polling fallback)
10. Update proposal card rendering to display `transcript_evidence`, `confidence`, `risk` badges

**Phase D — Safe apply and re-indexing enhancements:**
11. Add `verify_section_still_exists()` pre-flight to `_execute_single_change()`
12. Expand `_reindex_in_background()` to cover Neo4j page graph nodes
13. Add conflict status (`status: "conflict"`) handling in frontend card

**Why this order:**
- Phase A is pure backend with no breaking changes; safe to merge independently
- Phase B builds on A (extractor output feeds drafter input); cannot skip A
- Phase C is pure UI and streaming layer; can be built in parallel with B but needs B's output schema to be stable
- Phase D enhancements are standalone improvements to existing functions; can be built any time after Phase A establishes the expanded card schema

---

## Anti-Patterns to Avoid

### Anti-Pattern 1: Single Mega-Prompt for All Pages
**What:** Sending all retrieved candidate pages + full transcript to one LLM call for proposals (current behavior of `ProposedChangesAgent.propose()`).
**Why bad:** Context window saturation causes page blending; model starts referencing content from page A when proposing changes to page B. `MAX_RELEVANT_PAGES=6` and `MAX_PAGE_CONTEXT_CHARS=3500` are band-aids on this problem.
**Instead:** One drafter per candidate page, each with its own bounded context.

### Anti-Pattern 2: Blocking the Event Loop on LLM Calls During Apply
**What:** Running `commit_document_edit` synchronously during an async FastAPI handler.
**Why bad:** The existing `_commit_with_retry` uses `time.sleep()` for backoff — this blocks the asyncio event loop during retries. Current code wraps this in `asyncio.to_thread` implicitly because the editor agent runs in a thread via `asyncio.to_thread`. Verify this thread isolation is preserved when the multi-agent pipeline calls execute.
**Instead:** Ensure all `_commit_with_retry` calls remain inside `asyncio.to_thread` wrappers.

### Anti-Pattern 3: Re-indexing on the Same Thread as the Apply Response
**What:** Waiting for Pinecone + Neo4j re-indexing to complete before returning the execute response.
**Why bad:** Re-indexing a page (HTML fetch → docling conversion → embedding → upsert) takes 5–30 seconds. Blocking on this makes every accepted card take 30+ seconds to confirm.
**Instead:** The existing `threading.Thread(daemon=True)` pattern in `_reindex_in_background` is correct; do not change it to `await`.

### Anti-Pattern 4: Storing Proposals Only in In-Memory State
**What:** Current `state["pending_changes"]` is in-memory `MeetingStateProxy`. If the server restarts, all proposals are lost.
**Why bad:** Users may take time reviewing cards; a server restart between proposal generation and card execution loses all work.
**Instead:** `_persist_history_snapshot()` already writes `pending_changes` into `summary_json` in Supabase, and `_hydrate_state_from_history_item()` restores them. Ensure both paths are called after every proposal generation and after every execute. This is already partially wired — verify it is called in the new multi-agent pipeline path too.

### Anti-Pattern 5: Verifier as a Gatekeeper That Blocks All Cards
**What:** Making the verifier's `"reject"` verdict cause proposals to be dropped entirely and not returned to the frontend.
**Why bad:** Verifier may be wrong. The user is the final authority per the project safety requirement. A verifier that silently drops cards removes user agency.
**Instead:** The verifier adds `confidence`, `risk`, and `verifier_note` fields but all proposals are returned to the frontend. Rejected proposals appear with `risk: "risky"` and a note. The UI can sort/dim them, but the user can still accept them.

---

## Component Boundaries Summary

| Component | File | Responsibility |
|-----------|------|----------------|
| ExtractorAgent | `agents/extractor_agent.py` (new) | Transcript → FactBundle (decisions, action items, doc-worthy updates) |
| DrafterAgent | `agents/drafter_agent.py` (new) | FactBundle + one CandidatePage → ProposalDraft |
| VerifierAgent | `agents/verifier_agent.py` (new) | All ProposalDrafts + transcript → VerifiedProposal[] with confidence/risk |
| PipelineOrchestrator | `agents/pipeline.py` (new) | Async coordination of extractor → retriever → drafter pool → verifier; progress events |
| ProposedChangesAgent | `agents/proposed_changes_agent.py` (existing) | Kept as single-agent fallback path |
| review/api.py | (existing, modified) | Wire pipeline into propose endpoint; add SSE streaming endpoint; add section-anchor pre-flight to execute |
| tools.py | (existing, modified) | Expand `_reindex_in_background` to include Neo4j graph invalidation |
| confluence_page_graph.py | (existing, modified) | Add `_reindex_page_in_graph(page_id, user_id)` targeted node update |
| MeetingSummary.tsx | (existing, modified) | Consume SSE progress events; render new card fields (evidence, confidence, risk badge) |
| types.ts | (existing, modified) | Extend ChangeItem with transcript_evidence, confidence, risk, verifier_note |

---

*Architecture research: 2026-05-11*
