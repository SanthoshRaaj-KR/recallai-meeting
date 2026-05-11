# Domain Pitfalls

**Domain:** AI-generated Confluence change proposal system (multi-agent RAG pipeline)
**Project:** Jarvis — Meeting Intelligence Platform
**Researched:** 2026-05-11
**Confidence:** HIGH — findings grounded in codebase inspection of `confluence_logic/`, not only general theory

---

## Critical Pitfalls

Mistakes that cause rewrites, data loss, or user trust collapse.

---

### Pitfall 1: Hallucination of Decisions, Owners, and Metrics

**What goes wrong:** The proposal agent invents specific claims — "Alice owns this", "deadline is Friday", "we agreed on $50K budget" — that were never said in the meeting. These land in the `after_content` of a proposal card. A user skimming cards accepts without reading carefully. The fabrication is now in Confluence.

**Why it happens:** The `ProposedChangesAgent` receives the full transcript plus a meeting summary, but the summary itself is LLM-generated from the same transcript. Two LLM hops compound errors. `max_tokens=1400` creates pressure to fill the output budget even when the transcript supports fewer changes. GPT-5 mini will fill underspecified instructions with plausible-sounding content.

**Evidence in codebase:** The system prompt says "Never invent decisions, owners, dates, metrics, or page IDs." This instruction exists because the model will violate it without it — but a single negative instruction in a system prompt is not a reliable enforcement mechanism under context pressure. No post-generation grounding check exists: `_normalize_changes()` at line 210 validates structure only, not claim support.

**Consequences:** Misinformation in Confluence. User trust collapse if discovered. Once accepted and re-indexed, the fabricated content becomes future RAG retrieval context, compounding the error in subsequent meetings.

**Prevention:**
- Add a verifier pass after `propose()` returns: for each proposed `after_content`, verify that at least one named entity or claim in it is traceable to a transcript span. Reject or downgrade confidence on proposals that fail this check.
- Expose `transcript_evidence` snippets on each card (already in the active requirements list) so users have an anchor to spot-check claims.
- Cap `MAX_RELEVANT_PAGES` (currently 6) and `max_tokens` (1400) tightly — more context and more output budget increases hallucination surface, not accuracy.
- The verifier/critic agent (planned) must receive both the `after_content` and the raw transcript and return a grounded/ungrounded verdict with the specific passage that supports the claim.

**Detection warning signs:**
- Proposals referencing specific names, dates, or numbers that are absent from `transcript_highlights`
- High `generated_count` (e.g., 8–12 proposals) for a short or unfocused meeting
- `before_content` is empty on an edit proposal (agent invented content with no grounding in what exists)

**Build phase:** Must be addressed in the proposal generation phase, not deferred to UI. The verifier agent is the core mitigation and should be scoped into the same milestone as `ProposedChangesAgent`.

---

### Pitfall 2: Version Conflict Exhaustion on High-Traffic Pages

**What goes wrong:** When a user accepts a change to a frequently edited Confluence page (e.g., a team roadmap or sprint board), another team member edits the same page between when the proposal was generated and when the user clicks "accept." The optimistic lock version number stored at proposal time is stale. `_commit_with_retry` retries up to `_MAX_VERSION_RETRIES = 3` times with exponential backoff, then raises `ValueError`. The change silently fails with `status = "failed"` in `pending_changes`, and the user sees no actionable message in the current UI.

**Why it happens:** The Confluence API uses optimistic locking. The version number is captured during retrieval (either via `fetch_live_page` at proposal time in `_fallback_workspace_context`, or from the Neo4j graph which may be hours old). The gap between proposal generation (up to 20 minutes) and execution creates a conflict window. The retry loop re-fetches the version and retries the edit, but the retry applies the `old_block_html` → `new_block_html` transformation against the freshly fetched HTML. If a concurrent edit has changed the section structure, `edit_block_in_section` may match the wrong block or match nothing, producing a no-op diff that `preview_edit` would flag as "No visible DOM changes detected."

**Evidence in codebase:** `tools.py:88–127` implements `_commit_with_retry`. On `_MAX_VERSION_RETRIES` exhaustion it raises `ValueError`. `_execute_single_change` catches it and sets `change["status"] = "failed"` but the API response just returns `{"success": False, "error": "..."}`. The sync-sage-bot UI has no current handling for partially-failed batch executions.

**Consequences:** Silent data loss. The user believes the change was applied. Confluence page is unchanged. No re-try from the UI side.

**Prevention:**
- At execute time (not proposal time), always re-fetch the live page and diff the `before_content` against the current section HTML before committing. If the section has already been modified by someone else, flag the card as "conflict — please re-review" rather than silently failing.
- The UI must render per-card execution status (executed / failed / conflict) with a retry action.
- `_execute_changes_for_state` currently executes changes sequentially. If the user accepts 5 cards on the same page, the first success updates the version; cards 2–5 will conflict. The execution loop should re-fetch and update `expected_version` after each successful commit on the same `page_id` before proceeding to the next.

**Detection warning signs:**
- `change["status"] == "failed"` with `execution_error` containing "Version Conflict"
- Multiple accepted changes targeting the same `page_id` in a single execution batch

**Build phase:** Safe Confluence apply is already in the active requirements. Implement the per-page version chain in the execution loop before the UI card review milestone.

---

### Pitfall 3: Stale Graph Proposals Referencing Deleted or Moved Pages

**What goes wrong:** The Neo4j `CfPage` nodes are refreshed only when the TTL (`GRAPH_TTL_SECONDS = 7200`, i.e., 2 hours) has expired since `indexed_at`. A Confluence admin deletes or renames a page within that 2-hour window. The next meeting's proposal agent retrieves the deleted page from the graph and generates an edit proposal with a stale `page_id`. When the user accepts, `fetch_live_page` returns a 404, `_commit_with_retry` receives an HTTP error, and the change fails.

**Why it happens:**
1. TTL is in-process state. A server restart resets `indexed_at` to null (the graph still contains stale data) but `_is_fresh()` returns `False`, triggering a full re-index — which is correct. However, the `CONCERNS.md` note confirms the TTL and refresh logic "will not survive a server restart" because `indexed_at` is stored in Neo4j, not locally. On restart, the graph TTL check _will_ run correctly because `indexed_at` is persisted in Neo4j. The real risk is the 2-hour window itself.
2. The incremental refresh in `_write_pages_incremental` does delete pages no longer in `live_page_ids`, but this only runs at TTL expiry, not on-demand before proposal generation.
3. The fallback path (`_fallback_workspace_context`) calls `fetch_live_page` on each candidate, which would catch a 404 — but this path is only used when the graph query returns zero results. When the graph does return results, stale `page_id` values flow directly into the LLM payload without live validation.

**Evidence in codebase:** `confluence_page_graph.py:149–168` — `ensure_user_confluence_graph` skips re-index if fresh. `proposed_changes_agent.py:91–121` — `_graph_workspace_context` passes graph results directly to the LLM payload with no live `page_id` validation.

**Consequences:** Proposals targeting non-existent pages. Execution failures. User confusion when the card references a page they can see was deleted.

**Prevention:**
- Before the proposal agent formats the LLM payload, validate that all `page_id` values returned by the graph query resolve to live pages. A lightweight `HEAD` or metadata call to the Confluence API is sufficient — not a full HTML fetch.
- Alternatively: force a partial graph refresh (re-check deleted pages) before each proposal generation run, independent of TTL.
- Surface the graph's `indexed_at` timestamp in the proposal response so users know how fresh the knowledge base is.

**Detection warning signs:**
- Proposal cards with `page_id` that return 404 on execute
- `change["execution_error"]` containing "404" or "not found"
- Proposals for pages deleted in the last 2 hours

**Build phase:** Index staleness mitigation should be implemented in the same milestone as the graph-based RAG retrieval. A live validation step at proposal time is the minimum viable fix.

---

## Moderate Pitfalls

---

### Pitfall 4: Scope Creep — Agent Proposes Too Many Changes

**What goes wrong:** For a long meeting with broad discussion, the `ProposedChangesAgent` generates 10–12 proposals (the current cap is `raw_changes[:12]` in `_normalize_changes`). The user faces a wall of cards. Cognitive overload causes bulk-accept behavior, defeating the safety model of per-card review. Alternatively, the user rage-rejects everything, including genuinely useful changes.

**Why it happens:**
- The agent is prompted to cover "every useful Confluence update supported by the meeting context." Long transcripts provide many weak signals, each of which produces a proposal.
- `MAX_RELEVANT_PAGES = 6` means up to 6 pages can be targeted. With 2 proposals per page, the wall is 12 cards.
- No confidence scoring or priority ranking exists in the current output schema. All proposals are surfaced equally.
- The `JARVIS_PROPOSE_CHANGES_MAX_TOKENS = 1400` default is low, which creates pressure to compress many proposals into tokens, reducing rationale quality.

**Evidence in codebase:** `proposed_changes_agent.py:210-242` — `_normalize_changes` caps at 12 but does not rank or filter by confidence. No `confidence_score` or `risk_level` field exists in the current schema, though the active requirements list these as desired.

**Consequences:** User fatigue → bulk accept of bad changes, or bulk reject of good ones. Trust erosion in the system.

**Prevention:**
- The verifier agent should score each proposal (high/medium/low confidence) and the UI should default to showing only high-confidence proposals, with low-confidence ones collapsed behind a "show more" interaction.
- Cap the hard maximum at 6 proposals per pipeline run, not 12. If the meeting surface area exceeds 6, the verifier should select the 6 most impactful.
- Proposals targeting the same section as a previous accepted proposal in the same session should be suppressed (deduplication).
- Surface `rationale` prominently in the card UI — users skim rationale quickly to triage.

**Detection warning signs:**
- `generated_count > 6` on any single pipeline run
- Multiple proposals targeting the same `page_title`
- Proposals with empty or single-sentence `rationale`

**Build phase:** Proposal ranking and confidence scoring must be part of the verifier agent milestone. The UI card layout should default-collapse low-confidence proposals.

---

### Pitfall 5: RAG Retrieval Miss — Keyword Graph Returns Wrong Sections

**What goes wrong:** The Neo4j graph scoring (`query_user_confluence_graph`) uses term-frequency matching: `size([term IN $terms WHERE term IN coalesce(p.terms, [])])`. A meeting discussing "sprint velocity" and "story points" will not retrieve a Confluence page titled "Agile Delivery Framework" if its indexed terms are "framework, delivery, agile" — the intersection is zero. The agent never sees the most relevant page. It either hallucinates a proposal for a non-existent page, or misses the update entirely.

**Why it happens:**
- `_terms()` in `confluence_page_graph.py:59–68` does exact-word extraction with a 3-character minimum and stop-word filtering. No stemming, no semantic similarity.
- The Pinecone fallback (`_pinecone_search` via `search_workspace_knowledge`) does use embeddings but it's only invoked in the `_fallback_workspace_context` path (when the graph query succeeds with some results, the fallback is skipped entirely).
- `MAX_RETRIEVAL_QUERIES = 6` sends the summary title, key topics, decisions, and action items as separate queries. If the meeting summary itself uses different vocabulary than the Confluence page titles, all 6 queries miss.

**Evidence in codebase:** `confluence_page_graph.py:302–351` — `query_user_confluence_graph` is pure keyword/term overlap. `proposed_changes_agent.py:195–207` — if `_graph_workspace_context` returns any results, `_fallback_workspace_context` (which includes Pinecone) is skipped.

**Consequences:** Missed updates. Users notice that the system proposed changes to low-priority pages while ignoring the pages that actually needed updating. Trust erosion.

**Prevention:**
- The graph query should serve as a first-pass filter, not the sole retrieval mechanism. Always merge graph results with Pinecone vector results (the `search_workspace_knowledge` function already does this — the issue is that `_workspace_context` skips it when the graph returns hits).
- Modify `_workspace_context` to always run both graph and Pinecone retrieval in parallel, then merge and deduplicate results.
- Add a re-ranking step: after merging, re-score candidates using the full meeting summary (cosine similarity against embedding) to surface semantically relevant pages even with vocabulary mismatch.

**Detection warning signs:**
- Proposals targeting pages that don't obviously relate to the meeting topic
- No proposals for pages that the user expected to be updated
- `retrieval_source: "neo4j_confluence_graph"` on a run where the relevant pages are known to be in Pinecone

**Build phase:** The retrieval merge fix should be part of the RAG pipeline milestone, before the proposal agent is built out. This is an architecture issue that will compound every subsequent hallucination.

---

### Pitfall 6: Long-Running Pipeline Failures — Orphaned State and No Resume

**What goes wrong:** The 20-minute proposal pipeline runs as a single `await agent.propose(...)` call inside an HTTP request handler. If the FastAPI server restarts, the user closes the browser, or the OpenAI API times out mid-pipeline, the pipeline is killed with no intermediate state saved. The user returns to the UI and sees no proposals.

**Why it happens:**
- `_propose_changes_for_state` is a single-shot async function. There is no checkpointing.
- `pending_changes` is stored in in-memory `meeting_state` (a `MeetingStateProxy` dict), not in Supabase, until `_persist_history_snapshot` is called at the end of a successful run.
- OpenAI API calls within `ProposedChangesAgent.propose()` are single-request with `max_tokens=1400`. If the model is under load and takes >30 seconds, the default uvicorn request timeout (60s) may kill the connection before the response completes. The actual LLM work is done in `asyncio.to_thread`, but the HTTP response is still waiting on the outer coroutine.
- The `asyncio.run()` inside a running event loop bug (`CONCERNS.md`) shows the codebase has already encountered event loop instability — pipeline failures in a 20-minute run are plausible.

**Evidence in codebase:** `api.py:1066–1099` — `_propose_changes_for_state` has no save-on-progress. State is persisted only at the end. `CONCERNS.md` confirms broad `except Exception` swallowing in 60+ locations, meaning intermediate failures are likely to be silently discarded rather than surfaced.

**Consequences:** 20 minutes of compute wasted. User frustrated with a spinner that ends with nothing. No way to retry from where it left off.

**Prevention:**
- Move proposal generation to a background task (FastAPI `BackgroundTasks` or a proper task queue). Return a `task_id` immediately and let the UI poll a status endpoint.
- Persist intermediate state to Supabase after each worker agent completes, not just at the end. The `summary_json.pending_changes` field already exists in the schema — use it as a checkpoint store.
- Add a pipeline status field (`generating_proposals`, `proposals_ready`, `proposal_error`) to the bot status endpoint so the UI can show meaningful progress and distinguish "still running" from "failed silently."
- The pipeline progress indicator is already in the active requirements — implement it as a Supabase-backed status field, not just a frontend spinner.

**Detection warning signs:**
- `generated_count == 0` on a run that took more than 5 seconds
- `pending_changes` absent from `summary_json` in Supabase after a pipeline run
- HTTP 504 or connection reset from the `/review/changes/propose` endpoint

**Build phase:** Background task architecture should be established in the earliest pipeline milestone, before the actual proposal logic is built. Retrofitting background execution is harder than building it that way from the start.

---

### Pitfall 7: OpenAI Agents SDK Edge Cases in Multi-Agent Chains

**What goes wrong:** The `EditorAgent` uses the Agents SDK pattern of `.as_tool()` to compose specialists (resolver, editor, deleter, creator, lister) under a master orchestrator. In a prepared-query execution flow (`handle_prepared_query`), the master agent calls `edit_existing_page` which internally calls `Runner.run(self.edit_agent, ...)`. This is a nested `Runner.run` inside an already-running async context. The SDK internally manages its own event loop and context variables.

Known SDK edge cases in this pattern:

1. **Tool context variable isolation:** `_tool_run_state` and `_mutation_observer` are `ContextVar` values set in the outer call. When the SDK spawns the sub-agent in a thread via `.as_tool()`, the sub-agent's context may not inherit the parent's `contextvars` values, depending on the SDK's thread/task execution strategy. This means `get_tool_state()` in worker tools may return a fresh state dict instead of the one reset by `reset_tool_state()`.

2. **`Runner.run` is not reentrant in the same asyncio task:** If the master agent calls a tool that itself calls `Runner.run`, and all of this happens on the same asyncio event loop, the inner `Runner.run` may deadlock or raise if the SDK uses `asyncio.run()` internally (which errors inside a running loop). The `_run_async_blocking` workaround in `tools.py:140–159` spawns a thread to avoid this, but this pattern is fragile and creates thread-to-async boundaries.

3. **`final_output` attribute inconsistency:** `_run_editor` checks `hasattr(result, 'final_output')`. If the SDK version changes the result type (e.g., moves to a dataclass or changes the attribute name), this silently returns `str(result)` — the agent's full runner output stringified, which may include internal SDK state.

4. **Tool call count limits:** The SDK has a default max tool calls per run. A complex prepared query (search → fetch → preview → commit) requires 4 tool calls. If the Confluence API is slow, the model may retry searches, pushing toward the limit and causing truncated execution mid-sequence.

**Evidence in codebase:** `editor_agent.py:232` — `Runner.run(self.agent, enriched_query)` in an async context. `tools.py:140–159` — `_run_async_blocking` workaround already in place, indicating the team has already encountered this class of problem. `editor_agent.py:237` — `hasattr(result, 'final_output')` defensive check.

**Consequences:** Silent agent failures. Tool state not properly initialized between runs. Partial executions with no error surfaced.

**Prevention:**
- Pin the `openai-agents` package to a specific version in `requirements.txt` (currently `latest`). SDK API changes between minor versions have historically been breaking for agent orchestration patterns.
- Replace the `hasattr(result, 'final_output')` guard with an explicit type check against the SDK's documented result type.
- Test the `ContextVar` propagation through `.as_tool()` boundaries explicitly. If `_tool_run_state` is not inherited, move tool state to a function parameter or a dedicated context manager that wraps each `Runner.run` call.
- Add a max tool calls override if the SDK supports it, set to at least 12 for the editor master agent (resolve + search + fetch + preview + commit = 5 minimum, with retries).

**Detection warning signs:**
- `get_tool_state()` returns `{"last_action": None}` after a known successful commit (context bleed)
- `str(result)` in `_run_editor` output containing SDK-internal terms like `RunResult` or `ToolCallOutput`
- Agent correctly calls tools but `change["status"]` remains `"failed"` despite `success=True` from the tool

**Build phase:** SDK pinning must happen before any multi-agent chain is tested in integration. The ContextVar isolation test should be an explicit unit test case added alongside the first agent milestone.

---

### Pitfall 8: Trust Calibration — Users Blindly Accept or Reflexively Reject

**What goes wrong:** Two failure modes exist at opposite ends:

**Mode A — Over-trust (rubber-stamp):** The user clicks "Accept All" for every proposal because the cards are long, the `before`/`after` diff is hard to read in raw HTML, and they assume the AI is right. Bad proposals reach Confluence.

**Mode B — Under-trust (reject everything):** After one bad proposal (e.g., a hallucinated decision), the user rejects all subsequent proposals from every meeting, including correct ones. The product stops delivering value.

**Why it happens:**
- The current `ChangeItem` schema has `before_content` and `after_content` as raw HTML strings. Raw Confluence storage format HTML (`<ac:structured-macro>`, `<ac:rich-text-body>`, etc.) is unreadable to most users.
- No diff visualization is planned at the card level — users can't see what specifically changed at a glance.
- The `rationale` field exists but is generated by the same model that generated the change. A hallucinated change will have a confident-sounding rationale.
- Cards do not include `transcript_evidence` snippets (required in active requirements but not yet implemented), so users cannot cross-check claims.
- No `confidence_score` or `risk_level` is surfaced on cards (also in active requirements but not implemented). All proposals appear equally authoritative.

**Evidence in codebase:** `proposed_changes_agent.py:44–54` — system prompt produces rationale but no evidence snippets. `api.py:879–904` — `_replace_agent_generated_changes` does not add `confidence_score` or `risk_level` fields to the stored change object.

**Consequences:** Both modes destroy the product's core value proposition ("users just approve or reject"). Mode A is a data-quality problem; Mode B is a retention problem.

**Prevention:**
- Render `before`/`after` content as a visual diff (line-level), not raw HTML. The backend already has `difflib.unified_diff` in `tools.py:346–358` for preview — use the same approach to generate a displayable diff at proposal time and store it in the change object.
- Always include `transcript_evidence`: one or two verbatim transcript lines that justify the proposal, displayed inline on the card.
- Add a `confidence_score` (0.0–1.0) computed by the verifier agent. Display it visually (e.g., color-coded border) without exposing the number directly.
- For the first version, default to showing `before` vs `after` as plain text stripped of HTML tags for readability, with an "advanced" toggle for raw HTML.
- Never allow "Accept All" as a single click. The UX should require individual card acceptance with at minimum a brief read-time delay or a secondary confirmation for high-risk changes (`change_type == "delete"`).

**Detection warning signs:**
- User accepts all proposals in under 30 seconds (bulk-accept behavior)
- User rejects all proposals without opening any card detail
- Proposal `before_content` contains Confluence macro XML (`ac:structured-macro`) that is not decoded for display

**Build phase:** Card UX design must be informed by these trust signals. The `transcript_evidence` and `confidence_score` fields need to be in the data model from the first proposal generation milestone — retrofitting them into accepted/stored change objects in Supabase is expensive.

---

## Minor Pitfalls

---

### Pitfall 9: Post-Accept Re-indexing Race Condition

**What goes wrong:** `_reindex_in_background(page_id)` in `tools.py:76–85` fires a daemon thread immediately after a successful commit. If the user generates proposals again within seconds (or from a concurrent session), `ensure_user_confluence_graph` may run while the page is being re-indexed, reading a partially committed version of the page.

**Prevention:** Add a per-page re-index lock or a brief TTL-invalidation marker in Neo4j that forces a fresh fetch for that specific page on the next query, rather than re-indexing the entire graph immediately.

**Build phase:** Post-accept re-indexing milestone.

---

### Pitfall 10: `gpt-5-mini` Model Name Will Fail at Runtime

**What goes wrong:** `JARVIS_REVIEW_MODEL` defaults to `"gpt-5-mini"` which does not exist on the OpenAI API (`CONCERNS.md` confirms this). Any deployment that does not explicitly set the env var will fail at the first LLM call, not at startup.

**Prevention:** Add a startup validation function that calls `openai.models.list()` or a minimal completion with the configured model names, and fails fast with a clear error message rather than at the first user-triggered pipeline call. Set the default to a known-valid model (e.g., `gpt-4o-mini`) until the actual target models are confirmed available.

**Build phase:** Must be fixed before any integration testing. This is a pre-flight blocker.

---

### Pitfall 11: Section Anchor Mismatch on Apply

**What goes wrong:** A proposal is generated with `section_heading = "Goals & Objectives"`. Between proposal generation and execution, a Confluence author renames the section to "Objectives". `get_section_html()` in `html_parser.py` does an exact-string heading match. The section is not found, `edit_block_in_section` produces no diff, and the commit is a no-op. The change is marked as executed but nothing changed.

**Prevention:** After `fetch_live_page` at execution time, fuzzy-match the stored `section_heading` against `available_headings` using `difflib.SequenceMatcher`. If the match ratio is below 0.8, flag the change as "section anchor changed — please re-verify" instead of proceeding silently.

**Build phase:** Safe Confluence apply milestone, alongside version conflict handling.

---

### Pitfall 12: Transcript Truncation Drops Late-Meeting Content

**What goes wrong:** `_format_transcript` truncates long transcripts to `JARVIS_REVIEW_MAX_INPUT_CHARS` (default 0 = no limit, but the head+tail truncation at `api.py:225–232` keeps 2000 chars at the start and the rest from the end). A meeting where the key decisions happen in the middle (common in long standup-style calls) will have the middle section omitted. The proposal agent never sees those decisions.

**Prevention:** Use an extractive summary of the middle portion rather than dropping it. Alternatively, run the proposal agent in chunks (first half, second half, merge and deduplicate proposals) when the transcript exceeds a threshold length.

**Build phase:** Transcript preprocessing milestone, before proposal generation.

---

## Phase-Specific Warnings

| Phase Topic | Likely Pitfall | Mitigation |
|-------------|---------------|------------|
| Proposal agent (initial) | Hallucination of decisions/owners | Verifier agent required in same milestone, not deferred |
| RAG retrieval | Keyword mismatch misses relevant pages | Merge graph + Pinecone results regardless of graph hit count |
| Verifier/critic agent | Confidence scores without evidence anchors | Verifier must output `transcript_evidence` spans, not just a score |
| Safe Confluence apply | Version conflicts + section anchor mismatch | Re-fetch live version at execute time; fuzzy-match heading anchor |
| Pipeline progress UI | Background task assumption vs synchronous HTTP | Implement as background task from day one; poll a Supabase status field |
| Index staleness | 2h TTL window = stale page_ids in proposals | Live page_id validation before proposals are sent to LLM |
| Card review UI | Trust calibration (blind accept / reflexive reject) | Transcript evidence snippets + visual diff required before launch |
| OpenAI Agents SDK | ContextVar isolation through `.as_tool()` | Pin SDK version; add ContextVar propagation unit test |
| Model configuration | `gpt-5-mini` default does not exist | Startup validation before any integration test run |
| Post-accept re-index | Race condition on concurrent proposal generation | Per-page re-index lock or TTL invalidation marker |

---

## Sources

- Direct inspection of `confluence_logic/agents/proposed_changes_agent.py` (hallucination, scope creep, RAG miss)
- Direct inspection of `confluence_logic/agents/tools.py` (version conflicts, anchor mismatch, re-indexing race, SDK edge cases)
- Direct inspection of `confluence_logic/confluence_page_graph.py` (index staleness, TTL behavior, keyword-only scoring)
- Direct inspection of `confluence_logic/review/api.py` (pipeline failure modes, trust calibration, transcript truncation)
- Direct inspection of `confluence_logic/agents/editor_agent.py` (SDK composition patterns, `.as_tool()` nesting)
- `.planning/codebase/CONCERNS.md` (exception swallowing, model name bugs, `asyncio.run()` inside running loop)
- `.planning/codebase/ARCHITECTURE.md` (ContextVar patterns, session state, error handling strategy)
- `.planning/codebase/INTEGRATIONS.md` (Confluence API versioning, Pinecone index config, Neo4j driver)
