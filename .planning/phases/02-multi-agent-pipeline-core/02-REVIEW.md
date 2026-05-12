---
phase: 02-multi-agent-pipeline-core
reviewed: 2026-05-12T00:00:00Z
depth: standard
files_reviewed: 7
files_reviewed_list:
  - confluence_logic/tests/test_pipeline.py
  - confluence_logic/review/supabase_schema.sql
  - confluence_logic/review/supabase_store.py
  - confluence_logic/agents/fact_extraction_agent.py
  - confluence_logic/agents/drafter_agent.py
  - confluence_logic/agents/verifier_agent.py
  - confluence_logic/review/api.py
findings:
  critical: 5
  warning: 5
  info: 5
  total: 15
status: issues_found
---

# Phase 02: Code Review Report

**Reviewed:** 2026-05-12T00:00:00Z
**Depth:** standard
**Files Reviewed:** 7
**Status:** issues_found

## Summary

Seven files comprising the Phase 2 multi-agent pipeline core were reviewed: the test harness, Supabase schema, store helpers, three new agents (fact extraction, drafter, verifier), and the review API that orchestrates them. The pipeline architecture is sound in broad strokes — asyncio.gather for parallel drafting, incremental Supabase writes, xfail-guarded forward-looking tests. However, five critical defects were found: a schema contract mismatch between `ExtractedFacts` and its test fixture; a silent data-loss path when `user_id` is `None`; a `_format_transcript` buffer-math bug that blows token budgets on misconfigured deployments; a Supabase RLS policy that allows cross-user proposal injection; and a `job_id=None` return that makes the 202 pipeline-start response unpollable. Several warnings address duplicate code, a misleading `upsert_proposal` name (no actual upsert), and a covert `change_type` coercion in the drafter.

---

## Critical Issues

### CR-01: `ExtractedFacts.owners` and `.deadlines` typed as `List[str]` but test fixture assigns dicts — schema contract broken

**File:** `confluence_logic/agents/fact_extraction_agent.py:33-34`
**Also:** `confluence_logic/tests/test_pipeline.py:121-122`

**Issue:** `ExtractedFacts` declares `owners: List[str] = []` and `deadlines: List[str] = []`. The test fixture at lines 121–122 assigns `fake_final_output.owners = {"Prepare rollout checklist": "Ben"}` (a dict) and `fake_final_output.deadlines = {}` (a dict). The FACT_EXTRACTION_PROMPT at lines 48–55 also describes `owners` as "people responsible" (implying a list of names) while `action_items` already captures "owner and deadline if mentioned." The schema and the test expectation are inconsistent. If the LLM follows the prompt and returns `owners` as a list of strings (`["Asha", "Ben"]`), but downstream code expects a dict, any consumer that iterates `facts.owners.items()` will crash with `AttributeError`. The drafter currently ignores both fields, so the bug is latent but will surface when owners/deadlines are wired into drafting or display logic.

**Fix:** Decide on the correct type. If owners is a mapping of task→person, change the schema and prompt to match:
```python
class ExtractedFacts(BaseModel):
    decisions: List[str] = []
    action_items: List[str] = []
    new_requirements: List[str] = []
    owners: Dict[str, str] = {}      # task_description -> owner_name
    deadlines: Dict[str, str] = {}   # task_description -> deadline_string
    doc_worthy_updates: List[str] = []
    query_terms: List[str] = []
```
Then update `FACT_EXTRACTION_PROMPT` to request dict output for those fields and update the test fixture to match the schema.

---

### CR-02: `_format_transcript` produces output larger than `max_chars` when `max_chars < 2000`

**File:** `confluence_logic/review/api.py:268-272`

**Issue:** The truncation logic is:
```python
head = text[:2000]
tail = text[-(max_chars - 2000):]
```
When `max_chars` is between 1 and 1999 (e.g. `JARVIS_REVIEW_MAX_INPUT_CHARS=1000`), `max_chars - 2000` is negative (e.g. `-1000`). In Python, `text[-(-1000):]` evaluates to `text[1000:]`, returning most of the transcript. The final output is `head` (2000 chars) plus `tail` (most of the remaining transcript) — far exceeding `max_chars`. This silently blows the token budget on any deployment that sets `JARVIS_REVIEW_MAX_INPUT_CHARS` below 2000.

**Fix:**
```python
def _format_transcript(transcript_log: List[Dict[str, Any]], max_chars: Optional[int] = None) -> str:
    lines = [
        f"{entry.get('participant', 'Unknown')}: {entry.get('text', '')}"
        for entry in transcript_log
        if entry.get("text")
    ]
    text = "\n".join(lines)
    if max_chars is None or max_chars <= 0 or len(text) <= max_chars:
        return text
    # Guard: head must not exceed max_chars itself
    head_size = min(2000, max_chars // 2)
    tail_size = max_chars - head_size
    head = text[:head_size]
    tail = text[-tail_size:] if tail_size > 0 else ""
    return f"{head}\n[... middle transcript omitted ...]\n{tail}"
```

---

### CR-03: Silent data loss — all proposals dropped when `user_id` is `None` inside `_run_pipeline`

**File:** `confluence_logic/review/api.py:1527-1537`

**Issue:** `_run_pipeline` accepts `user_id: Optional[str]`. Inside `_draft_verify_persist`, `upsert_proposal` is called with `"user_id": user_id`. In `supabase_store.upsert_proposal` (line 223), the guard is `if not is_configured() or not row.get("user_id"): return`. If `user_id` is `None`, every `upsert_proposal` call silently returns without writing. The pipeline runs to completion, `update_pipeline_job` marks the job `"completed"`, but zero proposals were persisted. The user sees a completed job with no cards. The `start_pipeline` endpoint enforces auth so `user_id` is non-null in the normal path, but the `Optional[str]` signature invites callers (future background jobs, retries) to pass `None`.

**Fix:** Validate `user_id` at the top of `_run_pipeline` and fail early:
```python
async def _run_pipeline(
    session_id: str,
    job_id: str,
    user_id: Optional[str],
    graph_user_id: str,
) -> None:
    if not user_id:
        logger.error("Pipeline %s: user_id is required but was None — aborting", job_id)
        await asyncio.to_thread(
            supabase_store.update_pipeline_job,
            job_id, None, "failed", "user_id is required", _utc_now_iso(),
        )
        return
    # ... rest of pipeline
```

---

### CR-04: `proposals` RLS insert policy does not verify `job_id` ownership — cross-user proposal injection possible

**File:** `confluence_logic/review/supabase_schema.sql:139-143`

**Issue:** The insert policy for `proposals` is:
```sql
create policy "Users can insert their own proposals"
  on public.proposals
  for insert
  with check (auth.uid() = user_id);
```
This only checks that `user_id` matches the authenticated caller. It does NOT verify that the referenced `job_id` belongs to the same user. A malicious authenticated user can insert a proposal row with their own `user_id` but another user's `job_id` (obtained from a shared link or guessed UUID). Via the foreign key + `ON DELETE CASCADE`, this contaminates the victim's job. The select policy returns proposals where `auth.uid() = user_id`, so the victim would not see the injected rows — but the injected rows persist and inflate job proposal counts.

**Fix:** Add a subquery check to the insert policy:
```sql
create policy "Users can insert their own proposals"
  on public.proposals
  for insert
  with check (
    auth.uid() = user_id
    AND EXISTS (
      SELECT 1 FROM public.pipeline_jobs pj
      WHERE pj.job_id = proposals.job_id
        AND pj.user_id = auth.uid()
    )
  );
```

---

### CR-05: `POST /review/pipeline/start` returns `{"job_id": null}` when Supabase is unconfigured — 202 with unpollable job_id

**File:** `confluence_logic/review/api.py:1670-1680`

**Issue:** If `supabase_store.create_pipeline_job` returns `None` (Supabase not configured or network error), the response is `{"job_id": null, "status": "accepted"}`. The endpoint docstring says "Poll `/review/pipeline/{job_id}`" but a `null` `job_id` makes that impossible. The pipeline still runs as a background task, but the client has no way to track it. This is an API contract violation — a 202 that cannot be polled is functionally equivalent to fire-and-forget with no acknowledgment.

**Fix:** Either return a generated in-process UUID when Supabase is unavailable, or return 503 when Supabase is required:
```python
if not job_id:
    raise HTTPException(
        status_code=503,
        detail="Pipeline tracking unavailable: Supabase is not configured or unreachable.",
    )
```
Or generate a fallback in-memory job_id:
```python
import uuid
job_id = await asyncio.to_thread(...) or str(uuid.uuid4())
```

---

## Warnings

### WR-01: `upsert_proposal` is a plain INSERT with no conflict handling — misnamed and inserts duplicates on retry

**File:** `confluence_logic/review/supabase_store.py:221-236`

**Issue:** The function is named `upsert_proposal` but the Supabase REST call at line 229 is a plain `POST` to `/rest/v1/proposals` with no `on_conflict` query parameter and no `Prefer: resolution=merge-duplicates` header (unlike `upsert_history` which uses both). On network timeout + retry, a duplicate proposal row is inserted. Every retry creates a new row with a new `id` UUID, so duplicates are not detectable after the fact. Also, line 226 unconditionally overwrites `created_at` with the current time, erasing any caller-provided value.

**Fix:** Add upsert semantics if there is a unique key, or add a uniqueness constraint. If proposals are always insert-only (no update), rename the function to `insert_proposal` to avoid false expectations. At minimum, add the `Prefer` header for idempotency if a unique constraint exists:
```python
response = requests.post(
    f"{SUPABASE_URL}/rest/v1/proposals?on_conflict=job_id,page_id,section_heading",
    headers=_rest_headers("resolution=merge-duplicates,return=representation"),
    json=payload,
    timeout=8,
)
```

---

### WR-02: `_openai_completion_options` is duplicated verbatim in `api.py` and `verifier_agent.py`

**File:** `confluence_logic/review/api.py:198-205`
**Also:** `confluence_logic/agents/verifier_agent.py:71-78`

**Issue:** The exact same function body (model prefix check, `max_completion_tokens` vs `max_tokens`+`temperature`) exists in two files. Any change to the model name prefix list (e.g. adding `gpt-6`) must be applied in both places. The current prefix list `("gpt-5", "o1", "o3", "o4")` also omits `"gpt-4o"` — meaning gpt-4o models get the `max_tokens` + `temperature` path, which is correct, but a future maintainer may not realize this function exists twice.

**Fix:** Extract to a shared utility module, e.g. `confluence_logic/utils/openai_helpers.py`:
```python
def openai_completion_options(model: str, max_tokens: int, temperature: float = 0.2) -> Dict[str, Any]:
    opts: Dict[str, Any] = {"model": model}
    if model.startswith(("gpt-5", "o1", "o3", "o4")):
        opts["max_completion_tokens"] = max_tokens
    else:
        opts["max_tokens"] = max_tokens
        opts["temperature"] = temperature
    return opts
```

---

### WR-03: `_update_pipeline_job` status is NOT updated during "retrieval" and "drafting" stage transitions

**File:** `confluence_logic/review/api.py:1596-1598` and `1626-1628`

**Issue:** The calls to `update_pipeline_job` during stage transitions pass `None` for the `status` argument:
```python
# retrieval transition (line 1596-1598)
supabase_store.update_pipeline_job(job_id, "retrieval", None, None, None)
# drafting transition (line 1626-1628)
supabase_store.update_pipeline_job(job_id, "drafting", None, None, None)
```
Inside `update_pipeline_job`, `if status is not None: payload["status"] = status` (line 202) skips the status update. Only `stage` is written. If the pipeline fails mid-retrieval, the job row shows `status="running"` (from Stage 1) and `stage="retrieval"` — which looks identical to an in-progress job. The caller cannot distinguish a stuck/failed job from one that is actively retrieving. The error path at lines 1645–1651 correctly sets `status="failed"`, but any exception before that path is reached during stage transitions leaves a misleading state.

**Fix:** Pass `"running"` explicitly for intermediate transitions:
```python
supabase_store.update_pipeline_job(job_id, "retrieval", "running", None, None)
supabase_store.update_pipeline_job(job_id, "drafting", "running", None, None)
```

---

### WR-04: `_normalize_draft` silently coerces `change_type` to `"create"` for zero-RAG pages with no log

**File:** `confluence_logic/agents/drafter_agent.py:136-137`

**Issue:**
```python
if page_id is None and change_type not in {"create", "edit"}:
    change_type = "create"
```
If the LLM returns `change_type: "delete"` or `"title"` for a zero-RAG page (no `page_id`), it is silently overridden to `"create"`. The drafter returns a "create" card but the `rationale` still reflects a delete or rename intent, producing a confusing card for the reviewer. There is no log message indicating the coercion happened, making it invisible in production.

**Fix:**
```python
if page_id is None and change_type not in {"create", "edit"}:
    logger.warning(
        "DrafterAgent: coercing change_type %r to 'create' for zero-RAG page (page_id=None)",
        change_type,
    )
    change_type = "create"
```

---

### WR-05: `mock_pinecone_store` fixture patches the class method but `_get_store()` returns an already-instantiated singleton

**File:** `confluence_logic/tests/test_pipeline.py:53-60`

**Issue:** The fixture patches `confluence_logic.db.vector_store.PineconeStore.search` (the class's method). However, `_get_store()` in `fact_extraction_agent.py` (line 134–139) returns a module-level `_store` singleton that is instantiated at first call. If the module was imported before the test runs (which it will be due to the module-level `_fact_agent = Agent(...)` construction at line 67), `_store` may already be initialized. A patch on `PineconeStore.search` (the class attribute) will NOT affect the bound method on the already-created instance unless the instance delegates attribute lookup to the class (which Python normally does — so this should work). However, the `monkeypatch.setattr` targets `PineconeStore.search` as a `MagicMock`, not an `AsyncMock`. Since `_merged_rag_retrieval` wraps Pinecone in `asyncio.to_thread`, a sync `MagicMock` is correct. The deeper issue is that `_store` is a global — if test isolation fails (e.g., a previous test initialized `_store` with real credentials), the patch may not take effect. The tests should explicitly reset `_store = None` or use a more targeted patch on the instance.

**Fix:**
```python
@pytest.fixture
def mock_pinecone_store(monkeypatch):
    fake_results = [...]
    mock_search = MagicMock(return_value=fake_results)
    # Reset the singleton so a fresh mock instance is used
    monkeypatch.setattr("confluence_logic.agents.fact_extraction_agent._store", None)
    monkeypatch.setattr(
        "confluence_logic.db.vector_store.PineconeStore.search",
        mock_search,
    )
    return mock_search
```

---

## Info

### IN-01: `JARVIS_AGENT_MODEL` and `DRAFTER_MODEL` default to `"gpt-5-mini"` — invalid model name, broken by default

**File:** `confluence_logic/agents/fact_extraction_agent.py:18`, `confluence_logic/agents/drafter_agent.py:16`

**Issue:** Both files default to `"gpt-5-mini"` which does not exist. As documented in `CLAUDE.md`, this is a known invalid model name. Any deployment that does not set `JARVIS_AGENT_MODEL` will fail at first `Runner.run` call. The module-level `_fact_agent = Agent(model=JARVIS_AGENT_MODEL, ...)` at line 67 of `fact_extraction_agent.py` constructs the agent at import time with an invalid model — if the SDK validates at construction, the entire module fails to import.

**Fix:** Change both defaults to a valid model name per `CLAUDE.md` conventions:
```python
JARVIS_AGENT_MODEL = os.getenv("JARVIS_AGENT_MODEL", "gpt-4o-mini").strip()
```

---

### IN-02: `VERIFIER_MODEL` defaults to `"gpt-5.4-mini"` — also invalid

**File:** `confluence_logic/agents/verifier_agent.py:17`

**Issue:** Same pattern as IN-01. `"gpt-5.4-mini"` is not a valid OpenAI model name. The verifier will fail at runtime on any deployment without `JARVIS_VERIFIER_MODEL` set.

**Fix:**
```python
VERIFIER_MODEL = os.getenv("JARVIS_VERIFIER_MODEL", "gpt-4o-mini").strip()
```

---

### IN-03: `_merged_rag_retrieval` is implemented inside `fact_extraction_agent.py` — wrong module, violates single-responsibility

**File:** `confluence_logic/agents/fact_extraction_agent.py:129-218`

**Issue:** The file imports `PineconeStore` (line 129), instantiates a `_store` singleton (line 131), and defines `_merged_rag_retrieval` (line 147). None of this is fact extraction. The module docstring says "FactExtractionAgent — extracts structured facts" but the file now contains two unrelated responsibilities. `api.py` imports `_merged_rag_retrieval` from `fact_extraction_agent` which is a confusing and brittle coupling. The test for `pipeline_coordinator._merged_rag_retrieval` at `test_pipeline.py:20` imports from a different (not-yet-existing) module than where the function actually lives.

**Fix:** Move `_get_store`, `_merged_rag_retrieval`, and the `PineconeStore` import to a new `confluence_logic/agents/rag_retrieval.py` or the planned `pipeline_coordinator.py` module. Update `api.py` imports accordingly.

---

### IN-04: Test interface for `_verify_proposal` (single-arg `draft`) incompatible with `_run_verifier` (three args)

**File:** `confluence_logic/tests/test_pipeline.py:308`

**Issue:** The test calls `_verify_proposal(draft=draft)` with only one argument. The implemented verifier `_run_verifier` in `verifier_agent.py:86` requires three arguments: `draft`, `transcript_text`, and `page_content`. When `pipeline_coordinator._verify_proposal` is implemented, either it wraps `_run_verifier` with default empty strings for the missing args (degrading verifier quality), or the test signature is wrong. This ambiguity will cause the test to pass for the wrong implementation.

**Fix:** Decide the intended interface now. If `_verify_proposal` should extract transcript from a shared context, define a `PipelineContext` object. Otherwise update the test to reflect the actual three-argument signature:
```python
enriched = await _verify_proposal(
    draft=draft,
    transcript_text="We decided to launch next Friday.",
    page_content="Current Confluence content about launch.",
)
```

---

### IN-05: `_replace_agent_generated_changes` and `_append_agent_generated_changes` are nearly identical — dead duplication

**File:** `confluence_logic/review/api.py:908-948` and `951-989`

**Issue:** Both functions build the same per-proposal dict with identical field list. The only difference is that `_replace_agent_generated_changes` filters out existing `"meeting_proposal_agent"` source changes first. This is a ~80-line duplication. Any new field added to the proposal dict (e.g. `confidence` was added as a Phase 2 field) must be added to both functions.

**Fix:** Extract the shared proposal-building logic:
```python
def _build_proposal_entry(proposal: Dict, next_id: int, session_id: str, timestamp: str, source: str, query: str) -> Dict:
    return {
        "id": next_id,
        "change_type": proposal.get("change_type") or "edit",
        # ... shared fields
        "source": source,
    }

def _replace_agent_generated_changes(state, proposals, session_id, query):
    pending = [c for c in (state.get("pending_changes") or []) if c.get("source") != "meeting_proposal_agent"]
    return _append_proposals(state, proposals, pending, session_id, query, "meeting_proposal_agent")

def _append_agent_generated_changes(state, proposals, session_id, query, source="meeting_proposal_agent"):
    pending = state.get("pending_changes") or []
    return _append_proposals(state, proposals, pending, session_id, query, source)
```

---

_Reviewed: 2026-05-12T00:00:00Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
