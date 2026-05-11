# Technology Stack — Confluence Change Proposal Pipeline

**Project:** Jarvis — Meeting-to-Confluence Change Proposal Milestone
**Analysis Date:** 2026-05-11
**Milestone Scope:** Multi-agent pipeline that extracts facts from meeting transcripts, retrieves affected Confluence sections via RAG, drafts edit proposals, verifies them against transcript evidence, and surfaces them for per-card human approval.

---

## Current Stack Assessment

### What Is Already in Place

The existing codebase provides a strong foundation. The components below are working and should be extended rather than replaced:

| Component | File | Status | Assessment |
|-----------|------|--------|------------|
| `ProposedChangesAgent` | `confluence_logic/agents/proposed_changes_agent.py` | Working skeleton | Single-shot proposal generation; no verifier, no progress events, no streaming. Needs multi-agent expansion. |
| `PineconeStore` | `confluence_logic/db/vector_store.py` | Working | Uses `text-embedding-3-small` by default; configurable via `OPENAI_EMBEDDING_MODEL`. Good. |
| `confluence_page_graph` | `confluence_logic/confluence_page_graph.py` | Working | Per-user Neo4j graph, 2-hour TTL, keyword-based `_terms()` retrieval. Works but keyword search is brittle for semantic queries. |
| `IngestionPipeline` | `confluence_logic/ingestion/doc_pipeline.py` | Working | docling HTML→Markdown, section chunking, version-checked upsert. Solid. |
| `EditorAgent` | `confluence_logic/agents/editor_agent.py` | Working (live edits) | Uses OpenAI Agents SDK `Agent`/`Runner`. Apply pattern here too. |
| Review API | `confluence_logic/review/api.py` | Working | `asyncio.to_thread()` pattern for sync OpenAI calls. Correct. Needs SSE/progress events endpoint. |
| FastAPI + Uvicorn | `confluence_logic/jarvis_agentic.py` | Working | Single asyncio loop. WebSocket already used for live TTS. |

### Critical Bugs Found in Current Stack

**1. Model names are invalid.**
`JARVIS_AGENT_MODEL=gpt-5-mini` and `JARVIS_REVIEW_MODEL=gpt-5-mini` are set as defaults across the codebase. As of August 2025 (training cutoff), the correct OpenAI model identifiers are:
- `gpt-4o-mini` — the widely available small, fast, cheap model
- `gpt-4o` — the mid-tier capable model
- `gpt-4.1-mini` — newer smaller model (released April 2025, HIGH confidence it exists)
- `gpt-4.1` — released April 2025 (HIGH confidence)
- `o3-mini`, `o4-mini` — reasoning models

The string `gpt-5-mini` does not correspond to any known OpenAI model as of the knowledge cutoff. All `gpt-5*` references in the codebase will produce `model_not_found` API errors at runtime. This must be resolved before the pipeline can run.

**2. `neo4j==6.1.0` does not exist on PyPI.**
The `requirements.txt` pins `neo4j==6.1.0` but PyPI's `neo4j` package topped out at `5.x` in 2025. The correct version is `neo4j>=5.14,<6`. This will prevent `pip install` from succeeding. Must be fixed.

---

## Model Strategy

### Confidence note
External tool access is unavailable in this session. The following assessments are based on knowledge through August 2025. Model naming and pricing are subject to change. Flag all model IDs for validation against `https://platform.openai.com/docs/models` before coding.

### The `gpt-5-mini` Problem

The codebase uses `gpt-5-mini` as a default throughout. This name has two possible interpretations:

1. **It is a placeholder the team invented** to mean "whatever the cheapest/fastest model is at build time." In that case the env var system (`JARVIS_AGENT_MODEL`, `JARVIS_REVIEW_MODEL`) is the right abstraction — just correct the default values.
2. **It refers to a model announced but not yet released at knowledge cutoff.** OpenAI announced GPT-5 in May 2025; a `gpt-5-mini` variant may exist by the time this milestone ships. The env var abstraction means the codebase is already designed for this — the code is fine, just the default string needs to be validated.

**Recommendation:** Keep the env var indirection. Change all defaults to `gpt-4.1-mini` (confirmed April 2025 release, competitive with old `gpt-4o-mini` at lower cost) until `gpt-5-mini` is confirmed available and tested. The `_openai_completion_options()` helper already handles `gpt-5*` prefix branching — that code is forward-compatible.

### Recommended Model Assignments

| Pipeline Stage | Task | Recommended Model | Rationale | Confidence |
|---------------|------|------------------|-----------|------------|
| Fact extraction | Extract structured decisions/actions/owners from transcript | `gpt-4.1-mini` | Structured JSON output from constrained schema; `gpt-4.1-mini` handles this reliably with `response_format: json_object` | HIGH (re: capability) — MEDIUM (re: exact model name) |
| Candidate query generation | Turn summary fields into 3-6 RAG search strings | `gpt-4.1-mini` | Low complexity; already works in `_meeting_search_queries()` — no change needed | HIGH |
| Section draft writing | Write `after_content` for a specific page section | `gpt-4.1-mini` or `gpt-4o-mini` | Per-section drafts are bounded, templated prompts. Mini models do not hallucinate significantly more than larger models on this task when given explicit retrieved context and strict instructions to not invent facts. The existing system prompt in `ProposedChangesAgent` ("Never invent decisions, owners, dates, metrics, or page IDs") is already a strong guard. | MEDIUM |
| Verifier/critic | Check that each proposed change is supported by transcript evidence | `gpt-4.1` or `gpt-4o` | This is the hallucination firewall. Using a stronger model here is the highest-leverage quality investment. The verifier receives (transcript excerpt, proposed change, original section content) and must output a structured verdict with evidence citations. Budget 1 verification call per proposal card. | MEDIUM (capability well-established; exact model name MEDIUM confidence) |
| Orchestration | Decide which pages to target, sequence worker tasks | `gpt-4.1-mini` | Orchestration in this pipeline is mostly deterministic routing based on RAG scores and structured summaries — not open-ended reasoning. The existing `ProposedChangesAgent` already does this in one shot. No separate orchestrator LLM is needed if the pipeline is implemented as a deterministic workflow (see Architecture section). | HIGH |

### On Hallucination Risk for Draft Writing

`gpt-4o-mini` and `gpt-4.1-mini` do hallucinate when given open-ended generative tasks. For Confluence section drafting, the risk is bounded by the following already-present guardrails in the codebase:

- Retrieved page context is passed verbatim (`retrieved_page_context` in the proposal payload)
- System prompt explicitly prohibits inventing page IDs, owners, dates, metrics
- `_normalize_changes()` rejects changes with empty `page_title` or `after_content`
- Verifier/critic layer (to be built) provides a second-pass rejection gate

The remaining hallucination surface is: plausible-sounding but wrong section content. This is addressable at the verifier layer — do not over-engineer the drafter model selection.

**Do not upgrade to full `gpt-4o`/`gpt-4.1` for drafting.** The quality difference for structured, context-grounded drafts is marginal, and cost/latency for a 20-minute pipeline with 5-10 proposals is material.

---

## Embedding Model Assessment

### Current State

`PineconeStore` defaults to `text-embedding-3-small` (`OPENAI_EMBEDDING_MODEL` env var). This is already a strong, current-generation model.

### Assessment

| Model | Dimensions | Relative MTEB Score | Cost | Recommendation |
|-------|-----------|-------------------|------|---------------|
| `text-embedding-ada-002` | 1536 | Baseline | Low | Deprecated for new projects. Do not use. |
| `text-embedding-3-small` | 1536 (default) or truncated | ~5-7% better than ada-002 | Very low | **Current choice. Good. Keep.** |
| `text-embedding-3-large` | 3072 (default) or truncated | ~10-15% better than ada-002 | 2-5x vs small | Upgrade candidate only if retrieval precision is measurably poor after testing. |

**Recommendation: Keep `text-embedding-3-small`.** The retrieval bottleneck in this pipeline is not embedding quality — it is query construction. The current `_meeting_search_queries()` function generates queries from raw summary fields (key_topics, decisions, action_items text). These are already semantically rich strings that embed well. Upgrading to `text-embedding-3-large` before diagnosing actual retrieval failures is premature optimization.

**The real embedding gap:** The `confluence_page_graph` Neo4j graph uses keyword-based `_terms()` matching (a regex word extractor + stopword filter), NOT vector similarity. Vector similarity is only used in `PineconeStore`. For the new proposal pipeline, verify that candidate page retrieval goes through `PineconeStore.upsert_chunks` / `query()` (which uses embeddings), not solely through the keyword-based Neo4j graph path. The hybrid is correct — Pinecone for semantic, Neo4j for structural — but the Neo4j keyword path is brittle for abstract meeting language.

**Potential upgrade path (not recommended for this milestone):** `text-embedding-3-large` with dimension reduction to 1024 (`dimensions=1024`) provides a better quality-to-cost ratio than the default 3072-dimension config. Only pursue if retrieval recall is measured to be poor.

---

## Agent Framework Assessment

### OpenAI Agents SDK vs Alternatives

The codebase already uses the OpenAI Agents SDK (`openai-agents`, import name `agents`) in `editor_agent.py` for the live-editing path. The `ProposedChangesAgent` uses raw `openai.chat.completions.create()` calls, which is fine for a single-shot agent but needs structural extension.

#### OpenAI Agents SDK (current)

**Strengths for this pipeline:**
- Already installed and in use in the codebase — zero migration cost
- `Agent` + `Runner` + `function_tool` pattern fits the worker-agent shape well: each worker agent (fact extractor, section drafter, verifier) maps cleanly to an `Agent` with scoped `tools`
- `Runner.run()` supports streaming via `Runner.run_streamed()` — returns an event stream that can be forwarded to SSE
- Tool call tracing built in, which is useful for verifier citations (tool calls appear in the streamed events)
- Python-native async: `Runner.run()` is a coroutine; compatible with the existing `asyncio.to_thread()` pattern in `review/api.py`
- No new dependencies

**Weaknesses for this pipeline:**
- No built-in graph/DAG execution model — workflow sequencing must be coded manually
- No built-in state checkpointing or resumability — if the server restarts mid-pipeline the job is lost
- Streaming progress events require custom forwarding logic from `Runner.run_streamed()` to SSE

#### LangGraph

**Strengths:**
- Explicit DAG/graph execution with typed state; natural fit for multi-step pipelines with parallel branches
- Built-in streaming of intermediate node outputs
- State checkpointing to external store (Redis/Postgres) enables pipeline resumability on crash

**Weaknesses for this project:**
- Not currently installed; requires new dependency tree (`langgraph`, `langchain-core`, and LangChain's OpenAI bindings)
- The existing `Agent`/`Runner`/`function_tool` pattern would need rewriting as LangGraph `ToolNode` / `StateGraph` nodes — significant refactor of `editor_agent.py`, `reframer_agent.py`, and any new agents
- LangGraph's state model is verbose for this pipeline's size (5-8 agent steps)
- LangChain abstraction layer over OpenAI adds a dependency that may drift from OpenAI SDK updates

#### CrewAI

**Strengths:**
- High-level `Crew` / `Task` / `Agent` abstraction maps well conceptually to "worker agents plus orchestrator"

**Weaknesses for this project:**
- Not installed; requires migration of existing agent logic
- Less control over individual LLM call parameters; harder to mix model tiers (mini for workers, larger for verifier)
- CrewAI's sequential/hierarchical process model doesn't add value over manually sequenced coroutines for a 5-step pipeline
- The team has no prior familiarity with CrewAI in this codebase

#### AutoGen

**Weaknesses for this project:**
- Conversational multi-agent model (agents talk to each other) is a poor fit for a deterministic extract→retrieve→draft→verify pipeline
- Higher complexity overhead
- Not installed

### Recommendation: Keep OpenAI Agents SDK, Add Thin Orchestration Layer

**Do not migrate to LangGraph, CrewAI, or AutoGen for this milestone.**

Rationale:
- The pipeline has a fixed, linear topology: extract facts → retrieve candidates → (parallel) draft per page → verify → save. This does not require LangGraph's graph model.
- Migration cost is high and introduces risk on an existing working codebase.
- The OpenAI Agents SDK's `Runner.run_streamed()` is sufficient for progress streaming to the UI.
- LangGraph's checkpointing/resumability advantage is only relevant if the pipeline must survive server restarts. For a 20-minute pipeline with a single-user trigger, optimistic execution with a DB-written progress log is simpler and adequate.

**What to add instead of a framework swap:**

1. A lightweight `PipelineOrchestrator` class (pure Python) that sequences the pipeline stages as coroutines, writes progress events to Supabase as it goes, and exposes an async generator of `PipelineEvent` objects.
2. A FastAPI SSE endpoint (`GET /review/pipeline/{session_id}/stream`) that consumes the generator and forwards events to the UI.
3. The pipeline stages map to:
   - `FactExtractionAgent` — thin wrapper around an `Agent` with transcript as context
   - `CandidateRetrievalAgent` — calls `PineconeStore` + `confluence_page_graph`; pure Python, no LLM needed unless query rewriting is added
   - `SectionDraftAgent` — the existing `ProposedChangesAgent` refactored to draft one page at a time (enables per-page progress events)
   - `VerifierAgent` — new; receives (transcript, proposed change, original content) → returns `{verdict, evidence_snippets, confidence_score, risk_level}`
   - `PipelineOrchestrator` — sequences the above, checkpoints to Supabase, emits events

---

## Async Pipeline Pattern

### The Problem

The current proposal pipeline is a single `asyncio.to_thread()` call wrapping a synchronous OpenAI completion. For a 20-minute pipeline with ~5-10 proposal cards each requiring multiple LLM calls, this approach:
- Provides no progress visibility
- Will be killed by Nginx/load-balancer timeout if the HTTP connection is left open
- Cannot resume on failure

### Recommended Pattern: Fire-and-Forget Job + SSE Progress Stream

This is the correct pattern for long-running tasks in FastAPI. It uses two endpoints:

**Endpoint 1 — Job trigger (POST)**
```
POST /review/pipeline/start
Body: { session_id, query? }
Returns: { job_id } immediately (HTTP 202)
```
The endpoint creates a `job_id` (UUID), writes `status: "queued"` to Supabase `pipeline_jobs` table, and launches the pipeline as a background task via `asyncio.create_task()`. Returns immediately — no waiting.

**Endpoint 2 — Progress stream (GET)**
```
GET /review/pipeline/{job_id}/stream
Returns: text/event-stream (SSE)
```
The UI connects after receiving the `job_id`. The SSE endpoint reads from Supabase (polling a `pipeline_events` table) or from an in-process asyncio `Queue` that the background task writes to.

**Event schema:**
```json
{ "type": "stage_start", "stage": "fact_extraction", "job_id": "...", "ts": "..." }
{ "type": "stage_complete", "stage": "fact_extraction", "result_summary": "8 facts extracted", "ts": "..." }
{ "type": "proposal_ready", "page_title": "...", "change_type": "edit", "card_index": 1, "total": 5, "ts": "..." }
{ "type": "verification_complete", "card_index": 1, "verdict": "approved", "confidence": 0.91, "ts": "..." }
{ "type": "pipeline_complete", "proposal_count": 5, "job_id": "...", "ts": "..." }
{ "type": "pipeline_error", "stage": "...", "error": "...", "ts": "..." }
```

**In-process Queue vs Supabase polling:**
Use an in-process `asyncio.Queue` per `job_id` (stored in a module-level dict). This is simpler than polling Supabase every 2 seconds and has zero latency. The risk (lost events on server restart) is acceptable because:
- The pipeline is triggered by a single user action in the same process
- Proposal cards are written to Supabase by the pipeline as they complete — durable state is preserved even if the SSE stream drops
- The UI can re-fetch proposals via the existing `GET /review/changes` endpoint if the stream disconnects

**Why not WebSocket instead of SSE?**
The existing codebase uses WebSocket for bidirectional TTS/audio. Proposal streaming is unidirectional (server → client). SSE is simpler, browser-native, automatically reconnects, and does not require a persistent full-duplex connection. Use SSE.

**FastAPI SSE implementation:**
```python
from fastapi.responses import StreamingResponse

async def _event_generator(job_id: str):
    queue = _pipeline_queues.get(job_id)
    if not queue:
        yield f"data: {json.dumps({'type': 'error', 'error': 'job not found'})}\n\n"
        return
    while True:
        event = await asyncio.wait_for(queue.get(), timeout=30.0)
        yield f"data: {json.dumps(event)}\n\n"
        if event.get("type") in ("pipeline_complete", "pipeline_error"):
            break

@router.get("/review/pipeline/{job_id}/stream")
async def stream_pipeline(job_id: str):
    return StreamingResponse(
        _event_generator(job_id),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
```

**Timeout handling:**
The 20-minute pipeline budget is handled by setting `timeout=None` (or a large value) on the background task's OpenAI calls. The SSE connection has a 30-second keepalive via the `asyncio.wait_for` timeout — the server sends a heartbeat `data: {"type":"heartbeat"}\n\n` to prevent Nginx from closing the connection.

### Background Task Safety

The pipeline background task runs in the same event loop as FastAPI. This means:
- It must not block the loop — use `asyncio.to_thread()` for all synchronous OpenAI calls (already the pattern in `review/api.py`)
- Module-level singletons (`_openai_client`, Pinecone, Neo4j driver) are fine for concurrent access from one task at a time per job
- `contextvars` (`ContextVar` for `confluence_graph_user_id`) must be set at the start of the background task, not inherited from the HTTP handler — use `copy_context()` if needed

---

## Recommended Stack Adjustments

### Fix Immediately (Blockers)

| Item | Current | Fix | Priority |
|------|---------|-----|----------|
| Model names | `gpt-5-mini` everywhere | Set `JARVIS_AGENT_MODEL=gpt-4.1-mini`, `JARVIS_REVIEW_MODEL=gpt-4.1-mini`. Validate `gpt-5-mini` existence against live API docs before using. | P0 — runtime error |
| `neo4j` version | `neo4j==6.1.0` (does not exist) | Change to `neo4j>=5.14,<6` | P0 — pip install fails |

### Add for This Milestone

| Component | What to Add | Why |
|-----------|------------|-----|
| `FactExtractionAgent` | New class in `confluence_logic/agents/` | Structured extraction: decisions, action items, owners, deadlines, doc-worthy updates. Pydantic output schema. Wraps `Agent` + `response_format`. |
| `VerifierAgent` | New class in `confluence_logic/agents/` | Per-card verification: receives transcript excerpt + proposed change + current section → returns `{verdict, evidence_snippets, confidence_score, risk_level}`. Uses `gpt-4.1` or `gpt-4o` (not mini). |
| `PipelineOrchestrator` | New class in `confluence_logic/pipeline/` | Sequences stages, owns the `asyncio.Queue` for progress events, writes job state to Supabase. |
| `pipeline_jobs` table | New Supabase table | Columns: `job_id`, `session_id`, `status`, `created_at`, `completed_at`, `error`. Durable job state for UI recovery. |
| `pipeline_events` table | Optional Supabase table | Append-only event log. Only needed if multi-server or crash recovery is required; skip for MVP. |
| SSE endpoint | `GET /review/pipeline/{job_id}/stream` | Server-Sent Events stream; in-process `asyncio.Queue` per job. |
| Start endpoint | `POST /review/pipeline/start` | Returns `job_id` immediately; launches background task. |
| Extended `ChangeItem` schema | Add fields to Pydantic + Supabase | Add: `evidence_snippets: list[str]`, `confidence_score: float`, `risk_level: str`, `verifier_verdict: str`. |

### Do Not Change

| Item | Reason |
|------|--------|
| OpenAI Agents SDK | Already used and working. No migration. |
| Pinecone | Working. `text-embedding-3-small` is adequate. |
| Neo4j AuraDB | Working. Used for structural graph traversal alongside Pinecone semantic search. |
| FastAPI + Uvicorn | Solid. SSE and background tasks are native FastAPI patterns. |
| Supabase | Working persistence layer. Extend schema, do not replace. |
| `asyncio.to_thread()` pattern | Already correct in `review/api.py`. Continue using for all synchronous OpenAI calls. |
| `response_format: json_object` | Correct approach for structured agent outputs. Keep. |
| sync-sage-bot as UI | Already decided. Do not reopen. |

---

## Supporting Libraries — No New Dependencies Needed

The existing stack covers all needs for this milestone. Confirm:

| Need | Covered By | Confidence |
|------|-----------|-----------|
| LLM calls + structured output | `openai` (installed) | HIGH |
| Multi-agent orchestration | `openai-agents` (installed) | HIGH |
| Vector search | `pinecone` (installed) | HIGH |
| Graph RAG | `neo4j` (installed, version needs fix) | HIGH |
| HTTP/WebSocket/SSE server | `fastapi` + `uvicorn` (installed) | HIGH |
| Persistence | `supabase` via `requests` (installed) | HIGH |
| Schema validation | `pydantic` (installed) | HIGH |
| Confluence API | `confluence_logic/connectors/confluence.py` (existing) | HIGH |

The only new package that might be added is `anyio` or a timeout/cancellation utility, but `asyncio` stdlib is sufficient.

---

## Confidence Assessment

| Area | Level | Reason |
|------|-------|--------|
| Existing codebase issues (model names, neo4j version) | HIGH | Direct inspection of source files |
| `text-embedding-3-small` adequacy | HIGH | Model is well-established; dimension/API interface verified in codebase |
| OpenAI Agents SDK suitability | HIGH | Already in use; `Runner.run_streamed()` streaming is documented in the SDK |
| SSE + background task pattern | HIGH | Standard FastAPI pattern; `asyncio.Queue` is stdlib |
| Model name `gpt-4.1-mini` existence | MEDIUM | Announced April 2025; in training data as confirmed release but live API validation required |
| `gpt-5-mini` existence | LOW | Announced direction but not confirmed available at training cutoff; treat as unavailable until validated |
| Verifier model quality (gpt-4.1 vs gpt-4o for verification) | MEDIUM | Capability well-understood; exact model naming requires live API validation |
| LangGraph/CrewAI comparison | MEDIUM | Based on framework knowledge through Aug 2025; current feature sets may differ |

---

## Sources

- Direct codebase inspection: `confluence_logic/agents/proposed_changes_agent.py`, `confluence_logic/db/vector_store.py`, `confluence_logic/confluence_page_graph.py`, `confluence_logic/ingestion/doc_pipeline.py`, `confluence_logic/review/api.py`, `confluence_logic/agents/editor_agent.py`, `requirements.txt`
- OpenAI model knowledge: training data through August 2025 (gpt-4.1, gpt-4.1-mini confirmed April 2025 release)
- OpenAI Agents SDK streaming: training data through August 2025 (Runner.run_streamed API)
- FastAPI SSE pattern: training data through August 2025 (StreamingResponse + text/event-stream)
- NOTE: WebSearch and WebFetch were unavailable in this session. All external claims should be validated against https://platform.openai.com/docs/models and https://openai.github.io/openai-agents-python/ before implementation.
