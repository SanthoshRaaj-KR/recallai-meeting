# Phase 7: Confluence Document Q&A Agent - Research

**Researched:** 2026-05-16
**Domain:** OpenAI Agents SDK, Pinecone RAG retrieval, Neo4j confluence page graph, FastAPI async orchestration
**Confidence:** HIGH — all findings verified directly from the codebase

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

- **D-01:** Pinecone vector search is the primary retrieval path — fast semantic lookup (~100-200ms) for factual questions
- **D-02:** Neo4j `confluence_page_graph` is the secondary path — used only for relationship/structural queries or when Pinecone returns zero results
- **D-03:** If both graph sources return no relevant context, fall back to live Confluence REST API search — never say "I don't know" when the page exists
- **D-04:** Answer quality takes priority over raw speed. Deeper retrieval at cost of ~1 extra second is acceptable.
- **D-05:** Build a `ConfluenceQAAgent` class following the `editor_agent.py` pattern — OpenAI Agents SDK `Agent` object with `@function_tool` decorated tools
- **D-06:** Agent tools to expose: `search_confluence_pages(query: str)`, `get_full_page_content(page_id: str)`, `list_confluence_pages(limit: int)`
- **D-07:** The agent replaces `_answer_confluence_question()` in `jarvis_agentic.py` — the handle function becomes a thin wrapper that instantiates and runs the agent
- **D-08:** `_is_confluence_read_query()` regex gate stays in place — not changed, not moved
- **D-09:** Use `gpt-5-mini` for the agent's tool-use step (tool selection, multi-step orchestration)
- **D-10:** Use `gpt-4o-mini` for the final answer synthesis step — the LLM call that generates the spoken answer from retrieved context
- **D-11:** If the agent resolves the answer in a single tool call, it can synthesize with `gpt-4o-mini` directly. Multi-hop stays on `gpt-5-mini` throughout.
- **D-12:** The upstream `classifier.py` "confluence" intent + `_is_confluence_read_query()` gate are sufficient for routing — no changes to classification

### Claude's Discretion

- Token budget for the final answer synthesis call — keep concise for spoken delivery (~250-350 tokens max)
- Exact Pinecone `top_k` parameter — start with 8 chunks (up from current 6) given accuracy-first priority
- Whether to run Pinecone + Neo4j in parallel or strictly sequential — researcher should assess which is faster given actual latency characteristics

### Deferred Ideas (OUT OF SCOPE)

- Streaming answer token-by-token to TTS
- Multi-page aggregation answers ("across all docs, what's the status of...")
- Updating the classifier to use LLM-based intent detection for Confluence vs general
</user_constraints>

---

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| QA-01 | A factual Confluence question answered by the agent returns a correct spoken response within 3 seconds (including retrieval + synthesis); answer matches the relevant page/section content | Pinecone search is sync and ~100-200ms; Neo4j confluence graph query is async ~200ms; gpt-4o-mini synthesis is ~500ms; total under 3s with parallel gap-filler. Current code already hits this for simpler paths. |
| QA-02 | When the Pinecone index has no relevant chunks, the agent falls back to a live Confluence REST search and still returns an answer rather than "I don't know" | `ConfluenceConnector.search_pages()` already exists with two CQL queries (title and text match). This is the fallback tool: `get_full_page_content(page_id)` after a REST search surfaces a page_id. |
| QA-03 | The agent uses `gpt-5-mini` for tool orchestration and `gpt-4o-mini` for final answer synthesis; tool-use step never uses `gpt-4o-mini` | Two-step design: Agent with `model="gpt-5-mini"` runs tools; result is post-processed with a second `gpt-4o-mini` chat call in `_handle_confluence_question()` before speaking. |
| QA-04 | Edit/mutation queries ("update the SOC2 page", "add a section") are NOT routed to the Q&A agent — `_is_confluence_read_query()` gate holds | Gate already exists at `jarvis_agentic.py:406`. Tests already cover this (test_confluence_read_query_detection_excludes_mutations). No changes to the gate. |
</phase_requirements>

---

## Summary

Phase 7 replaces the existing `_answer_confluence_question()` function in `jarvis_agentic.py` (lines 455-507) with a proper `ConfluenceQAAgent` class following the `editor_agent.py` pattern. The current function already has the right retrieval call sequence — it calls the Neo4j confluence page graph and then synthesizes with `gpt-4o-mini` — but it misses Pinecone as primary retrieval and has no fallback to live REST. The new agent restructures these as proper tool calls, adds Pinecone-first retrieval, adds a REST fallback tool, and wires them into the OpenAI Agents SDK Agent/Runner pattern.

The codebase already contains every building block: `PineconeStore.search()`, `query_user_confluence_graph()`, `list_user_confluence_pages()`, and `ConfluenceConnector.search_pages()` / `fetch_page_html()`. The new agent wraps these in `@function_tool` decorated functions and wires them into an `Agent` object with `gpt-5-mini`. Final answer synthesis stays as a separate `gpt-4o-mini` chat completion call — called in `_handle_confluence_question()` after the agent returns raw context — matching D-10/D-11.

The integration point is minimal: `_handle_confluence_question()` at line 510 retains the same signature and the same gap-filler/speak pattern, but the inner `_answer_confluence_question()` call is replaced by `ConfluenceQAAgent.run(query, graph_user_id)`. This agent should live in a new file: `confluence_logic/agents/confluence_qa_agent.py`.

**Primary recommendation:** Build `ConfluenceQAAgent` as a thin wrapper class with three `@function_tool` functions defined at module level (not as class methods — see tools.py pattern). The class exposes a single `async def run(query, graph_user_id)` method that (1) pre-warms the Neo4j graph, (2) runs the Agent via `Runner.run()`, (3) takes `result.final_output` as raw context, (4) makes a second `gpt-4o-mini` synthesis call, and (5) returns the spoken answer string.

---

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Read query routing (is this a read vs mutation?) | Orchestration (`jarvis_agentic.py`) | — | Gate lives in orchestrator; no changes needed |
| Pinecone semantic retrieval | RAG (`db/vector_store.py`) | — | PineconeStore.search() is the canonical interface |
| Neo4j page graph retrieval | RAG (`confluence_page_graph.py`) | — | query_user_confluence_graph() is already scoped by user_id |
| Live REST fallback search | Connector (`connectors/confluence.py`) | — | search_pages() + fetch_page_html() already exists |
| Agent tool orchestration | Agents (`agents/confluence_qa_agent.py`) | — | New file, follows editor_agent.py pattern |
| Final answer synthesis (spoken) | Agents (inside run()) | Orchestration | gpt-4o-mini call after agent completes |
| TTS and gap-filler delivery | Orchestration (`jarvis_agentic.py`) | — | _speak_guarded / _speak_gap_filler unchanged |

---

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| `agents` (openai-agents) | latest (project) | `Agent`, `Runner`, `function_tool` | Project-standard for all agent classes |
| `openai` | latest (project) | gpt-4o-mini synthesis call | All LLM calls use this client |
| `pinecone` | latest (project) | Primary vector retrieval via PineconeStore | Already used in tools.py search_workspace_knowledge |
| `neo4j` | >=5.14,<6 (requirements.txt) | Secondary graph retrieval via confluence_page_graph | Already used across the project |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| `asyncio` | stdlib | `wait_for()` timeouts, `to_thread()` for sync calls | All async safety patterns |
| `logging` | stdlib | `logging.getLogger(__name__)` per module | All modules use this pattern |
| `confluence_page_graph` | internal | query_user_confluence_graph, list_user_confluence_pages, ensure_user_confluence_graph | Accessed via module import in new agent file |

**Installation:** No new packages needed — all dependencies already in `requirements.txt`.

---

## Architecture Patterns

### System Architecture Diagram

```
Voice query "hey Jarvis, when is SOC2 coming?"
        |
        v
jarvis_agentic.py
  classifier -> "confluence" intent
  _is_confluence_read_query() -> True (read gate passes)
        |
        v
_handle_confluence_question(query, bot_id)
  |-- asyncio.create_task(_speak_gap_filler(...))   [runs concurrently]
  |-- await ConfluenceQAAgent().run(query, graph_user_id)
        |
        v
ConfluenceQAAgent.run()
  1. ensure_user_confluence_graph(graph_user_id)  [pre-warm, 0.7s timeout]
  2. Runner.run(agent, query)                      [gpt-5-mini + tools]
        |
        +-- tool: search_confluence_pages(query)
        |     |-- PineconeStore.search(query, top_k=8)  [primary]
        |     |-- if empty: confluence_page_graph.query_user_confluence_graph()  [secondary]
        |     |-- if still empty: ConfluenceConnector.search_pages(query)  [fallback]
        |
        +-- tool: get_full_page_content(page_id)
        |     |-- ConfluenceConnector.fetch_page_html(page_id)  [on-demand section fetch]
        |
        +-- tool: list_confluence_pages(limit)
              |-- confluence_page_graph.list_user_confluence_pages()
  3. gpt-4o-mini synthesis call (max_tokens=300)  [converts raw context to spoken answer]
  4. return answer string
        |
        v
_handle_confluence_question (continued)
  await gap_filler_task
  await _speak_guarded(answer, bot_id, generation, allow_stale=True)
  meeting_state["last_jarvis_response"] = {...}
```

### Recommended Project Structure
```
confluence_logic/
├── agents/
│   ├── editor_agent.py          # existing — pattern to follow
│   ├── reframer_agent.py        # existing
│   ├── tools.py                 # existing — @function_tool pattern
│   └── confluence_qa_agent.py   # NEW — ConfluenceQAAgent
```

### Pattern 1: Agent Class with Module-Level Function Tools
**What:** `@function_tool` decorated functions are defined at module level (not as class methods). The class imports and wires them into its `Agent` object in `__init__`.
**When to use:** Always — this is how `editor_agent.py` and `tools.py` are structured.
**Example (from verified codebase — `confluence_logic/agents/editor_agent.py:17` and `tools.py:214`):**
```python
# module-level tools
@function_tool
def search_confluence_pages(query: str) -> str:
    """Searches Confluence for pages relevant to a question."""
    ...

# class wires tools into Agent
class ConfluenceQAAgent:
    def __init__(self, model: str = "gpt-5-mini"):
        self.model = model
        self.agent = Agent(
            name="Jarvis Confluence QA",
            model=model,
            instructions=(...),
            tools=[search_confluence_pages, get_full_page_content, list_confluence_pages],
        )
```
[VERIFIED: confluence_logic/agents/editor_agent.py, confluence_logic/agents/tools.py]

### Pattern 2: Runner.run() — Always Async, Result Has .final_output
**What:** `Runner.run(agent, input_string)` is the standard invocation. Returns an object; access answer via `result.final_output`.
**When to use:** All agent invocations in this project.
**Example (from `editor_agent.py:240`, `reframer_agent.py:40`):**
```python
result = await Runner.run(self.agent, query)
if hasattr(result, 'final_output'):
    answer = result.final_output.strip()
else:
    answer = str(result)
```
[VERIFIED: confluence_logic/agents/editor_agent.py:240, reframer_agent.py:40]

### Pattern 3: Lazy Singleton — Tools Use Module-Level Singletons
**What:** `_store = None; def get_store(): global _store; if _store is None: _store = PineconeStore(); return _store`
**When to use:** Wrapping PineconeStore and ConfluenceConnector inside tool functions so credentials are loaded lazily, not at import time.
**Example (from `tools.py:31-39`):**
```python
_store = None
_connector = None

def get_store():
    global _store
    if _store is None:
        _store = PineconeStore()
    return _store
```
[VERIFIED: confluence_logic/agents/tools.py:31-39]

### Pattern 4: asyncio.to_thread() for Synchronous Calls Inside Async Functions
**What:** Pinecone's `store.search()` is synchronous. From within an async function, wrap it with `asyncio.to_thread()` to avoid blocking the event loop.
**When to use:** Any sync call inside an async context — specifically `PineconeStore.search()` and `ConfluenceConnector` methods.
**Example (from `tools.py` and `_answer_confluence_question()`):**
```python
# PineconeStore.search() is sync — run it in a thread
results = await asyncio.to_thread(store.search, query, 8)
```
[VERIFIED: confluence_logic/jarvis_agentic.py:490-506 uses asyncio.to_thread for openai; tools.py uses concurrent.futures.ThreadPoolExecutor]

**Important:** The `@function_tool` functions called by the Agent SDK are invoked synchronously by the SDK runner, so sync calls inside them do NOT need `asyncio.to_thread()`. Only async methods called inside `@function_tool` bodies need `_run_async_blocking()` (see tools.py:141-160 for the established pattern).

### Pattern 5: _run_async_blocking() for Async Calls Inside @function_tool Bodies
**What:** `@function_tool` decorated functions must be synchronous (the SDK calls them in a sync context). To call async graph methods (like `query_user_confluence_graph`), use the existing `_run_async_blocking()` helper already in `tools.py`.
**When to use:** Calling `confluence_page_graph.query_user_confluence_graph()` or `confluence_page_graph.list_user_confluence_pages()` from inside a `@function_tool` body.
**Example (from `tools.py:141-186`):**
```python
def _run_async_blocking(coro):
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    result = {}
    def _runner():
        try:
            result["value"] = asyncio.run(coro)
        except Exception as exc:
            result["error"] = exc
    thread = threading.Thread(target=_runner)
    thread.start()
    thread.join()
    if "error" in result:
        raise result["error"]
    return result.get("value")

# Usage inside @function_tool:
matches = _run_async_blocking(
    confluence_page_graph.query_user_confluence_graph(user_id, query, limit=8)
)
```
[VERIFIED: confluence_logic/agents/tools.py:141-186]

### Pattern 6: Two-Model Split — gpt-5-mini for Tools, gpt-4o-mini for Synthesis
**What:** The Agent object uses `gpt-5-mini` for tool orchestration. After `Runner.run()` returns, the calling code makes a second direct `gpt-4o-mini` chat completion to produce the spoken answer from the agent's retrieved context.
**When to use:** This phase specifically — QA-03 requires tool-use to stay on gpt-5-mini, synthesis on gpt-4o-mini.
**Implementation:** `result.final_output` from the gpt-5-mini agent contains the raw retrieved context; the `run()` method then calls `get_openai_client().chat.completions.create(model="gpt-4o-mini", ...)` to produce the final spoken-quality answer string. This matches the existing `_answer_confluence_question()` pattern (line 491).
[VERIFIED: confluence_logic/jarvis_agentic.py:491, CONTEXT.md D-09/D-10]

### Pattern 7: asyncio.wait_for() for Graph Pre-Warm
**What:** `ensure_user_confluence_graph()` can take a few hundred milliseconds. The existing code uses `asyncio.wait_for(..., timeout=0.7)` and catches the timeout, firing the build as a background task.
**When to use:** At the start of `ConfluenceQAAgent.run()` — retain the pre-warm call and its timeout pattern exactly.
**Example (from `jarvis_agentic.py:458-460`):**
```python
try:
    await asyncio.wait_for(
        confluence_page_graph.ensure_user_confluence_graph(graph_user_id), timeout=0.7
    )
except (asyncio.TimeoutError, Exception):
    asyncio.create_task(confluence_page_graph.ensure_user_confluence_graph(graph_user_id))
```
[VERIFIED: confluence_logic/jarvis_agentic.py:458-460]

### Anti-Patterns to Avoid
- **Defining tools as instance methods with `self`:** The `@function_tool` decorator does not support `self` as a parameter. Tools must be module-level functions. See `tools.py` — all tools are plain functions, not methods.
- **Instantiating PineconeStore or ConfluenceConnector at module import time:** Anti-pattern documented in ARCHITECTURE.md. Use lazy singletons (get_store/get_connector pattern) so tests can patch without triggering real credential loads.
- **Calling async methods directly inside `@function_tool`:** Leads to "coroutine never awaited" errors. Use `_run_async_blocking()` for async calls inside sync tool bodies.
- **Threading the graph_user_id through tool parameters:** The `@function_tool` is a standalone function called by the SDK; it cannot receive runtime state via the Agent. Instead, use the `ContextVar("confluence_graph_user_id")` that is already set in `jarvis_agentic.py` via `confluence_page_graph.get_current_graph_user_id()`. The `search_confluence_pages` tool body should call `confluence_page_graph.get_current_graph_user_id()` directly (same pattern as `_graph_candidates()` in tools.py:164).
- **Returning "I don't know" when both Pinecone and Neo4j return empty:** Decision D-03 locks in a live REST fallback. The `search_confluence_pages` tool must attempt the REST fallback before returning empty.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Pinecone vector search | Custom embedding + similarity code | `PineconeStore.search(query, top_k=8)` | Already handles embedding generation, index query, and metadata extraction |
| Async-to-sync bridge for graph calls | Ad-hoc `loop.run_until_complete()` | `_run_async_blocking()` from `tools.py` | Already handles the running-loop detection edge case that causes `RuntimeError` |
| Confluence CQL search | Raw requests.get calls | `ConfluenceConnector.search_pages(query)` | Already runs title + text CQL queries, deduplicates, returns structured dicts |
| Page HTML fetch | Raw requests.get calls | `ConfluenceConnector.fetch_page_html(page_id)` | Already handles authentication, storage format extraction |
| Lazy singleton pattern | Class-level `__init__` clients | `get_store()` / `get_connector()` module-level functions | Already implemented in tools.py — reuse the same pattern |

**Key insight:** The entire retrieval infrastructure (Pinecone, Neo4j, REST) is already built. This phase is purely about wiring them into the Agent SDK tool framework.

---

## Common Pitfalls

### Pitfall 1: @function_tool with Async Context Mismatch
**What goes wrong:** A `@function_tool` decorated function calls `await` directly, which crashes because the SDK calls tools synchronously.
**Why it happens:** The Agent SDK invokes tool functions in a sync context via its internal task runner.
**How to avoid:** Use `_run_async_blocking()` (copy from tools.py) for any async calls inside tool bodies. Pinecone's `store.search()` is already sync, so no wrapping needed there.
**Warning signs:** `RuntimeError: coroutine was never awaited` or `RuntimeError: This event loop is already running` in tool invocations.

### Pitfall 2: graph_user_id Threading Problem
**What goes wrong:** The `graph_user_id` string needs to reach the `search_confluence_pages` tool body, but tool functions have no access to instance state.
**Why it happens:** `@function_tool` functions are standalone module-level callables — the Agent SDK calls them without any class context.
**How to avoid:** Use the existing `ContextVar("confluence_graph_user_id")` — `confluence_page_graph.get_current_graph_user_id()` retrieves the current value set by `_handle_confluence_question()` before the agent runs. This is the same mechanism used by `_graph_candidates()` in tools.py:164.
**Warning signs:** Tool always returns empty results because user_id is empty string.

### Pitfall 3: Model Split Implementation — final_output Misuse
**What goes wrong:** Using `result.final_output` directly as the spoken answer, bypassing the gpt-4o-mini synthesis step, so QA-03 is violated.
**Why it happens:** `Runner.run()` returns whatever the gpt-5-mini agent chose to output — raw JSON context, partial reasoning, or a structured dump. This is not spoken-quality text.
**How to avoid:** `result.final_output` from the agent is treated as the retrieved context. Pass it to a second `gpt-4o-mini` chat completion (the synthesis step). Only the synthesis output is returned as the spoken answer.
**Warning signs:** Spoken response contains JSON structures, tool call descriptions, or page IDs.

### Pitfall 4: Pinecone Search Returns Matches Without Checking Score
**What goes wrong:** Pinecone returns matches even for semantically unrelated queries (just lower-scored), so the fallback path never fires when it should.
**Why it happens:** `store.search()` always returns `top_k` matches unless the index is empty.
**How to avoid:** Apply a minimum score threshold in the `search_confluence_pages` tool. Pinecone match objects have a `score` field. Matches below ~0.3 (cosine similarity) should be treated as "no relevant results" and trigger the fallback. [ASSUMED — threshold needs tuning in practice]
**Warning signs:** Agent answers a question about SOC2 with content from an unrelated page.

### Pitfall 5: Timeout on Gap-Filler/Answer Race
**What goes wrong:** The agent takes >3s, the gap filler already finished, and the answer arrives after the generation has already been bumped.
**Why it happens:** `_speak_guarded` checks `generation != meeting_state["output_generation"]` before speaking. If a new query arrived while the agent was running, the old generation is stale.
**How to avoid:** Use `allow_stale=True` in `_speak_guarded` for the answer — exactly as the existing `_handle_confluence_question()` does at line 517. Do not change this. [VERIFIED: jarvis_agentic.py:517]

### Pitfall 6: Circular Import Risk
**What goes wrong:** `confluence_qa_agent.py` imports from `tools.py`, which imports from `confluence_page_graph`, which imports from `graph_rag`. If `jarvis_agentic.py` then imports `confluence_qa_agent.py` at the top level, this can create import-time side effects (client initialization).
**Why it happens:** Module-level code in tools.py and graph_rag.py initializes logging and sets up module-level state.
**How to avoid:** Import `ConfluenceQAAgent` inside `_handle_confluence_question()` using a deferred import, or define a module-level lazy singleton `_qa_agent = None` in `jarvis_agentic.py` with a getter. The CONTEXT.md recommends the lazy singleton pattern.

---

## Code Examples

Verified patterns from existing codebase:

### PineconeStore.search() Call
```python
# Source: confluence_logic/db/vector_store.py:93-110
# Synchronous — safe to call directly inside @function_tool body
store = get_store()  # lazy singleton
results = store.search(query, top_k=8)
# results is List[Dict] with keys: 'metadata' (has page_id, title, heading, text_summary, space_key)
# Example: results[0]["metadata"]["page_id"], results[0]["metadata"]["heading"]
```
[VERIFIED: confluence_logic/db/vector_store.py:93-110]

### query_user_confluence_graph() Return Format
```python
# Source: confluence_logic/confluence_page_graph.py:329-378
# Async — must use _run_async_blocking() inside @function_tool body
results = _run_async_blocking(
    confluence_page_graph.query_user_confluence_graph(user_id, query, limit=8)
)
# results is List[Dict] with keys:
#   page_id, title, space_key, version, heading, relevant_content, score, source
```
[VERIFIED: confluence_logic/confluence_page_graph.py:360-375]

### list_user_confluence_pages() Return Format
```python
# Source: confluence_logic/confluence_page_graph.py:382-416
# Async — must use _run_async_blocking()
pages = _run_async_blocking(
    confluence_page_graph.list_user_confluence_pages(user_id, limit=10)
)
# pages is List[Dict] with keys: page_id, title, space_key, version
```
[VERIFIED: confluence_logic/confluence_page_graph.py:403-411]

### ConfluenceConnector.search_pages() Return Format
```python
# Source: confluence_logic/connectors/confluence.py:37-77
# Synchronous
connector = get_connector()
results = connector.search_pages(query, limit=5)
# results is List[Dict] with keys: page_id, title, space_key, version, excerpt
# Note: does NOT include section content — only excerpt
# For section content, call fetch_page_html(page_id) separately
```
[VERIFIED: confluence_logic/connectors/confluence.py:37-77]

### Full _handle_confluence_question() Integration Point
```python
# Source: confluence_logic/jarvis_agentic.py:510-522
# This function is what gets modified — inner call changes, outer structure stays
async def _handle_confluence_question(query: str, bot_id: str) -> None:
    generation = meeting_state["output_generation"]
    try:
        answer_task = asyncio.create_task(_answer_confluence_question(query))  # <- replace this
        gap_filler_task = asyncio.create_task(_speak_gap_filler(query, bot_id, generation))
        answer = await answer_task
        await gap_filler_task
        await _speak_guarded(answer, bot_id, generation, allow_stale=True)
        if answer:
            meeting_state["last_jarvis_response"] = {"intent": "confluence_question", "query": query, "answer": answer}
    except Exception as exc:
        logger.error("Confluence question handling failed: %s", exc)
        await _speak_guarded("I hit an issue reading the Confluence graph.", bot_id, generation, allow_stale=True)
```
[VERIFIED: confluence_logic/jarvis_agentic.py:510-522]

### Existing @function_tool Pattern with Pydantic Return
```python
# Source: confluence_logic/agents/tools.py:214-223
@function_tool
def list_workspace_pages(limit: int = 100) -> SearchResponse:
    """Lists recent Confluence pages..."""
    try:
        items = get_connector().list_pages(limit=limit)
        candidates = [_candidate_from_metadata(item) for item in items if item.get("page_id")]
        return SearchResponse(candidates=candidates, message="Success")
    except Exception as e:
        logger.error(f"List workspace pages failed: {e}")
        return SearchResponse(candidates=[], message=f"Error: {e}")
```
[VERIFIED: confluence_logic/agents/tools.py:214-223]

**Note on tool return types for QA agent:** The `search_confluence_pages` tool for the QA agent can return a plain `str` (JSON-formatted context) rather than a Pydantic model. The Agent SDK accepts `str` return from tools — the gpt-5-mini model will reason over the string. Using `str` avoids defining new Pydantic models when the return is only meant for LLM consumption, not structured parsing. [ASSUMED — str returns are standard in the SDK; Pydantic returns are optional]

---

## Parallel vs Sequential Retrieval Assessment (Discretion Item)

The CONTEXT.md left this to the researcher. Based on the actual code:

- **Pinecone `store.search()`** is synchronous and calls the OpenAI embedding API + Pinecone index. Estimated latency: 150-300ms (embedding ~100ms + Pinecone query ~100ms). [ASSUMED based on embedding model characteristics]
- **Neo4j `query_user_confluence_graph()`** is async and runs a Cypher query. Estimated latency: 100-300ms when the driver is connected. [ASSUMED based on Neo4j AuraDB typical latency]

**Recommendation:** Run Pinecone first. If Pinecone returns results above a relevance threshold, skip Neo4j (sequential). This avoids the complexity of running both in parallel inside a `@function_tool` body (which would require nested thread pools). The `search_confluence_pages` tool should implement the waterfall: Pinecone → Neo4j → REST. This is architecturally simpler and matches D-01/D-02 (primary/secondary distinction).

If parallel retrieval is desired, it should be implemented at the `_handle_confluence_question()` level before the agent runs — pre-fetching context and injecting it into the agent prompt — rather than inside the tool body.

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| `_answer_confluence_question()` plain function | `ConfluenceQAAgent` class with OpenAI Agents SDK tools | This phase | Multi-tool orchestration, proper fallback chain, model split |
| Neo4j-only retrieval (current function) | Pinecone-first + Neo4j secondary + REST fallback | This phase | Better semantic recall; correct fallback when index is sparse |
| Single `gpt-4o-mini` call for both retrieval routing and synthesis | gpt-5-mini for tool orchestration, gpt-4o-mini for synthesis | This phase | Matches model split requirement; gpt-4o-mini is cheaper and sufficient for final synthesis |

**Deprecated in this phase:**
- `_answer_confluence_question()` in `jarvis_agentic.py`: replaced entirely by `ConfluenceQAAgent.run()`. The function body is deleted; the call site in `_handle_confluence_question()` is updated.

---

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | Pinecone match score threshold of ~0.3 distinguishes relevant from irrelevant results | Common Pitfalls #4 | Threshold may need tuning; if too high, REST fallback fires unnecessarily; if too low, irrelevant context pollutes the synthesis |
| A2 | Pinecone embedding call takes ~100ms, Pinecone index query ~100ms (total ~150-300ms) | Parallel vs Sequential assessment | If embedding is slower, total latency for QA-01 may exceed 3s; should be validated with a real query |
| A3 | `str` return type is acceptable from `@function_tool` (not required to be Pydantic) | Code Examples note | If SDK requires Pydantic, a new QAResult schema would be needed; low risk based on SDK design |
| A4 | gpt-5-mini model ID is valid for `agents.Agent(model=...)` in the current project environment | Standard Stack | CLAUDE.md notes JARVIS_AGENT_MODEL defaults to "gpt-5-mini" which it flags as an invalid model name; project accepts this as a convention |

---

## Open Questions

1. **gpt-5-mini model validity in this environment**
   - What we know: CLAUDE.md flags `JARVIS_AGENT_MODEL = "gpt-5-mini"` as potentially invalid. The existing `EditorAgent` uses it.
   - What's unclear: Whether the current deployment actually has gpt-5-mini access or falls back to a mapped model.
   - Recommendation: Follow the existing pattern (use `"gpt-5-mini"` as the model string). If it's a known invalid model, the EditorAgent would already be broken.

2. **Whether `result.final_output` from the QA agent will be a string or structured output**
   - What we know: When Agent has no `output_type` set, `final_output` is a string. When `output_type` is a Pydantic model, it's that model.
   - What's unclear: Whether the agent should use `output_type` to return structured context (page_id, heading, content) for the synthesis step.
   - Recommendation: Start without `output_type` (plain string). The agent instructions should tell it to return the retrieved text context as its final output. The synthesis call then uses that string. This is simpler and avoids needing a new Pydantic schema.

3. **Whether `_run_async_blocking()` should be copied or imported from tools.py**
   - What we know: `_run_async_blocking()` is a private function in `tools.py`.
   - What's unclear: Whether to copy it to `confluence_qa_agent.py` or make it a shared utility.
   - Recommendation: Import from tools.py if the @function_tool functions for the QA agent are defined in `tools.py` (co-located with existing tools). If the new tools are defined in `confluence_qa_agent.py`, copy the helper with a comment referencing the source.

---

## Environment Availability

Step 2.6: SKIPPED (no new external dependencies — all required services are already in use by the existing codebase: Pinecone, Neo4j AuraDB, Confluence REST, OpenAI API).

---

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest + pytest-asyncio |
| Config file | none detected (no pytest.ini, pyproject.toml, setup.cfg) |
| Quick run command | `conda run -n ml pytest confluence_logic/tests/test_confluence_qa_agent.py -x -q` |
| Full suite command | `conda run -n ml pytest confluence_logic/tests/ -x -q` |

### Phase Requirements → Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| QA-01 | Factual question returns correct spoken answer matching page content | unit (mock Pinecone + graph) | `pytest confluence_logic/tests/test_confluence_qa_agent.py::test_qa_returns_answer_from_pinecone -x` | ❌ Wave 0 |
| QA-02 | Pinecone empty → REST fallback still returns answer | unit (mock Pinecone empty, mock REST) | `pytest confluence_logic/tests/test_confluence_qa_agent.py::test_qa_fallback_to_rest_when_pinecone_empty -x` | ❌ Wave 0 |
| QA-03 | Tool orchestration uses gpt-5-mini; synthesis uses gpt-4o-mini | unit (mock Runner + openai client) | `pytest confluence_logic/tests/test_confluence_qa_agent.py::test_qa_model_split_tools_gpt5mini_synthesis_gpt4omini -x` | ❌ Wave 0 |
| QA-04 | Mutation queries are not routed to QA agent | unit (existing test coverage) | `pytest confluence_logic/tests/test_jarvis_agentic.py::test_confluence_read_query_detection_excludes_mutations -x` | ✅ exists |

### Sampling Rate
- **Per task commit:** `conda run -n ml pytest confluence_logic/tests/test_confluence_qa_agent.py -x -q`
- **Per wave merge:** `conda run -n ml pytest confluence_logic/tests/ -x -q`
- **Phase gate:** Full suite green before `/gsd-verify-work`

### Wave 0 Gaps
- [ ] `confluence_logic/tests/test_confluence_qa_agent.py` — covers QA-01, QA-02, QA-03

*(QA-04 is already covered by `test_jarvis_agentic.py::test_confluence_read_query_detection_excludes_mutations`)*

---

## Security Domain

Security enforcement is enabled (not explicitly disabled in config.json).

### Applicable ASVS Categories

| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | no | No new auth surface — reads Confluence via existing credentials |
| V3 Session Management | no | user_id scoping uses existing ContextVar mechanism |
| V4 Access Control | yes | `get_current_graph_user_id()` scopes all Neo4j queries to the calling user — preserve this |
| V5 Input Validation | yes | Query string passed from voice transcript — already sanitized before reaching this layer |
| V6 Cryptography | no | No new crypto; credentials via env vars as established |

### Known Threat Patterns for This Stack

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| Cross-user graph data leak | Information Disclosure | `graph_user_id` scoped via ContextVar; Neo4j queries always include `user_id` parameter — do NOT remove this parameter from tool queries |
| CQL injection via voice query | Tampering | `ConfluenceConnector.search_pages()` already escapes double quotes in the CQL query (line 40: `safe_query = (query or "").replace('"', '\\"')`); the new tool reuses this method |
| Prompt injection via Confluence page content | Tampering | Confluence page content is presented as user-role context to the synthesis LLM, not as system instructions — acceptable risk within the existing design |

---

## Sources

### Primary (HIGH confidence)
- `confluence_logic/agents/editor_agent.py` — Agent class pattern, Runner.run() usage, tool wiring
- `confluence_logic/agents/tools.py` — @function_tool pattern, lazy singleton, _run_async_blocking(), _graph_candidates()
- `confluence_logic/jarvis_agentic.py:406-522` — _is_confluence_read_query(), _answer_confluence_question(), _handle_confluence_question(), _speak_gap_filler(), _speak_guarded()
- `confluence_logic/db/vector_store.py` — PineconeStore.search() signature and return format
- `confluence_logic/confluence_page_graph.py` — query_user_confluence_graph(), list_user_confluence_pages(), ensure_user_confluence_graph() signatures and return formats
- `confluence_logic/connectors/confluence.py` — search_pages(), fetch_page_html() signatures
- `confluence_logic/core/schemas.py` — existing Pydantic models
- `confluence_logic/agents/reframer_agent.py` — minimal agent class pattern (Runner.run + model_validate)
- `confluence_logic/tests/test_jarvis_agentic.py` — QA-04 coverage already present
- `.planning/phases/07-confluence-document-q-a-agent/07-CONTEXT.md` — all locked decisions

### Secondary (MEDIUM confidence)
- ARCHITECTURE.md — module structure, anti-patterns
- CLAUDE.md — project conventions, model naming

### Tertiary (LOW confidence)
- None — all claims in this research are verified from the codebase directly.

---

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — all dependencies already in use, no new packages
- Architecture: HIGH — verified by reading all relevant source files
- Pitfalls: HIGH for verified patterns; MEDIUM for score threshold (A1) which requires runtime validation
- Test patterns: HIGH — follows exact same structure as test_flow.py and test_apply_hardening.py

**Research date:** 2026-05-16
**Valid until:** 2026-07-16 (stable codebase, no external library churn expected)
