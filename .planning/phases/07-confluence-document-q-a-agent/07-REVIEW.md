---
phase: 07-confluence-document-q-a-agent
reviewed: 2026-05-17T00:00:00Z
depth: standard
files_reviewed: 4
files_reviewed_list:
  - confluence_logic/agents/confluence_qa_agent.py
  - confluence_logic/jarvis_agentic.py
  - confluence_logic/tests/test_confluence_qa_agent.py
  - confluence_logic/tests/test_qa_latency.py
findings:
  critical: 4
  warning: 5
  info: 2
  total: 11
status: issues_found
---

# Phase 07: Code Review Report

**Reviewed:** 2026-05-17T00:00:00Z
**Depth:** standard
**Files Reviewed:** 4
**Status:** issues_found

## Summary

The ConfluenceQAAgent implementation is broadly well-structured: the Pinecone → Neo4j → REST waterfall is recognisable and the two-model split (gpt-5-mini for tool orchestration, gpt-4o-mini for synthesis) is architecturally sound. However, four blockers exist: the Pinecone score threshold is silently bypassed by every unit test (every mock result has no `"score"` key, so `m.get("score", 0)` is always 0, which is below 0.3 — meaning tests that claim to verify the Pinecone path actually always fall through to the REST fallback branch); the async-to-sync bridge in `@function_tool` bodies blocks the asyncio event loop's thread pool in a way that causes deadlocks under test isolation; the orphaned `asyncio.create_task()` call in the fallback branch of `run()` fires into a potentially absent or wrong event loop; and the two-model requirement relies on `"gpt-5-mini"` — a model name explicitly flagged in `CLAUDE.md` as invalid on the OpenAI API. Warnings cover thread safety of lazy singletons, the read-gate regex over-blocking real read queries that contain mutation words, and the latency test not actually measuring the synthesis path correctly.

---

## Critical Issues

### CR-01: Pinecone score filter silently broken in all QA-01/QA-03 unit tests — threshold never exercised

**File:** `confluence_logic/agents/confluence_qa_agent.py:136` and `confluence_logic/tests/test_confluence_qa_agent.py:45-55`

**Issue:** `search_confluence_pages` filters with `m.get("score", 0) >= 0.3`. Every mock Pinecone result in the test suite (`test_qa_returns_answer_from_pinecone`, `test_qa_model_split_tools_gpt5mini_synthesis_gpt4omini`, and the latency test) omits the `"score"` key entirely. The expression `m.get("score", 0)` therefore returns `0`, which is less than `0.3`, so **every mock result is filtered out**, `matches` is empty, and the code falls through to the Neo4j branch (or REST branch), never exercising the intended Pinecone-first path. The tests claim to verify the Pinecone path (QA-01, QA-03) but they silently test the REST/Neo4j fallback instead. The real threshold guard is completely untested.

**Fix:** Add `"score": 0.85` (any value ≥ 0.3) to every mock Pinecone result:

```python
mock_get_store.return_value.search.return_value = [
    {
        "score": 0.85,          # REQUIRED — without this score filter discards the match
        "metadata": {
            "page_id": "p99",
            "title": "Security Roadmap",
            "heading": "SOC2 Timeline",
            "text_summary": "SOC2 is planned for Q3 2025.",
            "space_key": "SEC",
        }
    }
]
```

Add a dedicated test that asserts the threshold boundary: a result with `"score": 0.29` must be discarded and a result with `"score": 0.30` must be kept.

---

### CR-02: `asyncio.create_task()` in fallback branch fires into wrong event loop — silent coroutine leak

**File:** `confluence_logic/agents/confluence_qa_agent.py:232`

**Issue:** When `ensure_user_confluence_graph` times out (or raises any `Exception`), the except block calls:

```python
asyncio.create_task(confluence_page_graph.ensure_user_confluence_graph(graph_user_id))
```

`asyncio.create_task()` requires a running event loop in the **current thread**. If this code is ever called from a non-async context or from inside the `_run_async_blocking` thread pool (which spawns bare `threading.Thread` instances with no event loop attached), the call raises `RuntimeError: no running event loop` — and because it is inside the `except` clause of `except (asyncio.TimeoutError, Exception)`, that `RuntimeError` is silently swallowed by the broad catch above it (the outer `try/except Exception` at line 241 will catch it), completely hiding the failure. Even when called from the correct async context, the created task is detached and never awaited — if it raises, the exception is unhandled and logged by Python's "Task exception was never retrieved" machinery rather than surfaced to the caller.

**Fix:** Replace the naked `create_task` with `asyncio.ensure_future` only if a loop is actually running, or schedule the background warm-up at the `_handle_confluence_question` site where the loop is known-good:

```python
except (asyncio.TimeoutError, Exception) as _exc:
    logger.debug("Confluence graph pre-warm timed out or failed (%s); continuing", _exc)
    try:
        loop = asyncio.get_running_loop()
        loop.create_task(
            confluence_page_graph.ensure_user_confluence_graph(graph_user_id),
            name="confluence_graph_prewarm_bg",
        )
    except RuntimeError:
        pass  # no running loop — skip background warm-up
```

---

### CR-03: `_run_async_blocking` inside `@function_tool` bodies deadlocks when the OpenAI Agents SDK runs tools on the event loop thread

**File:** `confluence_logic/agents/confluence_qa_agent.py:60-79` (and usage at lines 144, 186)

**Issue:** The OpenAI Agents SDK (`Runner.run`) is an async function. When `Runner.run` calls a `@function_tool`, the tool is invoked from within the running async event loop (either directly as an async call, or via `asyncio.to_thread` — depending on SDK version). `_run_async_blocking` detects a running loop and spawns a new `threading.Thread` that calls `asyncio.run(coro)`. `asyncio.run()` creates a **new event loop in that thread** and runs the coroutine. The Neo4j driver used inside `query_user_confluence_graph` and `list_user_confluence_pages` is typically an async driver; if it was created on the original event loop, its internal connection pool is **not safe to use from a separate event loop in a different thread**. This creates an intermittent deadlock or `Event loop is closed` / `Task attached to a different loop` errors that are test-environment-dependent and only surface under real Neo4j connections.

The pattern is explicitly flagged in the architecture notes (`confluence_logic/agents/tools.py` already uses the same bridge, so this is a known accepted risk) but the bridge is **copied** into `confluence_qa_agent.py` as a local duplicate (lines 57-79) without the comment from `tools.py` acknowledging the risk. The duplicate creates a second maintenance surface.

**Fix (minimum):** Remove the local copy and import the shared bridge from `tools.py`:

```python
from .tools import _run_async_blocking  # single canonical copy
```

**Fix (proper):** Convert the `@function_tool` functions to `async def` if the Agents SDK version in use supports async tools (which the OpenAI Agents SDK v0.x does). This eliminates the bridge entirely:

```python
@function_tool
async def search_confluence_pages(query: str) -> str:
    ...
    graph_results = await confluence_page_graph.query_user_confluence_graph(user_id, query, limit=8)
    ...
```

---

### CR-04: Both model names (`gpt-5-mini`, `gpt-4o-mini` for the Agent) are invalid on OpenAI — `gpt-5-mini` does not exist

**File:** `confluence_logic/agents/confluence_qa_agent.py:203` and `confluence_logic/jarvis_agentic.py:67`

**Issue:** `CLAUDE.md` explicitly flags `gpt-5-mini` as an **invalid model name**: "JARVIS_AGENT_MODEL: gpt-5-mini **(invalid model name)**" and "JARVIS_REVIEW_MODEL: gpt-5-mini **(invalid model name)**". `ConfluenceQAAgent.__init__` defaults `model="gpt-5-mini"` and passes it to the `Agent(model=model, ...)` constructor. `jarvis_agentic.py` line 67 reads `JARVIS_AGENT_MODEL = os.getenv("JARVIS_AGENT_MODEL", "gpt-5-mini")` and then constructs `ConfluenceQAAgent()` via `_get_qa_agent()` with no model override — so the default is always `gpt-5-mini`. Every invocation of the agent in production will fail at the OpenAI API with a model-not-found error.

The unit test `test_qa_model_split_tools_gpt5mini_synthesis_gpt4omini` **structurally asserts** `agent.agent.model == "gpt-5-mini"` (line 126), which bakes the invalid name into the test contract and would cause that assertion to fail the moment the model name is corrected to a valid value.

**Fix:** Replace `"gpt-5-mini"` with the intended valid model. Based on project context the most likely intended model is `"gpt-4o-mini"` for the tool orchestration layer (or `"gpt-4.1-mini"` if GPT-4.1 tier is intended). Align the env-var default, the class default, and the test assertion:

```python
# jarvis_agentic.py
JARVIS_AGENT_MODEL = os.getenv("JARVIS_AGENT_MODEL", "gpt-4o-mini")

# confluence_qa_agent.py
class ConfluenceQAAgent:
    def __init__(self, model: str = "gpt-4o-mini"):
```

Update `test_qa_model_split_tools_gpt5mini_synthesis_gpt4omini` to assert the correct valid model name.

---

## Warnings

### WR-01: `get_store()`, `get_connector()`, `_get_openai_client()` are not thread-safe — double-initialisation possible under concurrent tool calls

**File:** `confluence_logic/agents/confluence_qa_agent.py:35-54`

**Issue:** All three lazy singleton getters use the check-then-set pattern without a lock:

```python
def get_store():
    global _store
    if _store is None:       # read
        _store = PineconeStore()   # write — not atomic with read
    return _store
```

`PineconeStore.__init__` constructs a Pinecone client, makes an OpenAI client, and potentially connects to the Pinecone index. If two `@function_tool` calls run concurrently (the Agents SDK can parallelise tool calls), both can pass the `is None` check and both construct a `PineconeStore` — resulting in two live Pinecone connections and the second silently replacing the first in the global. This is a race condition, not a crash, but it wastes resources and the discarded client may leave connection state open.

**Fix:** Use a module-level lock, or Python's `threading.Lock`, for each singleton. In an asyncio context, a simpler pattern is to initialise eagerly at import time (inside `if TYPE_CHECKING:` guards if needed) or use `asyncio.Lock` at the `run()` call site. Minimum fix:

```python
_store_lock = threading.Lock()

def get_store():
    global _store
    if _store is None:
        with _store_lock:
            if _store is None:
                _store = PineconeStore()
    return _store
```

---

### WR-02: `_get_qa_agent()` in `jarvis_agentic.py` is not thread-safe — same double-init race as WR-01

**File:** `confluence_logic/jarvis_agentic.py:458-464`

**Issue:** `_get_qa_agent()` uses the identical check-then-set pattern:

```python
if _qa_agent is None:
    from confluence_logic.agents.confluence_qa_agent import ConfluenceQAAgent
    _qa_agent = ConfluenceQAAgent()
```

If `_handle_confluence_question` is invoked twice in rapid succession (two near-simultaneous wake-word triggers), both coroutines can reach `_get_qa_agent()` concurrently, both find `_qa_agent is None`, and both construct a `ConfluenceQAAgent` — which in turn constructs an `Agent` object and calls the OpenAI Agents SDK. The second `ConfluenceQAAgent` instance silently replaces the first in the module global. Since this is pure asyncio (not multi-threaded), the actual interleaving can only happen across `await` points, but the deferred `from ... import` statement itself is not an await point, so in practice the race window is narrow. Still, it is architecturally unsafe.

**Fix:** Add a simple None-check-and-assign inside the existing asyncio event loop (no lock needed since asyncio is single-threaded), but guard the construction behind the deferred import. Alternatively, initialise the agent at application startup in the `lifespan` context manager where the import is already resolved.

---

### WR-03: `_is_confluence_read_query` read-gate blocks read queries that happen to contain any mutation word

**File:** `confluence_logic/jarvis_agentic.py:136-143`

**Issue:** `_CONFLUENCE_MUTATION_PATTERN` matches any of `create|edit|update|delete|remove|rename|add|append|write|change|make|draft|save|put|move|replace` as whole words anywhere in the query. The pattern is checked first and is an automatic rejection. This means:

- "What **changes** were made to the roadmap?" — blocked (`change` matches)
- "**Tell** me **how to add** users to a Confluence space" — blocked (`add` matches)
- "**Show** me the **update** history" — blocked (`update` matches)
- "**List** the **draft** pages" — blocked (`draft` matches)

All of these are unambiguously read queries that users would reasonably expect to work. The mutation check fires on individual words rather than intent phrases, causing false negatives from the read gate. These queries fall through to the Confluence mutation/proposal path instead of the Q&A path.

**Fix:** Narrow the mutation pattern to require additional context (action + object) or use phrase-level patterns. Minimum fix — require the mutation verb to be followed by an object noun within a few words:

```python
_CONFLUENCE_MUTATION_PATTERN = re.compile(
    r"\b(?:create|edit|update|delete|remove|rename|add|append|write|change|make|draft|save|put|move|replace)\s+(?:a\s+|the\s+|that\s+|this\s+)?(?:page|section|heading|content|doc|document|space|table|block|text)\b",
    re.IGNORECASE,
)
```

---

### WR-04: QA-02 test does not actually verify the Neo4j skip path — `query_user_confluence_graph` is set up but never reached by `search_confluence_pages`

**File:** `confluence_logic/tests/test_confluence_qa_agent.py:99`

**Issue:** In `test_qa_fallback_to_rest_when_pinecone_empty`, the test mocks `mock_graph.query_user_confluence_graph = AsyncMock(return_value=[])` but `search_confluence_pages` calls `_run_async_blocking(confluence_page_graph.query_user_confluence_graph(...))` where `confluence_page_graph` is the **module-level import** from line 24 (`from confluence_logic import confluence_page_graph`). The test patches `confluence_logic.agents.confluence_qa_agent.confluence_page_graph` as a `MagicMock()`, which replaces the entire module object. `mock_graph.query_user_confluence_graph` is therefore an attribute on the mock, not on the real module. However, `_run_async_blocking` calls `asyncio.run(coro)` in a new thread — and the coroutine returned by `AsyncMock()` is an asyncio coroutine, which `asyncio.run` will attempt to run in a new event loop in that thread. If the test's outer `asyncio` event loop is still active (pytest-asyncio), `asyncio.run()` will raise `RuntimeError: This event loop is already running` inside the helper thread, and `result["error"]` will be set — causing the Neo4j branch to raise instead of returning `[]` and letting the code fall through to REST. The test may be accidentally passing because the Neo4j exception is caught at line 149 (`except Exception as graph_exc`) and falls through to REST, but for the wrong reason.

**Fix:** The test must be adjusted to verify the Neo4j branch is actually entered and returns empty before REST is tried. Additionally, the `_run_async_blocking` / `AsyncMock` interaction in a `threading.Thread` under a live pytest-asyncio event loop is fragile. Consider converting the tool function to `async def` (see CR-03) which removes this complexity entirely.

---

### WR-05: Latency test synthesis mock uses a synchronous `MagicMock` but `asyncio.to_thread` runs it in a thread pool — the mock call count assertion may be flaky

**File:** `confluence_logic/tests/test_qa_latency.py:50-51, 76`

**Issue:** The synthesis call in `ConfluenceQAAgent.run()` is wrapped in `asyncio.to_thread(lambda: _get_openai_client().chat.completions.create(...))`. The mock is:

```python
mock_oai_client.chat.completions.create.return_value = mock_oai_response
```

`asyncio.to_thread` runs the lambda in `ThreadPoolExecutor`. The `MagicMock` is a regular Python object accessed from a worker thread. While `MagicMock` itself is generally thread-safe for attribute access, the assertion `mock_oai_client.chat.completions.create.assert_called_once()` is checked after `await agent.run(...)` completes, which is correct. However, the `20ms simulated latency` comment in the docstring (line 30) is misleading: the mock's `create` call returns synchronously in ~microseconds, not 20ms. The latency budget claimed for synthesis does not reflect any actual overhead. The test therefore cannot detect if a real `asyncio.to_thread` call introduces blocking overhead — the test validates structure, not latency contributions from the synthesis step.

This is a **documentation/accuracy bug** in the test specification. The 3000ms SLA assertion is satisfied trivially (total time will be ~50ms from `asyncio.sleep(0.050)` in the runner mock) and gives false confidence that the synthesis step is latency-accounted.

**Fix:** Either add an explicit `asyncio.sleep` in the synthesis mock to simulate realistic latency, or document clearly that the synthesis contribution is not mocked:

```python
# Synthesis mock with 20ms simulated latency
original_to_thread = asyncio.to_thread
async def _mock_to_thread(func, *args, **kwargs):
    await asyncio.sleep(0.020)
    return func(*args, **kwargs)
```

---

## Info

### IN-01: `_run_async_blocking` is duplicated — identical copy exists in both `tools.py` and `confluence_qa_agent.py`

**File:** `confluence_logic/agents/confluence_qa_agent.py:60-79`

**Issue:** The function `_run_async_blocking` in `confluence_qa_agent.py` (lines 60-79) is byte-for-byte identical to the one in `confluence_logic/agents/tools.py` (lines 141-160). The comment on line 57 says "copied from confluence_logic/agents/tools.py". This creates two maintenance surfaces for the same critical bridge. Any fix to the bridge must be applied in both places.

**Fix:** Import the shared copy:
```python
from .tools import _run_async_blocking
```
Remove lines 57-79 from `confluence_qa_agent.py`.

---

### IN-02: `except (asyncio.TimeoutError, Exception)` is redundant — `Exception` already covers `asyncio.TimeoutError`

**File:** `confluence_logic/agents/confluence_qa_agent.py:231`

**Issue:** `asyncio.TimeoutError` is a subclass of `Exception` (Python 3.11+) and of `concurrent.futures.TimeoutError` (Python 3.10). In all supported Python versions this means `except (asyncio.TimeoutError, Exception)` is exactly equivalent to `except Exception`. The explicit listing of `asyncio.TimeoutError` does not add any behaviour, only confusion (a reader might wonder if the ordering matters or if there is a subtlety being handled).

**Fix:**
```python
except Exception:
    asyncio.create_task(confluence_page_graph.ensure_user_confluence_graph(graph_user_id))
```
Or, after applying CR-02's fix, the entire except clause is restructured and this becomes moot.

---

_Reviewed: 2026-05-17T00:00:00Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
