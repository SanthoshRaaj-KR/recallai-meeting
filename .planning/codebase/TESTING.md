# Testing Patterns

**Analysis Date:** 2026-05-11

## Framework & Setup

- **Framework:** `pytest` + `pytest-asyncio`
- **No config files** found (`pytest.ini`, `setup.cfg`, `pyproject.toml` test sections absent)
- **No `conftest.py`** present in any package

### Running Tests

```bash
# Unit/integration tests per package
pytest confluence_logic/tests/
pytest local_office_logic/tests/

# Run a specific test file
pytest confluence_logic/tests/test_classifier.py
pytest confluence_logic/tests/test_general_responder.py
pytest confluence_logic/tests/test_review_api.py
pytest confluence_logic/tests/test_confluence_page_graph.py

# Root-level E2E evaluation harnesses (NOT pytest — run directly)
python -m tests.e2e_pipeline_eval
python -m tests.pipeline_comprehensive_eval
python -m tests.pipeline_comprehensive_eval --quality-only   # Mock TTS, CI-safe
python -m tests.pipeline_comprehensive_eval --full           # Real TTS, ~3-5 min
```

## Test Organization

Tests are co-located in `tests/` subdirectories within each domain package — not in a single top-level test tree.

```
confluence_logic/tests/
├── __init__.py
├── test_flow.py                   # Integration flow through confluence pipeline
├── test_classifier.py             # Intent routing — context-dependent vs standalone questions
├── test_general_responder.py      # Web search LLM router (_needs_web_search, _quick_web_search)
├── test_graph_rag.py              # Graph RAG unit tests
├── test_jarvis_agentic.py         # Agentic orchestrator tests
├── test_review_api.py             # Review API: session state machine, Recall status parsing
└── test_confluence_page_graph.py  # Page graph HTML parsing, user_id scoping, Neo4j query

local_office_logic/tests/
├── __init__.py
└── test_local_office_flow.py      # Integration flow through local office pipeline

tests/                             # Root-level evaluation harnesses (not pytest)
├── e2e_pipeline_eval.py           # Synthetic 40-min meeting → full pipeline quality eval
├── pipeline_comprehensive_eval.py # Quality + TTS latency eval with two modes
└── test_meeting_responder.py
```

## Types of Tests

### Unit Tests (mocked I/O)
Individual functions tested in isolation with all external calls patched. Covers:
- `test_classifier.py` — parametrized pytest tests asserting correct intent classification for context-dependent and standalone queries, using `@pytest.mark.asyncio` + `@pytest.mark.parametrize`
- `test_general_responder.py` — tests `_needs_web_search()` routing (weather/sports/factual/error fallback), all using `patch.object` on the OpenAI client
- `test_review_api.py` — tests session state machine helpers: `_latest_recall_status_code`, `_refresh_session_status_from_recall` with 404 bot handling
- `test_confluence_page_graph.py` — tests `_sections_from_html()` HTML parsing and `query_user_confluence_graph()` Neo4j driver scoping

### Integration Flow Tests
Tool-chain tests that exercise multiple components together with select mocks at system boundaries (LLM calls, DB calls). Found in `test_flow.py` and `test_local_office_flow.py`.

### E2E Evaluation Scripts
Standalone scripts in `tests/` that run full pipelines against synthetic meeting transcripts. These are not pytest-collected — run directly via `python -m`.

**`e2e_pipeline_eval.py`:**
- Injects a realistic 40-participant synthetic meeting transcript (Indian SaaS startup, ~40 minutes)
- Runs classifier → handler → LLM → captures spoken text (TTS mocked)
- Reports answer quality, timing, filler words, scored quality summary

**`pipeline_comprehensive_eval.py`:**
- Two modes: `--quality-only` (mock TTS, CI-safe) and `--full` (real TTS synthesis, ~3-5 min)
- Measures time-to-first-word (TTFW): elapsed from question arrival to first audio byte
- Uses 6-speaker 32-minute synthetic transcript with realistic STT artifacts

## Fixtures & Mocking

No pytest fixtures defined except one `autouse` fixture in `test_graph_rag.py`.

**Dominant mocking patterns:**

| Pattern | Usage |
|---------|-------|
| `@patch` decorator | Most common — patches module-level callables (e.g., `@patch.object(gr, "_get_client")`) |
| `patch.object` context manager | Patching methods on specific instances or module attributes |
| `AsyncMock` | Mocking coroutines (LLM calls, async DB ops, Neo4j driver) |
| `SimpleNamespace` | Lightweight stubs for data objects and OpenAI response shapes |
| `MagicMock` | General-purpose mock for sync callables |
| `Mock(status_code=...)` | Simulating HTTP response objects for Recall/Supabase error paths |

**Typical mock for OpenAI responses:**
```python
def _mock_openai_response(content: str):
    choice = SimpleNamespace(message=SimpleNamespace(content=content))
    return SimpleNamespace(choices=[choice])

with patch.object(gr, "_get_client") as mock_client:
    mock_client.return_value.chat.completions.create.return_value = _mock_openai_response("yes")
    result = await gr._needs_web_search("what's the weather in Paris")
```

**Typical mock for Neo4j driver:**
```python
record = SimpleNamespace(data=lambda: {"page_id": "page-1", "title": "Roadmap", ...})
driver = AsyncMock()
driver.execute_query = AsyncMock(return_value=([record], None, []))
with patch.object(confluence_page_graph, "_driver", return_value=driver):
    results = await confluence_page_graph.query_user_confluence_graph("supabase:user-1", "launch")
```

**Typical mock for Recall API state machine:**
```python
with patch.object(api, "_fetch_recall_bot_payload", return_value={"status_changes": [{"code": "call_ended"}]}), \
     patch.object(api, "_RECALL_STATUS_CACHE_SECONDS", 0.0):
    api._refresh_session_status_from_recall(state)
```

## Async Patterns

- `@pytest.mark.asyncio` — marks native async test functions for pytest-asyncio collection
- `asyncio.run()` wrappers — used in some test functions to exercise async code from a sync test (anti-pattern; prefer `@pytest.mark.asyncio`)
- `await asyncio.sleep(0)` — yields control to background tasks during event-loop tests

## Parametrize Pattern

`@pytest.mark.parametrize` is used in `test_classifier.py` to run the same assertion over multiple query strings:
```python
@pytest.mark.asyncio
@pytest.mark.parametrize("query", [
    "what is the fix?",
    "how do we fix the problem?",
    "should we do that?",
])
async def test_context_dependent_questions_are_meeting_opinion(query):
    assert await classifier.classify_intent(query) == "meeting_opinion"
```

## Coverage

- **No coverage threshold enforced**
- **Gaps identified:**
  - `local_office_logic/` has sparse test coverage vs `confluence_logic/`
  - `local_repl.py` (both packages) — no dedicated tests
  - `meeting_responder.py` — limited test coverage (only via eval harnesses)
  - `ingestion/doc_pipeline.py` (both packages) — zero coverage
  - TTS pipeline (all providers) — zero coverage
  - `utils/sandbox.py`, `utils/office_runtime.py` — zero coverage
  - `local_office_logic/general_responder.py`, `meeting_responder.py` — zero coverage
  - Evaluation harnesses in `tests/` are not integrated into a CI pipeline

## Frontend Testing

The `review-ui/` Next.js app has **no test files**. No `vitest`, `jest`, or `@testing-library/react` is configured in `review-ui/package.json`. The `sync-sage-bot/` has vitest configured but is outside the main codebase.
