# Testing Patterns

**Analysis Date:** 2026-04-10

## Test Framework

**Runner:**
- pytest (listed in requirements.txt)
- No explicit pytest.ini or setup.cfg configuration found
- Tests discovered via standard pytest convention (test_*.py files)

**Assertion Library:**
- pytest assertions (built-in `assert` statements)
- Manual assertions: `assert len(resp.candidates) == 1`, `assert resp.success is True`

**Run Commands:**
```bash
pytest                          # Run all tests
pytest confluence_logic/tests/  # Run all tests in tests directory
pytest confluence_logic/tests/test_flow.py  # Run specific test file
pytest -k test_name            # Run specific test by name
```

## Test File Organization

**Location:**
- Co-located in dedicated `confluence_logic/tests/` directory
- Separate from production code (not alongside modules)

**Naming:**
- Test files: `test_flow.py`, `test_jarvis_agentic.py`
- Test functions: `test_*()` prefix (pytest convention)
- Helper functions: `_reset_meeting_state()` prefix with underscore for internal utilities

**Structure:**
```
confluence_logic/tests/
├── __init__.py
├── test_flow.py              # Tool and agent integration tests
└── test_jarvis_agentic.py    # Voice request and bot lifecycle tests
```

## Test Structure

**Suite Organization:**

Tests are organized by feature/module being tested:
- `test_flow.py`: Tests for search, fetch, preview, commit, delete, and creation workflows
- `test_jarvis_agentic.py`: Tests for voice request handling, bot creation, and speech synthesis

**Test function patterns:**

```python
def test_page_selection_disambiguation(mock_get_store, mock_get_connector):
    # Arrange: Setup mocks
    mock_connector.search_pages.return_value = [
        {"page_id": "1", "title": "Quarterly Goals", "space_key": "DEV", "excerpt": "Live page result"},
    ]

    # Act: Call function
    resp = search_workspace_knowledge("Quarterly")

    # Assert: Verify results
    assert len(resp.candidates) == 1
    assert resp.candidates[0].page_id == "1"
    assert "Success" in resp.message
```

**Patterns:**
- **Setup:** Mock objects configured with `.return_value` or `.side_effect`
- **Execution:** Direct function calls (synchronous) or `asyncio.run()` for async functions
- **Teardown:** Implicit via mocks; cleanup helper `_reset_meeting_state()` for stateful tests
- **Assertion:** Simple `assert` statements with boolean conditions or equality checks

## Mocking

**Framework:** Python `unittest.mock` (standard library)

**Usage pattern:**
```python
from unittest.mock import patch, Mock, AsyncMock

@patch('confluence_logic.agents.tools.get_connector')
@patch('confluence_logic.agents.tools.get_store')
def test_search(mock_get_store, mock_get_connector):
    mock_connector = mock_get_connector.return_value
    mock_connector.search_pages.return_value = [...]
```

**Decorators used:**
- `@patch()`: Mock entire dependencies (see test_flow.py lines 11-12, test_jarvis_agentic.py lines 113-114)
- `@patch.object()`: Mock specific object attributes (test_jarvis_agentic.py line 102, 124)

**Patterns:**

Example 1 - Mock connector calls:
```python
@patch('confluence_logic.agents.tools.get_connector')
def test_version_conflict_handling(mock_get_connector):
    mock_connector = mock_get_connector.return_value
    mock_connector.push_update.side_effect = ValueError("Version Conflict: ...")
    resp = commit_document_edit("123", 4, "H1", "<p>test</p>", "<p>new</p>")
    assert resp.success is False
```

Example 2 - Mock async functions:
```python
with patch.object(ja.session_agent, "plan_voice_turn", new=AsyncMock(return_value=...)):
    await ja.handle_spoken_request("update page", "bot-123")
```

Example 3 - Mock with call tracking:
```python
@patch('confluence_logic.ingestion.doc_pipeline.IngestionPipeline.process_page')
def test_commit_delete(mock_pipeline, mock_get_connector):
    # ... test code ...
    mock_pipeline.assert_called_once_with("123")
```

**What to Mock:**
- External API calls: `get_connector()`, `get_store()` (Confluence and Pinecone)
- Agent runners: `Runner.run()` when testing orchestration
- Async operations: `AsyncMock()` for async function behavior
- System interactions: `requests.post`, `gTTS`, file operations
- Side effects: Use `.side_effect` to simulate failures (exceptions, version conflicts)

**What NOT to Mock:**
- Internal utility functions: HTML parsing (`edit_block_in_section`, `delete_content_in_section`) - tested directly
- Pydantic models: Response objects created directly, not mocked
- Data structures: Lists, dicts used as test data directly
- Pure logic: String normalization, heading detection functions tested without mocks

## Fixtures and Factories

**Test Data:**
Test data created inline within test functions. No separate fixtures file.

Example from test_flow.py:
```python
mock_connector.search_pages.return_value = [
    {"page_id": "1", "title": "Quarterly Goals", "space_key": "DEV", "excerpt": "Live page result"},
]
```

Example with Pydantic models:
```python
CandidatePage(page_id="1", title="Sample AI Page", space_key="DEV", snippet="About AI"),
CandidatePage(page_id="2", title="ML Notes", space_key="DEV", snippet="About ML"),
```

Helper function for state cleanup:
```python
def _reset_meeting_state():
    ja.meeting_state["bot_id"] = None
    ja.meeting_state["transcript_log"] = []
    ja.meeting_state["is_active"] = False
    # ... resets 12 state fields ...
```

**Location:**
- Fixtures defined inline at top of test files: `_reset_meeting_state()` (test_jarvis_agentic.py:10)
- Test data hardcoded in test functions
- No conftest.py or centralized fixture management

## Coverage

**Requirements:** No explicit coverage requirements enforced (no .coveragerc or pytest config)

**View Coverage:**
```bash
pytest --cov=confluence_logic confluence_logic/tests/
pytest --cov=confluence_logic --cov-report=html  # Generate HTML report
```

## Test Types

**Unit Tests:**
- Scope: Individual functions in isolation
- Examples: `test_fragment_replacement()`, `test_root_level_blank_edit()`, `test_format_page_titles_for_user_omits_metadata()`
- Approach: Call function directly, mock dependencies, assert outputs
- Characteristic: Fast, no external dependencies

Unit test example:
```python
def test_fragment_replacement():
    html = "<body><h2>Heading 1</h2><p>Old text</p><h2>Heading 2</h2></body>"
    result = edit_block_in_section(html, "Heading 1", "<p>Old text</p>", "<p>New text</p>")
    assert "Old text" not in result
    assert "New text" in result
```

**Integration Tests:**
- Scope: Multiple components working together
- Examples: `test_page_selection_disambiguation()`, `test_commit_delete()`, `test_handle_spoken_request_master_clarifies_when_needed()`
- Approach: Mock external dependencies (Confluence, Pinecone), verify tool orchestration
- Characteristic: Test tool flows, agent decision logic, state management

Integration test example:
```python
@patch('confluence_logic.agents.tools.get_connector')
@patch('confluence_logic.agents.tools.get_store')
def test_page_selection_disambiguation(mock_get_store, mock_get_connector):
    # Mocks represent Confluence and Pinecone
    # Test verifies search combines both sources
    resp = search_workspace_knowledge("Quarterly")
    assert resp.candidates[0].page_id == "1"
```

**E2E Tests:**
- Framework: Not found (no Playwright, Cypress, etc. in requirements.txt)
- Approach: Would require live Confluence/Pinecone instances
- Coverage: Voice request flow (test_jarvis_agentic.py) simulates end-to-end voice handling

Simulation of E2E flow:
```python
async def run_test():
    with patch.object(ja.session_agent, "plan_voice_turn", new=AsyncMock(...)):
        with patch.object(ja.session_agent, "handle_prepared_query", new=AsyncMock(...)):
            await ja.handle_spoken_request("update the roadmap page", "bot-123")
            # Verifies: voice input → planning → execution → state updates
```

## Common Patterns

**Async Testing:**

Tests wrap async functions in `asyncio.run()`:
```python
def test_handle_prepared_query_bypasses_reframer(mock_runner_run):
    agent = EditorAgent(model="gpt-5-mini")
    result = asyncio.run(agent.handle_prepared_query(
        "Edit the Sample AI Page and add current AI trends.",
        original_query="update sample ai page",
    ))
    assert result == "Prepared edit completed."
```

For async mocks with await:
```python
async def run_test():
    with patch.object(ja.session_agent, "plan_voice_turn", new=AsyncMock(...)) as mock_plan:
        await ja.handle_spoken_request("request", "bot-123")
        assert mock_plan.await_count == 1
```

**Error Testing:**

Tests explicitly trigger error conditions:
```python
def test_version_conflict_handling(mock_get_connector):
    mock_connector.push_update.side_effect = ValueError("Version Conflict: Expected version 4, but live version is 5.")
    resp = commit_document_edit("123", 4, "H1", "<p>test</p>", "<p>new</p>")
    assert resp.success is False
    assert "ConflictError" in resp.message
```

Tests verify error messages:
```python
def test_disambiguation_error():
    html_dup = "<body><h2>Heading 1</h2><p>A</p><h2>Heading 1</h2><p>B</p></body>"
    with pytest.raises(ValueError, match="Multiple headings match"):
        edit_block_in_section(html_dup, "Heading 1", "", "")
```

**State Verification:**

Tests check side effects and state changes:
```python
def test_handle_spoken_request_queues_non_overriding_when_busy():
    # ... setup ...
    await ja.handle_spoken_request("also update notes", "bot-123")
    assert len(ja.meeting_state["pending_requests"]) == 1
    assert ja.meeting_state["pending_requests"][0].request == "also update notes"
    mock_speak.assert_called_once_with(ja.QUEUE_ACK, "bot-123")
```

**Mock Call Verification:**

Tests verify mocks were called correctly:
```python
def test_update_page_title_tool(mock_pipeline, mock_get_connector):
    # ... test code ...
    mock_connector.push_update.assert_called_once_with(
        "123",
        "<h2>Overview</h2><p>Hello</p>",
        expected_version=4,
        title_override="Why Donuts Are Awesome",
    )
    mock_pipeline.assert_called_once_with("123")
```

---

*Testing analysis: 2026-04-10*
