# Testing Patterns

**Analysis Date:** 2026-04-04

## Test Framework

**Status:** No testing infrastructure detected

- No test runner configured (pytest, unittest, nose, etc.)
- No test dependencies in `requirements.txt`
- No test configuration files (`pytest.ini`, `setup.cfg`, `tox.ini`, `conftest.py`)
- No test files found in codebase (no `*_test.py`, `test_*.py`, `*_spec.py` files)

**Implications:**
- All testing is manual
- No automated verification of functionality
- No regression test suite
- No CI/CD test pipeline in place

## Test File Organization

**Current State:**
- Not applicable - no test files exist
- No established pattern for test file placement or naming

**Recommended Structure (if tests were added):**
```
recallai-meeting/
├── jarvis.py           # Main source file
├── tests/              # Test directory (parallel structure)
│   ├── __init__.py
│   ├── test_jarvis.py
│   ├── test_recall_helpers.py
│   ├── test_wake_word.py
│   ├── test_query_handler.py
│   └── fixtures/       # Test data
│       └── sample_transcripts.json
```

**Naming Convention (to adopt):**
- `test_*.py` for test modules
- `test_<function_name>()` for test functions
- Group tests by functionality area: recall API, wake word detection, query handling, websocket

## Testing Gaps

**Critical untested areas:**

### API Integration (`create_bot()`)
- Location: `jarvis.py`, lines 73-114
- Tests needed:
  - Successful bot creation with valid API key
  - Handling of invalid API key (401)
  - Network timeout scenarios
  - Malformed response handling
  - WebSocket URL construction
- Current validation: Only manual/integration testing

### Wake Word Detection (`extract_wake_and_query()`)
- Location: `jarvis.py`, lines 269-277
- Tests needed:
  - "Hey Jarvis" with full query: `"Hey Jarvis, what's the weather?"`
  - Bare "Jarvis" wake word
  - Case insensitivity: `"HEY JARVIS"`, `"hey jarvis"`
  - With trailing punctuation: `"Jarvis, ..."`
  - Without wake word should return None
  - Query extraction accuracy
- Regex pattern: `r"(?:hey\s+)?jarvis[,.]?\s*(.*)"` (lines 263-266)
- Current validation: Manual in-meeting testing only

### Tool Execution (`_run_tool()`)
- Location: `jarvis.py`, lines 188-193
- Tests needed:
  - Valid tool name execution
  - Invalid tool name handling
  - Argument passing to `_fetch_weather()`
  - Argument passing to `_get_meeting_summary()`
  - Edge cases: empty args, missing args, malformed args
- Current validation: Runtime error handling only

### Weather Service (`_fetch_weather()`)
- Location: `jarvis.py`, lines 144-150
- Tests needed:
  - Valid city name returns weather
  - Invalid city name handling
  - Network timeout (timeout=5)
  - HTTP error status codes
  - Response parsing (assumes text format)
- Current validation: Try-except with generic error message

### Meeting Transcript (`_get_meeting_transcript()`)
- Location: `jarvis.py`, lines 153-158
- Tests needed:
  - Empty transcript returns "[No transcript yet]"
  - Multiple entries format correctly
  - Special characters in participant names
  - Special characters in transcript text
- Current validation: None - assumes format correctness

### Agentic Loop (`handle_query()`)
- Location: `jarvis.py`, lines 199-256
- Tests needed:
  - Tool call loop: max 5 iterations enforced
  - Tool call chaining: multiple tools in sequence
  - Early termination when finish_reason != "tool_calls"
  - Graceful handling of OpenAI API errors:
    - Insufficient quota (error 429)
    - Invalid API key (401)
    - Generic API failures
  - Message history construction
  - Speech output after successful query
- Current validation: Manual testing; error handling only logs and speaks error

### WebSocket Handler (`websocket_endpoint()`)
- Location: `jarvis.py`, lines 283-341
- Tests needed:
  - Connection acceptance
  - Non-transcript.data events ignored
  - Transcript data parsing and extraction
  - Empty sentence filtering
  - Bot's own speech ignored (case-insensitive match)
  - State updates to `meeting_state["transcript_log"]`
  - Wake word detection integration
  - Thread spawning for query and speech
  - Disconnect handling
  - Exception handling
  - Message format validation
- Current validation: Runtime websocket protocol only

### Text-to-Speech (`speak()`)
- Location: `jarvis.py`, lines 117-138
- Tests needed:
  - Successful audio generation and upload
  - Temporary file cleanup on success
  - Temporary file cleanup on error
  - gTTS language parameter
  - API response status code validation (200)
  - Network timeout handling
  - Malformed response handling
- Current validation: Status code check; no test coverage

## Mocking Strategy (if tests were implemented)

**What to Mock:**
- `requests.post()` for all Recall.ai API calls
- `requests.get()` for weather API calls
- `gTTS()` for text-to-speech generation
- `OpenAI().chat.completions.create()` for LLM calls
- `websocket.receive_json()` for WebSocket messages
- `time.sleep()` to speed up tests
- File I/O operations (`open()`, `os.path.exists()`, `os.remove()`)

**What NOT to Mock:**
- `re.compile()` and regex matching - test actual wake word detection
- Logger calls - verify correct log messages are generated
- Dictionary operations - test state management directly
- String operations - test message formatting

## Recommended Testing Approach

**Given the single-file structure and production nature:**

1. **Unit Test Layer:**
   - Test pure functions: `extract_wake_and_query()`, `_fetch_weather()`, `_get_meeting_transcript()`, `_run_tool()`
   - Mock external dependencies (requests, OpenAI, gTTS)
   - Fast, isolated, deterministic tests

2. **Integration Test Layer:**
   - Test `handle_query()` with mocked OpenAI responses
   - Test agentic loop tool chaining
   - Test API call construction and error handling
   - Test WebSocket message processing (async)

3. **System/E2E Tests:**
   - Deploy with test Recall.ai account
   - Join actual test meeting
   - Verify end-to-end flow with real services

4. **Manual Testing Checklist:**
   - Wake word detection accuracy across different speaking styles
   - Response latency perception
   - Audio quality of TTS output
   - Meeting transcription accuracy
   - Tool execution reliability

## Test Framework Recommendation

**Recommended Framework:** `pytest` with `pytest-asyncio`

**Rationale:**
- Lightweight, no boilerplate compared to `unittest`
- Better assertion messages for debugging
- Excellent fixture system for shared test data
- Native async support for WebSocket testing
- Community plugins for coverage, mocking, etc.

**Setup:**
```bash
pip install pytest pytest-asyncio pytest-mock pytest-cov
```

**Configuration file:** `pytest.ini` or `[tool:pytest]` in `setup.cfg`

```ini
[pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
asyncio_mode = auto
addopts = --cov=. --cov-report=html --cov-report=term-missing
```

## Sample Test Structure (if implemented)

```python
# tests/test_wake_word.py
import pytest
from jarvis import extract_wake_and_query

@pytest.mark.parametrize("text,expected", [
    ("Hey Jarvis, what's the weather?", "what's the weather?"),
    ("jarvis summarize the meeting", "summarize the meeting"),
    ("JARVIS help me", "help me"),
    ("Hey Jarvis,", ""),
    ("jarvis", ""),
    ("Hey Jarvis", ""),
    ("No wake word here", None),
])
def test_extract_wake_and_query(text, expected):
    result = extract_wake_and_query(text)
    assert result == expected
```

```python
# tests/test_tools.py
import pytest
from unittest.mock import patch, MagicMock
from jarvis import _run_tool, _fetch_weather

@patch("jarvis.requests.get")
def test_fetch_weather_success(mock_get):
    mock_get.return_value.status_code = 200
    mock_get.return_value.text = "Clear +72°F"

    result = _fetch_weather("London")
    assert result == "Clear +72°F"
    mock_get.assert_called_once_with("https://wttr.in/London?format=3", timeout=5)

@patch("jarvis.requests.get")
def test_fetch_weather_error(mock_get):
    mock_get.side_effect = Exception("Connection timeout")
    result = _fetch_weather("Unknown")
    assert "unavailable" in result
```

```python
# tests/test_api.py
import pytest
from unittest.mock import patch, MagicMock
from jarvis import create_bot

@patch("jarvis.requests.post")
def test_create_bot_success(mock_post):
    mock_response = MagicMock()
    mock_response.status_code = 201
    mock_response.json.return_value = {"id": "bot-123"}
    mock_post.return_value = mock_response

    result = create_bot("https://meet.google.com/test")
    assert result == "bot-123"

@patch("jarvis.requests.post")
def test_create_bot_auth_failure(mock_post):
    mock_response = MagicMock()
    mock_response.status_code = 401
    mock_response.text = "Unauthorized"
    mock_post.return_value = mock_response

    result = create_bot("https://meet.google.com/test")
    assert result is None
```

## Coverage Goals

**Not Currently Enforced:** No coverage threshold exists

**Recommended Targets:**
- 80% overall coverage
- 95% coverage for critical paths: API calls, wake word detection, tool execution, error handling
- 100% coverage for pure utility functions: `_fetch_weather()`, `_get_meeting_transcript()`, `_run_tool()`

**View Coverage (once tests added):**
```bash
pytest --cov=. --cov-report=html
# View: htmlcov/index.html
```

---

*Testing analysis: 2026-04-04*
