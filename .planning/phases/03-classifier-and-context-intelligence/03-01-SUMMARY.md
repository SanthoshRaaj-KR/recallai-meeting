---
phase: 03-classifier-and-context-intelligence
plan: 01
subsystem: general-responder
tags: [sliding-window, web-search, llm-routing, topic-drift, tdd]
dependency_graph:
  requires: []
  provides: [TOPIC-01, WEBSEARCH-01]
  affects: [confluence_logic/general_responder.py, confluence_logic/jarvis_agentic.py]
tech_stack:
  added: [neo4j==6.1.0, pytest-asyncio]
  patterns: [asyncio.to_thread, gpt-4o-mini yes/no routing, sliding window history cap]
key_files:
  created:
    - confluence_logic/tests/test_general_responder.py
  modified:
    - requirements.txt
    - confluence_logic/jarvis_agentic.py
    - confluence_logic/tests/test_jarvis_agentic.py
    - confluence_logic/general_responder.py
decisions:
  - "_MAX_GENERAL_HISTORY set to 3 (from 4): stale topic context ages out after 3 Q&A exchanges"
  - "async _needs_web_search uses gpt-4o-mini with max_tokens=5, temperature=0.0 for deterministic yes/no routing"
  - "_FRESHNESS_PATTERNS regex deleted: LLM routing generalizes beyond static keyword list"
metrics:
  duration: "~3 min"
  completed: "2026-04-13"
  tasks_completed: 2
  tasks_total: 2
  files_modified: 5
---

# Phase 03 Plan 01: Foundation Changes — Sliding Window and LLM Web Search Router Summary

**One-liner:** Capped general question history to 3 exchanges for topic drift control and replaced regex web search trigger with an async gpt-4o-mini yes/no classifier covering weather, sports, and any real-time data query.

## What Was Built

### Task 1: Sliding window cap and dependency install

Changed `_MAX_GENERAL_HISTORY` from 4 to 3 in `jarvis_agentic.py`. The existing `_format_general_history()` and `_remember_general_exchange()` already use `[-_MAX_GENERAL_HISTORY:]` slice logic, so this constant change is sufficient to enforce a 3-exchange window. Added `neo4j==6.1.0` and `pytest-asyncio` to `requirements.txt`.

Added three new tests to `test_jarvis_agentic.py`:
- `test_format_general_history_returns_at_most_3_exchanges` — verifies only last 3 Q&A pairs are returned (q1, q2 absent; q3, q4, q5 present; 6 lines total)
- `test_remember_general_exchange_caps_at_3` — verifies appending 5 exchanges leaves exactly 3 in state
- `test_format_general_history_empty` — verifies empty history returns `[none]`

Also updated `_reset_meeting_state()` in tests to reset `general_history` and `last_jarvis_response` keys.

### Task 2: Async LLM web search router (TDD)

**RED:** Created `test_general_responder.py` with 4 failing async tests for the new `_needs_web_search` behavior.

**GREEN:** Replaced the sync `_needs_web_search` and `_FRESHNESS_PATTERNS` regex with:
- `_WEB_SEARCH_ROUTER_PROMPT`: a routing prompt covering weather, sports, news, stock prices, current events
- `async _needs_web_search(question: str) -> bool`: calls gpt-4o-mini via `asyncio.to_thread`, `max_tokens=5`, `temperature=0.0`, returns True if response starts with "yes", falls back to False on exception
- Updated `answer_general_question` call site: `if await _needs_web_search(question):`
- Removed unused `import re` from `general_responder.py`
- `_quick_web_search()` is completely unchanged per D-09

## Decisions Made

- `_MAX_GENERAL_HISTORY = 3` to match D-04 through D-06 (stale topic context ages out after 3 turns)
- LLM routing over regex: regex missed paraphrases and new question forms; gpt-4o-mini generalizes
- `asyncio.to_thread` wrapper for the LLM call (matches established codebase pattern)
- Graceful fallback: on any LLM exception, `_needs_web_search` returns False (non-fatal, DuckDuckGo skipped)

## Test Results

All plan-required tests pass:
```
pytest confluence_logic/tests/test_jarvis_agentic.py -x -k "test_format_general_history or test_remember_general_exchange" → 3 passed
pytest confluence_logic/tests/test_general_responder.py -x -v → 4 passed
```

Note: `test_build_create_bot_payload_supports_assembly_provider_opt_in` in `test_jarvis_agentic.py` has a pre-existing failure (assembly_ai provider dict mismatch) that predates this plan — logged to deferred items.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Pinecone package rename blocked test collection**
- **Found during:** Task 1 test run
- **Issue:** `pinecone-client` package raises an exception on import, blocking the entire test module: "The official Pinecone python package has been renamed from `pinecone-client` to `pinecone`"
- **Fix:** Ran `pip install pinecone` to install the new package alongside the existing `pinecone-client`
- **Files modified:** None (environment fix only)
- **Commit:** N/A (no code change needed)

**2. [Rule 2 - Missing functionality] _reset_meeting_state missing general_history reset**
- **Found during:** Task 1 — test setup
- **Issue:** The `_reset_meeting_state()` helper in `test_jarvis_agentic.py` did not reset `general_history` or `last_jarvis_response`, which could cause test state bleed when running tests that manipulate `meeting_state["general_history"]`
- **Fix:** Added `ja.meeting_state["general_history"] = []` and `ja.meeting_state["last_jarvis_response"] = None` to `_reset_meeting_state()`
- **Files modified:** `confluence_logic/tests/test_jarvis_agentic.py`
- **Commit:** f483bcc

## Known Stubs

None — all changes are wired and functional. The LLM web search router will make real API calls in production; the tests mock the OpenAI client as designed.

## Commits

| Hash | Message |
|------|---------|
| f483bcc | feat(03-01): cap sliding window to 3 exchanges and install dependencies |
| 5865330 | test(03-01): add failing tests for LLM web search router (WEBSEARCH-01 RED) |
| ca69501 | feat(03-01): replace regex _needs_web_search with async LLM yes/no router (WEBSEARCH-01) |

## Self-Check: PASSED

### Files Created/Modified
- [x] `requirements.txt` — contains `neo4j==6.1.0` and `pytest-asyncio`
- [x] `confluence_logic/jarvis_agentic.py` — `_MAX_GENERAL_HISTORY = 3`
- [x] `confluence_logic/tests/test_jarvis_agentic.py` — new sliding window tests
- [x] `confluence_logic/general_responder.py` — async `_needs_web_search`, `_WEB_SEARCH_ROUTER_PROMPT`, no `_FRESHNESS_PATTERNS`, no standalone `import re`
- [x] `confluence_logic/tests/test_general_responder.py` — 4 async web search router tests

### Commits Verified
- [x] f483bcc — sliding window + dependency changes
- [x] 5865330 — RED phase failing tests
- [x] ca69501 — GREEN phase LLM router implementation
