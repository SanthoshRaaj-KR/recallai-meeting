---
phase: 06-pipeline-intelligence-and-safety-improvements
plan: 02
subsystem: web-search, garbled-query-recovery
tags: [tavily, web-search, garbled-detection, safety, pipeline]
dependency_graph:
  requires: []
  provides: [TAVILY-01, GARBLED-01]
  affects: [confluence_logic/general_responder.py, confluence_logic/jarvis_agentic.py]
tech_stack:
  added: [Tavily REST API (api.tavily.com)]
  patterns: [env-var-gated feature, heuristic text detection, early-exit guard]
key_files:
  created: []
  modified:
    - confluence_logic/general_responder.py
    - confluence_logic/jarvis_agentic.py
decisions:
  - "TAVILY-01: Tavily replaces DuckDuckGo — DuckDuckGo instant answer API returns empty for most specific queries; Tavily provides AI-synthesized answers with include_answer=True"
  - "GARBLED-01: Heuristic-only garbled detection (no LLM) — zero latency, zero cost; catches transcription noise before any LLM/classification call"
  - "GARBLED-01: Early exit in _debounced_dispatch saves classification LLM call on garbled input"
  - "GARBLED-01: JARVIS_GARBLED_RECOVERY_ENABLED defaults to true — opt-out via env var"
metrics:
  duration: ~6 min
  completed: "2026-04-14"
  tasks: 2/2
  files: 2
---

# Phase 06 Plan 02: Tavily Web Search and Garbled Query Recovery Summary

Upgraded web search from DuckDuckGo Instant Answer API to Tavily for AI-synthesized results, and added heuristic garbled/unintelligible query detection so Jarvis asks users to repeat instead of wasting LLM calls on transcription noise.

## Tasks Completed

### Task 1: Replace DuckDuckGo with Tavily web search

**Commit:** 3e86730

**Changes to `confluence_logic/general_responder.py`:**
- Added `TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "").strip()` module-level env var
- Replaced `_quick_web_search` from DuckDuckGo GET to Tavily POST at `https://api.tavily.com/search`
- Tavily request includes `search_depth: "basic"`, `max_results: 3`, `include_answer: True`
- Prefers AI-synthesized `answer` field (up to 800 chars), falls back to concatenated result snippets
- API key gate: returns `""` silently if `TAVILY_API_KEY` is unset
- Added `force_web_search: bool = False` parameter to `answer_general_question` signature
- Web search conditional updated to `if force_web_search or await _needs_web_search(question):`

### Task 2: Garbled query detection and re-prompt

**Commit:** ef4d451

**Changes to `confluence_logic/jarvis_agentic.py`:**
- Added `JARVIS_GARBLED_RECOVERY_ENABLED` env var (default `"true"`)
- Added `_is_garbled_query(text: str) -> bool` heuristic function with four checks:
  - Empty or too-short text (<= 2 chars)
  - Low alphabetic ratio (< 40% alpha chars for text > 3 chars)
  - All words are single characters (e.g., "a b c d")
  - Excessive character repetition (e.g., "aaaaaa", set size <= 2 for len > 4)
- Added garbled check in `handle_spoken_request` after status-query check, before intent classification
- Added garbled check in `_debounced_dispatch` before calling `handle_spoken_request`
- Both paths respond with `"Sorry, I didn't catch that. Could you say that again?"`

## Deviations from Plan

None — plan executed exactly as written.

## Verification Results

- `from confluence_logic.general_responder import answer_general_question; from confluence_logic.jarvis_agentic import _is_garbled_query` imports successfully
- `grep -c "tavily" confluence_logic/general_responder.py` returns 4 (>= 3)
- `grep -c "_is_garbled_query" confluence_logic/jarvis_agentic.py` returns 3 (>= 3)
- `_is_garbled_query('a')` returns `True`
- `_is_garbled_query('!!!')` returns `True`
- `_is_garbled_query('what is the weather today')` returns `False`
- `force_web_search` in `answer_general_question` signature confirmed

## Known Stubs

None — both features are fully wired with real API calls and functional heuristics.

## Self-Check: PASSED

- `confluence_logic/general_responder.py` — modified, contains `api.tavily.com/search` and `force_web_search`
- `confluence_logic/jarvis_agentic.py` — modified, contains `_is_garbled_query` and `JARVIS_GARBLED_RECOVERY_ENABLED`
- Commit 3e86730 — Task 1 (Tavily web search)
- Commit ef4d451 — Task 2 (garbled query detection)
