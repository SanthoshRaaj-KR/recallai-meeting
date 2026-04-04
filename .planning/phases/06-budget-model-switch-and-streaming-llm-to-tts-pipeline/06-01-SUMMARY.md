---
phase: 06-budget-model-switch-and-streaming-llm-to-tts-pipeline
plan: "01"
subsystem: testing
tags: [budget-model, regression-tests, gpt-4o-mini, audit]
dependency_graph:
  requires: []
  provides: [budget-model-regression-tests]
  affects: [tests/test_budget_model.py]
tech_stack:
  added: []
  patterns: [unittest.mock.patch for AsyncOpenAI isolation in tests]
key_files:
  created:
    - tests/test_budget_model.py
  modified: []
decisions:
  - OrchestratorAgent tests patch agents.orchestrator.AsyncOpenAI to prevent OPENAI_API_KEY requirement at test time
  - Jarvis OPENAI_MODEL test reads module constant directly instead of reloading (avoids fragile importlib.reload side effects)
metrics:
  duration: "1 minute"
  completed: "2026-04-04"
  tasks_completed: 1
  files_created: 1
  files_modified: 0
requirements_satisfied: [PERF-01]
---

# Phase 06 Plan 01: Budget Model Audit and Regression Tests Summary

## One-liner

Confirmed all 4 LLM-calling agents default to gpt-4o-mini; added 7-test regression suite to lock in the constraint permanently.

## What Was Built

Audited every OpenAI model call in the codebase (`jarvis.py`, `agents/summarizer.py`, `agents/answer_agent.py`, `agents/orchestrator.py`). Confirmed no bare `gpt-4o` usage anywhere — all agents already default to `gpt-4o-mini`. Created `tests/test_budget_model.py` with 7 regression tests covering:

- Default model for SummarizerAgent, AnswerAgent, OrchestratorAgent
- jarvis.OPENAI_MODEL env var defaults to "gpt-4o-mini"
- Model override propagation for all three agents (confirms the `model` param is wired through `__init__`)

## Tasks Completed

| Task | Name | Commit | Files |
|------|------|--------|-------|
| 1 | Audit LLM model usage and write regression tests | f7523f0 | tests/test_budget_model.py |

## Test Results

- `python -m pytest tests/test_budget_model.py` — 7 passed
- `python -m pytest tests/` — 197 passed, 1 skipped, 1 pre-existing failure (test_speak_chunked.py::test_abbreviation_not_split — unrelated to this plan, references `_split_sentences` not yet implemented in jarvis.py, targeted by plan 06-02)

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] OrchestratorAgent tests required AsyncOpenAI mock**
- **Found during:** Task 1 (TDD GREEN phase)
- **Issue:** `OrchestratorAgent.__init__` instantiates `AsyncOpenAI()` which immediately raises `OpenAIError` when `OPENAI_API_KEY` is unset in test environment
- **Fix:** Added `patch("agents.orchestrator.AsyncOpenAI")` context manager to both orchestrator tests
- **Files modified:** tests/test_budget_model.py
- **Commit:** f7523f0

## Known Stubs

None — this plan is audit-and-test only, no production code changes.

## Self-Check: PASSED

- tests/test_budget_model.py: FOUND
- Commit f7523f0: FOUND
