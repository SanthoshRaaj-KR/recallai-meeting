---
phase: 05-orchestration-answer-path
plan: 05-03
subsystem: slack-integration
tags: [slack, slash-command, disambiguation, orchestration, tdd]
dependency_graph:
  requires: [05-01, 05-02]
  provides: [jarvis-ask-command, jarvis-disambiguation, jarvis-memory-routing]
  affects: [jarvis.py]
tech_stack:
  added: []
  patterns: [standalone-handler-functions-for-testability, module-level-singletons]
key_files:
  created:
    - tests/test_slack_ask.py
  modified:
    - jarvis.py
decisions:
  - "_handle_ask and _handle_message_disambig defined at module level (not inside if-block) so they are importable in tests regardless of SLACK_BOT_TOKEN being set"
  - "_handle_memory_query returns OrchestratorResult with 'not configured' answer when orchestrator is None — no exception raised — callers need no try/except for the unconfigured case"
  - "slack_app.command('/ask')(_handle_ask) pattern used instead of decorator inside if-block — keeps functions importable"
metrics:
  duration: "3 minutes"
  completed_date: "2026-04-04"
  tasks_completed: 2
  files_changed: 2
---

# Phase 05 Plan 03: Slack /ask Command + Disambiguation Wiring Summary

## One-liner

Wired OrchestratorAgent into jarvis.py as the /ask Slack slash command with full disambiguation flow and memory-query routing in handle_query().

## What Was Built

- **`_is_memory_query(query)`** — heuristic function that detects memory/history questions to route through the orchestrator rather than the weather/live-transcript tool loop
- **`_handle_memory_query(query, user_id, channel_id)`** — delegates to OrchestratorAgent.run() or returns a static "not configured" OrchestratorResult when Pinecone is absent
- **`_handle_ask(ack, say, client, command)`** — Slack `/ask` slash command: acks immediately, routes through orchestrator, posts numbered disambiguation list or final answer
- **`_handle_message_disambig(message, say, client)`** — resolves disambiguation by matching user digit reply to `_pending_disambig[user_id]`, re-runs scoped query, clears pending state
- **`_pending_disambig`** — module-level dict keyed by user_id holding pending disambiguation option lists
- **`orchestrator` singleton** — initialized with RetrieverAgent + DateResolutionAgent + AnswerAgent when `pinecone_client is not None`
- **`handle_query()` update** — memory routing check inserted before existing weather/tool loop; falls through to tool loop on error
- **18 TDD tests** across 4 test classes covering all behaviors

## Tasks Completed

| Task | Description | Commit | Files |
|------|-------------|--------|-------|
| 1 | Write failing tests (RED) | 14b48d3 | tests/test_slack_ask.py |
| 2 | Implement jarvis.py changes (GREEN) | c920202 | jarvis.py |

## Test Results

- New tests: 18/18 passed
- Existing integration tests: 15/15 passed
- Full suite: 181 passed, 1 skipped

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Handler functions moved to module level for test importability**

- **Found during:** Task 2 - initial implementation put `_handle_ask` and `_handle_message_disambig` inside `if slack_app is not None:` block using decorator syntax
- **Issue:** When `SLACK_BOT_TOKEN` is not set (test environment), the `if` block never executes, making the functions undefined and causing `ImportError` in tests
- **Fix:** Defined both handler functions at module level before the `if slack_app is not None:` block; registered them with Bolt using `slack_app.command("/ask")(_handle_ask)` and `slack_app.event("message")(_handle_message_disambig)` patterns
- **Files modified:** jarvis.py
- **Commit:** c920202

## Requirements Satisfied

- RETR-03: Disambiguation list posted when date matches >1 meeting; user selects via number; bot re-runs scoped query
- QUERY-01: /ask returns precise answer with meeting attribution via AnswerAgent
- QUERY-02: /ask resolves date expressions via DateResolutionAgent
- QUERY-03: Cross-meeting queries handled via AnswerAgent cross_meeting type
- QUERY-04: Action item queries attributed with meeting context

## Self-Check: PASSED
