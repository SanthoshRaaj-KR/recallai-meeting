---
phase: 01-intelligent-question-classification-and-conversational-response
plan: "03"
subsystem: jarvis-agentic
tags: [clarification, general-questions, no-wake-word, conversation-state]
dependency_graph:
  requires: ["01-01", "01-02"]
  provides: ["CLARIFY-01"]
  affects: ["confluence_logic/jarvis_agentic.py"]
tech_stack:
  added: []
  patterns: ["pending state with expiry timeout", "no-wake-word listening window", "enriched context re-ask"]
key_files:
  created: []
  modified:
    - confluence_logic/jarvis_agentic.py
decisions:
  - "15-second clarification timeout default (JARVIS_GENERAL_CLARIFICATION_TIMEOUT env var) — short enough to feel conversational, long enough for natural response"
  - "Clarification state cleared immediately at handler start — prevents stale state if handler errors mid-way"
  - "Enriched history built from original_query + clarification_question + user_answer — gives LLM full context for final answer"
  - "Only one clarification exchange per general question (no chaining) — deferred per CONTEXT.md"
metrics:
  duration: "~4 min"
  completed: "2026-04-11"
  tasks: 1/1
  files: 1
---

# Phase 01 Plan 03: General Question Clarification Flow Summary

**One-liner:** 15-second no-wake-word clarification window for general questions using pending_general_clarification state and enriched context re-ask.

## What Was Built

When Jarvis answers a general question with a clarifying question (detected by `_looks_like_clarification_prompt`), it now enters a no-wake-word listening mode. The user can respond naturally without saying "Hey Jarvis". Their response is routed to a dedicated handler that builds enriched conversation context (original question + clarification exchange) and calls `answer_general_question` again for a final answer.

**Key additions to `confluence_logic/jarvis_agentic.py`:**
- `JARVIS_GENERAL_CLARIFICATION_TIMEOUT` constant (default 15.0s, env-configurable)
- `"pending_general_clarification": None` key in `meeting_state` dict
- `_handle_general_question()` expanded: sets `pending_general_clarification` when LLM answer is itself a clarifying question
- `_handle_general_clarification_answer()`: new handler that clears state, builds enriched history, re-calls `answer_general_question`
- `process_transcript_event()`: checks `pending_general_clarification` after existing `pending_clarification` and before `jarvis_listening`, with expiry timeout logic
- `handle_spoken_request()`: routes to `_handle_general_clarification_answer` when pending state is active and not expired

## Deviations from Plan

None - plan executed exactly as written.

## Tasks Completed

| Task | Name | Commit | Files |
|------|------|--------|-------|
| 1 | Add general question clarification state to pipeline | 9b03cbe | confluence_logic/jarvis_agentic.py |

## Known Stubs

None — all new state is fully wired into the pipeline.

## Self-Check: PASSED

- [x] `confluence_logic/jarvis_agentic.py` modified and committed (9b03cbe)
- [x] `pending_general_clarification` appears 6 times in file (meets >= 5 requirement)
- [x] `JARVIS_GENERAL_CLARIFICATION_TIMEOUT` appears 3 times (meets >= 2 requirement)
- [x] `pending_clarification` (confluence flow) still present 13 times (untouched)
- [x] `_execute_editor_task` still present (confluence flow untouched)
- [x] `ast.parse()` passes — valid Python
