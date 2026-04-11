---
phase: quick
plan: 260411-vfq
subsystem: jarvis-conversation
tags: [conversation-memory, cross-handler, follow-up, opinion, summary]
dependency_graph:
  requires: []
  provides: [cross-handler-memory, followup-context-injection]
  affects: [confluence_logic/classifier.py, confluence_logic/jarvis_agentic.py]
tech_stack:
  added: []
  patterns: [shared-state-slot, word-boundary-regex-guard]
key_files:
  created: []
  modified:
    - confluence_logic/classifier.py
    - confluence_logic/jarvis_agentic.py
decisions:
  - "_is_followup uses re.search with \\b for short words 'it'/'that' to prevent false positives on words like 'iterate'"
  - "last_jarvis_response persists until next opinion/summary — not cleared after use — allows chained follow-ups"
  - "cross_context prepended to conversation_history (not appended) so LLM sees meeting context before general history"
metrics:
  duration: ~4 min
  completed: 2026-04-11
  tasks: 2
  files: 2
---

# Quick Task 260411-vfq: Cross-handler conversation memory

**One-liner:** Shared `last_jarvis_response` slot in `meeting_state` with `_is_followup()` word-boundary helper enables follow-up queries like "explain that simpler" to carry prior opinion/summary context into the general handler.

## What Was Done

### Task 1: Opinion triggers and state/helper additions

- Added `"feel"` and `"thoughts"` to `_OPINION_TRIGGERS` frozenset in `classifier.py` so queries like "what do you feel about X" fast-path to `meeting_opinion` intent without LLM classifier overhead.
- Added `"last_jarvis_response": None` to `meeting_state` dict — holds `{"intent": str, "query": str, "answer": str}` after any opinion or summary response.
- Added `_is_followup(query: str) -> bool` helper that uses `re.search(r"\bit\b")` and `re.search(r"\bthat\b")` for word-boundary matching, plus simple `in` substring check for longer unambiguous follow-up words.

### Task 2: Cross-handler memory wiring

- `_handle_meeting_opinion`: after `answer = await opinion_task`, writes `meeting_state["last_jarvis_response"]` with intent=`"meeting_opinion"`.
- `_handle_meeting_summary`: in the `if specified:` branch only (where an actual answer is generated), writes `meeting_state["last_jarvis_response"]` with intent=`"meeting_summary"` after `answer = await summary_task`.
- `_handle_general_question`: after `_format_general_history()`, checks `prior = meeting_state.get("last_jarvis_response")` and if truthy and `_is_followup(query)`, prepends `"User: {prior['query']}\nAssistant: {prior['answer']}"` to `conversation_history` before calling `answer_general_question`.

## Commits

| Task | Commit | Description |
|------|--------|-------------|
| 1    | 5041ddb | feat(quick-260411-vfq): add opinion triggers and cross-handler memory state |
| 2    | 6ef2fe3 | feat(quick-260411-vfq): wire cross-handler conversation memory |

## Deviations from Plan

None - plan executed exactly as written.

## Known Stubs

None.

## Self-Check: PASSED

- `confluence_logic/classifier.py` modified and verified: `feel` and `thoughts` in `_OPINION_TRIGGERS`
- `confluence_logic/jarvis_agentic.py` modified and verified: `last_jarvis_response` in `meeting_state`, `_is_followup` defined, all three handlers reference `last_jarvis_response`
- Both commits exist: 5041ddb, 6ef2fe3
