---
phase: 02-meeting-transcript-access-with-summarization-and-opinion-generation
plan: "02"
subsystem: classifier-routing
tags: [openai, asyncio, classifier, routing, intent-classification, meeting-transcript]
dependency_graph:
  requires:
    - confluence_logic/meeting_responder.py (summarize_meeting, generate_opinion — from Plan 02-01)
    - confluence_logic/classifier.py (classify_intent — extended here)
    - confluence_logic/jarvis_agentic.py (handle_spoken_request — routing extended here)
  provides:
    - 4-intent classify_intent() in confluence_logic/classifier.py
    - _handle_meeting_summary() and _handle_meeting_opinion() in confluence_logic/jarvis_agentic.py
    - Routing branches for meeting_summary and meeting_opinion in handle_spoken_request()
  affects:
    - Full intent routing pipeline: user speech -> classify_intent -> handler -> meeting_responder -> _speak_guarded
tech_stack:
  added: []
  patterns:
    - Fast-path heuristic classification (keyword sets + phrase matching)
    - asyncio.create_task() fire-and-forget handler dispatch
    - Defensive transcript copy (list(meeting_state["transcript_log"])) before await
    - 4-intent LLM prompt with single-token response constraint
key_files:
  created: []
  modified:
    - confluence_logic/classifier.py
    - confluence_logic/jarvis_agentic.py
decisions:
  - D-04: meeting_summary and meeting_opinion branches placed after general branch but before Confluence pipeline — matches design intent of routing before Confluence queue
  - Fast-path heuristics already present in classifier.py; LLM prompt and validation guard updated to 4-intent
  - transcript_log copied defensively with list() before passing to async meeting_responder — avoids mutation during await
metrics:
  duration: ~5 min
  completed: "2026-04-11"
  tasks: 2/2
  files: 2
---

# Phase 02 Plan 02: Classifier Routing Extension Summary

**One-liner:** Extended classifier.py to 4-intent LLM fallback (meeting_summary/meeting_opinion added) and wired _handle_meeting_summary/_handle_meeting_opinion handler functions in jarvis_agentic.py via asyncio.create_task() routing branches.

## What Was Built

### Task 1: Extended classifier.py

`confluence_logic/classifier.py` already had fast-path heuristics (_SUMMARY_TRIGGERS, _SUMMARY_PHRASES, _OPINION_TRIGGERS, _OPINION_PHRASES) from a prior partial implementation. This plan completed the remaining changes:

- Updated the LLM fallback system_prompt to describe all four intents: 'confluence', 'general', 'meeting_summary', 'meeting_opinion' — instructs the model to respond with ONLY one of these four words
- Updated the valid-result guard from `result in ("confluence", "general")` to `result in ("confluence", "general", "meeting_summary", "meeting_opinion")`
- Updated `classify_intent()` docstring to reflect 4-way return

The fast-path catches high-confidence meeting phrases ("catch me up", "summarize the meeting", "what do you think", "what's your take") without LLM call. Ambiguous cases fall through to the LLM with the updated prompt.

### Task 2: Wired routing in jarvis_agentic.py

`confluence_logic/jarvis_agentic.py` received three additions:

**Import:** `from .meeting_responder import summarize_meeting, generate_opinion` added after the existing `general_responder` import.

**Two private handler functions** (after `_handle_general_clarification_answer`, before `handle_spoken_request`):

- `_handle_meeting_summary(bot_id)` — reads `meeting_state["transcript_log"]` defensively with `list()`, calls `summarize_meeting()`, speaks result via `_speak_guarded()`
- `_handle_meeting_opinion(query, bot_id)` — same pattern, calls `generate_opinion(transcript_log, query=query)`

**Routing branches in `handle_spoken_request()`** — inserted after the `intent == "general"` branch, before the `pending_general_clarification` check and Confluence pipeline:

```python
if intent == "meeting_summary":
    asyncio.create_task(_handle_meeting_summary(bot_id))
    return

if intent == "meeting_opinion":
    asyncio.create_task(_handle_meeting_opinion(spoken_query, bot_id))
    return
```

Both handlers use `asyncio.create_task()` — fire-and-forget, non-blocking, matching the pattern established for `_handle_general_question`.

## Deviations from Plan

### Auto-fixed Issues

None.

### Pre-existing Partial Implementation

The fast-path heuristics (_SUMMARY_TRIGGERS, _SUMMARY_PHRASES, _OPINION_TRIGGERS, _OPINION_PHRASES, and corresponding _fast_classify() checks) were already present in classifier.py from a prior implementation. The plan description treats these as new additions, but they were already present. This task only updated the LLM prompt and validation guard.

This is not a deviation — the plan's acceptance criteria and done criteria were all met. The pre-existing heuristics are correct and match the plan's specified values exactly.

## Known Stubs

None. Both handler functions are fully wired to real meeting_responder functions. The full pipeline is active: user speech -> classify_intent -> asyncio.create_task -> handler -> meeting_responder (LLM) -> _speak_guarded -> TTS.

## Self-Check: PASSED

Files verified:
- confluence_logic/classifier.py — exists, AST-valid, contains meeting_summary (7x), meeting_opinion (7x), "ONLY one of these four", 4-intent valid-result guard
- confluence_logic/jarvis_agentic.py — exists, AST-valid, contains import, both handler functions, both routing branches, both create_task calls, defensive list() transcript copies

Commits verified:
- 10d6e5a — feat(02-02): extend classifier.py with 4-intent LLM prompt and validation guard
- 7488d3b — feat(02-02): wire meeting_summary and meeting_opinion routing in jarvis_agentic.py
