---
quick_task: 260411-le0
title: Smart time-filler and no-wake-word persistence after clarification cycles
subsystem: jarvis_agentic
tags: [ux, tts, filler, clarification, no-wake-word, conversational-flow]
dependency_graph:
  requires: []
  provides: [filler-phrases-before-slow-ops, clarification-re-arm]
  affects: [_handle_general_question, _handle_general_clarification_answer, _handle_summary_clarification_answer, _handle_meeting_summary, _handle_meeting_opinion]
tech_stack:
  added: []
  patterns: [random.choice for filler variation, re-arm pending state on follow-up question]
key_files:
  modified:
    - confluence_logic/jarvis_agentic.py
decisions:
  - JARVIS_FILLER_PHRASES uses 4 short phrases to vary audio feedback and avoid repetition
  - _speak_filler uses allow_stale=True so it works in fire-and-forget async tasks
  - Filler is wired BEFORE slow calls (LLM/summarize/opinion) so user hears audio immediately
  - No filler before clarifying questions (else branch keeps short "Sure!" ack instead)
  - Re-arm uses fresh expires_at so the no-wake-word window resets on each follow-up
  - Clear-at-start pattern preserved — state cleared before try block prevents stale state on error
metrics:
  duration: ~5 min
  completed_date: "2026-04-11T09:58:40Z"
  tasks: 2/2
  files: 1
---

# Quick Task 260411-le0: Smart time-filler and no-wake-word persistence after clarification cycles

**One-liner:** Random filler phrase plays immediately before slow LLM/summary/opinion calls; clarification handlers re-arm no-wake-word mode when the LLM's follow-up response is itself a question.

## What Was Built

Two focused UX fixes to eliminate awkward silences and forced wake-word re-invocations in mid-conversation flows.

### Task 1: JARVIS_FILLER_PHRASES + _speak_filler()

Added a module-level constant `JARVIS_FILLER_PHRASES` with 4 short phrases after `_INSTANT_ACKS`, and a new `_speak_filler()` async helper that picks randomly and calls `_speak_guarded(allow_stale=True)`.

Wired into 4 slow-operation sites:
- `_handle_general_question()` — before `answer_general_question()`
- `_handle_summary_clarification_answer()` — before `summarize_meeting()`
- `_handle_meeting_summary()` specified branch — before `summarize_meeting()`
- `_handle_meeting_opinion()` — before `generate_opinion()`

Also restructured `_handle_meeting_summary()`: removed the unconditional hardcoded `"Sure! Just give me a sec."` before the if/else. The specified branch now calls `_speak_filler()`, while the else (clarifying question) branch calls `_speak_guarded("Sure!")` followed by the clarifying question — preserving immediate audio without filler before the question.

### Task 2: Re-arm no-wake-word state on follow-up question

Both clarification handlers now check `_looks_like_clarification_prompt()` on the LLM's response after speaking it:

- `_handle_general_clarification_answer()`: if `final_answer` ends with `?` and looks like a question, re-arms `meeting_state["pending_general_clarification"]` with fresh context and a new `expires_at`
- `_handle_summary_clarification_answer()`: if `answer` (from summarize_meeting) looks like a question, re-arms `meeting_state["pending_summary_clarification"]` with fresh `expires_at`

Both handlers preserve the existing clear-at-start pattern (`= None` before the try block), ensuring stale state is never left if the handler errors mid-execution.

## Commits

| Task | Commit | Message |
|------|--------|---------|
| 1    | de9860e | feat(260411-le0): add JARVIS_FILLER_PHRASES and _speak_filler() before slow ops |
| 2    | 3c19f1c | feat(260411-le0): re-arm no-wake-word state when clarification answer is a question |

## Deviations from Plan

None — plan executed exactly as written.

## Known Stubs

None.

## Self-Check: PASSED

- confluence_logic/jarvis_agentic.py: exists and syntax valid
- de9860e: confirmed in git log
- 3c19f1c: confirmed in git log
- All 6 plan verification criteria confirmed via automated checks
