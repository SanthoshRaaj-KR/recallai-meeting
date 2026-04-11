---
phase: 02-meeting-transcript-access-with-summarization-and-opinion-generation
plan: "01"
subsystem: meeting-responder
tags: [openai, asyncio, meeting-transcript, tts, summarization, opinion]
dependency_graph:
  requires:
    - confluence_logic/general_responder.py (structural pattern reference)
    - confluence_logic/jarvis_agentic.py (transcript_log schema: participant/text/timestamp)
  provides:
    - confluence_logic/meeting_responder.py (summarize_meeting, generate_opinion)
  affects:
    - Phase 02 Plan 02 (classifier routing will call these functions)
tech_stack:
  added:
    - openai (via asyncio.to_thread, same lazy client pattern as general_responder)
  patterns:
    - TDD red-green cycle for async functions
    - Lazy OpenAI client singleton (_get_client())
    - asyncio.to_thread for non-blocking LLM API calls
    - env-configurable token caps and model selection
key_files:
  created:
    - confluence_logic/meeting_responder.py
    - tests/test_meeting_responder.py
  modified: []
decisions:
  - D-11: JARVIS_SUMMARY_MAX_TOKENS defaults to 400 — enough for multi-topic summaries without verbosity
  - D-12: JARVIS_OPINION_MAX_TOKENS defaults to 200 — constrains to 2-4 sentences as prompted
  - D-14: MEETING_RESPONDER_MODEL reads JARVIS_GENERAL_MODEL — shares model config with general responder
  - D-07: Empty transcript fallback is "I haven't heard anything in the meeting yet." — consistent for both functions
  - D-09: Opinion system prompt mandates grounding phrase opener (Based on what I heard / From the discussion / Given what the team discussed)
  - D-08: Opinion prompt instructs Jarvis to pick a side — no hedging
  - D-13: No markdown/bullet points in system prompts — TTS-safe spoken output
metrics:
  duration: ~4 min
  completed: "2026-04-11"
  tasks: 1/1
  files: 2
---

# Phase 02 Plan 01: Meeting Responder Module Summary

**One-liner:** async meeting_responder.py with summarize_meeting() and generate_opinion() using lazy OpenAI client, env-configurable token caps (400/200), and mandatory grounding-phrase opinion opener.

## What Was Built

Created `confluence_logic/meeting_responder.py` — the meeting transcript handler module that Plan 02-02 will call from classifier routing. The module follows the `general_responder.py` pattern exactly:

- Lazy `_get_client()` singleton for OpenAI
- `asyncio.to_thread` wrapping so LLM calls are non-blocking
- Env-var-driven model and token caps
- Error fallback strings for TTS playback
- TTS-safe system prompts (no markdown, no bullet points)

Two exported async functions:

**`summarize_meeting(transcript_log)`** — Formats transcript as "Participant: text" lines, truncates to 8000 chars from the end (most recent), sends to OpenAI with a "summarize aloud" system prompt at temperature=0.5. Returns the empty-transcript fallback for empty input.

**`generate_opinion(transcript_log, query="")`** — Same transcript formatting and truncation. System prompt mandates a grounding opener phrase and a confident, non-hedging opinion. Optional `query` argument adds the user's specific question to the user message. Temperature=0.7 for more dynamic responses.

## Tests Written (TDD)

`tests/test_meeting_responder.py` — 12 tests covering:
- Empty transcript fallback (both functions)
- Env var default values (400, 200, gpt-4o-mini)
- API mock: correct token cap passed per function
- Grounding phrase requirement validated against mock response
- `query` argument inclusion in API call messages
- API error → error fallback string
- Source-level check that `asyncio.to_thread` appears at least twice

## Deviations from Plan

None — plan executed exactly as written. The implementation file matches the code block in `<action>` verbatim.

## Known Stubs

None. Both functions are fully implemented with real OpenAI API calls. Routing to call these functions is deferred to Plan 02-02 (classifier extension).

## Self-Check: PASSED

Files verified:
- confluence_logic/meeting_responder.py — exists, importable, syntax valid
- tests/test_meeting_responder.py — exists, 12/12 tests passing

Commits verified:
- 585354f — test(02-01): add failing tests (RED)
- 286fac5 — feat(02-01): implement meeting_responder (GREEN)
