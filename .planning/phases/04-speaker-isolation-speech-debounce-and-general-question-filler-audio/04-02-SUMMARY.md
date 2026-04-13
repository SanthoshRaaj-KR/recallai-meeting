---
phase: 04-speaker-isolation-speech-debounce-and-general-question-filler-audio
plan: 02
subsystem: general-question-handler
tags: [filler-audio, contextual-ack, general-question, FILLER-02, ux]
dependency_graph:
  requires: [04-01-PLAN.md]
  provides: [FILLER-02]
  affects: [_handle_general_question, _generate_contextual_gap_filler, _speak_guarded]
tech_stack:
  added: []
  patterns: [contextual gap filler before LLM answer, allow_stale=True speak guard bypass]
key_files:
  created: []
  modified:
    - confluence_logic/jarvis_agentic.py
    - confluence_logic/tests/test_jarvis_agentic.py
decisions:
  - "Filler awaited directly (not via create_task) in _handle_general_question — no concurrent work to overlap, filler IS the gap"
  - "allow_stale=True used for filler speak — prevents generation guard from suppressing it while main LLM call is in-flight"
  - "force_web_search parameter added to _handle_general_question — enables web_search intent routing from handle_spoken_request"
metrics:
  duration: ~5 min
  completed_date: "2026-04-13"
  tasks: 2/2
  files: 2
---

# Phase 04 Plan 02: General Question Filler Audio Summary

**One-liner:** Contextual gap filler spoken via _generate_contextual_gap_filler before answer_general_question in _handle_general_question, eliminating the 3-5 second silence between wake detection and response delivery.

## What Was Built

Added FILLER-02 implementation to `_handle_general_question` in `jarvis_agentic.py`:

1. **Filler call before answer** — Two lines inserted immediately before `answer = await answer_general_question(...)`:
   ```python
   # Speak contextual filler immediately while LLM generates the answer (FILLER-02, D-10)
   filler = await _generate_contextual_gap_filler(query)
   await _speak_guarded(filler, bot_id, generation, allow_stale=True)
   ```

2. **force_web_search parameter** — `_handle_general_question` now accepts `force_web_search: bool = False`, passed through to `answer_general_question`. The `web_search` intent in `handle_spoken_request` routes to `_handle_general_question(spoken_query, bot_id, force_web_search=True)`.

3. **FILLER-02 test** — New test `test_handle_general_question_speaks_filler_before_answer` appended to `test_jarvis_agentic.py`. Uses call_order tracking to assert `filler_generated` appears before `answer_generated`. Mock signature includes `force_web_search=False` to match the updated function signature.

## Commits

| Task | Commit | Description |
|------|--------|-------------|
| 1 | 3977bbf | feat(04-02): add contextual gap filler to _handle_general_question (FILLER-02) |
| 2 | 07c9896 | test(04-02): add FILLER-02 test verifying filler precedes answer in _handle_general_question |

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed mock_answer signature mismatch in FILLER-02 test**
- **Found during:** Task 2 verification
- **Issue:** The FILLER-02 test's `mock_answer` had signature `(q, history, graph_context="")` but `_handle_general_question` passes `force_web_search=force_web_search` as a keyword argument, causing `TypeError: mock_answer() got an unexpected keyword argument 'force_web_search'`
- **Fix:** Added `force_web_search=False` to `mock_answer` signature
- **Files modified:** `confluence_logic/tests/test_jarvis_agentic.py`
- **Commit:** 07c9896

## Known Stubs

None. The filler call is fully wired — `_generate_contextual_gap_filler` is an existing LLM-backed function, and `_speak_guarded` is the production TTS pipeline.

## Test Results

- 1 new test: PASS (`test_handle_general_question_speaks_filler_before_answer`)
- Previously passing tests: all 18 still pass
- 8 pre-existing failures confirmed unrelated to this plan (same failures present before any changes in this plan — involve `SWITCH_ACK` attribute missing and assembly provider config mismatch)

## Self-Check: PASSED

Files confirmed present:
- `confluence_logic/jarvis_agentic.py` — modified with filler call + force_web_search param + web_search routing
- `confluence_logic/tests/test_jarvis_agentic.py` — modified with FILLER-02 test

Commits confirmed:
- `3977bbf` present in git log
- `07c9896` present in git log
