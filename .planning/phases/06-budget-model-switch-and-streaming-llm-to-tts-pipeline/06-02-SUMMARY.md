---
phase: 06-budget-model-switch-and-streaming-llm-to-tts-pipeline
plan: "02"
subsystem: tts-pipeline
tags: [tts, streaming, sentence-splitting, asyncio, performance]
dependency_graph:
  requires: []
  provides: [_split_sentences, speak_chunked]
  affects: [jarvis.py]
tech_stack:
  added: []
  patterns: [asyncio.to_thread, TDD, sentence-boundary-detection]
key_files:
  created: [tests/test_speak_chunked.py]
  modified: [jarvis.py]
decisions:
  - "_ABBREVS frozenset used to skip known abbreviation periods (Dr, Mr, etc.) — prevents false sentence splits"
  - "finditer-based approach chosen over re.split lookbehind — variable-width lookbehind not supported in Python re, finditer is more flexible"
  - "_SENTENCE_END matches .!? followed by whitespace; abbreviation check filters out non-boundary periods"
metrics:
  duration: "2 minutes"
  completed: "2026-04-04T16:37:31Z"
  tasks_completed: 1
  files_changed: 2
---

# Phase 06 Plan 02: Sentence Splitter and speak_chunked Pipeline Summary

**One-liner:** Abbreviation-aware sentence splitter + sequential asyncio.to_thread TTS pipeline using finditer-based boundary detection with known abbreviation frozenset.

## What Was Built

Two new utilities in `jarvis.py` that enable the streaming TTS pipeline:

1. `_split_sentences(text)` — splits answer text into sentence-level chunks at `.!?` boundaries, using a `finditer`-based approach with a `_ABBREVS` frozenset to skip abbreviations like "Dr.", "Mr.", "Mrs.", etc.

2. `speak_chunked(text, bot_id)` — async coroutine that calls `speak()` via `asyncio.to_thread` for each sentence chunk sequentially, keeping blocking gTTS/HTTP calls off the event loop.

## Tasks Completed

| Task | Description | Commit | Files |
|------|-------------|--------|-------|
| RED | Failing tests for _split_sentences and speak_chunked | 0a6bf45 | tests/test_speak_chunked.py |
| GREEN | Implement both functions in jarvis.py | 22af767 | jarvis.py |

## Test Results

- 10 new tests added in `tests/test_speak_chunked.py` — all pass
- 198 total tests pass (was 188 before this plan), 1 skipped (unchanged)

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed abbreviation detection failing with initial `re.split` lookbehind approach**

- **Found during:** GREEN phase implementation
- **Issue:** Plan specified `_SENTENCE_END = re.compile(r'(?<=[.!?])\s+')` but this splits "Dr. Smith" because Python `re` does not support variable-width lookbehinds, and even fixed-width lookbehind `(?<=[.!?])\s+(?=[A-Z])` incorrectly split "Dr. Smith" since "Smith" starts with uppercase "S"
- **Fix:** Replaced `re.split` approach with `re.finditer` iterating over `[.!?]\s+` match objects, checking each match against an `_ABBREVS` frozenset to skip known abbreviation boundaries
- **Files modified:** jarvis.py
- **Commit:** 22af767

## Known Stubs

None.

## Self-Check

- `jarvis.py` modified: FOUND
- `tests/test_speak_chunked.py` created: FOUND
- Commit 0a6bf45 (RED tests): FOUND
- Commit 22af767 (GREEN implementation): FOUND
- 10 tests pass: VERIFIED
- All 198 tests pass: VERIFIED
