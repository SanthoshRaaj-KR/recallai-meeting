---
phase: 06-budget-model-switch-and-streaming-llm-to-tts-pipeline
plan: "03"
subsystem: streaming-tts-wiring
tags: [tts, streaming, openai, asyncio, performance, latency]
dependency_graph:
  requires: [speak_chunked, _split_sentences, client, OPENAI_MODEL]
  provides: [_stream_llm_and_speak]
  affects: [jarvis.py, handle_query, _handle_memory_query]
tech_stack:
  added: []
  patterns: [asyncio.to_thread, streaming-llm, sentence-boundary-tts, TDD]
key_files:
  created: [tests/test_streaming_tts.py]
  modified: [jarvis.py]
decisions:
  - "_run_stream inner function collected entirely via asyncio.to_thread before token processing — avoids async/thread boundary complexity while still freeing event loop during generation"
  - "Fallback to speak_chunked(answer, bot_id) on _stream_llm_and_speak exception — guarantees voice output even if streaming fails"
  - "Memory query paths (both direct-answer and disambiguation) use speak_chunked() — replaces asyncio.to_thread(speak, ...) to enable sentence-level pipelining"
metrics:
  duration: "4 minutes"
  completed: "2026-04-04T16:41:40Z"
  tasks_completed: 1
  files_changed: 2
---

# Phase 06 Plan 03: Streaming TTS Pipeline Wiring Summary

**One-liner:** _stream_llm_and_speak() coroutine wires OpenAI streaming + sentence-level asyncio.to_thread TTS into handle_query's final-answer and memory-query paths, cutting perceived latency from (full_generation + full_tts) to (first_sentence_generation + first_sentence_tts).

## What Was Built

Two code changes to `jarvis.py` that enable the full streaming TTS pipeline:

1. `_stream_llm_and_speak(messages, bot_id)` — new async coroutine inserted after `speak_chunked`. Runs `client.chat.completions.create(stream=True)` inside `asyncio.to_thread(_run_stream)` to collect tokens without blocking the event loop. Processes token buffer for sentence boundaries (`.!?\s` pattern) and calls `speak()` via `asyncio.to_thread` as each sentence completes. Remaining buffer spoken after stream ends.

2. `handle_query()` call sites updated:
   - **Final answer round** (tool-calling loop `else` branch): replaced `asyncio.to_thread(speak, answer, bot_id)` with `await _stream_llm_and_speak(messages, bot_id)` plus `speak_chunked` fallback on exception.
   - **Memory query path** (`result.needs_disambiguation == False`): replaced `asyncio.to_thread(speak, result.answer, bot_id)` with `await speak_chunked(result.answer, bot_id)`.
   - **Disambiguation path**: replaced `asyncio.to_thread(speak, spoken, bot_id)` with `await speak_chunked(spoken, bot_id)`.

## Tasks Completed

| Task | Description | Commit | Files |
|------|-------------|--------|-------|
| RED | Failing tests for _stream_llm_and_speak and handle_query wiring | 5d4c741 | tests/test_streaming_tts.py |
| GREEN | Implement _stream_llm_and_speak + update call sites in jarvis.py | e440f7c | jarvis.py |

## Test Results

- 8 new tests added in `tests/test_streaming_tts.py` — all pass
- 206 total tests pass (was 198 before this plan), 1 skipped (unchanged)

## Deviations from Plan

None - plan executed exactly as written.

## Known Stubs

None.

## Self-Check: PASSED

- `jarvis.py` modified: FOUND
- `tests/test_streaming_tts.py` created: FOUND
- Commit 5d4c741 (RED tests): FOUND
- Commit e440f7c (GREEN implementation): FOUND
- 8 new tests pass: VERIFIED
- All 206 tests pass: VERIFIED
- `asyncio.to_thread(speak, ...)` in handle_query direct: NOT PRESENT (only in speak_chunked/_stream_llm_and_speak bodies)
- `_stream_llm_and_speak` called in handle_query final answer branch: FOUND (line 654)
- `speak_chunked` called in handle_query memory paths (x2): FOUND (lines 599, 606)
