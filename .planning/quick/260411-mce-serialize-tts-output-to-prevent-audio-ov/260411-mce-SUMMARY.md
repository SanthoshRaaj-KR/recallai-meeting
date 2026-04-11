---
id: 260411-mce
type: quick
phase: quick
plan: 260411-mce
subsystem: tts-serialization
tags: [audio, tts, concurrency, lock, speak]
key-files:
  modified:
    - confluence_logic/jarvis_agentic.py
decisions:
  - "Hold output_lock for estimated playback duration after successful audio POST to prevent concurrent audio overlap"
  - "Use 2.5 words/sec for text-based duration estimate (min 0.5s) — conservative estimate for TTS pacing"
  - "Use 4000 bytes/sec for MP3 byte-based duration estimate (min 0.3s) — matches ~32 kbps encoding"
  - "Early-exit on generation change inside sleep loop — new wake-word/interruption is not blocked by previous playback wait"
metrics:
  duration: ~3 min
  completed: "2026-04-11"
  tasks: 2/2
  files: 1
---

# Quick Task 260411-mce: Serialize TTS Output to Prevent Audio Overlap

**One-liner:** Added post-POST lock holding with estimated playback duration in both guarded speak functions to prevent concurrent audio clip overlap.

## Problem

`_speak_guarded` and `_speak_cached_guarded` released `_output_lock` as soon as the HTTP POST response arrived from the Recall API, but audio was still playing on the client. A concurrent handler could acquire the lock immediately and send its own audio clip, causing both clips to play simultaneously.

## Solution

Both guarded speak functions now hold the lock for the estimated playback duration after a successful POST, then release it. An early-exit check on `generation != meeting_state["output_generation"]` inside the sleep loop allows user interruptions (new wake-word events) to cancel the wait without unnecessary delay.

## Tasks Completed

| Task | Name | Commit | Files |
|------|------|--------|-------|
| 1 | Add `_estimate_speech_duration` and update `_speak_guarded` | f786107 | confluence_logic/jarvis_agentic.py |
| 2 | Add `_estimate_cached_duration` and update `_speak_cached_guarded` | 2d00290 | confluence_logic/jarvis_agentic.py |

## Changes Made

### `_estimate_speech_duration(text: str) -> float`
- New helper added immediately before `_speak_guarded` (line ~553)
- Estimates duration from word count at 2.5 words/sec, minimum 0.5s

### `_speak_guarded` updated
- Captures `asyncio.to_thread(speak, ...)` result into `ok`
- On success, sleeps in 0.1s increments up to estimated duration
- Breaks early if `generation != meeting_state["output_generation"]`
- Returns `ok` instead of the direct coroutine result

### `_estimate_cached_duration(audio_bytes: bytes) -> float`
- New helper added immediately before `_speak_cached_guarded` (line ~451)
- Estimates duration from byte length at 4000 bytes/sec (~32 kbps), minimum 0.3s

### `_speak_cached_guarded` updated
- Captures `asyncio.to_thread(speak_cached_audio, ...)` result into `ok`
- On success, sleeps in 0.1s increments up to byte-based estimated duration
- Breaks early if `generation != meeting_state["output_generation"]`
- Returns `ok` instead of the direct coroutine result

## Deviations from Plan

None - plan executed exactly as written.

## Verification

- Python AST parse check passed after each task
- No other files modified

## Self-Check: PASSED
