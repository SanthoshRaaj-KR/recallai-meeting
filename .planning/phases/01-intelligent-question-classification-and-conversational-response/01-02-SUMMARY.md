---
phase: "01"
plan: "02"
subsystem: "confluence_logic"
tags: [classifier, general-responder, cached-audio, pipeline-integration, intent-routing]
dependency_graph:
  requires:
    - "confluence_logic.classifier.classify_intent"
    - "confluence_logic.audio_cache.get_random_ack_audio"
  provides:
    - "confluence_logic.general_responder.answer_general_question"
    - "confluence_logic.jarvis_agentic.handle_spoken_request (updated with routing)"
    - "confluence_logic.jarvis_agentic.speak_cached_audio"
    - "confluence_logic.jarvis_agentic._speak_cached_guarded"
    - "confluence_logic.jarvis_agentic._handle_general_question"
  affects:
    - "confluence_logic/jarvis_agentic.py"
tech_stack:
  added: []
  patterns:
    - "intent-based early routing before task queue"
    - "cached audio playback with graceful TTS fallback"
    - "lazy OpenAI client singleton pattern (reused in general_responder)"
key_files:
  created:
    - "confluence_logic/general_responder.py"
  modified:
    - "confluence_logic/jarvis_agentic.py"
decisions:
  - "gpt-4o-mini default for general responder (configurable via JARVIS_GENERAL_MODEL env var)"
  - "max_tokens=150 caps response length for TTS-optimized concise answers"
  - "classifier runs before state_lock acquisition — lightweight, non-blocking routing"
  - "general question task created with asyncio.create_task() — fire-and-forget, non-blocking"
  - "cached ack falls back to live TTS when audio cache is empty — zero-downtime before assets generated"
metrics:
  duration: "~3 minutes"
  completed: "2026-04-11"
  tasks_completed: 2
  tasks_total: 2
  files_created: 1
  files_modified: 1
---

# Phase 01 Plan 02: Classifier Integration and General Responder Summary

**One-liner:** Integrated intent classification into the main pipeline — general questions get LLM-generated TTS answers, confluence intents get cached MP3 acks, all existing edit flows unchanged.

## What Was Built

### Task 1: General Question Responder (`confluence_logic/general_responder.py`)

- `answer_general_question(question, conversation_history="")` — async function returning a TTS-ready string
- Uses `gpt-4o-mini` by default (configurable via `JARVIS_GENERAL_MODEL` env var)
- `max_tokens=150` — caps response to 1-3 sentences suitable for voice playback
- System prompt instructs: no markdown, no bullet points, conversational tone
- Accepts `conversation_history` for context-aware answers
- Returns fallback string on error: `"Sorry, I couldn't process that question right now."`
- Lazy OpenAI client singleton (same pattern as existing jarvis_agentic.py)

### Task 2: Pipeline Integration (`confluence_logic/jarvis_agentic.py`)

Six targeted changes — all additive, nothing removed from existing flows:

1. **New imports** — `classify_intent`, `get_random_ack_audio`, `answer_general_question`

2. **`speak_cached_audio(audio_bytes, bot_id)`** — sends pre-cached MP3 bytes directly to Recall API, bypassing TTS synthesis

3. **`_speak_cached_guarded(audio_bytes, bot_id, generation)`** — like `_speak_guarded` but for cached bytes; respects output generation and speech hold timing

4. **`_handle_general_question(query, bot_id)`** — orchestrates: captures output generation, gets conversation history from session agent, calls `answer_general_question()`, speaks via `_speak_guarded(allow_stale=True)`

5. **`handle_spoken_request()` classifier gate** — after status query check, before `state_lock` acquisition: classifies intent; if `general`, fires `_handle_general_question` as a background task and returns early

6. **`_run_voice_task()` cached ack** — replaces hardcoded `random.choice(_INSTANT_ACKS)` with `get_random_ack_audio()`; falls back to live TTS when cache is empty

## Commits

| Task | Commit | Description |
|------|--------|-------------|
| Task 1 | `9668d92` | feat(01-02): create general question responder module |
| Task 2 | `50c9b1e` | feat(01-02): integrate classifier routing and cached ack playback into main pipeline |

## Decisions Made

1. **gpt-4o-mini for general responder** — Fast, cost-effective; matches classifier model choice from Plan 01.
2. **`asyncio.create_task()` for general questions** — Fire-and-forget ensures `handle_spoken_request` returns immediately without blocking the WebSocket handler.
3. **Cached ack fallback to live TTS** — Zero-downtime: works before audio assets are generated; degrades gracefully.
4. **Classifier before `state_lock`** — Classification is lightweight and non-blocking; avoids holding the lock during an LLM call.
5. **`allow_stale=True` for general question answer** — General questions don't need strict generation tracking; answer should be spoken even if another task starts.

## Deviations from Plan

None - plan executed exactly as written.

## Known Stubs

None - all integrations are fully wired. Note that `get_random_ack_audio()` returns `None` until the audio generation script (`confluence_logic/scripts/generate_wav_assets.py`) has been run, in which case the fallback to live TTS activates automatically.

## Self-Check: PASSED

Files created:
- `/Users/akshathr/Clones/recallai-meeting/confluence_logic/general_responder.py` — FOUND

Files modified:
- `/Users/akshathr/Clones/recallai-meeting/confluence_logic/jarvis_agentic.py` — FOUND

Commits verified:
- `9668d92` — FOUND (feat(01-02): create general question responder module)
- `50c9b1e` — FOUND (feat(01-02): integrate classifier routing and cached ack playback into main pipeline)
