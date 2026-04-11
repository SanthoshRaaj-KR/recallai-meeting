---
phase: "01"
plan: "01"
subsystem: "confluence_logic"
tags: [classifier, audio-cache, tts, intent-classification]
dependency_graph:
  requires: []
  provides:
    - "confluence_logic.classifier.classify_intent"
    - "confluence_logic.audio_cache.get_random_ack_audio"
    - "confluence_logic.audio_cache.load_audio_cache"
    - "confluence_logic.scripts.generate_wav_assets.generate_all"
  affects:
    - "confluence_logic/jarvis_agentic.py"
tech_stack:
  added: []
  patterns:
    - "lazy OpenAI client singleton pattern"
    - "fast-path heuristic + LLM fallback classifier pattern"
    - "in-memory file cache with lazy loading"
key_files:
  created:
    - "confluence_logic/classifier.py"
    - "confluence_logic/audio_cache.py"
    - "confluence_logic/scripts/generate_wav_assets.py"
    - "confluence_logic/scripts/__init__.py"
    - "confluence_logic/assets/audio/.gitkeep"
  modified: []
decisions:
  - "Used gpt-4o-mini with temperature=0 for deterministic LLM fallback classification"
  - "Default to 'confluence' on classifier error (safe fallback — existing pipeline handles it)"
  - "Used MP3 format for audio cache to match Recall API 'kind': 'mp3' requirement"
  - "Lazy cache loading — no startup cost if audio cache not needed"
metrics:
  duration: "~2 minutes"
  completed: "2026-04-11"
  tasks_completed: 2
  tasks_total: 2
  files_created: 5
  files_modified: 0
---

# Phase 01 Plan 01: Intent Classifier and Audio Cache Infrastructure Summary

**One-liner:** Fast-path heuristic + gpt-4o-mini LLM classifier for confluence vs general routing, with lazy-loaded MP3 audio cache and idempotent pre-generation script.

## What Was Built

### Task 1: Intent Classifier Module (`confluence_logic/classifier.py`)
- `classify_intent(text)` — async function returning `'confluence'` or `'general'`
- `_fast_classify(text)` — synchronous heuristic that handles obvious cases without an LLM call:
  - Action verbs (`create`, `edit`, `delete`, etc.) as first word → `'confluence'`
  - Question words (`what`, `why`, `how`, etc.) without confluence nouns → `'general'`
  - Returns `None` for ambiguous cases (triggers LLM fallback)
- LLM fallback uses `gpt-4o-mini` with `temperature=0.0` for deterministic classification
- Defaults to `'confluence'` on any error (safe: existing pipeline handles it)

### Task 2: Audio Cache Module and Generation Script
- `confluence_logic/audio_cache.py` — In-memory MP3 cache:
  - `load_audio_cache(directory?)` — loads all `.mp3` files from `assets/audio/` into memory
  - `get_random_ack_audio()` — returns random `(stem, bytes)` tuple; `None` if cache empty
  - `get_cache_size()` — returns count of cached files
  - Lazy loading on first access
- `confluence_logic/scripts/generate_wav_assets.py` — Idempotent pre-generation script:
  - 10 acknowledgement phrases (`on_it`, `sure_thing`, `give_me_a_sec`, etc.)
  - Reads TTS settings from env vars (`JARVIS_TTS_MODEL`, `JARVIS_TTS_VOICE`, `JARVIS_TTS_SPEED`)
  - Skips existing files; uses `mp3` format matching Recall API requirement
- `confluence_logic/assets/audio/.gitkeep` — directory placeholder for generated audio files

## Commits

| Task | Commit | Description |
|------|--------|-------------|
| Task 1 | `14f34fd` | feat(01-01): add intent classifier module |
| Task 2 | `5acaff8` | feat(01-01): add WAV generation script and audio cache module |

## Decisions Made

1. **gpt-4o-mini for LLM fallback** — Lightweight, fast, cost-effective for a single-token classification response.
2. **Default to 'confluence' on error** — Safe fallback; existing agentic pipeline handles it gracefully.
3. **MP3 format** — Matches the Recall API `"kind": "mp3"` requirement used in `speak()`.
4. **Lazy cache loading** — Avoids startup cost if audio features are not used (e.g., in test scenarios).
5. **Idempotent generation script** — Skips existing files so it can be safely re-run.

## Deviations from Plan

None - plan executed exactly as written.

## Known Stubs

None - all modules are fully functional. Note that `get_random_ack_audio()` returns `None` when no MP3 files have been generated yet (which is the expected behavior before running `generate_wav_assets.py`). Plan 02 will integrate these modules into the main pipeline.

## Self-Check: PASSED

Files created:
- `/Users/akshathr/Clones/recallai-meeting/confluence_logic/classifier.py` — FOUND
- `/Users/akshathr/Clones/recallai-meeting/confluence_logic/audio_cache.py` — FOUND
- `/Users/akshathr/Clones/recallai-meeting/confluence_logic/scripts/generate_wav_assets.py` — FOUND
- `/Users/akshathr/Clones/recallai-meeting/confluence_logic/scripts/__init__.py` — FOUND
- `/Users/akshathr/Clones/recallai-meeting/confluence_logic/assets/audio/.gitkeep` — FOUND

Commits verified:
- `14f34fd` — FOUND (feat(01-01): add intent classifier module)
- `5acaff8` — FOUND (feat(01-01): add WAV generation script and audio cache module)
