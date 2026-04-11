# Quick Task 260411-mk0 Summary

## What was done
Added `is_final` guard in the websocket transcript handler. Partial streaming transcript events (mid-sentence) now only update `last_user_speech_at` for silence detection and skip command dispatch. `process_transcript_event` and `handle_spoken_request` are only called when `data_block.get("is_final", True)` is truthy — i.e. when Recall signals the sentence segment is complete.

## Files changed
- `confluence_logic/jarvis_agentic.py` — added is_final guard in websocket handler (~3 lines)

## Key decisions
- Default `is_final` to `True` so events still process if field is absent
- `last_user_speech_at` updated on ALL events (including partials) for correct silence detection
- transcript_log now only receives final segments (no duplicate partial entries)
