---
type: quick-task-summary
id: 260411-kvc
title: Add immediate acknowledgment and brief/detailed summary clarification flow
completed: "2026-04-11"
duration: ~5 min
tasks: 2/2
files_modified:
  - confluence_logic/meeting_responder.py
  - confluence_logic/jarvis_agentic.py
commits:
  - 80e2831: feat(260411-kvc): add detail_level parameter to summarize_meeting()
  - 8c8288b: feat(260411-kvc): add ack + clarification flow to _handle_meeting_summary()
---

# Quick Task 260411-kvc Summary

**One-liner:** Immediate "Sure! Just give me a sec." acknowledgment with brief/detailed clarification flow and dual-prompt/token summarize_meeting() for meeting summary requests.

## What Was Done

### Task 1: detail_level parameter in summarize_meeting() (meeting_responder.py)

- Added `JARVIS_SUMMARY_BRIEF_MAX_TOKENS` constant (default 150, env-configurable via `JARVIS_SUMMARY_BRIEF_MAX_TOKENS`)
- Changed `summarize_meeting()` signature: `detail_level: str = "detailed"` parameter added
- `detail_level="brief"` uses bullet-point style prompt capped at `JARVIS_SUMMARY_BRIEF_MAX_TOKENS`
- `detail_level="detailed"` preserves the existing narrative prompt capped at `JARVIS_SUMMARY_MAX_TOKENS`

### Task 2: ack + clarification flow in jarvis_agentic.py

- Added `pending_summary_clarification` to `meeting_state` init
- Added `JARVIS_SUMMARY_CLARIFICATION_TIMEOUT` constant (default 15s)
- New `_handle_summary_clarification_answer()` function: resolves "brief"/"short"/"quick"/"concise" keywords to `detail_level="brief"`, all else to `"detailed"`
- Replaced `_handle_meeting_summary(bot_id)` with `_handle_meeting_summary(query, bot_id)`:
  - Detects brief/detailed keywords in the original query
  - Immediately speaks "Sure! Just give me a sec." via `_speak_guarded`
  - If type specified: calls `summarize_meeting(transcript_log, detail_level=...)` directly
  - If type unspecified: asks "Do you want a detailed or a brief summary?" and sets `pending_summary_clarification`
- Updated `handle_spoken_request()` to pass `spoken_query` to `_handle_meeting_summary()`
- Added `pending_summary_clarification` routing in `handle_spoken_request()` before intent classification
- Added `pending_summary_clarification` wake word bypass in `process_transcript_event()` with timeout handling

## Deviations from Plan

None - plan executed exactly as written.

## Self-Check

- [x] `confluence_logic/meeting_responder.py` modified
- [x] `confluence_logic/jarvis_agentic.py` modified
- [x] Both files syntax-verified clean (`ast.parse`)
- [x] Commits 80e2831 and 8c8288b exist
- [x] All plan verification grep checks passed
