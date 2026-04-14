---
phase: 06-pipeline-intelligence-and-safety-improvements
plan: "04"
subsystem: meeting-intelligence
tags: [action-items, speaker-query, classifier, meeting-responder, jarvis-agentic]
dependency_graph:
  requires: [06-02]
  provides: [action_items-intent, speaker_query-intent]
  affects: [classifier.py, jarvis_agentic.py, meeting_responder.py]
tech_stack:
  added: []
  patterns: [gap-filler-parallelism, speech-rewrite, fuzzy-speaker-matching, regex-name-extraction]
key_files:
  created: []
  modified:
    - confluence_logic/meeting_responder.py
    - confluence_logic/classifier.py
    - confluence_logic/jarvis_agentic.py
decisions:
  - "ACTIONITEMS-01: JARVIS_ACTION_ITEMS_MAX_TOKENS defaults to 300, JARVIS_SPEAKER_QUERY_MAX_TOKENS to 250 — token caps for action item extraction and speaker summarization"
  - "SPEAKERQ-01: Fuzzy participant name matching uses substring containment (speaker_lower in p.lower() or p.lower() in speaker_lower) — handles partial name input gracefully"
  - "_extract_speaker_name uses regex patterns covering 'what did X say/mention/think/contribute' and 'summarize what X said' — rejects common pronouns as non-name words"
  - "Action items and speaker_query heuristics placed before opinion heuristics in _fast_classify — avoids false opinion classification for action-item phrasing"
  - "LLM fallback prompt updated to list 6 intents (up from 4); valid result set expanded to include action_items and speaker_query"
metrics:
  duration: "~8 min"
  completed_date: "2026-04-14"
  tasks_completed: 2
  tasks_total: 2
  files_modified: 3
---

# Phase 06 Plan 04: Action Items Extraction and Speaker Query Summary

**One-liner:** Action items extraction and per-participant speaker summarization as two new meeting intelligence intents, routed through heuristic classifier and spoken with gap-filler parallelism.

## What Was Built

Added two new meeting intelligence capabilities to the Jarvis voice assistant pipeline:

1. **Action items extraction** — Users can ask "what are the action items?" and get a spoken list of tasks, decisions, and commitments extracted from the full meeting transcript via LLM.

2. **Speaker query** — Users can ask "what did Alice say?" and get a concise spoken summary of a specific participant's contributions, using fuzzy name matching against the transcript log.

Both features follow the same established pattern: classify intent heuristically, generate a contextual gap-filler in parallel with the LLM call, speak the gap-filler first, then speak the answer after speech rewrite.

## Tasks Completed

| Task | Name | Commit | Files |
|------|------|--------|-------|
| 1 | Add extract_action_items and summarize_speaker to meeting_responder | 3a1ecd1 | confluence_logic/meeting_responder.py |
| 2 | Wire action_items and speaker_query intents through classifier and handler | 7d0174f | confluence_logic/classifier.py, confluence_logic/jarvis_agentic.py |

## Decisions Made

- **ACTIONITEMS-01:** `JARVIS_ACTION_ITEMS_MAX_TOKENS` defaults to 300, `JARVIS_SPEAKER_QUERY_MAX_TOKENS` to 250 — appropriate token caps for spoken-length lists and summaries
- **SPEAKERQ-01:** Fuzzy name matching uses substring containment in both directions — handles first names, last names, and partial inputs
- Heuristic triggers placed before opinion heuristics to prevent "next steps" from matching opinion patterns
- LLM fallback expanded from 4 to 6 valid intents with descriptive prompts for each

## Deviations from Plan

None - plan executed exactly as written.

## Self-Check: PASSED

- `confluence_logic/meeting_responder.py` — FOUND, contains `extract_action_items` and `summarize_speaker`
- `confluence_logic/classifier.py` — FOUND, contains `_ACTION_ITEMS_TRIGGERS`, `_ACTION_ITEMS_PHRASES`, `_SPEAKER_QUERY_PHRASES`, 6 occurrences of `action_items`, 5 occurrences of `speaker_query`
- `confluence_logic/jarvis_agentic.py` — FOUND, contains `_extract_speaker_name`, `_handle_action_items`, `_handle_speaker_query`, and routing in `handle_spoken_request`
- Commit `3a1ecd1` — FOUND
- Commit `7d0174f` — FOUND
