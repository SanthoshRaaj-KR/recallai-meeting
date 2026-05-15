---
phase: 4
plan: "04-01"
subsystem: fact-extraction
tags: [bug-fix, dedup, prompt-engineering, verbatim-capture]
dependency_graph:
  requires: []
  provides: [correct-fact-dedup, verbatim-content-field, final-state-extraction]
  affects: [confluence_logic/agents/fact_extraction_agent.py]
tech_stack:
  added: []
  patterns: [last-wins-dedup, normalized-key-dedup]
key_files:
  created: []
  modified:
    - confluence_logic/agents/fact_extraction_agent.py
decisions:
  - "Use (normalized_subject, action) as dedup key instead of (subject, new_value, action) — new_value varies with phrasing, subject+action uniquely identifies topic"
  - "LAST occurrence wins in _merge_facts — transcript order = temporal order = final state of discussion"
  - "_norm() caps at 60 chars and strips punctuation to handle minor phrasing variations across chunks"
metrics:
  duration: "~10 min"
  completed: "2026-05-15"
  tasks_completed: 4
  files_modified: 1
---

# Phase 4 Plan 01: Fix FactExtractionAgent — Final-State Extraction, Dedup, and Verbatim Capture

## One-Liner

Fixed three proposal-quality bugs in `FactExtractionAgent`: contradiction intents from intermediate discussion states, duplicate intents from repeated mentions, and missing verbatim content capture for add/create actions.

## What Was Done

### Task 1 — Add `verbatim_content` to `ChangeIntent`

Added `verbatim_content: str = ""` as the 8th field on `ChangeIntent`. For add/create actions where participants name specific items (lists of concerns, features, metrics), this field captures the exact quoted text from the transcript. Default empty string ensures backward compatibility with all existing LLM responses that predate this field.

### Task 2 — Three new critical rules in `FACT_EXTRACTION_PROMPT`

Inserted three rules before the "Use [] only if..." closing line:

- **Rule A — FINAL STATE ONLY**: Extract the net final agreed state. If "change Q3 to Q1, actually Q3 is fine", produce ZERO intents. Never produce two intents for the same topic with different new_values.
- **Rule B — NO DUPLICATES**: Each distinct change appears exactly once. If the same update is mentioned multiple times across the meeting, extract it once with the most complete information.
- **Rule C — verbatim_content for add/create**: For add/create actions where participants named specific items, copy the EXACT items verbatim into `verbatim_content`. Leave empty for replace/remove/rename.

### Task 3 — Fix `_merge_facts` dedup: (subject, action) keeping LAST

Replaced the `seen_intents` set approach (which deduplicated on `(subject, new_value, action)`) with a new `intent_key_order` + `intent_last` dict approach:

- Key is now `(_norm(subject), action)` — `new_value` removed from key so same-topic revisions collapse
- `_norm()` lowercases, strips punctuation, collapses whitespace, and caps at 60 chars — handles phrasing variations across overlapping chunks
- LAST occurrence wins, preserving transcript order = final agreed state of the discussion
- Moved change_intents accumulation outside the `for chunk in chunks:` loop for clarity

### Task 4 — Verify `ExtractedFacts` return statement and backward compatibility

Verified that:
- `ExtractedFacts` return in `_merge_facts` already passes `change_intents` correctly
- `ExtractedFacts.model_validate()` in `_extract_chunk` handles the new `verbatim_content` field via default value (no schema migration needed)
- No code changes required for this task

## Commits

| Task | Commit | Description |
|------|--------|-------------|
| 1 | f88d7c5 | feat(04-01): add verbatim_content field to ChangeIntent |
| 2 | 567909a | feat(04-01): add FINAL STATE ONLY, NO DUPLICATES, verbatim_content rules to prompt |
| 3 | 2163c46 | fix(04-01): replace seen_intents dedup with (subject, action) last-wins strategy |
| 4 | — | Verify-only, no code change |

## Verification Results

```
Test 1 PASSED: verbatim_content field exists
Test 2 PASSED: last intent wins (Q3 = final state)
ALL CHECKS PASSED
```

Full verification scenario: chunk1 contains `ChangeIntent(subject='SOC quarter', action='replace', new_value='Q1')`, chunk2 contains `ChangeIntent(subject='SOC quarter', action='replace', new_value='Q3')`. After `_merge_facts([chunk1, chunk2])`: exactly 1 intent with `new_value='Q3'` (final state wins).

## Deviations from Plan

None — plan executed exactly as written.

The plan's Task 4 included a typo (`confluece_logic`) in the verification script, noted by the plan itself. The actual verification was run with the correct module path.

## Known Stubs

None. `verbatim_content` will be populated by the LLM when it follows Rule C; the default empty string is intentional (not a UI stub).

## Threat Flags

None. Changes are entirely within the fact extraction data model and dedup logic. No new network endpoints, auth paths, or schema changes at trust boundaries.

## Self-Check

Files modified:
- `confluence_logic/agents/fact_extraction_agent.py` — FOUND

Commits:
- f88d7c5 — FOUND
- 567909a — FOUND
- 2163c46 — FOUND

## Self-Check: PASSED
