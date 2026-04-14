---
phase: 06-pipeline-intelligence-and-safety-improvements
plan: 03
subsystem: wake-word-detection, confidence-signaling
tags: [wake-word, fuzzy-matching, phonetic, confidence, safety, pipeline]
dependency_graph:
  requires: []
  provides: [WAKEALIAS-01, CONFIDENCE-01]
  affects: [confluence_logic/jarvis_agentic.py]
tech_stack:
  added: []
  patterns: [env-var-gated feature, regex aliases, post-processing confidence prefix]
key_files:
  created: []
  modified:
    - confluence_logic/jarvis_agentic.py
decisions:
  - "WAKEALIAS-01: _WAKE_ALIASES replaces hardcoded 'jarvis' in _WAKE_PATTERN — phonetic variants jarvas/jervis/jarvus/jarves/jarvi/jarv plus prefixes hey/yo/ok/hi prevent missed activations from transcription errors"
  - "WAKEALIAS-01: JARVIS_WAKE_ALIASES env var allows pipe-separated custom aliases — user-extensible without code changes"
  - "CONFIDENCE-01: Post-processing prefix 'Based on what I found, ' added only when force_web_search=True — regular answers not affected; web-sourced answers get explicit attribution"
  - "CONFIDENCE-01: _LOW_CONFIDENCE_MARKERS tuple defined for future low-confidence detection — _add_confidence_signal() helper provided as extension point"
metrics:
  duration: ~2 min
  completed: "2026-04-14"
  tasks: 2/2
  files: 1
---

# Phase 06 Plan 03: Fuzzy Wake Word Aliases and Confidence Signaling Summary

Expanded wake word detection with phonetic near-misses and common prefix aliases (hey/yo/ok/hi) to handle transcription errors, and added a confidence attribution prefix for web-search-sourced answers so Jarvis signals its information source rather than presenting web results as absolute fact.

## Tasks Completed

### Task 1: Fuzzy wake word aliases with phonetic near-misses

**Commit:** 80611ca

**Changes to `confluence_logic/jarvis_agentic.py`:**
- Replaced hardcoded `_WAKE_PATTERN = re.compile(r"(?:hey\s+)?jarvis[,.]?\s*(.*)", re.IGNORECASE)` with an expanded two-variable approach
- Added `_WAKE_ALIASES` string containing `(?:jarvis|jarvas|jervis|jarvus|jarves|jarvi|jarv)` — seven phonetic variants
- Updated `_WAKE_PATTERN` to use `rf"(?:(?:hey|yo|ok|hi)\s+)?{_WAKE_ALIASES}[,.\s!?]*\s*(.*)"` — adds yo/ok/hi as valid prefixes
- Added `_CUSTOM_WAKE_ALIASES = os.getenv("JARVIS_WAKE_ALIASES", "").strip()` env var for user-configurable aliases
- If `JARVIS_WAKE_ALIASES` is set, the aliases are injected into the pattern (pipe-separated)
- `extract_wake_and_query` and `is_bare_wake_invocation` continue to work unchanged — they use `_WAKE_PATTERN` which is still the same variable name

### Task 2: Confidence signaling in general question responses

**Commit:** 55115db

**Changes to `confluence_logic/jarvis_agentic.py`:**
- Added `JARVIS_CONFIDENCE_SIGNAL_ENABLED = os.getenv("JARVIS_CONFIDENCE_SIGNAL_ENABLED", "true").strip().lower() == "true"` module-level env var
- Added `_LOW_CONFIDENCE_MARKERS` tuple with 11 common hedging phrases for future low-confidence detection
- Added `_add_confidence_signal(answer: str) -> str` helper function as an extension point
- In `_handle_general_question`, after `_rewrite_for_speech`, added attribution prefix logic:
  - Only applies when `force_web_search=True` (web_search intent routing)
  - Skips if answer already starts with "according to", "based on", or "from what i found"
  - Prepends `"Based on what I found, "` with lowercase-normalized first character
  - Entire block gated by `JARVIS_CONFIDENCE_SIGNAL_ENABLED`

## Deviations from Plan

None - plan executed exactly as written.

## Self-Check: PASSED

- `confluence_logic/jarvis_agentic.py` modified: confirmed
- Commit 80611ca: feat(06-03): fuzzy wake word aliases with phonetic near-misses
- Commit 55115db: feat(06-03): confidence signaling for web-search answers
- All verification tests passed (wake word assertions + grep checks)
