---
phase: 04-pipeline-quality-fixes
plan: "02"
subsystem: agents
tags: [drafter, grounding, verbatim, transcript-window, hallucination-fix]

# Dependency graph
requires:
  - phase: 04-01
    provides: verbatim_content field on ChangeIntent (fact_extraction_agent.py)
provides:
  - _find_relevant_transcript_window helper for targeted transcript access
  - RULE 0 verbatim grounding in INTENT_DRAFTER_PROMPT (highest-priority check)
  - verbatim_content passed through _run_intent_drafter to drafter JSON
  - relevant_transcript_window passed through _run_intent_drafter for long-meeting coverage
affects: [drafter_agent, proposed_changes_pipeline, hallucination-reduction]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Verbatim grounding: extract exact quoted items into verbatim_content, pass to drafter, enforce via RULE 0"
    - "Transcript windowing: find relevant 1500-char excerpt around subject for targeted context"

key-files:
  created: []
  modified:
    - confluence_logic/agents/drafter_agent.py

key-decisions:
  - "RULE 0 placed as highest-priority rule in INTENT_DRAFTER_PROMPT — drafter must check verbatim_content before any other rule"
  - "Window total capped at window parameter (1500 chars) by computing end = start + window rather than pos + window"
  - "relevant_transcript_window passed as top-level JSON key so drafter can distinguish it from broader meeting_context"

patterns-established:
  - "Verbatim pipeline: fact extractor captures exact quoted items → drafter receives them → RULE 0 enforces use"
  - "Transcript window: _find_relevant_transcript_window centers on first match of subject with pre-400 / remaining-1100 split"

requirements-completed:
  - QUAL-02

# Metrics
duration: 15min
completed: 2026-05-15
---

# Phase 4 Plan 02: Fix DrafterAgent Grounding Summary

**DrafterAgent now grounds after_content in verbatim_content (RULE 0) and receives a targeted 1500-char transcript window for the relevant discussion, eliminating hallucinated concerns/features**

## Performance

- **Duration:** ~15 min
- **Started:** 2026-05-15T00:00:00Z
- **Completed:** 2026-05-15T00:15:00Z
- **Tasks:** 4 (+ 1 auto-fix)
- **Files modified:** 1

## Accomplishments

- Added `_find_relevant_transcript_window` helper: finds a 1500-char transcript excerpt centered on the first mention of a subject, solving the "middle-of-meeting" blind spot in the head+tail excerpt approach
- Added RULE 0 to `INTENT_DRAFTER_PROMPT` before RULE 1: when `intent.verbatim_content` is non-empty, drafter must use it as the sole factual source with no additions or paraphrasing
- Wired `verbatim_content` and `relevant_transcript_window` into `_run_intent_drafter` drafter JSON so both new fields reach the LLM at inference time
- All plan verification checks pass: window captures correct text, RULE 0 precedes RULE 1, ChangeIntent.verbatim_content round-trips correctly

## Task Commits

Each task was committed atomically:

1. **Task 1: Add _find_relevant_transcript_window helper** - `22e969d` (feat)
2. **Task 2: Add RULE 0 verbatim grounding to INTENT_DRAFTER_PROMPT** - `c5d49bf` (feat)
3. **Task 3: Wire verbatim_content and relevant_transcript_window into _run_intent_drafter** - `dc82e5b` (feat)
4. **Task 4: Document relevant_transcript_window in INTENT_DRAFTER_PROMPT inputs** - `59257bd` (feat)
5. **Auto-fix: Correct window size calculation** - `60ffa2e` (fix)

**Plan metadata:** (this commit)

## Files Created/Modified

- `confluence_logic/agents/drafter_agent.py` — Added `_find_relevant_transcript_window`, RULE 0, `verbatim_content` in drafter JSON, `relevant_transcript_window` in drafter JSON, fixed window size calculation

## Decisions Made

- Placed RULE 0 (verbatim content) as highest-priority rule before RULE 1 (transcript grounding) — verbatim_content is stronger evidence than transcript inference
- Capped window total at 1500 chars total (not 1900) by computing `end = start + window` to stay within the spec budget
- Added `relevant_transcript_window` as a top-level JSON key distinct from `meeting_context` so the drafter can clearly identify the targeted evidence vs. the broader meeting brief

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Window size calculation exceeded 1500-char spec**
- **Found during:** Final verification run
- **Issue:** Plan code used `start = pos - 400`, `end = pos + 1500` producing up to 1900 chars; plan's own verification test required `len(window) <= 1600`
- **Fix:** Changed to `pre = min(400, pos); start = pos - pre; end = start + window` so total span equals `window` (1500)
- **Files modified:** `confluence_logic/agents/drafter_agent.py`
- **Verification:** `len(window) = 1500` on test input; all plan checks passed
- **Committed in:** `60ffa2e`

---

**Total deviations:** 1 auto-fixed (Rule 1 - bug in window calculation)
**Impact on plan:** Fix required for spec compliance — window was 27% larger than max allowed. No scope creep.

## Issues Encountered

None beyond the window size auto-fix documented above.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- DrafterAgent grounding chain is complete: fact extractor captures verbatim_content → drafter receives it via verbatim_content field → RULE 0 enforces verbatim use
- Long-meeting transcript window also in place so discussions in the middle of meetings are no longer silently dropped
- Ready for phase 4 plan 03 (if any) or E2E verification of the full pipeline quality improvements

---
*Phase: 04-pipeline-quality-fixes*
*Completed: 2026-05-15*
