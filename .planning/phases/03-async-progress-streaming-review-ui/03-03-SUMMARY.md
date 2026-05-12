---
phase: 03-async-progress-streaming-review-ui
plan: 03
subsystem: ui
tags: [frontend, sync-sage-bot, sse, components, react, typescript, review-ui]

# Dependency graph
requires:
  - phase: 03-async-progress-streaming-review-ui
    plan: 02
    provides: openPipelineStream, executeProposal, startPipeline, PipelinePage stub, /pipeline/:jobId route, badges.ts, PipelineStage types

provides:
  - StageIndicator component with 4 named stages, live pulse dot, and pipeline_complete collapse
  - ProposalCard component with all 9 UI-SPEC regions, delete variant with checkbox gate, accept/reject flow
  - ProposalCardGroup component with page-title grouping and confidence sort (high > medium > low)
  - ProposalCardSkeleton shimmer placeholder
  - Full PipelinePage with SSE wiring, 5 connection states, accept/reject handlers, localStorage cleanup

affects: [04-confluence-apply]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - openPipelineStream async promise with mounted-flag + esRef cleanup pattern
    - useMemo groupBy(page_title) for incremental card grouping
    - setCurrentStage functional updater with side-effect completedStages push
    - animate-in fade-in slide-in-from-bottom-2 duration-200 for card entry animation
    - deleteConfirmed local state gate on Accept button for delete cards
    - CardStatus state machine (pending -> applying -> accepted | rejected)

key-files:
  created:
    - sync-sage-bot/src/components/StageIndicator.tsx
    - sync-sage-bot/src/components/ProposalCard.tsx
    - sync-sage-bot/src/components/ProposalCardGroup.tsx
    - sync-sage-bot/src/components/ProposalCardSkeleton.tsx
  modified:
    - sync-sage-bot/src/pages/PipelinePage.tsx

key-decisions:
  - "ProposalCardGroup passes onAccept(id) and onReject(id) to ProposalCard — card callbacks are id-bound wrappers; PipelinePage owns all cardStatuses state"
  - "Navbar import uses default import (not named) — Navbar.tsx only has default export"
  - "Tasks 1 and 2 committed together per plan instruction since ProposalCardGroup imports ProposalCard which didn't exist until Task 2"
  - "PipelinePage mounted flag pattern used because openPipelineStream is async — EventSource not available synchronously at mount"

# Metrics
duration: 4min
completed: 2026-05-12
---

# Phase 3 Plan 03: Review UI Components + Full PipelinePage Summary

**StageIndicator (4 named stages + collapse), ProposalCard (9 UI-SPEC regions + delete variant), ProposalCardGroup (page grouping + confidence sort), ProposalCardSkeleton, and a fully SSE-wired PipelinePage handling all five connection states**

## Performance

- **Duration:** ~4 min
- **Started:** 2026-05-12T16:30:38Z
- **Completed:** 2026-05-12T16:34:11Z
- **Tasks completed:** 3 of 4 (Task 4 is checkpoint:human-verify — awaiting user)
- **Files created:** 4
- **Files modified:** 1

## Accomplishments

- Created `StageIndicator.tsx` with STAGE_SEQUENCE constant, per-step completed/active/pending visual states (CheckCircle2/Loader2/Circle icons), live pulse dot in top-right corner, and collapse to single "Pipeline complete — N proposals ready" row on `isComplete=true`
- Created `ProposalCard.tsx` implementing all 9 UI-SPEC regions: header badges (change_type + confidence + risk), section heading, delete warning (AlertTriangle), rationale, before/after monospace blocks, transcript evidence blockquotes, verifier note (Info icon + blue tinted box), delete-confirm checkbox gate, and accept/reject action row
- ProposalCard delete variant: 2px destructive border, AlertTriangle warning, checkbox that must be checked before Accept is enabled; accepted/rejected states with opacity + styling transitions
- Created `ProposalCardGroup.tsx` grouping cards by `page_title`, sorting within group by CONFIDENCE_ORDER (high=0, medium=1, low=2), FileText group header with change count
- Created `ProposalCardSkeleton.tsx` with exact UI-SPEC shimmer layout (3 badge skeletons + content + 2 button skeletons)
- Replaced stub `PipelinePage.tsx` with full implementation: SSE stream via `openPipelineStream(jobId, handlers)` opened on mount with mounted-flag + esRef cleanup on unmount; all 5 SSE states handled (connecting, streaming, complete, error, disconnected); accept/reject handlers with `executeProposal` + toast feedback; sticky StageIndicator (top-4 z-10); `localStorage.removeItem('activeJob:{sessionId}')` on both `pipeline_complete` and `pipeline_error`
- Build verified green (✓ built in 5.96s, 1769 modules)

## Task Commits

Each task was committed atomically in the sync-sage-bot repo:

1. **Task 1+2: StageIndicator, ProposalCardSkeleton, ProposalCardGroup, ProposalCard** - `b64293b` (feat)
2. **Task 3: Full PipelinePage SSE wiring** - `af8b69f` (feat)
3. **Task 4** — checkpoint:human-verify (pending user sign-off)

## Files Created/Modified

- `sync-sage-bot/src/components/StageIndicator.tsx` — 4-stage progress indicator with live pulse, stage icons, collapse on complete
- `sync-sage-bot/src/components/ProposalCard.tsx` — 9-region proposal card with delete variant, checkbox gate, accept/reject + toast feedback
- `sync-sage-bot/src/components/ProposalCardGroup.tsx` — per-page grouping + CONFIDENCE_ORDER sort + FileText header
- `sync-sage-bot/src/components/ProposalCardSkeleton.tsx` — shimmer placeholder for streaming state
- `sync-sage-bot/src/pages/PipelinePage.tsx` — full page replacing stub: SSE wiring, 5 states, sticky StageIndicator, grouped cards, empty state, error banner

## Decisions Made

- Tasks 1 and 2 committed together because ProposalCardGroup imports ProposalCard — build would fail with only Task 1 committed (per plan note, this was intentional)
- Navbar import changed from `{ Navbar }` to default import `Navbar` — Navbar.tsx only exports a default; the plan template used named import which caused a build error (Rule 1 auto-fix)
- PipelinePage uses a `mounted` flag alongside `esRef` because `openPipelineStream` is async — the EventSource arrives via Promise resolution after the component may have already unmounted; the mounted flag prevents stale state updates and the returned ES is still closed in the Promise `.then()` handler

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed named import of Navbar to default import in PipelinePage**
- **Found during:** Task 3 (build verification after writing PipelinePage.tsx)
- **Issue:** Plan template used `import { Navbar } from "@/components/Navbar"` but `Navbar.tsx` only has `export default Navbar` — build error: `"Navbar" is not exported by "src/components/Navbar.tsx"`
- **Fix:** Changed to `import Navbar from "@/components/Navbar"` (default import)
- **Files modified:** sync-sage-bot/src/pages/PipelinePage.tsx
- **Verification:** Build passes (✓ built in 5.96s)
- **Committed in:** af8b69f (Task 3 commit)

---

**Total deviations:** 1 auto-fixed (1 Rule 1 bug — import style mismatch)
**Impact on plan:** Fix required for correct build. No behavior change — identical runtime result.

## Known Stubs

None — all components implemented with complete functionality. PipelinePage stub from Plan 02 has been fully replaced.

## Threat Flags

No new threat surface introduced beyond what was catalogued in the plan threat model (T-3-10 through T-3-13). All user-supplied content (verifier_note, transcript_evidence) is rendered via React text interpolation — no `dangerouslySetInnerHTML` used, no XSS surface added.

## Checkpoint Status

Task 4 (checkpoint:human-verify) reached. Human must complete the 25-step browser walk-through defined in the plan to verify UI-02 through UI-06 + D-08 + polish (animations, responsiveness, console-error-free).

## Next Phase Readiness

- All Phase 3 UI deliverables are implemented (Plan 01 SSE backend + Plan 02 types/routing/entry + Plan 03 components/page)
- Phase 4 (Confluence apply hardening) can add backend validation to the `/sessions/{sessionId}/review/execute` endpoint called by ProposalCard's Accept button
- No blockers for Phase 4 — `executeProposal(sessionId, id)` is already wired and functional

---

## Self-Check: PASSED

- FOUND: sync-sage-bot/src/components/StageIndicator.tsx
- FOUND: sync-sage-bot/src/components/ProposalCard.tsx
- FOUND: sync-sage-bot/src/components/ProposalCardGroup.tsx
- FOUND: sync-sage-bot/src/components/ProposalCardSkeleton.tsx
- FOUND: sync-sage-bot/src/pages/PipelinePage.tsx
- FOUND commit: b64293b (Tasks 1+2)
- FOUND commit: af8b69f (Task 3)
- Build: ✓ built in 5.96s (1769 modules transformed)

---
*Phase: 03-async-progress-streaming-review-ui*
*Completed: 2026-05-12 (Tasks 1–3; Task 4 checkpoint pending human verification)*
