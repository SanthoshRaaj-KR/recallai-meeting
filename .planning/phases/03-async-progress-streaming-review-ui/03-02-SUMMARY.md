---
phase: 03-async-progress-streaming-review-ui
plan: 02
subsystem: ui
tags: [frontend, sync-sage-bot, sse, routing, react, typescript, vite]

# Dependency graph
requires:
  - phase: 03-async-progress-streaming-review-ui
    plan: 01
    provides: GET /review/pipeline/{job_id}/stream SSE endpoint, POST /review/pipeline/start endpoint

provides:
  - PipelineJob, PipelineStage, StageStatus, PipelineEventType types exported from sync-sage-bot/src/types.ts
  - changeBadgeClass, confidenceBadgeClass, riskBadgeClass, confidenceBadgeLabel, riskBadgeLabel exported from sync-sage-bot/src/lib/badges.ts
  - startPipeline(), openPipelineStream(), executeProposal() exported from sync-sage-bot/src/lib/api.ts
  - SsePipelineHandlers interface exported from sync-sage-bot/src/lib/api.ts
  - PipelinePage stub at sync-sage-bot/src/pages/PipelinePage.tsx
  - /pipeline/:jobId route registered in App.tsx wrapped in ProtectedRoute
  - "Generate Confluence Changes" button (4th action button) + in-progress banner on MeetingSummary

affects: [03-03]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - startPipeline() calls POST /review/pipeline/start with existing request<T>() Bearer wrapper
    - openPipelineStream() retrieves Supabase access_token and passes via ?token= query param to EventSource
    - localStorage.setItem("activeJob:{sessionId}", job_id) persists pipeline tracking across page navigations
    - Banner reads localStorage on component mount via useState lazy initializer using sessionId (URL param)

key-files:
  created:
    - sync-sage-bot/src/lib/badges.ts
    - sync-sage-bot/src/pages/PipelinePage.tsx
  modified:
    - sync-sage-bot/src/types.ts
    - sync-sage-bot/src/lib/api.ts
    - sync-sage-bot/src/App.tsx
    - sync-sage-bot/src/pages/MeetingSummary.tsx

key-decisions:
  - "activeJobId lazy initializer uses sessionId (URL param) not resolvedSessionId — avoids temporal dead zone since resolvedSessionId derives from summary state loaded asynchronously"
  - "startPipeline added to existing combined api.ts import in MeetingSummary.tsx — avoids duplicate import while satisfying the intent of the plan requirement"
  - "sync-sage-bot has its own .git repo (was a submodule) — Task commits are made within sync-sage-bot's git repo on main branch; parent repo tracks sync-sage-bot as untracked subdir"

patterns-established:
  - "Pattern 1: Pipeline LocalStorage key — activeJob:{sessionId} stores job_id; banner reads on mount, navigate writes on start, dismiss clears"
  - "Pattern 2: EventSource token passing — ?token= query param populated from supabase.auth.getSession() in openPipelineStream, guarded by isSupabaseConfigured"

requirements-completed: [UI-01]

# Metrics
duration: 8min
completed: 2026-05-12
---

# Phase 3 Plan 02: Frontend Infrastructure for Pipeline Entry Point Summary

**Pipeline types, badge utilities, startPipeline/openPipelineStream/executeProposal API functions, PipelinePage stub, /pipeline/:jobId route, and Generate Confluence Changes button with in-progress banner on MeetingSummary**

## Performance

- **Duration:** 8 min
- **Started:** 2026-05-12T16:17:13Z
- **Completed:** 2026-05-12T16:26:04Z
- **Tasks:** 3
- **Files modified:** 6 (4 modified, 2 created)

## Accomplishments
- Added `PipelineJob`, `PipelineStage`, `StageStatus`, `PipelineEventType` to `types.ts` — complete Phase 3 SSE type vocabulary
- Created `src/lib/badges.ts` shared utility with `changeBadgeClass`, `confidenceBadgeClass`, `riskBadgeClass` records (exact UI-SPEC color tokens); removed inline definition from MeetingSummary.tsx
- Added `startPipeline()`, `openPipelineStream()`, `executeProposal()` to `api.ts` — full API client coverage for Phase 3 and Phase 4 (apply)
- Created `PipelinePage.tsx` stub with page shell (Navbar, back link, header) and registered `/pipeline/:jobId` in ProtectedRoute in `App.tsx`
- Added "Generate Confluence Changes" 4th action button with gradient-prism styling + in-progress banner below Navbar on MeetingSummary

## Task Commits

Each task was committed atomically in the sync-sage-bot repo:

1. **Task 1: Pipeline types + badges.ts + api.ts functions** - `1199538` (feat)
2. **Task 2: PipelinePage stub + /pipeline/:jobId route** - `a7088d3` (feat)
3. **Task 3: Generate Confluence Changes button + in-progress banner** - `52b4c7c` (feat)

## Files Created/Modified
- `sync-sage-bot/src/types.ts` — Appended PipelineJob, PipelineStage, StageStatus, PipelineEventType
- `sync-sage-bot/src/lib/badges.ts` — New shared badge-class records file (changeBadgeClass, confidenceBadgeClass, riskBadgeClass, labels)
- `sync-sage-bot/src/lib/api.ts` — Added startPipeline, openPipelineStream, SsePipelineHandlers, executeProposal
- `sync-sage-bot/src/App.tsx` — Added PipelinePage import and /pipeline/:jobId ProtectedRoute
- `sync-sage-bot/src/pages/PipelinePage.tsx` — New stub page (Navbar + back link + header + placeholder body)
- `sync-sage-bot/src/pages/MeetingSummary.tsx` — Added useNavigate, Loader2, X imports; generatingPipeline/activeJobId state; handleGenerateChanges handler; dismissPipelineBanner; 4th button; in-progress banner; imports changeBadgeClass from @/lib/badges

## Decisions Made
- `activeJobId` lazy initializer uses `sessionId` (URL param) rather than `resolvedSessionId` because `resolvedSessionId` derives from async-loaded summary state and is not yet available during first render; at mount time both reduce to the same value
- `startPipeline` added to the existing combined import line in MeetingSummary.tsx (avoids duplicate `@/lib/api` import)
- Commits go directly to `sync-sage-bot`'s own git repo on `main` (the directory has its own `.git` — was a former submodule)

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Used sessionId instead of resolvedSessionId in activeJobId lazy initializer**
- **Found during:** Task 3 (adding state to MeetingSummary.tsx)
- **Issue:** The plan's code used `resolvedSessionId` in the useState lazy initializer, but `resolvedSessionId` is defined AFTER the useState calls using summary state. At first render (before data loads), `resolvedSessionId` would be `sessionId` anyway, so using `sessionId` directly is semantically equivalent and avoids the temporal dead zone.
- **Fix:** Changed lazy initializer to use `sid = sessionId` (from URL params) instead of `sid = resolvedSessionId`
- **Files modified:** sync-sage-bot/src/pages/MeetingSummary.tsx
- **Verification:** Build passes; behavior is identical since resolvedSessionId = data.session_id || sessionId || null and data.session_id is null at mount time
- **Committed in:** 52b4c7c (Task 3 commit)

---

**Total deviations:** 1 auto-fixed (1 Rule 1 bug)
**Impact on plan:** Fix necessary to avoid temporal dead zone in React render. No behavior change — semantically identical at mount time. No scope creep.

## Issues Encountered
- `sync-sage-bot/` has its own `.git` directory (was previously a submodule, removed with `removing submodule` commit). The parent repo's `.gitignore` working copy had `sync-sage-bot/` added (unstaged); this was restored to HEAD to allow the parent repo to track sync-sage-bot as untracked. All Task commits were made inside sync-sage-bot's own git repo on `main` branch.

## Known Stubs
- `sync-sage-bot/src/pages/PipelinePage.tsx` — Full page body (StageIndicator, ProposalCards, SSE wiring) intentionally stubbed. Placeholder text "Pipeline page ready. Stream wiring lands in Plan 03." Plan 03 fills in the complete component body. This stub is intentional per plan spec and does not block the plan's goal (route registration and navigation from MeetingSummary).

## Next Phase Readiness
- All API client functions ready for Plan 03: `startPipeline`, `openPipelineStream`, `executeProposal` are exported
- PipelinePage route resolves — Plan 03 can edit `PipelinePage.tsx` to add StageIndicator + ProposalCard + SSE wiring
- Pipeline types (`PipelineJob`, `PipelineStage`, `StageStatus`, `PipelineEventType`) exported — Plan 03 components can import from `@/types`
- Badge utilities exported — Plan 03 ProposalCard can import from `@/lib/badges`
- No blockers for Plan 03 execution

---
*Phase: 03-async-progress-streaming-review-ui*
*Completed: 2026-05-12*
