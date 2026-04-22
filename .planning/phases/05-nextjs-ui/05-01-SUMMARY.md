---
phase: 05-nextjs-ui
plan: 01
subsystem: review-ui
tags: [nextjs, typescript, tailwind, frontend, api-client]
dependency_graph:
  requires: []
  provides: [review-ui/package.json, review-ui/next.config.ts, review-ui/src/lib/api.ts, review-ui/src/app/page.tsx, review-ui/src/app/results/page.tsx]
  affects: [confluence_logic/review/api.py]
tech_stack:
  added: [next@14, react@18, tailwindcss@3, typescript@5]
  patterns: [App Router, "use client", rewrites proxy]
key_files:
  created:
    - review-ui/package.json
    - review-ui/tsconfig.json
    - review-ui/next.config.ts
    - review-ui/tailwind.config.ts
    - review-ui/postcss.config.js
    - review-ui/src/app/globals.css
    - review-ui/src/app/layout.tsx
    - review-ui/src/types.ts
    - review-ui/src/lib/api.ts
    - review-ui/src/app/page.tsx
    - review-ui/src/app/results/page.tsx
  modified: []
decisions:
  - "Used next@14.2.29 (latest stable 14.x) — App Router, supports rewrites natively in next.config.ts"
  - "All API calls go through /api/* prefix which next.config.ts proxies to localhost:8000 — no CORS issues in dev"
  - "Polling interval is 5s in landing page; clears on component unmount to prevent memory leaks"
  - "Results page selects all pending changes by default; user can deselect before approving"
  - "per-change result indicators shown inline after executeChanges response — no full page reload"
metrics:
  duration: "2m 27s"
  completed_date: "2026-04-22"
  tasks_completed: 3
  tasks_total: 3
  files_created: 11
  files_modified: 0
---

# Phase 05 Plan 01: Next.js UI Scaffold Summary

## One-liner

Next.js 14 App Router project in `review-ui/` with typed API client, meeting join flow (auto-redirecting to results), and results page with meeting summary + Confluence diff approval UI.

## What Was Built

### Task 1: Next.js 14 project scaffold (commit: 08aa6f5)

Initialized `review-ui/` with:
- `package.json`: next@14.2.29, react@18, tailwindcss@3, typescript@5
- `tsconfig.json`: strict mode, `@/*` alias to `src/`
- `next.config.ts`: rewrites `/api/:path*` to `http://localhost:8000/:path*` (FastAPI proxy)
- `tailwind.config.ts` + `postcss.config.js`: standard Tailwind setup
- `src/app/globals.css`: Tailwind directives
- `src/app/layout.tsx`: root layout, gray-50 background
- `src/types.ts`: `SessionStatus`, `ChangeItem`, `ChangeResult`, `ExecuteChangesResponse`, `MeetingSummary`, `ActionItem`
- `src/lib/api.ts`: typed API client — `startBot`, `getSession`, `getChanges`, `executeChanges`, `getMeetingSummary`

### Task 2: Landing page (commit: 4e3b33c)

`src/app/page.tsx` — four states:
- **idle**: meeting URL input (`id="meeting_url"`) + Join Meeting button
- **joining**: spinner "Starting bot..."
- **active**: green status indicator, live change count (polls `getSession()` every 5s), manual "View Results" link
- **error**: error message + retry button
- Auto-redirects to `/results` when `session.status === "ended"`

### Task 3: Results page (commit: c8486fa)

`src/app/results/page.tsx`:
- Fetches `getMeetingSummary()` and `getChanges()` in parallel on mount
- Section 1 (Meeting Summary): title, date, summary text, key topics, action items with owner, decisions, participants chips
- Section 2 (Confluence Changes): count heading, empty state, change cards with checkbox (checked by default), page title, change type badge (create/edit/delete/title), section heading
- Diff view: before block (`bg-red-50`) and after block (`bg-green-50`) with monospace pre-formatted text
- Approve Selected Changes button: calls `executeChanges(selectedIds)`, disabled while executing, shows per-change result badges (Applied/Failed) inline

## Deviations from Plan

None — plan executed exactly as written.

## Known Stubs

None. All API calls are wired to real endpoints (`/api/*` proxied to FastAPI). The UI will render real data when the FastAPI server is running on localhost:8000 with `/bot/start`, `/bot/status`, `/review/changes`, `/review/execute`, and `/review/summary` endpoints implemented.

## Self-Check: PASSED

Files verified:
- review-ui/package.json: FOUND
- review-ui/next.config.ts: FOUND
- review-ui/src/lib/api.ts: FOUND
- review-ui/src/app/page.tsx: FOUND
- review-ui/src/app/results/page.tsx: FOUND

Commits verified:
- 08aa6f5: scaffold
- 4e3b33c: landing page
- c8486fa: results page
