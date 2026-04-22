---
phase: 05-nextjs-ui
plan: 02
subsystem: review-api
tags: [fastapi, review, summary, graph-rag, transcript]
dependency_graph:
  requires:
    - confluence_logic/jarvis_agentic.py (meeting_state runtime dict)
    - confluence_logic/graph_rag.py (_local_nodes in-memory graph)
    - confluence_logic/meeting_responder.py (transcript_log source)
  provides:
    - GET /review/summary endpoint at http://localhost:8000/review/summary
  affects:
    - review-ui results page (getMeetingSummary() in api.ts calls this)
tech_stack:
  added:
    - FastAPI APIRouter (review subpackage)
  patterns:
    - Late-import pattern for meeting_state to avoid circular imports
    - In-memory graph_rag._local_nodes for zero-latency entity read
key_files:
  created:
    - confluence_logic/review/__init__.py
    - confluence_logic/review/api.py
  modified:
    - confluence_logic/jarvis_agentic.py
decisions:
  - Late-import meeting_state in api.py to avoid circular dependency with jarvis_agentic
  - Summary text built from last 20 transcript entries (synchronous, no LLM call at review time)
  - action_items returned as empty list — structured extraction is a future plan item
  - participants fall back to transcript participant names when graph_rag has no Person nodes
metrics:
  duration: "< 5 minutes"
  completed: "2026-04-22"
  tasks_completed: 1
  tasks_total: 1
  files_created: 2
  files_modified: 1
---

# Phase 05 Plan 02: Add GET /review/summary Endpoint Summary

GET /review/summary FastAPI route returning meeting summary aggregated from transcript and graph RAG state.

## What Was Built

Added `confluence_logic/review/api.py` as a FastAPI `APIRouter` with a single `GET /review/summary` route. The router is mounted into the main `jarvis_agentic` app via `app.include_router()`. The endpoint is now served at `http://localhost:8000/review/summary` and is proxied by the Next.js frontend at `/api/review/summary`.

## Response Shape

```json
{
  "session_id": "...",
  "meeting_url": "...",
  "started_at": null,
  "ended_at": null,
  "summary": "Speaker: text\nSpeaker: text...",
  "topics": ["Topic A", "Topic B"],
  "action_items": [],
  "decisions": ["decision text"],
  "participants": ["Alice", "Bob"]
}
```

## Data Sources

| Field | Source |
|-------|--------|
| `session_id` | `meeting_state["bot_id"]` |
| `meeting_url` | `meeting_state["meeting_url"]` or `MEETING_URL` env var |
| `started_at` / `ended_at` | `meeting_state["started_at/ended_at"]` (None until Phase 02 DB keys added) |
| `summary` | Last 20 transcript entries from `meeting_state["transcript_log"]` |
| `topics` | `graph_rag._local_nodes` — type == "Topic" |
| `decisions` | `graph_rag._local_nodes` — type == "Decision" |
| `participants` | `graph_rag._local_nodes` — type == "Person" (falls back to transcript participants) |
| `action_items` | Empty list (future plan) |

## Commits

| Task | Commit | Description |
|------|--------|-------------|
| 1 | 206f6a5 | feat(05-02): add GET /review/summary endpoint to review API |

## Deviations from Plan

### Context Mismatch

**Found during:** Task 1
**Issue:** Plan references `confluence_logic/review/api.py` as if it already exists (from Phase 02). However, Phase 02 was never executed in this repository — the file did not exist.
**Fix:** Created the full `confluence_logic/review/` package from scratch, including `__init__.py` and `api.py`. No other endpoints were added beyond the required `/review/summary` — out of scope for this plan.
**Files created:** `confluence_logic/review/__init__.py`, `confluence_logic/review/api.py`

### `started_at` / `ended_at` Not in meeting_state

**Found during:** Task 1 (Rule 1 - observation)
**Issue:** `meeting_state` dict (line 177 of `jarvis_agentic.py`) does not contain `started_at` or `ended_at` keys — those were planned for Phase 02's DB work.
**Fix:** Endpoint reads these keys defensively with `.get()`, returning `null` when absent. No behavioral regression.

## Known Stubs

| Stub | File | Reason |
|------|------|--------|
| `action_items: []` always empty | `confluence_logic/review/api.py:169` | Structured action item extraction requires an LLM call or dedicated extractor; planned as future work |
| `started_at: null` / `ended_at: null` | `confluence_logic/review/api.py:155-158` | Phase 02 DB keys not yet wired into `meeting_state` — future plan will resolve |

Note: these stubs do not prevent the plan goal (results page can fetch and display the summary). Topics, decisions, and participants are live-wired.

## Self-Check: PASSED

- confluence_logic/review/__init__.py: FOUND
- confluence_logic/review/api.py: FOUND
- commit 206f6a5: FOUND
