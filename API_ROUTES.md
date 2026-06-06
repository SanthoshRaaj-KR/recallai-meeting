# API Routes Reference

All routes are served by the FastAPI backend (`confluence_logic/jarvis_agentic.py`) on **port 8000**.

The `sync-sage-bot` frontend proxies `/api/*` → `http://localhost:8000/*` via Vite, so the frontend calls `/api/bot/start` and the backend receives `POST /bot/start`.

---

## Authentication

Most endpoints accept an optional `Authorization: Bearer <supabase_jwt>` header.
When provided, the user's meeting history is persisted to Supabase.
Some endpoints (pipeline start, history, pipeline stream) **require** auth and return `401` without it.

---

## Bot Lifecycle

### `POST /bot/start`
Start the Recall.ai meeting bot (no session isolation).

**Body:**
```json
{ "meeting_url": "https://meet.google.com/...", "session_id": "optional-uuid" }
```
**Returns:** `SessionStatus`
```json
{
  "status": "in_meeting",
  "session_id": "uuid",
  "bot_id": "recall-bot-id",
  "meeting_url": "https://...",
  "change_count": 0
}
```

---

### `POST /sessions/{session_id}/bot/start`
Start bot, scoped to a specific session ID.

**Body:** same as above.
**Returns:** same `SessionStatus` shape.

---

### `GET /bot/status`
Get current bot/session status (global session).

**Returns:** `SessionStatus`
```json
{
  "status": "idle | in_meeting | ended | error",
  "session_id": "uuid | null",
  "bot_id": "string | null",
  "meeting_url": "string | null",
  "change_count": 0,
  "ended_at": "ISO timestamp | null",
  "end_reason": "string | null",
  "recall_status_code": "string | null"
}
```

---

### `GET /sessions/{session_id}/bot/status`
Get bot status for a specific session. Hydrates from Supabase history if auth is provided.

**Returns:** same `SessionStatus` shape.

---

## Meeting Review

### `GET /review/summary`
Generate and return the AI executive summary for the current session.

**Returns:** `MeetingSummary`
```json
{
  "title": "Meeting — meet.google.com/...",
  "session_id": "uuid | null",
  "date": "May 30, 2026 10:00 UTC",
  "summary": "Executive summary paragraph...",
  "key_topics": ["Topic A", "Topic B"],
  "decisions": ["Decision text..."],
  "action_items": [
    { "description": "...", "owner": "Alice | null", "due": "2026-06-01 | null" }
  ],
  "participants": ["Alice", "Bob"],
  "mom": [
    { "topic": "Roadmap Discussion", "summary": "..." }
  ],
  "transcript_highlights": [
    { "time": "02:15", "speaker": "Alice", "text": "..." }
  ],
  "stats": {
    "transcript_entries": 42,
    "topic_count": 3,
    "decision_count": 2,
    "action_item_count": 5
  }
}
```

---

### `GET /sessions/{session_id}/review/summary`
Same as above but for a specific session. Returns cached Supabase summary when available (avoids re-running LLM).

**Returns:** same `MeetingSummary` shape.

---

## Changes (Confluence Proposals)

### `GET /review/changes`
List all pending Confluence change proposals for the current session.

**Returns:** `ChangeItem[]`
```json
[
  {
    "id": 1,
    "change_type": "edit | create | delete | title",
    "page_id": "confluence-page-id | null",
    "page_title": "Engineering Runbook",
    "section_heading": "On-call Rotation | null",
    "before_content": "old text | null",
    "after_content": "new text",
    "timestamp": "2026-05-30T10:00:00Z",
    "session_id": "uuid",
    "status": "pending",
    "source": "meeting_proposal_agent | pipeline | null",
    "rationale": "Why this change is needed",
    "confidence": "high | medium | low",
    "risk": "safe | review | risky",
    "verifier_note": "string | null",
    "transcript_evidence": []
  }
]
```

---

### `GET /sessions/{session_id}/review/changes`
Same as above but session-scoped.

**Returns:** `ChangeItem[]`

---

### `POST /review/changes/propose`
Ask the AI to generate Confluence change proposals from the current transcript.

**Body:**
```json
{ "query": "optional focus hint, e.g. focus on action items | null" }
```
**Returns:** `ProposeChangesResponse`
```json
{
  "changes": [ /* ChangeItem[] — full updated list */ ],
  "generated_count": 3
}
```

---

### `POST /sessions/{session_id}/review/changes/propose`
Same, session-scoped.

**Returns:** same `ProposeChangesResponse`.

---

### `POST /review/execute`
Execute selected change proposals by ID (applies them to Confluence via EditorAgent).

**Body:**
```json
{ "ids": [1, 2, 3] }
```
**Returns:** `ExecuteChangesResponse`
```json
{
  "results": [
    { "id": 1, "success": true },
    { "id": 2, "success": false, "error": "Text not found on page" }
  ]
}
```

---

### `POST /sessions/{session_id}/review/execute`
Execute changes for a session. Handles three modes based on body fields:

| Body field | Mode |
|---|---|
| `ids: number[]` | Legacy in-memory changes |
| `proposal_id: string` | Single pipeline proposal (UUID) |
| `proposal_ids: string[]` | Batch — groups by page to prevent version conflicts |

**Returns (single proposal):**
```json
{ "success": true, "message": "Change applied to Confluence." }
```

**Returns (batch by page):**
```json
{
  "results": [
    { "page": "Engineering Runbook", "success": true, "proposal_count": 2, "message": "Changes applied to Confluence." }
  ]
}
```

---

### `POST /sessions/{session_id}/review/regenerate/{proposal_id}`
Re-draft a stale proposal against the **current live Confluence page**. Use when Accept fails because the page was edited after the proposal was generated.

**Auth:** Required.
**Body:** `{}` (empty)

**Returns:** Updated `ChangeItem` with status reset to `"pending"`.

---

## Meeting Chat

### `POST /sessions/{session_id}/review/chat`
Ask a question about the meeting. Combines transcript, summary, decisions, and action items as context.

**Body:**
```json
{
  "messages": [
    { "role": "user", "content": "What were the key decisions?" },
    { "role": "assistant", "content": "..." }
  ]
}
```
**Returns:** `MeetingChatResponse`
```json
{
  "answer": "The team decided to...",
  "session_id": "uuid",
  "context": {
    "transcript_entries": 42,
    "has_summary": true
  }
}
```

---

## AI Proposal Pipeline

The pipeline runs multi-agent fact extraction → RAG retrieval → drafting → verification asynchronously, streaming results via SSE.

### `POST /review/pipeline/start`
Start the pipeline. Returns immediately (202 Accepted) — poll the SSE stream for progress.

**Auth:** Required.
**Body:**
```json
{ "session_id": "uuid" }
```
**Returns:**
```json
{ "job_id": "uuid", "status": "accepted" }
```

---

### `GET /review/pipeline/{job_id}/stream`
SSE stream for live pipeline progress. Connect via `EventSource`.

**Auth:** Bearer token via `Authorization` header OR `?token=<jwt>` query param (EventSource cannot set custom headers).

**SSE Events:**

| Event name | Payload |
|---|---|
| `stage_start` | `{ "stage": "fact_extraction \| rag_retrieval \| drafting \| verification" }` |
| `stage_progress` | `{ "stage": "...", ...extra fields }` |
| `proposal_ready` | Full `ChangeItem` shape with Supabase-assigned `id` (UUID string) |
| `pipeline_complete` | `{ "proposal_count": 5 }` |
| `pipeline_error` | `{ "detail": "error message" }` |
| `ping` | `{}` (keepalive every 30s) |

---

## History

### `GET /history`
List all meeting history items for the authenticated user.

**Auth:** Required.
**Returns:** `HistoryItem[]`
```json
[
  {
    "session_id": "uuid",
    "title": "Meeting — meet.google.com/...",
    "meeting_url": "https://...",
    "status": "ended",
    "started_at": "2026-05-30T09:00:00Z",
    "ended_at": "2026-05-30T10:00:00Z",
    "summary": "Executive summary text",
    "change_count": 3,
    "stats": { "transcript_entries": 42, ... },
    "updated_at": "2026-05-30T10:05:00Z"
  }
]
```

---

### `GET /history/{session_id}`
Get a single history item.

**Auth:** Required.
**Returns:** Single `HistoryItem`.
**Errors:** `404` if not found.

---

## Utility

### `GET /health`
Health check for the backend process.

**Returns:**
```json
{
  "status": "healthy_agentic",
  "bot_id": "recall-bot-id | null",
  "transcript_provider": "recallai_streaming",
  "tts_provider": "edge_tts"
}
```

---

### `POST /review/confluence-webhook`
Receives Confluence page events and triggers Pinecone + Neo4j re-indexing.
Register in Confluence admin → Webhooks.

**Body:** Confluence webhook payload (page_created / page_updated / page_removed).
**Returns:**
```json
{ "status": "accepted", "page_id": "12345", "event": "page_updated" }
```

---

## CORS

The backend adds CORS middleware for the origins listed in `CORS_ORIGINS` (env var, comma-separated).

**Default:** `http://localhost:3000,http://localhost:5173`

Override example:
```
CORS_ORIGINS=http://localhost:3000,https://your-prod-domain.com
```

In development, the Vite proxy (`/api → http://localhost:8000`) handles CORS automatically — no browser preflight needed.

---

## Quick Start

```bash
# Backend
cd Confluence
uvicorn confluence_logic.jarvis_agentic:app --host 0.0.0.0 --port 8000 --reload

# Frontend
cd sync-sage-bot
npm run dev   # runs on :3000, proxies /api/* to :8000
```

The frontend `src/lib/api.ts` exports typed wrappers for every route above. Import them directly — no raw `fetch` needed.
