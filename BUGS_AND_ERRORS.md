# Bugs & Errors — To Fix Later

---

## BUG-1 — 404 on bot join ✅ FIXED

**Symptom:** Frontend sees a 404 immediately after `/bot/start`, then the bot enters the meeting and works normally within a few seconds. Also: mid-meeting "Backend connection issue" 404 during waiting room.

**Root cause (two issues):**
1. Race condition: `session_store.upsert` + `_bot_index` were written AFTER `_create_recall_bot()` returned, so Recall's instant webhook and frontend polls had nothing to find.
2. `session_store.get()` returned `None` on Supabase 200+empty-array without falling back to SQLite local cache — transient Supabase hiccup = 404.
3. `_RECALL_STATUS_MAP` had wrong keys (`"joining"` / `"error"` instead of `"joining_call"` / `"fatal"`); `"in_waiting_room"` was missing entirely.

**Fix applied:**
- `bot_service.py` `start_bot()`: session upserted with `status:"pending"` BEFORE `_create_recall_bot()`; `_bot_index` + patch to `status:"joining"` set immediately after bot_id is known.
- `session_store.py` `get()`: removed early `return None` when Supabase returns empty; now falls through to `_sqlite_get()`.
- `_RECALL_STATUS_MAP` corrected: `joining_call`, `in_waiting_room`, `fatal`, `recording_permission_denied` added/fixed.
- `types.ts` `BotStatus`: added `"pending"` to the union.

**Bot kick verified working post-fix. ✅**

---

## BUG-2 — Organisation details are hardcoded / not real-time

**Symptom:** Bot timings, org name, meeting metadata, and org-level stats shown in the dashboard are seeded/static rather than pulled from live Supabase data.

**Root cause:** `org_activity.py` records meeting activity against the `team_id` passed in the `StartBotRequest`, but the org dashboard reads org-level aggregates from Supabase tables that were seeded with dummy data (see `seed_data.py`). The per-user weekly/monthly meeting stats come from those seeded rows, not from real meeting events.

**Fix (do later):**
- Remove seed data dependency; derive all dashboard stats purely from `jarvis_sessions` + `meeting_activity` tables populated by real bot runs.
- Ensure `team_id` (and therefore `org_id`) is always passed from the frontend when starting a bot — currently it's `Optional` and often omitted, so activity is never linked to an org.

**Files:** `my-agent/src/org_activity.py`, `my-agent/src/bot_service.py` (`StartBotRequest.team_id`), `Confluence/seed_data.py`.

---

## BUG-3 — Inviting participants to meetings not verified / possibly broken

**Symptom:** Unknown — not tested end-to-end.

**Root cause / observation:**
- `StartBotRequest` has no `participants` field; there is no invite flow in `bot_service.py`.
- The Recall bot creation payload (`_create_recall_bot`) sends only `meeting_url`, `bot_name`, `metadata`, and `output_media`. No participant list or calendar invite is constructed.
- If participants are invited via the org-service (port 8003), the wiring from the frontend → org-service → bot-service → Recall has not been confirmed working.

**Fix (do later):** Trace the full invite flow from the frontend through org-service; add a `participants` field to `StartBotRequest` if invites should be sent at bot-start time; confirm Recall or calendar integration handles the actual invite delivery.

**Files:** `my-agent/src/bot_service.py`, `user_service/` (org-service invite endpoints).

---

## BUG-4 — Architecture: Recall diarized transcript must be the source of truth for logs

**Symptom:** Transcript entries stored in `jarvis_sessions.transcript` (which feed the Confluence proposal pipeline — decisions, changes, action items) come from LiveKit with `participant: "Meeting"` — no real speaker names, no diarization.

**Clarified architecture (what needs to change):**

Current (wrong) flow:
```
LiveKit STT → agent.py → /livekit-transcript/{session_id} → session_store.transcript
                                                                      ↓
                                                          Confluence pipeline (decisions)
```

Target (correct) flow:
```
Recall bot in meeting → AssemblyAI v3 diarization → bot.transcript webhook
                                                            ↓
                                               /recall-webhook handler (new branch)
                                                            ↓
                                    session_store.transcript  ← stored as "Speaker Name: text"
                                                            ↓
                                          Confluence pipeline (decisions, changes, action items)
```

**Constraint — DO NOT TOUCH:**
- `agent.py` — the Jarvis voice bot answering pipeline stays completely untouched.
- The LiveKit STT pipeline — it still runs for the bot's real-time responses.
- `/livekit-transcript/{session_id}` endpoint — leave as-is.

**What to add (only):**
- Inside the existing `/recall-webhook` handler in `bot_service.py`, add a new `bot.transcript` event branch.
- Recall fires `bot.transcript` events continuously during the meeting with diarized words (AssemblyAI v3 gives `speaker_id` + `words` per utterance).
- Parse the Recall payload → build entries shaped `{"participant": "<resolved speaker name>", "text": "<utterance>", "timestamp": ..., "source": "recall"}`.
- `session_store.patch` to append these to `jarvis_sessions.transcript`.
- These entries (with real names) are what the Confluence pipeline consumes — replacing the anonymous LiveKit entries.

**Format to store:**
```json
{
  "participant": "Santhosh",
  "text": "We should update the SLA from 4 hours to 2 hours.",
  "timestamp": 1234567890.0,
  "source": "recall"
}
```

**Files to touch:** `my-agent/src/bot_service.py` — `/recall-webhook` handler only (~line 509). Nothing else.

---

## BUG-5 — One-on-one Jarvis calls must run on a separate service

**Symptom / risk:** When a user talks to Jarvis one-on-one (outside a meeting — Q&A, Confluence queries, voice assistant mode), it runs on the same LiveKit room / agent worker as the active meeting bot. This causes:
- Agent resource contention during a live meeting.
- Jarvis responding to both the meeting and the 1-on-1 conversation simultaneously.
- If the 1-on-1 session crashes or hangs, it can bring down the in-meeting agent.

**Fix (do later):**
- One-on-one Jarvis sessions should be dispatched to a **separate named agent** (different `agent_name` in `AgentDispatchRequest`) running in its own isolated LiveKit room.
- The agent worker for 1-on-1 should be a separate process / separate `AgentServer` registration, so a crash in one doesn't affect the other.
- Routing: bot-service dispatches meeting bots to `AGENT_NAME` (current); a new endpoint (e.g. `/assistant/start`) dispatches 1-on-1 sessions to `AGENT_NAME_ASSISTANT` (new env var).
- Both share the same `agent.py` codebase but are registered under different names so LiveKit routes them independently.

**Files to touch (when coding):**
- `my-agent/src/bot_service.py` — new `/assistant/start` endpoint + separate dispatch.
- `my-agent/src/agent.py` — register a second `AgentServer` for the assistant agent name.
- `.env.local` / deploy `.env` — add `AGENT_NAME_ASSISTANT=jarvis-assistant`.

---

## Priority Order

1. **BUG-4** — Highest. Recall diarized transcript as source of truth. One file, one new webhook branch. No risk to existing pipeline.
2. **BUG-1** — Quick fix. Move `session_store.upsert` above `_create_recall_bot()`. Stops the 404 noise.
3. **BUG-5** — Medium. Separate 1-on-1 agent service. Needs arch decision on agent naming.
4. **BUG-2** — Requires org-service coordination.
5. **BUG-3** — Requires full invite flow trace first.
