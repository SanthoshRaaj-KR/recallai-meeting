# Session Work Log — Networking, DB Optimization & Per-Person Analytics

> A record of everything changed in this working session, why, how it was verified,
> and **what you still need to do to activate it** (migrations + backfill + a live
> Recall payload check). Pairs with `SERVICE_MAP.md` (what runs where).
>
> **Nothing here is applied to Supabase or pushed to origin yet** — all commits are
> local on `master`. Apply steps are in §6.

---

## 0. TL;DR

Three bodies of work, all committed as small independent commits:

1. **Networking fixes** — the cross-host URLs the services use to reach each other.
2. **DB optimization (P1–P6)** — canonical `jarvis_sessions`, transcript moved off the
   O(N²) write path, column projection, analytics pushed into Postgres RPCs.
3. **Per-person meeting analytics (Phases 1–5)** — replace "credit the whole team"
   attendance with **who actually joined and for how long**, keyed to real org users.

The transcript source split is the load-bearing rule to remember:

> **LiveKit STT → live meeting / voice call (fast). Recall diarized → everything
> post-meeting (summary, MOM, proposals, action items, attendance).**

---

## 1. Networking fixes

**Context:** bot-service + agent run on Azure **VM-A** (`bot.104-43-112-6.nip.io`);
confluence-service + org-service + frontend run locally. Several configs still pointed
at Docker-compose service names or `localhost`, which don't resolve across hosts.

| Fix | File | Change |
|---|---|---|
| Admin "kick bot" proxy | `user_service/.env` | `BOT_SERVICE_URL`: `http://bot-service:8000` → `https://bot.104-43-112-6.nip.io` (compose DNS doesn't resolve cross-host) |
| Frontend bot-service proxy | `sync-sage-bot/.env.local` | `VITE_API_PROXY_TARGET`: `localhost:8000` → the VM-A domain; confluence+org stay local |
| Docker model re-download | `my-agent/Dockerfile` | BuildKit cache mount on `download-files` so source edits don't force a full model re-download |

**Still stale (not active, fix when you deploy VM-B):** `deploy/vm-b/.env` +
`deploy/vm-b/Caddyfile` have placeholder/ngrok values; `deploy/vm-a/.env`
`BRIDGE_SERVER_URL` in the repo is a stale ngrok URL (the VM runs its own edited copy).

**Full audit result:** every URL active in the current split points where it should.
The one thing unverifiable from here is `BRIDGE_SERVER_URL` **on the VM itself**
(bot-service → Recall). VM-A was powered off during the session (`HTTP 000` on all
ports), so live connectivity couldn't be confirmed — that's a hosting state, not a
config bug.

---

## 2. Transcript pipeline correction (important)

An earlier DB change (3a) wrongly routed the **live** transcript through Recall.
Corrected so the split is exactly:

- **LiveKit STT** → `jarvis_sessions.transcript` blob → live meeting view, `/history`
  count, jarvis voice-call context. (Fast; unchanged.)
- **Recall diarized** → `session_transcript_turns` → proposal pipeline, summary/MOM,
  meeting chat, action-item extraction, attendance.

Also removed the LiveKit `transcript_memory_text` from post-meeting paths so
post-meeting is **100% Recall** (commits `4a911b9`, `5447fef`). The real-time voice
agent (in-process STT→LLM→TTS in `agent.py`) was never touched.

---

## 3. DB optimization (P1–P6)

Problems identified (ranked) and the fix for each:

| ID | Problem | Fix | Commit |
|---|---|---|---|
| **P5** | `jarvis_sessions` defined twice, incompatibly (text vs timestamptz, text vs uuid team_id) | One canonical definition; `006` converges the live DB (text→timestamptz, text→uuid, safe on data) | `9a6c354` (3b) |
| **P6** | List/board views load the `changes` blob just to count | Generated `change_count` column | `9a6c354` (3b) |
| **P1** | Transcript append = read-modify-write of a whole jsonb array per turn → O(N²) + 2 HTTP calls/turn + a concurrent-writer race | Append-only `session_transcript_turns` (O(1) INSERT). **Recall only**; LiveKit blob kept for live | `4027f61` (3a) + `4a911b9` (fix) |
| **P2** | Every read is `SELECT *` incl. big blobs | `database.select(..., columns=)` projection; live board + analytics fetch only needed cols | `a693259` (3c) |
| **P3/P4** | Analytics looped per-team/per-user (~120 sequential HTTP calls for one org page) | Postgres RPCs aggregate in-DB (1–3 calls) | `5f02995` (3d) |

**Note:** 3b's canonical table is a *safe superset*, not the fully-slimmed version —
`transcript`/`transcript_memory_text` stay inline (live path needs them). It also
declared columns the app writes but no migration had (`pipeline_cache`,
`confluence_enabled`) — that was a latent bug (silent write failures).

---

## 4. Per-person meeting analytics (Phases 1–5)

**The problem:** attendance was a **team proxy** — `record_meeting_activity` credited
*every team member* for *every meeting*, and `/me/stats` counted all of a user's
team's meetings with full duration. A no-show looked identical to a full attendee.

**The crux — identity:** Recall identifies people by **display name**; the org by
**email/user_id**. Bridging that is the whole job.

| Phase | Delivers | Key files | Commit |
|---|---|---|---|
| **1** | Real presence capture — subscribe to Recall `participant_events.join/.leave`; store `jarvis_sessions.participants` jsonb; authoritative `GET /bot/{id}` backfill at meeting end | `bot_service.py`, `migrations/003_session_participants.sql` | `1048440` |
| **2** | `meeting_participants` table (1 row/real attendee) + resolve Recall name → org_user (exact=`high`, fuzzy≥0.85=`medium`, else `guest`) + real per-person minutes from join/leave | `org_activity.py`, `migrations/008_meeting_participants.sql` | `3858669` |
| **3** | `org_users.display_name_aliases` for deterministic matching + admin API to list participants and assign guests (auto-adds alias) | `org_activity.py`, `admin_meetings.py`, `migrations/009_user_display_aliases.sql` | `5568f2b` |
| **4** | Rewrite `/me/stats` + `org_top_users`/`team_member_usage` RPCs on `meeting_participants`. **No-show now = 0**; real time-in-call | `users.py`, `migrations/010_real_attendance_stats.sql` | `60bfeb2` |
| **5** | Stop writing the proxy; `TRUNCATE user_meeting_activity`; backfill history from transcript speaker names (`source='backfill'`, best-effort) | `org_activity.py`, `backfill_participants.py`, `migrations/011_scrap_proxy_activity.sql` | `43a0983` |

### Identity resolution, precisely
1. **Presence** → participant id, name, host flag, timestamp (+email if the platform
   exposes it), backfilled from `GET /bot/{id}`.
2. **Name → user** → normalize (lowercase, strip punctuation/whitespace); exact = `high`,
   `difflib` ratio ≥ 0.85 = `medium`, else `guest`. Raw name + confidence always stored
   (auditable, never silently wrong).
3. **Aliases** → admin maps "iPhone"/"Al B" once → deterministic `high` thereafter.

### Now computable per person
Meetings actually attended · real minutes in call · last attended · this week/month ·
(join/left timestamps enable punctuality & speaking-share later).

### Honest limitations
- **Historical backfill is speaking-only** — past meetings have no join/leave, so
  backfilled rows credit whole-meeting duration and miss silent attendees
  (`source='backfill'`). Only meetings **going forward** get true per-person timing.
- **Guest email** depends on the meeting platform granting it; name/alias is the
  reliable path.

---

## 5. Verification done

All logic tested against **ephemeral Postgres 15** (Docker) and isolated Python tests;
the production Supabase was never touched.

- Migrations: fresh install, full `001→011` chain, and upgrade-from-text-table-with-data
  all apply clean; `change_count` generated + rejects direct writes; idempotent re-runs.
- Transcript turns: ordering, FK cascade + reject, SQLite fallback roundtrip.
- Analytics RPCs: exact expected aggregates incl. zero-session teams; **no-show = 0/0/0**,
  attendee = real minutes.
- Identity: exact=high, typo=medium, device/unknown=guest; alias → deterministic high;
  duration full/no-leave/no-join.
- Backfill: dedups speakers, excludes the bot, team-scoped, dry-run touches no DB.
- Org-service suite: **18 passed**. All changed Python compiles.
- Pre-existing failures confirmed unrelated (a RAG-rerank test + a missing `deepgram`
  plugin in the `--no-sync` env both fail identically on clean HEAD).

---

## 6. What YOU must do to activate

### 6.1 Apply migrations (Supabase SQL editor, in order)
`my-agent/migrations`: `001_initial`, `002_transcript_turns`, `003_session_participants`
`user_service/migrations`: `001` → `002` → `003` → `004` → `005` → `006_canonical_jarvis_sessions`
→ `007_analytics_rpcs` → `008_meeting_participants` → `009_user_display_aliases`
→ `010_real_attendance_stats` → `011_scrap_proxy_activity`

All are idempotent and safe on existing data.

### 6.2 Backfill historical attendance
```bash
cd my-agent
uv run --no-sync python src/backfill_participants.py           # dry-run (counts only)
uv run --no-sync python src/backfill_participants.py --write   # actually write
```

### 6.3 Verify the live Recall payload (the one thing untested)
Field names in `participant_events` and `GET /bot/{id}` `meeting_participants` vary by
Recall API version/platform. The code has defensive fallbacks, but confirm against one
real meeting's logs (`participant join → session … name=…`) before trusting attribution.

### 6.4 Restart services to pick up env/code
- Local: `docker compose -f docker-compose.local.yml up -d --build confluence-service org-service`
- Frontend: restart `npm run dev` (Vite reads `.env.local` only at startup)
- VM-A: rebuild/redeploy the `my-agent` image for the bot-service changes

---

## 7. Commit list (local, on `master`)

```
43a0983 analytics(phase5): scrap team-proxy attendance + backfill history from transcripts
60bfeb2 analytics(phase4): compute stats from REAL attendance (fixes the no-show lie)
5568f2b analytics(phase3): deterministic display-name aliases + guest correction API
3858669 analytics(phase2): meeting_participants table + Recall name -> org_user resolution
1048440 analytics(phase1): capture real participant presence (join/leave)
7d199f1 chore: stop tracking local SQLite session-store artifacts
4a911b9 db(3a-fix): keep LiveKit transcript for the LIVE meeting; Recall only post-meeting
5f02995 db(3d): push analytics aggregation into Postgres RPCs (fixes P3, P4)
a693259 db(3c): column projection on hot jarvis_sessions reads (fixes P2)
4027f61 db(3a): move meeting transcript to append-only turns table (fixes P1)
9a6c354 db(3b): canonical slimmed jarvis_sessions (P5 schema unification + P6 change_count)
```
*(`6f34edb`/`5447fef` in between are vishwa's review-pipeline commits + captured
working-tree edits — see §2.)*

### Networking fixes (env files — not committed; local config)
`user_service/.env`, `sync-sage-bot/.env.local`, `my-agent/Dockerfile` (Dockerfile is committed).

---

## 8. Not done / possible next steps
- Punctuality (late-join delta) and speaking-share % — data is now captured; just needs
  RPCs/UI.
- Fully slim `jarvis_sessions` (drop inline `transcript`) once the turns table is
  confirmed in production.
- VM-B deploy config cleanup (§1).
