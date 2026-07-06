# Jarvis — Database Design for Organisation & User Metrics

> **Purpose:** Canonical reference for how data is stored, what's missing, and exactly what
> needs to change so that the org dashboard, team usage tab, user profile, and MyTasks page
> can show real numbers instead of zeros or stale guesses.
>
> **Read before:** touching `migrations/`, `analytics.py`, `admin_meetings.py`, or the Usage/
> Profile tabs in the frontend.

---

## 1. Current Table Inventory

### 1.1 `organizations`
The top-level tenant. One row per company.

| Column | Type | Notes |
|--------|------|-------|
| `id` | uuid PK | |
| `name` | text | |
| `created_at` | timestamptz | |
| `updated_at` | timestamptz | |

### 1.2 `org_users`
One row per person who has joined an org.

| Column | Type | Notes |
|--------|------|-------|
| `id` | uuid PK | |
| `email` | text UNIQUE | |
| `name` | text | |
| `password_hash` | text nullable | null for Google-auth users |
| `role` | text | CEO / ADMIN / MANAGER / MEMBER / ASSOCIATE |
| `org_id` | uuid → organizations | |
| `is_active` | boolean | |
| `job_title` | text nullable | |
| `department` | text nullable | |
| `phone` | text nullable | |
| `bio` | text nullable | |
| `avatar_url` | text nullable | |
| `supabase_user_id` | text UNIQUE nullable | links Google OAuth identity |
| `created_at` | timestamptz | |
| `updated_at` | timestamptz | |

Indexes: `org_id`, `role`, `supabase_user_id`.

### 1.3 `org_teams`
One row per team inside an org.

| Column | Type | Notes |
|--------|------|-------|
| `id` | uuid PK | |
| `name` | text | |
| `org_id` | uuid → organizations | |
| `description` | text nullable | |
| `created_at` | timestamptz | |
| `updated_at` | timestamptz | |

### 1.4 `org_team_members`
Many-to-many: which users belong to which team, with a team-level role.

| Column | Type | Notes |
|--------|------|-------|
| `team_id` | uuid → org_teams PK part | |
| `user_id` | uuid → org_users PK part | |
| `role` | text | MANAGER / MEMBER / ASSOCIATE |
| `joined_at` | timestamptz | |

### 1.5 `org_reporting_hierarchy`
Closure table: stores every ancestor→descendant pair at every depth.
`depth=0` = self, `depth=1` = direct report, `depth=N` = transitive.
Enables O(1) sub-tree queries ("everyone under manager X").

| Column | Type |
|--------|------|
| `ancestor_id` | uuid → org_users PK part |
| `descendant_id` | uuid → org_users PK part |
| `depth` | int ≥ 0 |

### 1.6 `org_team_bots`
One bot config per team (UNIQUE on team_id).

| Column | Type |
|--------|------|
| `id` | uuid PK |
| `team_id` | uuid → org_teams UNIQUE |
| `name` | text |
| `config` | jsonb |
| `created_at` | timestamptz |

### 1.7 `jarvis_sessions`
One row per Jarvis bot meeting session. Written by bot-service (port 8000),
read by org-service (port 8003) for all admin and analytics queries.

**⚠️ Two conflicting schemas exist:**
- `my-agent/migrations/001_initial.sql` (original): `team_id TEXT`, `started_at TEXT`, `ended_at TEXT`
- `user_service/migrations/001_initial.sql` (org service): `team_id UUID`, `started_at TIMESTAMPTZ`, `ended_at TIMESTAMPTZ`

In practice, whichever ran first won the CREATE TABLE. Both code paths handle ISO string timestamps, so it works — but the org service FK constraint on `team_id` may or may not be active depending on order. **Migration 006 must lock this down (see §6).**

| Column | Type | Notes |
|--------|------|-------|
| `session_id` | text PK | Recall-assigned ID |
| `bot_id` | text | Recall bot ID |
| `meeting_url` | text | original Google Meet / Zoom / Teams URL |
| `status` | text | `joining` / `in_meeting` / `ended` / `error` |
| `error` | text nullable | |
| `changes` | jsonb `[]` | array of proposed Confluence change objects |
| `transcript` | jsonb `[]` | array of `{speaker, text, ts}` entries |
| `transcript_memory_text` | text | flattened transcript for in-meeting RAG |
| `summary` | jsonb nullable | `{title, summary, key_topics, decisions, action_items, mom_minutes, ...}` |
| `extracted_meeting` | jsonb nullable | raw LLM extraction |
| `pipeline_diagnostics` | jsonb `[]` | step-by-step pipeline log |
| `team_id` | uuid → org_teams (or text) | **keystone** — org-scopes the session |
| `started_at` | timestamptz (or text) | when bot joined |
| `ended_at` | timestamptz nullable | when bot left |
| `updated_at` | timestamptz | last write |

Indexes: `team_id`, `status`, `updated_at DESC`.

### 1.8 `user_meeting_activity`
Per-user attendance record written when a meeting ends.
Populated by `org_activity.py` in confluence-service.

| Column | Type | Notes |
|--------|------|-------|
| `id` | uuid PK | |
| `user_id` | uuid → org_users | |
| `session_id` | text | ← NOT a FK (jarvis_sessions.session_id is text) |
| `team_id` | uuid → org_teams | |
| `joined_at` | timestamptz | when Recall transcript said this user spoke first |
| `duration_mins` | decimal nullable | time they were present (approximate) |
| `created_at` | timestamptz | |

UNIQUE on `(user_id, session_id)`. Indexes: `user_id`, `team_id`, `session_id`, `joined_at`.

**Current limitation:** Recall's native transcription provides display names, not org emails.
`org_activity.py` does a fuzzy name match against `org_users.name` to assign `user_id`.
This means rows may be missing for users whose Recall display name differs from their org name.

### 1.9 `meeting_action_items`
Action items extracted by the AI from a meeting transcript.

| Column | Type | Notes |
|--------|------|-------|
| `id` | uuid PK | |
| `session_id` | text | ← NOT a FK |
| `team_id` | **text** | **⚠️ should be uuid — see Gap 5** |
| `org_id` | uuid → organizations | |
| `description` | text | |
| `owner_user_id` | uuid → org_users nullable | null if LLM couldn't match |
| `owner_name` | text nullable | raw LLM name |
| `due` | text nullable | free-text date |
| `status` | text | `open` / `submitted` / `closed` / `cancelled` |
| `member_note` | text nullable | completion note |
| `reviewed_by` | uuid → org_users nullable | |
| `reviewed_at` | timestamptz nullable | |
| `source` | text | `ai` / `manual` |
| `created_at` | timestamptz | |
| `updated_at` | timestamptz | |

UNIQUE on `(session_id, description)` for idempotent re-extraction.

### 1.10 `org_team_invitations`
Pending invites (1-hour TTL, hard-deleted on expiry or acceptance).

| Column | Type |
|--------|------|
| `id` | uuid PK |
| `team_id` | uuid → org_teams |
| `email` | text |
| `role` | text |
| `code` | text UNIQUE |
| `status` | text — `pending` / `accepted` |
| `inviter_id` | uuid → org_users |
| `created_at` | timestamptz |
| `expires_at` | timestamptz |

---

## 2. How Meeting Data Flows Into the Database

```
1. User picks a team in MeetingInput → startBot(url, teamId)
          │
          ▼
2. bot-service (8000): POST /bot/start
   → creates jarvis_sessions row: {session_id, team_id, status="joining", started_at=now()}
          │
          ▼
3. Recall bot joins → status updated to "in_meeting"
   Recall publishes transcript events → bot-service appends to transcript[]
          │
          ▼
4. Meeting ends (user stops or admin kicks)
   → bot-service: status="ended", ended_at=now()
   → confluence-service pipeline runs:
       a. LLM extracts summary + action items → jarvis_sessions.summary updated
       b. org_activity.py fuzzy-matches speakers → inserts user_meeting_activity rows
       c. action_items_extractor.py assigns owners → inserts meeting_action_items rows
```

**Everything hangs on `team_id` being stamped at step 2.** If it's null, the session is
invisible to all org-scoped queries.

---

## 3. Metrics Catalogue — What Each Dashboard Needs

### 3.1 Org-wide Dashboard (CEO / ADMIN — Usage tab)

| Widget | Source tables | Query pattern |
|--------|--------------|---------------|
| Total meetings (all time) | `jarvis_sessions` | COUNT WHERE team_id IN org_teams AND status='ended' |
| Meetings this week / month | `jarvis_sessions` | COUNT + date filter on `started_at` |
| Total meeting minutes | `jarvis_sessions` | SUM(ended_at - started_at) per row |
| Avg meeting duration | `jarvis_sessions` | AVG(duration) |
| Most active team | `jarvis_sessions` GROUP BY team_id | COUNT DESC, JOIN org_teams for name |
| Top users by attendance | `user_meeting_activity` | GROUP BY user_id, COUNT DESC, JOIN org_users |
| Action items open / closed | `meeting_action_items` | COUNT GROUP BY status WHERE org_id = X |

### 3.2 Team Usage Tab

| Widget | Source tables | Query pattern |
|--------|--------------|---------------|
| Team total sessions | `jarvis_sessions` | WHERE team_id = X AND status='ended' |
| Team meetings this week/month | `jarvis_sessions` | + date filter |
| Per-member attendance | `user_meeting_activity` + `org_team_members` + `org_users` | JOIN, GROUP BY user_id |
| Per-member meeting minutes | `user_meeting_activity` | SUM(duration_mins) GROUP BY user_id |

### 3.3 User Profile Page

| Widget | Source tables | Query pattern |
|--------|--------------|---------------|
| My total meetings | `user_meeting_activity` | COUNT WHERE user_id = me |
| My meeting minutes | `user_meeting_activity` | SUM(duration_mins) WHERE user_id = me |
| Meetings this week/month | `user_meeting_activity` | + date filter on joined_at |
| Avg meeting duration | `user_meeting_activity` | AVG(duration_mins) |
| Last meeting | `user_meeting_activity` | MAX(joined_at) |
| My open action items | `meeting_action_items` | COUNT WHERE owner_user_id = me AND status='open' |

### 3.4 Admin Meetings Live Board

| Widget | Source tables | Query pattern |
|--------|--------------|---------------|
| Running meetings | `jarvis_sessions` | WHERE team_id IN (...) AND status IN ('joining','in_meeting') |
| Meeting title | `jarvis_sessions.summary->>'title'` | JSONB extract (slow on large tables) |
| Live elapsed time | computed from `started_at` | in Python: `now() - started_at` |

### 3.5 Admin Meeting History

| Widget | Source tables | Query pattern |
|--------|--------------|---------------|
| History list | `jarvis_sessions` | ORDER BY started_at DESC, filter by team/status |
| Title, team, duration | `jarvis_sessions` + `org_teams` | JOIN for team_name, duration from timestamps |
| Change count | `jsonb_array_length(changes)` | or a pre-stored column |

### 3.6 MyTasks (any user) + Manager Review Queue

| Widget | Source tables | Query pattern |
|--------|--------------|---------------|
| My action items by status | `meeting_action_items` | WHERE owner_user_id = me GROUP BY status |
| Items to review (manager) | `meeting_action_items` + `org_team_members` | JOIN: user manages teams, items in those teams |
| Meeting context for item | `jarvis_sessions` | WHERE session_id = item.session_id |

---

## 4. Gaps & Problems (why real data doesn't show up)

### Gap 1 — Title buried in JSONB
`jarvis_sessions.summary` is a big JSONB blob. Every list query that shows a meeting title
must deserialise the full `summary` column just to get one field.
**Effect:** slow list queries; no SQL index on title for search.

### Gap 2 — Duration computed at query time
`ended_at - started_at` is done in Python on every analytics request. With 1000 sessions
this means deserialising 1000 rows and computing floats in a loop.
**Effect:** analytics.py O(N) loop; the 60s in-memory cache is a band-aid, not a fix.

### Gap 3 — `org_id` not on `jarvis_sessions`
To scope sessions to an org the code does:
```python
teams = select("org_teams", {"org_id": f"eq.{org_id}"})
team_ids = [t["id"] for t in teams]
# then: WHERE team_id IN (team_ids)
```
For a org with 20 teams this is fine. But it's 2 round-trips every time.
**Effect:** every admin and analytics query costs one extra Supabase call.

### Gap 4 — `user_meeting_activity` not reliably populated
The fuzzy name match in `org_activity.py` may fail if:
- The user's Recall display name (e.g. "Santhosh") doesn't exactly match `org_users.name` ("Santhosh Raaj K R")
- The user joined late or wasn't captured in Recall's transcript
**Effect:** per-user stats show 0 even when the user was in the meeting.

### Gap 5 — `meeting_action_items.team_id` is TEXT
```sql
team_id text   -- should be uuid references org_teams(id)
```
This prevents a proper FK and index. Joining with `org_teams` on `team_id::uuid` is an
implicit cast that bypasses the index.
**Effect:** action-item queries by team are slow; no referential integrity.

### Gap 6 — No trend / time-series table
The Usage tab currently shows totals. Adding a week-over-week chart requires either:
- A slow `GROUP BY date_trunc('week', started_at)` across all sessions, or
- A pre-aggregated daily/weekly snapshot table
**Effect:** charts that show "meetings per week" can't be added without either slow queries
or a new table.

### Gap 7 — `jarvis_sessions` type inconsistency
Two migrations both try to create `jarvis_sessions` with `IF NOT EXISTS`. One uses
`TEXT` for `team_id`/timestamps; the other uses `UUID`/`TIMESTAMPTZ`. The actual schema
in production depends on which migration ran first. Analytics and admin routes handle
both by parsing strings, but the `org_users` FK on `team_id` may not be enforced.
**Effect:** referential integrity not guaranteed; subtle bugs if team is deleted.

---

## 5. Recommended Schema Changes — Migration 006

Run in Supabase SQL Editor. All statements are idempotent (`IF NOT EXISTS`, `IF EXISTS`).

```sql
-- ============================================================
-- Migration 006: Schema hardening + metrics columns
-- ============================================================

-- ── 5.1 Lock jarvis_sessions timestamps to proper types ──────
-- Only runs if the column is still TEXT (no-op if already TIMESTAMPTZ)
DO $$
BEGIN
  IF (SELECT data_type FROM information_schema.columns
      WHERE table_name='jarvis_sessions' AND column_name='started_at') = 'text' THEN
    ALTER TABLE jarvis_sessions
      ALTER COLUMN started_at TYPE timestamptz USING started_at::timestamptz,
      ALTER COLUMN ended_at   TYPE timestamptz USING ended_at::timestamptz;
  END IF;
END $$;

-- Lock team_id to uuid with FK
DO $$
BEGIN
  IF (SELECT data_type FROM information_schema.columns
      WHERE table_name='jarvis_sessions' AND column_name='team_id') = 'text' THEN
    ALTER TABLE jarvis_sessions
      ALTER COLUMN team_id TYPE uuid USING team_id::uuid;
    ALTER TABLE jarvis_sessions
      ADD CONSTRAINT fk_jarvis_sessions_team FOREIGN KEY (team_id)
        REFERENCES org_teams(id) ON DELETE SET NULL;
  END IF;
END $$;

-- ── 5.2 Add denormalised columns to jarvis_sessions ──────────

-- title: extracted from summary.title when meeting ends, avoids JSONB read on lists
ALTER TABLE jarvis_sessions ADD COLUMN IF NOT EXISTS title text;

-- org_id: denormalised so org-scoped queries need no team JOIN
ALTER TABLE jarvis_sessions
  ADD COLUMN IF NOT EXISTS org_id uuid REFERENCES organizations(id) ON DELETE SET NULL;

-- duration_mins: precomputed when meeting ends, avoids timestamp arithmetic on every read
ALTER TABLE jarvis_sessions ADD COLUMN IF NOT EXISTS duration_mins decimal;

-- attendee_count: number of known org users who attended (from user_meeting_activity)
ALTER TABLE jarvis_sessions ADD COLUMN IF NOT EXISTS attendee_count int NOT NULL DEFAULT 0;

-- change_count: len(changes[]) stored so list queries don't deserialise the full array
ALTER TABLE jarvis_sessions ADD COLUMN IF NOT EXISTS change_count int NOT NULL DEFAULT 0;

-- ── 5.3 Backfill new columns from existing data ───────────────
UPDATE jarvis_sessions
SET
  title        = summary->>'title',
  duration_mins = EXTRACT(EPOCH FROM (ended_at - started_at)) / 60,
  change_count  = jsonb_array_length(COALESCE(changes, '[]'::jsonb)),
  org_id       = ot.org_id
FROM org_teams ot
WHERE jarvis_sessions.team_id = ot.id
  AND jarvis_sessions.ended_at IS NOT NULL;

-- ── 5.4 Fix meeting_action_items.team_id to uuid ─────────────
-- Safe only if existing team_id values are valid UUIDs or NULL.
-- Check first: SELECT DISTINCT team_id FROM meeting_action_items WHERE team_id !~ '^[0-9a-f-]{36}$';
ALTER TABLE meeting_action_items
  ALTER COLUMN team_id TYPE uuid USING team_id::uuid;

ALTER TABLE meeting_action_items
  ADD CONSTRAINT IF NOT EXISTS fk_action_items_team
    FOREIGN KEY (team_id) REFERENCES org_teams(id) ON DELETE SET NULL;

-- ── 5.5 New indexes for fast metrics queries ─────────────────
CREATE INDEX IF NOT EXISTS idx_js_org_id     ON jarvis_sessions(org_id);
CREATE INDEX IF NOT EXISTS idx_js_started_at ON jarvis_sessions(started_at DESC);
CREATE INDEX IF NOT EXISTS idx_js_org_status ON jarvis_sessions(org_id, status);

-- Composite index for per-user attendance date filtering
CREATE INDEX IF NOT EXISTS idx_uma_user_joined ON user_meeting_activity(user_id, joined_at DESC);

-- Action items per user by status
CREATE INDEX IF NOT EXISTS idx_ai_owner_status ON meeting_action_items(owner_user_id, status);
CREATE INDEX IF NOT EXISTS idx_ai_org_status   ON meeting_action_items(org_id, status);
```

---

## 6. Supabase Views for Metrics Queries

Create these as **regular views** (not materialised — Supabase doesn't support `REFRESH MATERIALIZED VIEW` in the free tier). The analytics Python cache (60s TTL) is the materialisation layer for now.

```sql
-- ── 6.1 Per-team meeting summary ─────────────────────────────
-- Used by analytics.py team_usage + org_usage endpoints
CREATE OR REPLACE VIEW v_team_meeting_stats AS
SELECT
  js.team_id,
  ot.name                                           AS team_name,
  ot.org_id,
  COUNT(*)                                          AS total_sessions,
  COALESCE(SUM(js.duration_mins), 0)               AS total_duration_mins,
  ROUND(AVG(js.duration_mins)::numeric, 1)         AS avg_duration_mins,
  COUNT(*) FILTER (WHERE js.started_at >= now() - interval '7 days')   AS sessions_this_week,
  COUNT(*) FILTER (WHERE js.started_at >= now() - interval '30 days')  AS sessions_this_month,
  MAX(js.started_at)                                AS last_meeting_at
FROM jarvis_sessions js
JOIN org_teams ot ON ot.id = js.team_id
WHERE js.status = 'ended'
GROUP BY js.team_id, ot.name, ot.org_id;

-- ── 6.2 Per-user meeting summary ─────────────────────────────
-- Used by users.py /users/me/stats + analytics.py /usage/me
CREATE OR REPLACE VIEW v_user_meeting_stats AS
SELECT
  uma.user_id,
  ou.name                                                             AS user_name,
  ou.email,
  ou.org_id,
  COUNT(*)                                                            AS total_meetings,
  COALESCE(SUM(uma.duration_mins), 0)                               AS total_duration_mins,
  ROUND(AVG(uma.duration_mins)::numeric, 1)                         AS avg_duration_mins,
  COUNT(*) FILTER (WHERE uma.joined_at >= now() - interval '7 days')  AS meetings_this_week,
  COUNT(*) FILTER (WHERE uma.joined_at >= now() - interval '30 days') AS meetings_this_month,
  MAX(uma.joined_at)                                                  AS last_meeting_at
FROM user_meeting_activity uma
JOIN org_users ou ON ou.id = uma.user_id
GROUP BY uma.user_id, ou.name, ou.email, ou.org_id;

-- ── 6.3 Org-wide action item counts ──────────────────────────
CREATE OR REPLACE VIEW v_org_action_item_stats AS
SELECT
  org_id,
  COUNT(*) FILTER (WHERE status = 'open')       AS open_count,
  COUNT(*) FILTER (WHERE status = 'submitted')  AS submitted_count,
  COUNT(*) FILTER (WHERE status = 'closed')     AS closed_count,
  COUNT(*) FILTER (WHERE status = 'cancelled')  AS cancelled_count,
  COUNT(*)                                       AS total_count,
  ROUND(
    100.0 * COUNT(*) FILTER (WHERE status = 'closed')
    / NULLIF(COUNT(*) FILTER (WHERE status != 'cancelled'), 0),
    1
  )                                              AS completion_pct
FROM meeting_action_items
GROUP BY org_id;
```

---

## 7. Backend Changes Required to Populate the New Columns

After applying Migration 006, three service locations need updating:

### 7.1 bot-service — on meeting start (`/bot/start`)

```python
# When creating the jarvis_sessions row, also write org_id.
# team_id is already passed from the frontend.
team = db.select_one("org_teams", {"id": f"eq.{body.team_id}"})
org_id = team["org_id"] if team else None

db.upsert("jarvis_sessions", {
    "session_id": session_id,
    "team_id": body.team_id,
    "org_id": org_id,          # ← NEW
    "status": "joining",
    "started_at": now_iso(),
    ...
})
```

### 7.2 confluence-service — on meeting end (pipeline completion)

```python
# After summary is generated, extract scalar columns and write them back.
title = summary.get("title") or f"Meeting {session_id[:8]}"
duration_mins = (ended_at - started_at).total_seconds() / 60 if ended_at else None
change_count = len(changes)

db.update("jarvis_sessions", session_id, {
    "summary": summary,
    "title": title,             # ← NEW
    "duration_mins": round(duration_mins, 1),   # ← NEW
    "change_count": change_count,               # ← NEW
    "status": "ended",
    "ended_at": ended_at_iso,
})
```

### 7.3 confluence-service — `org_activity.py` — after writing `user_meeting_activity`

```python
# After inserting all attendance rows, update the attendee_count on the session.
attendee_count = len(matched_users)
db.update("jarvis_sessions", session_id, {
    "attendee_count": attendee_count,   # ← NEW
})
```

### 7.4 analytics.py — switch from raw scans to the views

Once the views exist, replace the in-Python aggregation loops with direct view reads:

```python
# Before (expensive):
all_s = select("jarvis_sessions", {"team_id": f"eq.{team_id}", "status": "eq.ended"})
stats = _compute_stats(all_s, recent_s)  # Python loop over all rows

# After (cheap):
stats = select_one("v_team_meeting_stats", {"team_id": f"eq.{team_id}"})
# Returns pre-aggregated row — one row, one Supabase call
```

---

## 8. Week-over-Week Trend Table (optional, for charts)

If the frontend ever needs a chart showing "meetings per week for the last 12 weeks",
add this table and populate it nightly (or after every meeting end):

```sql
CREATE TABLE IF NOT EXISTS org_meeting_weekly_stats (
  org_id       uuid NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
  team_id      uuid REFERENCES org_teams(id) ON DELETE CASCADE,
  week_start   date NOT NULL,    -- Monday of the week (date_trunc('week', started_at)::date)
  session_count    int NOT NULL DEFAULT 0,
  total_duration_mins decimal NOT NULL DEFAULT 0,
  attendee_count   int NOT NULL DEFAULT 0,
  created_at   timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY (org_id, team_id, week_start)
);

CREATE INDEX IF NOT EXISTS idx_omws_org_week ON org_meeting_weekly_stats(org_id, week_start DESC);
```

Populate after each meeting ends (or via a Supabase cron / pg_cron job):

```sql
INSERT INTO org_meeting_weekly_stats (org_id, team_id, week_start, session_count, total_duration_mins)
SELECT
  org_id,
  team_id,
  date_trunc('week', started_at)::date AS week_start,
  COUNT(*),
  COALESCE(SUM(duration_mins), 0)
FROM jarvis_sessions
WHERE status = 'ended'
GROUP BY org_id, team_id, date_trunc('week', started_at)::date
ON CONFLICT (org_id, team_id, week_start)
DO UPDATE SET
  session_count = EXCLUDED.session_count,
  total_duration_mins = EXCLUDED.total_duration_mins;
```

---

## 9. Entity-Relationship Summary

```
organizations (1)
  └── org_teams (N)                    ← org_id FK
       ├── org_team_members (N)        ← team_id FK + user_id FK
       ├── org_team_bots (1)           ← team_id FK (UNIQUE)
       ├── org_team_invitations (N)    ← team_id FK
       └── jarvis_sessions (N)         ← team_id FK + org_id (denorm)
            ├── user_meeting_activity (N)     ← session_id (text, no FK)
            └── meeting_action_items (N)      ← session_id (text, no FK)

org_users (1)
  ├── org_team_members (N)             ← user_id FK (member of teams)
  ├── org_reporting_hierarchy (N)      ← ancestor_id + descendant_id FK
  ├── user_meeting_activity (N)        ← user_id FK
  └── meeting_action_items (N)         ← owner_user_id FK
```

**Note:** `session_id` on `user_meeting_activity` and `meeting_action_items` is a plain `text`
column, not a FK, because `jarvis_sessions.session_id` is also `text` (Recall-assigned) and
cross-service FK enforcement is impractical. Treat it as a logical key.

---

## 10. Missing Endpoints (APIs That Don't Exist Yet)

The tables and backend exist for these, but no route is wired up:

| Metric | Needed endpoint | Source tables |
|--------|----------------|---------------|
| Action item completion rate per org | `GET /analytics/action-items` | `meeting_action_items` |
| Per-manager team stats | `GET /analytics/usage/teams/{id}/members` | `user_meeting_activity` + `org_team_members` |
| Confluence change acceptance rate | `GET /analytics/changes` | `jarvis_sessions.changes[]` aggregation |
| Week-over-week trend | `GET /analytics/usage/org/trend?weeks=12` | `org_meeting_weekly_stats` (see §8) |

---

## 11. Frontend Wiring Gap (Critical — No Real Data Shown)

**The most impactful blocker for "showing real data" is that `OrgSettings.tsx` uses
hardcoded `DUMMY_*` data for every tab except Knowledge.**

| Tab | Currently shows | Should call |
|-----|----------------|-------------|
| Members | `DUMMY_MEMBERS` (static array) | `GET /users` |
| Teams | `DUMMY_TEAMS` (static counts) | `GET /teams` |
| Hierarchy | Static hardcoded tree | `GET /org/hierarchy` |
| Meetings → Live | `DUMMY_LIVE` | `GET /admin/meetings/live` |
| Meetings → History | `DUMMY_HISTORY` | `GET /admin/meetings` |
| Usage | `DUMMY_USAGE_TEAMS` ("248 sessions", "9,120m") | `GET /analytics/usage/org` |

All the backend routes and analytics logic are implemented and tested. The data is in the
database. **The only work needed to show real numbers is replacing the DUMMY constants
with API calls** using the existing `api.ts` client functions.

The DB schema changes in §5 and view creation in §6 make those API calls return
accurate, fast data — but even before those changes, the raw endpoints return real numbers
for any meetings that have already been recorded.

---

## 12. Priority Order for Implementation

| Priority | Change | Effort | Unblocks |
|----------|--------|--------|----------|
| **P0** | Wire `OrgSettings.tsx` tabs to real API calls (remove DUMMY_* constants) | 3–4 hours | real data visible immediately |
| **P0** | Run Migration 006 SQL | ~5 min | schema correctness |
| **P0** | Write `org_id`, `title`, `duration_mins`, `change_count` on meeting end | 1 hour | accurate list queries + fast analytics |
| **P1** | Create the 3 Supabase views (§6) | ~10 min | analytics.py refactor |
| **P1** | Update `analytics.py` to read from views instead of full-scan loops | 2 hours | fast org/team/user metrics at scale |
| **P1** | Write `attendee_count` after `org_activity.py` runs | 30 min | attendance widget accuracy |
| **P2** | Improve `org_activity.py` name-matching (fuzzy → exact on supabase_user_id) | 2 hours | accurate per-user stats |
| **P2** | Add missing analytics endpoints (action items, changes, per-manager) | 3 hours | richer dashboard widgets |
| **P2** | Create `org_meeting_weekly_stats` + populate on meeting end | 2 hours | trend/chart data |
| **P3** | RLS policies | ongoing | multi-tenant security hardening |
