-- ============================================================
-- Migration 008: meeting_participants — real per-attendee attendance (Phase 2)
--
-- Normalises the Phase-1 jarvis_sessions.participants jsonb into one row per real
-- attendee, with the Recall identity resolved to an org_user where possible
-- (name-match now; alias table in Phase 3). This is what per-person stats will be
-- computed from in Phase 4 — replacing the team-proxy user_meeting_activity.
--
-- match_confidence:
--   'high'   exact normalised name match to a team member
--   'medium' fuzzy name match (>= threshold)
--   'none'   no match  -> is_guest = true, user_id NULL
--
-- Idempotent per (session_id, recall_participant_id). Run once in the SQL editor.
-- ============================================================

create table if not exists meeting_participants (
    id                    uuid        primary key default gen_random_uuid(),
    session_id            text        not null,
    team_id               uuid,
    org_id                uuid,
    recall_participant_id text,
    recall_name           text,
    user_id               uuid references org_users(id) on delete set null,
    match_confidence      text        not null default 'none'
                                       check (match_confidence in ('high', 'medium', 'none')),
    is_guest              boolean     not null default false,
    joined_at             timestamptz,
    left_at               timestamptz,
    duration_mins         numeric,
    source                text        not null default 'presence',  -- 'presence' | 'backfill'
    created_at            timestamptz not null default now(),
    unique (session_id, recall_participant_id)
);

create index if not exists idx_mp_session on meeting_participants(session_id);
create index if not exists idx_mp_user    on meeting_participants(user_id);
create index if not exists idx_mp_team    on meeting_participants(team_id);

alter table meeting_participants disable row level security;
