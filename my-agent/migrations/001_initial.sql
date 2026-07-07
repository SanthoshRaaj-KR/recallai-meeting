-- Jarvis shared session store — CANONICAL definition (single source of truth).
-- Run once in Supabase SQL Editor: Dashboard → SQL Editor → New Query → paste → Run.
--
-- This is the standalone (bot-service-only) shape: identical columns/types to the
-- org-service definition in user_service/migrations/001_initial.sql, MINUS the
-- team_id → org_teams foreign key (bot-only dev may not have the org tables).
-- On the shared Supabase both definitions are `create table if not exists`, so
-- whichever runs first wins and the other is a no-op — the column TYPES now match,
-- which removes the old text-vs-timestamptz conflict (P5).
--
-- NOTE: transcript + transcript_memory_text still live inline here. A later
-- migration (the 3a transcript-turns split) moves them to their own table; until
-- then the live meeting write-path depends on these columns, so they stay.

create table if not exists jarvis_sessions (
    session_id             text primary key,
    bot_id                 text,
    meeting_url            text,
    status                 text        not null default 'joining',
    error                  text,
    changes                jsonb       not null default '[]'::jsonb,
    transcript             jsonb       not null default '[]'::jsonb,   -- 3a will move to session_transcript_turns
    transcript_memory_text text        not null default '',           -- 3a will move alongside transcript
    summary                jsonb,
    extracted_meeting      jsonb,
    pipeline_diagnostics   jsonb       not null default '[]'::jsonb,
    pipeline_cache         jsonb       not null default '{}'::jsonb,   -- written by review_pipeline (stage cache)
    participants           jsonb       not null default '{}'::jsonb,   -- real presence map (Phase 1 attendance)
    confluence_enabled     boolean     not null default false,        -- written by /bot/start
    team_id                uuid,                                       -- FK added by org-service migration
    started_at             timestamptz not null default now(),
    ended_at               timestamptz,
    updated_at             timestamptz not null default now(),
    -- Derived count so meeting-list queries never have to load the `changes` blob (P6).
    change_count           int generated always as (jsonb_array_length(changes)) stored
);

-- Fast lookups for the hot read paths.
create index if not exists idx_sessions_bot                 on jarvis_sessions(bot_id);
create index if not exists idx_sessions_team_status_started on jarvis_sessions(team_id, status, started_at desc);
create index if not exists idx_sessions_updated             on jarvis_sessions(updated_at desc);

-- Disable RLS so both services can read/write with the service-role/anon key.
alter table jarvis_sessions disable row level security;
