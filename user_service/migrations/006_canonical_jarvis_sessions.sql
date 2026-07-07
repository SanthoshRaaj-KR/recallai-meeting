-- ============================================================
-- Migration 006: Converge jarvis_sessions to the CANONICAL shape (fixes P5 + P6)
--
-- P5 — jarvis_sessions was defined twice, incompatibly:
--        my-agent/migrations/001_initial.sql  → text timestamps, text team_id, no FK
--        user_service/migrations/001_initial.sql → timestamptz, uuid team_id + FK
--      Whichever ran last won. This script converges an EXISTING table (with data)
--      to the single canonical shape, regardless of which one created it.
--
-- P6 — adds the generated `change_count` column so meeting-list / live-board
--      queries can show a count without loading the (large) `changes` jsonb.
--
-- Also backfills two columns the app writes but neither 001 declared:
--   pipeline_cache (review_pipeline stage cache) and confluence_enabled (/bot/start).
--
-- Idempotent + safe to re-run. Run once in the Supabase SQL editor.
-- NOTE: transcript + transcript_memory_text are intentionally KEPT here — the
-- 3a transcript-turns split removes them in a later migration.
-- ============================================================

-- 1. Add columns the app writes but earlier migrations omitted.
alter table jarvis_sessions add column if not exists pipeline_cache         jsonb   not null default '{}'::jsonb;
alter table jarvis_sessions add column if not exists confluence_enabled     boolean not null default false;
alter table jarvis_sessions add column if not exists transcript_memory_text text    not null default '';
alter table jarvis_sessions add column if not exists extracted_meeting      jsonb;
alter table jarvis_sessions add column if not exists pipeline_diagnostics   jsonb   not null default '[]'::jsonb;

-- 2. Normalise timestamp columns text -> timestamptz (only if currently text).
--    ISO-8601 strings with a trailing 'Z' cast cleanly; '' becomes NULL.
do $$
declare c text;
begin
  foreach c in array array['started_at','ended_at','updated_at'] loop
    if exists (
      select 1 from information_schema.columns
      where table_name = 'jarvis_sessions' and column_name = c
        and data_type in ('text', 'character varying')
    ) then
      execute format(
        'alter table jarvis_sessions alter column %I type timestamptz using nullif(%I, '''')::timestamptz',
        c, c
      );
    end if;
  end loop;
end $$;

-- 3. Normalise team_id text -> uuid (only if currently text).
do $$
begin
  if exists (
    select 1 from information_schema.columns
    where table_name = 'jarvis_sessions' and column_name = 'team_id'
      and data_type in ('text', 'character varying')
  ) then
    alter table jarvis_sessions alter column team_id type uuid using nullif(team_id, '')::uuid;
  end if;
end $$;

-- 4. Guarantee `changes` is a non-null jsonb array before deriving change_count.
update jarvis_sessions
   set changes = '[]'::jsonb
 where changes is null
    or jsonb_typeof(changes) is distinct from 'array';

alter table jarvis_sessions alter column changes set not null;
alter table jarvis_sessions alter column changes set default '[]'::jsonb;

-- 5. Add the generated change_count column (P6) if it isn't already present.
do $$
begin
  if not exists (
    select 1 from information_schema.columns
    where table_name = 'jarvis_sessions' and column_name = 'change_count'
  ) then
    alter table jarvis_sessions
      add column change_count int generated always as (jsonb_array_length(changes)) stored;
  end if;
end $$;

-- 6. Ensure the team_id -> org_teams(id) FK exists (skip if org_teams is absent).
do $$
begin
  if to_regclass('public.org_teams') is not null
     and not exists (
       select 1 from information_schema.table_constraints
       where table_name = 'jarvis_sessions'
         and constraint_name = 'jarvis_sessions_team_id_fkey'
     ) then
    alter table jarvis_sessions
      add constraint jarvis_sessions_team_id_fkey
      foreign key (team_id) references org_teams(id) on delete set null;
  end if;
end $$;

-- 7. Canonical indexes for the hot read paths.
create index if not exists idx_sessions_bot                 on jarvis_sessions(bot_id);
create index if not exists idx_sessions_team_status_started on jarvis_sessions(team_id, status, started_at desc);
create index if not exists idx_sessions_updated             on jarvis_sessions(updated_at desc);
