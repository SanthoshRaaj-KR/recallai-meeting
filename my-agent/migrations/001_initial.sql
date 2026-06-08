-- Jarvis shared session store
-- Run once in Supabase SQL Editor: Dashboard → SQL Editor → New Query → paste → Run

create table if not exists jarvis_sessions (
    session_id          text primary key,
    bot_id              text,
    meeting_url         text,
    status              text,
    error               text,
    changes             jsonb    default '[]'::jsonb,
    transcript          jsonb    default '[]'::jsonb,
    transcript_memory_text text  default '',
    summary             jsonb,
    extracted_meeting   jsonb,
    pipeline_diagnostics jsonb   default '[]'::jsonb,
    team_id             text,
    started_at          text,
    ended_at            text,
    updated_at          text
);

-- Disable RLS so both services can read/write with the anon key
alter table jarvis_sessions disable row level security;
