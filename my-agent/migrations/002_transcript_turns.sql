-- ============================================================
-- Migration 002 (session store): append-only transcript turns (fixes P1)
--
-- Append-only store for the diarized RECALL meeting transcript — one INSERT per
-- turn (O(1), no blob rewrite). This is the POST-MEETING source of truth used by
-- the proposal pipeline, summary/MOM, and action-item extraction.
--
-- SOURCE SPLIT (important):
--   * LiveKit STT (fast, real-time)  -> jarvis_sessions.transcript blob, used
--     DURING the live meeting/voice call. UNCHANGED — not stored here.
--   * Recall native transcription (diarized) -> THIS table, used post-meeting.
--
-- Run once in the Supabase SQL editor. Requires jarvis_sessions to exist.
-- ============================================================

create table if not exists session_transcript_turns (
    session_id  text        not null references jarvis_sessions(session_id) on delete cascade,
    seq         bigint      generated always as identity,
    participant text,
    text        text        not null,
    ts          double precision,                 -- relative meeting time (seconds)
    source      text        not null default 'recall',
    created_at  timestamptz not null default now(),
    primary key (session_id, seq)
);

-- Ordered read of a session's turns (participant/text/ts) in chronological order.
create index if not exists idx_transcript_turns_session
    on session_transcript_turns(session_id, seq);

alter table session_transcript_turns disable row level security;
