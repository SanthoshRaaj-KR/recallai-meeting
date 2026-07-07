-- ============================================================
-- Migration 002 (session store): append-only transcript turns (fixes P1)
--
-- The meeting transcript used to live as one big jsonb array on
-- jarvis_sessions.transcript, appended via read-modify-write on every spoken
-- turn — O(N^2) bytes + 2 HTTP round-trips per turn. This table replaces that
-- with a one-row INSERT per turn (O(1) append, no blob rewrite, race-free).
--
-- SOURCE POLICY: we store ONLY the Recall native-transcription turns (the real
-- meeting transcript, with speaker names). The LiveKit/agent STT stream is
-- ignored — it no longer writes here (see bot_service /livekit-transcript).
--
-- jarvis_sessions.transcript is intentionally LEFT in place for now (a later
-- cleanup migration can drop it once this is verified in production).
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

-- Denormalised count so /history and list views never scan the turns table.
alter table jarvis_sessions
    add column if not exists transcript_turn_count int not null default 0;

create or replace function bump_transcript_turn_count()
returns trigger language plpgsql as $$
begin
    update jarvis_sessions
       set transcript_turn_count = transcript_turn_count + 1
     where session_id = new.session_id;
    return new;
end $$;

drop trigger if exists trg_bump_transcript_turn_count on session_transcript_turns;
create trigger trg_bump_transcript_turn_count
    after insert on session_transcript_turns
    for each row execute function bump_transcript_turn_count();

alter table session_transcript_turns disable row level security;
