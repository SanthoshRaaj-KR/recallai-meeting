-- ============================================================
-- Jarvis — Meeting Action Items (auto-assign workflow)
-- Run in your Supabase SQL editor (Dashboard → SQL Editor → New Query).
-- ============================================================

create table if not exists meeting_action_items (
  id            uuid primary key default gen_random_uuid(),
  session_id    text not null,
  team_id       text,
  org_id        uuid,
  description   text not null,
  owner_user_id uuid references org_users(id) on delete set null,
  owner_name    text,                       -- raw LLM name when no roster match
  due           text,
  status        text not null default 'open' check (status in ('open','submitted','closed')),
  member_note   text,                       -- member's completion note
  reviewed_by   uuid references org_users(id) on delete set null,
  reviewed_at   timestamptz,
  source        text not null default 'ai' check (source in ('ai','manual')),
  created_at    timestamptz not null default now(),
  updated_at    timestamptz not null default now(),
  unique (session_id, description)          -- idempotent re-extraction
);

create index if not exists idx_action_items_owner   on meeting_action_items(owner_user_id);
create index if not exists idx_action_items_session  on meeting_action_items(session_id);
create index if not exists idx_action_items_status   on meeting_action_items(status);

-- Disable RLS so the services can read/write with the service-role key
alter table meeting_action_items disable row level security;
