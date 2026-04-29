create table if not exists public.meeting_history (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null,
  session_id text not null unique,
  title text,
  meeting_url text,
  status text,
  started_at timestamptz,
  ended_at timestamptz,
  summary text,
  summary_json jsonb,
  transcript_compressed text,
  transcript_codec text,
  transcript_entry_count integer default 0,
  transcript_uncompressed_bytes integer default 0,
  transcript_compressed_bytes integer default 0,
  change_count integer default 0,
  stats jsonb,
  created_at timestamptz default now(),
  updated_at timestamptz default now()
);

alter table public.meeting_history
  add column if not exists transcript_compressed text,
  add column if not exists transcript_codec text,
  add column if not exists transcript_entry_count integer default 0,
  add column if not exists transcript_uncompressed_bytes integer default 0,
  add column if not exists transcript_compressed_bytes integer default 0;

create index if not exists meeting_history_user_updated_idx
  on public.meeting_history (user_id, updated_at desc);

alter table public.meeting_history enable row level security;

drop policy if exists "Users can read their own meeting history" on public.meeting_history;
create policy "Users can read their own meeting history"
  on public.meeting_history
  for select
  using (auth.uid() = user_id);

drop policy if exists "Users can insert their own meeting history" on public.meeting_history;
create policy "Users can insert their own meeting history"
  on public.meeting_history
  for insert
  with check (auth.uid() = user_id);

drop policy if exists "Users can update their own meeting history" on public.meeting_history;
create policy "Users can update their own meeting history"
  on public.meeting_history
  for update
  using (auth.uid() = user_id)
  with check (auth.uid() = user_id);
