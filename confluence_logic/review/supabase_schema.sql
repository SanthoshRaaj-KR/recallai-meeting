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

-- ---------------------------------------------------------------------------
-- pipeline_jobs: background pipeline execution tracking (Phase 2+)
-- ---------------------------------------------------------------------------

create table if not exists public.pipeline_jobs (
  job_id uuid primary key default gen_random_uuid(),
  user_id uuid not null,
  session_id text not null,
  status text not null default 'pending',
  stage text,
  created_at timestamptz default now(),
  completed_at timestamptz,
  error text
);

-- NOTE: Run this entire block in the Supabase SQL Editor before starting Phase 2.
-- Status values: 'pending' | 'running' | 'completed' | 'failed'
-- Stage holds the current named pipeline stage (e.g. 'fact_extraction', 'drafting').

create index if not exists pipeline_jobs_session_created_idx
  on public.pipeline_jobs (session_id, created_at desc);

alter table public.pipeline_jobs enable row level security;

drop policy if exists "Users can read their own pipeline jobs" on public.pipeline_jobs;
create policy "Users can read their own pipeline jobs"
  on public.pipeline_jobs
  for select
  using (auth.uid() = user_id);

drop policy if exists "Users can insert their own pipeline jobs" on public.pipeline_jobs;
create policy "Users can insert their own pipeline jobs"
  on public.pipeline_jobs
  for insert
  with check (auth.uid() = user_id);

drop policy if exists "Users can update their own pipeline jobs" on public.pipeline_jobs;
create policy "Users can update their own pipeline jobs"
  on public.pipeline_jobs
  for update
  using (auth.uid() = user_id)
  with check (auth.uid() = user_id);

-- ---------------------------------------------------------------------------
-- proposals: incremental per-card proposal storage (Phase 2+)
-- ---------------------------------------------------------------------------

create table if not exists public.proposals (
  id uuid primary key default gen_random_uuid(),
  job_id uuid not null references public.pipeline_jobs(job_id) on delete cascade,
  session_id text not null,
  user_id uuid not null,
  change_type text not null,
  page_id text,
  page_title text not null,
  section_heading text,
  before_content text,
  after_content text,
  rationale text,
  transcript_evidence jsonb default '[]',
  confidence text not null default 'low',
  risk text not null default 'safe',
  verifier_note text,
  status text not null default 'pending',
  source text default 'pipeline',
  created_at timestamptz default now()
);

-- NOTE: Run this entire block in the Supabase SQL Editor before executing Phase 2.
-- Requires pipeline_jobs table to already exist (Phase 1 checkpoint).
-- confidence values: 'high' | 'medium' | 'low'
-- risk values: 'safe' | 'review' | 'risky'
-- status values: 'pending' | 'accepted' | 'rejected'

create index if not exists proposals_job_created_idx
  on public.proposals (job_id, created_at desc);

alter table public.proposals enable row level security;

drop policy if exists "Users can read their own proposals" on public.proposals;
create policy "Users can read their own proposals"
  on public.proposals
  for select
  using (auth.uid() = user_id);

drop policy if exists "Users can insert their own proposals" on public.proposals;
create policy "Users can insert their own proposals"
  on public.proposals
  for insert
  with check (auth.uid() = user_id);

drop policy if exists "Users can update their own proposals" on public.proposals;
create policy "Users can update their own proposals"
  on public.proposals
  for update
  using (auth.uid() = user_id)
  with check (auth.uid() = user_id);
