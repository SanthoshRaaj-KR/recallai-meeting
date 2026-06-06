-- ============================================================
-- Jarvis Org Service — Initial Schema
-- Run this in your Supabase SQL editor (or via psql).
-- ============================================================

-- ── Organizations ────────────────────────────────────────────
create table if not exists organizations (
  id          uuid primary key default gen_random_uuid(),
  name        text not null,
  created_at  timestamptz not null default now(),
  updated_at  timestamptz not null default now()
);

-- ── Users ────────────────────────────────────────────────────
-- role hierarchy: CEO > MANAGER > MEMBER > ASSOCIATE
create table if not exists org_users (
  id            uuid primary key default gen_random_uuid(),
  email         text unique not null,
  name          text not null,
  password_hash text not null,
  role          text not null check (role in ('CEO', 'MANAGER', 'MEMBER', 'ASSOCIATE')),
  org_id        uuid references organizations(id) on delete cascade,
  is_active     boolean not null default true,
  created_at    timestamptz not null default now(),
  updated_at    timestamptz not null default now()
);

create index if not exists idx_org_users_org_id on org_users(org_id);
create index if not exists idx_org_users_role   on org_users(role);

-- ── Teams ────────────────────────────────────────────────────
create table if not exists org_teams (
  id         uuid primary key default gen_random_uuid(),
  name       text not null,
  org_id     uuid not null references organizations(id) on delete cascade,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create index if not exists idx_org_teams_org_id on org_teams(org_id);

-- ── Team Members ─────────────────────────────────────────────
create table if not exists org_team_members (
  team_id   uuid not null references org_teams(id) on delete cascade,
  user_id   uuid not null references org_users(id) on delete cascade,
  role      text not null check (role in ('MANAGER', 'MEMBER', 'ASSOCIATE')),
  joined_at timestamptz not null default now(),
  primary key (team_id, user_id)
);

create index if not exists idx_team_members_user_id on org_team_members(user_id);

-- ── Reporting Hierarchy (Closure Table) ──────────────────────
-- Stores ALL ancestor→descendant pairs at every depth.
-- depth=0: self-loop  depth=1: direct report  depth=N: transitive
-- Enables O(1) "is X a subordinate of Y?" and full sub-tree queries.
create table if not exists org_reporting_hierarchy (
  ancestor_id   uuid not null references org_users(id) on delete cascade,
  descendant_id uuid not null references org_users(id) on delete cascade,
  depth         int  not null check (depth >= 0),
  primary key (ancestor_id, descendant_id)
);

create index if not exists idx_hierarchy_descendant on org_reporting_hierarchy(descendant_id);

-- ── Team Bots ─────────────────────────────────────────────────
-- Each team has at most one bot assigned.
create table if not exists org_team_bots (
  id         uuid primary key default gen_random_uuid(),
  team_id    uuid not null references org_teams(id) on delete cascade,
  name       text not null,
  config     jsonb not null default '{}',
  created_at timestamptz not null default now(),
  unique(team_id)
);

-- ── Meeting Sessions (extend jarvis_sessions) ─────────────────
-- Links the shared session store to a team + org for access control.
-- jarvis_sessions.team_id already carries this FK reference.
create table if not exists jarvis_sessions (
  session_id             text primary key,
  bot_id                 text,
  meeting_url            text,
  status                 text not null default 'joining',
  error                  text,
  changes                jsonb not null default '[]',
  transcript             jsonb not null default '[]',
  transcript_memory_text text not null default '',
  summary                jsonb,
  extracted_meeting      jsonb,
  pipeline_diagnostics   jsonb not null default '[]',
  started_at             timestamptz not null default now(),
  ended_at               timestamptz,
  updated_at             timestamptz not null default now(),
  team_id                uuid references org_teams(id) on delete set null
);

create index if not exists idx_jarvis_sessions_team_id  on jarvis_sessions(team_id);
create index if not exists idx_jarvis_sessions_status   on jarvis_sessions(status);
create index if not exists idx_jarvis_sessions_updated  on jarvis_sessions(updated_at desc);

-- ── Row-level security (optional, recommended for Supabase) ──
-- Enable RLS if you want Supabase to enforce access at DB level.
-- alter table org_users          enable row level security;
-- alter table org_teams          enable row level security;
-- alter table org_team_members   enable row level security;
-- alter table jarvis_sessions    enable row level security;
-- (Define policies based on your JWT claims when ready.)
