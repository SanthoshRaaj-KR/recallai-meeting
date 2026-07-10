-- ============================================================
-- Migration 012: durable, org-global RAG knowledge-base sync status
--
-- The knowledge base is already global (one Pinecone index/namespace fed from one
-- Confluence space). This makes the *sync status* global + durable too: one row per
-- org, upserted by the bot-service on sync start/finish, so every admin/manager sees
-- the same "last synced" across sessions and after a bot-service restart.
--
-- org_id is text to match the bot-service ORG_ID env verbatim (no cast needed).
-- Run once in the Supabase SQL editor. Idempotent.
-- ============================================================

create table if not exists rag_sync_status (
    org_id       text primary key,
    job_id       text,
    status       text,
    total        int  default 0,
    total_stale  int  default 0,
    checked      int  default 0,
    changed      int  default 0,
    skipped      int  default 0,
    failed       int  default 0,
    deleted      int  default 0,
    current_page text,
    error        text,
    started_at   timestamptz,
    finished_at  timestamptz,
    synced_by    text,
    synced_by_name text,
    updated_at   timestamptz default now()
);

-- If the table already existed from an earlier run, add the name column.
alter table rag_sync_status add column if not exists synced_by_name text;
