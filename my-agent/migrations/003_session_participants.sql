-- ============================================================
-- Migration 003 (session store): real participant presence (Phase 1)
--
-- Captures the REAL people in a meeting (from Recall participant_events.join /
-- .leave + the GET /bot/{id} backfill), so attendance analytics can stop crediting
-- "the whole team" and instead reflect who actually joined and for how long.
--
-- Phase 1 stores presence as a jsonb map on the session, keyed by Recall
-- participant id:
--   { "<pid>": { name, is_host, joined_at, left_at, email? }, ... }
-- Phase 2 will normalise this into a meeting_participants table + resolve each
-- Recall name to an org_user (name-match, then alias table).
--
-- Idempotent. Run once in the Supabase SQL editor.
-- ============================================================

alter table jarvis_sessions
    add column if not exists participants jsonb not null default '{}'::jsonb;
