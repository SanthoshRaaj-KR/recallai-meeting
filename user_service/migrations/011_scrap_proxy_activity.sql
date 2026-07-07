-- ============================================================
-- Migration 011: scrap the team-proxy attendance data (Phase 5)
--
-- user_meeting_activity credited EVERY team member for EVERY meeting (a proxy).
-- Phase 4 switched all reads to meeting_participants (real attendance) and Phase 5
-- stopped writing the proxy. This clears the misleading historical data.
--
-- The table is KEPT (seed/cleanup scripts still reference it) but emptied. Real
-- historical attendance is repopulated by src/backfill_participants.py, which
-- derives attendees from stored transcripts into meeting_participants.
--
-- Run once in the SQL editor, AFTER 006-010 and AFTER you've run the backfill
-- script (order doesn't strictly matter — they touch different tables — but this
-- makes the intent clear). Idempotent.
-- ============================================================

truncate table user_meeting_activity;
