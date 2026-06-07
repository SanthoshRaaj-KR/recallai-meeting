-- ============================================================
-- Migration 002: Extended user/team fields
-- Run this in your Supabase SQL editor.
-- ============================================================

ALTER TABLE org_users
  ADD COLUMN IF NOT EXISTS job_title  text,
  ADD COLUMN IF NOT EXISTS department text,
  ADD COLUMN IF NOT EXISTS phone      text,
  ADD COLUMN IF NOT EXISTS bio        text,
  ADD COLUMN IF NOT EXISTS avatar_url text;

ALTER TABLE org_teams
  ADD COLUMN IF NOT EXISTS description text;
