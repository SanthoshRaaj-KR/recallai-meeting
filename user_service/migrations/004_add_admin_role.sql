-- ============================================================
-- Migration 004: Allow the ADMIN org role
--
-- The original schema (001_initial.sql) constrained org_users.role to
-- ('CEO','MANAGER','MEMBER','ASSOCIATE') — but the application code (OrgRole.ADMIN,
-- rbac.require_admin_or_above, the frontend role dropdowns, and analytics access
-- checks) all use ADMIN. Result: setting any user to ADMIN failed with a CHECK
-- violation (SQLSTATE 23514). This adds ADMIN to the allowed set.
--
-- Run once in the Supabase SQL editor (Dashboard → SQL Editor → paste → Run),
-- or via psql with your DB connection string.
-- ============================================================

ALTER TABLE org_users DROP CONSTRAINT IF EXISTS org_users_role_check;

ALTER TABLE org_users
  ADD CONSTRAINT org_users_role_check
  CHECK (role IN ('CEO', 'ADMIN', 'MANAGER', 'MEMBER', 'ASSOCIATE'));
