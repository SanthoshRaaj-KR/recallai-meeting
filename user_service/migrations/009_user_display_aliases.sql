-- ============================================================
-- Migration 009: deterministic display-name aliases (Phase 3)
--
-- Recall shows display names ("Al", "Alice (mobile)", "iPhone") that don't match
-- org_users.name. This column lets an admin map those to a real user ONCE; the
-- resolver then matches them deterministically (high confidence) on every future
-- meeting — no fuzzy guessing.
--
-- Aliases are stored normalised (lowercase, punctuation stripped) so lookup is a
-- plain array-contains. Idempotent. Run once in the SQL editor.
-- ============================================================

alter table org_users
    add column if not exists display_name_aliases text[] not null default '{}';

-- GIN index for fast "which user has this alias" lookups.
create index if not exists idx_org_users_aliases
    on org_users using gin (display_name_aliases);
