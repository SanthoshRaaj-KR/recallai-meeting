-- ============================================================
-- Migration 003: Google Auth Support
-- - supabase_user_id column on org_users (links Google identity)
-- - password_hash made nullable (Google-auth users have no password)
-- - user_meeting_activity table for per-person meeting tracking
-- Run in Supabase SQL editor.
-- ============================================================

-- Allow null passwords (Google OAuth users don't register with a password)
ALTER TABLE org_users ALTER COLUMN password_hash DROP NOT NULL;

-- Link Supabase Google auth identity to org user
ALTER TABLE org_users ADD COLUMN IF NOT EXISTS supabase_user_id TEXT UNIQUE;

CREATE INDEX IF NOT EXISTS idx_org_users_supabase_uid ON org_users(supabase_user_id);

-- Per-user meeting activity — populated when a meeting ends
-- Enables: attendance counts, time-in-meetings per person, weekly/monthly breakdowns
CREATE TABLE IF NOT EXISTS user_meeting_activity (
    id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id       UUID NOT NULL REFERENCES org_users(id) ON DELETE CASCADE,
    session_id    TEXT NOT NULL,
    team_id       UUID NOT NULL REFERENCES org_teams(id) ON DELETE CASCADE,
    joined_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    duration_mins DECIMAL,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (user_id, session_id)
);

CREATE INDEX IF NOT EXISTS idx_uma_user_id    ON user_meeting_activity(user_id);
CREATE INDEX IF NOT EXISTS idx_uma_team_id    ON user_meeting_activity(team_id);
CREATE INDEX IF NOT EXISTS idx_uma_session_id ON user_meeting_activity(session_id);
CREATE INDEX IF NOT EXISTS idx_uma_joined_at  ON user_meeting_activity(joined_at);
