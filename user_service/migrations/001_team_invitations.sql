-- Run this in your Supabase SQL editor (once)
-- Creates the org_team_invitations table for the email invite flow

CREATE TABLE IF NOT EXISTS org_team_invitations (
    id          UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    team_id     UUID        NOT NULL,
    email       TEXT        NOT NULL,
    role        TEXT        NOT NULL DEFAULT 'MEMBER',
    code        TEXT        NOT NULL UNIQUE,
    status      TEXT        NOT NULL DEFAULT 'pending'
                            CHECK (status IN ('pending', 'accepted', 'expired')),
    inviter_id  UUID,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    expires_at  TIMESTAMPTZ NOT NULL DEFAULT (now() + INTERVAL '48 hours')
);

CREATE INDEX IF NOT EXISTS idx_org_team_invitations_code
    ON org_team_invitations (code);

CREATE INDEX IF NOT EXISTS idx_org_team_invitations_email
    ON org_team_invitations (email);
