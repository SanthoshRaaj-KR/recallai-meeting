-- ============================================================
-- Migration 014: align invite expiry with the code, and stop duplicate
--                live invites for the same person on the same team
--
-- 001_team_invitations.sql defaulted expires_at to now() + 48 hours, but
-- routes/teams.py has always set expires_at explicitly to now() + 1 hour (and
-- the invite email says "expires in 1 hour"). The default was therefore dead
-- but misleading: anyone reading the schema would conclude invites last two
-- days. Realigned to 1 hour so the column agrees with INVITE_TTL.
--
-- Also adds a partial unique index so one email can hold only ONE pending
-- invite per team. invite_member now supersedes previous pending invites
-- before inserting; the index makes that an invariant rather than a
-- convention, since every live code is an independent way into the org.
--
-- Idempotent. Run once in the Supabase SQL editor.
-- ============================================================

alter table org_team_invitations
    alter column expires_at set default (now() + interval '1 hour');

-- Collapse any existing duplicates before the index is created, keeping the
-- most recently issued invite for each (team, lower(email)) pair.
delete from org_team_invitations a
 using org_team_invitations b
 where a.status = 'pending'
   and b.status = 'pending'
   and a.team_id = b.team_id
   and lower(a.email) = lower(b.email)
   and (a.created_at, a.id) < (b.created_at, b.id);

-- Case-insensitive: invites are matched to org_users case-insensitively
-- everywhere else, so "Jane@x.com" and "jane@x.com" are the same invitee.
create unique index if not exists uq_pending_invite_per_team_email
    on org_team_invitations (team_id, lower(email))
    where status = 'pending';
