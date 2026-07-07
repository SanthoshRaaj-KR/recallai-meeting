-- ============================================================
-- Migration 007: Analytics aggregation RPCs (fixes P3 + P4)
--
-- The analytics endpoints looped in Python — for /usage/org that was
-- ~2 queries per team + 1 per user (100+ sequential HTTP round-trips) with all
-- counting/summing done on the event loop. These functions push the aggregation
-- into Postgres so each endpoint makes 1-3 RPC calls instead.
--
-- Idempotent (create or replace). Run once in the Supabase SQL editor.
-- Duration math mirrors the old Python _compute_stats:
--   total_minutes = sum of positive (ended - started) minutes over ENDED sessions
--   dur_count     = number of ENDED sessions with a positive duration
--   avg (client)  = total_minutes / dur_count
-- ============================================================

-- Per-team usage for an explicit set of teams (left join so zero-session teams
-- still appear). Used by /usage/org, /usage/me, /usage/teams.
create or replace function team_usage_stats(
    p_team_ids uuid[],
    p_week     timestamptz,
    p_month    timestamptz
)
returns table (
    team_id        uuid,
    team_name      text,
    total_sessions bigint,
    total_minutes  numeric,
    dur_count      bigint,
    sessions_week  bigint,
    sessions_month bigint
) language sql stable as $$
    select
        t.id,
        t.name,
        count(s.session_id) filter (where s.status = 'ended'),
        coalesce(round(sum(
            extract(epoch from (s.ended_at - s.started_at)) / 60
        ) filter (where s.status = 'ended'
                    and s.ended_at   is not null
                    and s.started_at is not null
                    and s.ended_at > s.started_at)), 0),
        count(s.session_id) filter (where s.status = 'ended'
                    and s.ended_at   is not null
                    and s.started_at is not null
                    and s.ended_at > s.started_at),
        count(s.session_id) filter (where s.status = 'ended' and s.started_at >= p_week),
        count(s.session_id) filter (where s.status = 'ended' and s.started_at >= p_month)
    from org_teams t
    left join jarvis_sessions s on s.team_id = t.id
    where t.id = any(p_team_ids)
    group by t.id, t.name;
$$;

-- Top users in an org by meetings attended (inner join — only users with activity).
create or replace function org_top_users(p_org uuid, p_limit int default 20)
returns table (
    user_id           uuid,
    name              text,
    email             text,
    org_role          text,
    sessions_attended bigint,
    total_minutes     numeric
) language sql stable as $$
    select
        u.id, u.name, u.email, u.role,
        count(a.id),
        coalesce(round(sum(a.duration_mins)), 0)
    from org_users u
    join user_meeting_activity a on a.user_id = u.id
    where u.org_id = p_org
    group by u.id, u.name, u.email, u.role
    order by count(a.id) desc
    limit p_limit;
$$;

-- Per-member breakdown for a single team (left join so all members appear).
-- Counts only activity tied to that team's ENDED sessions.
create or replace function team_member_usage(p_team uuid)
returns table (
    user_id           uuid,
    name              text,
    email             text,
    team_role         text,
    sessions_attended bigint,
    total_minutes     numeric
) language sql stable as $$
    select
        u.id, u.name, u.email, m.role,
        count(a.session_id),
        coalesce(round(sum(a.duration_mins)), 0)
    from org_team_members m
    join org_users u on u.id = m.user_id
    left join user_meeting_activity a
        on a.user_id = m.user_id
       and a.team_id = p_team
       and a.session_id in (
            select session_id from jarvis_sessions
            where team_id = p_team and status = 'ended'
       )
    where m.team_id = p_team
    group by u.id, u.name, u.email, m.role;
$$;
