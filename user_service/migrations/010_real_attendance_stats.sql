-- ============================================================
-- Migration 010: point analytics at REAL attendance (Phase 4)
--
-- Repoints the per-user aggregations from the team-proxy user_meeting_activity
-- to meeting_participants (real attendees, real per-person minutes from join/leave).
-- Team-level session counts (team_usage_stats) stay on jarvis_sessions — those are
-- meeting-level, not attendance-level, so they're already correct.
--
-- create-or-replace with identical return columns → analytics.py is unchanged;
-- it just gets real numbers. Idempotent. Run once in the SQL editor.
-- ============================================================

-- Top users in an org by meetings ACTUALLY attended (was user_meeting_activity).
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
        count(mp.id),
        coalesce(round(sum(mp.duration_mins)), 0)
    from org_users u
    join meeting_participants mp on mp.user_id = u.id
    where u.org_id = p_org
    group by u.id, u.name, u.email, u.role
    order by count(mp.id) desc
    limit p_limit;
$$;

-- Per-member breakdown for a team, from real attendance (was user_meeting_activity).
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
        count(mp.id),
        coalesce(round(sum(mp.duration_mins)), 0)
    from org_team_members m
    join org_users u on u.id = m.user_id
    left join meeting_participants mp
        on mp.user_id = m.user_id and mp.team_id = p_team
    where m.team_id = p_team
    group by u.id, u.name, u.email, m.role;
$$;

-- Real per-user rollup for GET /users/me/stats.
create or replace function user_attendance_stats(
    p_user  uuid,
    p_week  timestamptz,
    p_month timestamptz
)
returns table (
    meetings_attended bigint,
    total_minutes     numeric,
    dur_count         bigint,
    attended_week     bigint,
    attended_month    bigint,
    last_attended     timestamptz
) language sql stable as $$
    select
        count(*),
        coalesce(round(sum(duration_mins)), 0),
        count(*) filter (where duration_mins is not null and duration_mins > 0),
        count(*) filter (where joined_at >= p_week),
        count(*) filter (where joined_at >= p_month),
        max(coalesce(left_at, joined_at))
    from meeting_participants
    where user_id = p_user;
$$;
