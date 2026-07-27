-- ============================================================
-- Migration 013: count MEETINGS attended, not attendance ROWS
--
-- meeting_participants is unique on (session_id, recall_participant_id), and the
-- meeting platform issues a NEW participant id when someone drops and rejoins.
-- One person with one reconnect therefore produces two rows for a single
-- meeting, and the 010 RPCs' count(mp.id) reported that as two meetings
-- attended. Minutes were always additive across those rows and stay correct.
--
-- Fix: count(distinct mp.session_id) everywhere a MEETING count is intended.
--
-- Also excludes guest rows (user_id is null) from the team breakdown's join, so
-- an unresolved attendee can't be credited to a team member.
--
-- create-or-replace with identical return columns -> analytics.py unchanged.
-- Idempotent. Run once in the Supabase SQL editor.
-- ============================================================

-- Top users in an org by meetings ACTUALLY attended.
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
        count(distinct mp.session_id),
        coalesce(round(sum(mp.duration_mins)), 0)
    from org_users u
    join meeting_participants mp on mp.user_id = u.id
    where u.org_id = p_org
    group by u.id, u.name, u.email, u.role
    order by count(distinct mp.session_id) desc
    limit p_limit;
$$;

-- Per-member breakdown for a team, from real attendance.
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
        count(distinct mp.session_id),
        coalesce(round(sum(mp.duration_mins)), 0)
    from org_team_members m
    join org_users u on u.id = m.user_id
    left join meeting_participants mp
        on mp.user_id = m.user_id and mp.team_id = p_team
    where m.team_id = p_team
    group by u.id, u.name, u.email, m.role;
$$;

-- Real per-user rollup for GET /users/me/stats.
--
-- The week/month/last-attended figures are per MEETING, so they count distinct
-- sessions whose FIRST join falls in the window rather than raw rows.
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
    with per_session as (
        select
            session_id,
            sum(duration_mins)                     as mins,
            min(joined_at)                         as first_joined,
            max(coalesce(left_at, joined_at))      as last_seen
        from meeting_participants
        where user_id = p_user
        group by session_id
    )
    select
        count(*),
        coalesce(round(sum(mins)), 0),
        count(*) filter (where mins is not null and mins > 0),
        count(*) filter (where first_joined >= p_week),
        count(*) filter (where first_joined >= p_month),
        max(last_seen)
    from per_session;
$$;
