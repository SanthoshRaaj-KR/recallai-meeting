"""Team CRUD + member management routes."""

import datetime
import logging
import secrets

from fastapi import APIRouter, Depends, HTTPException, status

from ..auth import get_current_user
from ..database import select, select_one, insert, update, delete, DBError, find_by_text_ci
from ..models import (
    TeamCreate, TeamOut, TeamUpdate,
    AddMemberRequest, MemberOut, UserOut,
    TeamInviteCreate, TeamInviteOut,
    OrgRole, TeamRole,
)
from ..rbac import (
    require_ceo, require_admin_or_above, require_manager_or_above,
    can_manage_team,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/teams", tags=["teams"])

# How long a team invite stays valid. Single source of truth: the code always
# sets expires_at explicitly, and migration 014 realigns the column default to
# match so reading the schema doesn't suggest a different (48h) lifetime.
INVITE_TTL = datetime.timedelta(hours=1)


def _enrich_team(row: dict, claims: dict) -> TeamOut:
    members = select("org_team_members", {"team_id": f"eq.{row['id']}"})
    bot = select_one("org_team_bots", {"team_id": f"eq.{row['id']}"})
    # The caller's role *within* this team; falls back to their org role so an
    # ADMIN/CEO who isn't a member still reads as a manager on the client.
    viewer_role = next(
        (m["role"] for m in members if m["user_id"] == claims.get("sub")), None
    )
    if viewer_role is None and claims.get("role") in OrgRole.admin_and_above:
        viewer_role = claims.get("role")
    return TeamOut(
        id=row["id"],
        name=row["name"],
        org_id=row["org_id"],
        created_at=row["created_at"],
        member_count=len(members),
        bot_assigned=bot is not None,
        description=row.get("description"),
        viewer_team_role=viewer_role,
    )


@router.get("", response_model=list[TeamOut])
def list_teams(claims: dict = Depends(get_current_user)):
    if claims["role"] in (OrgRole.CEO, OrgRole.ADMIN):
        rows = select("org_teams", {"org_id": f"eq.{claims['org_id']}"})
    else:
        # Return only teams this user belongs to
        memberships = select("org_team_members", {"user_id": f"eq.{claims['sub']}"})
        team_ids = [m["team_id"] for m in memberships]
        rows = [select_one("org_teams", {"id": f"eq.{tid}"}) for tid in team_ids]
        rows = [r for r in rows if r]
    return [_enrich_team(r, claims) for r in rows]


@router.post("", response_model=TeamOut, status_code=status.HTTP_201_CREATED)
def create_team(body: TeamCreate, claims: dict = Depends(require_admin_or_above())):
    team_data: dict = {"name": body.name, "org_id": body.org_id}
    if body.description is not None:
        team_data["description"] = body.description
    try:
        row = insert("org_teams", team_data)
    except DBError as e:
        raise HTTPException(500, str(e))
    return _enrich_team(row, claims)


@router.get("/{team_id}", response_model=TeamOut)
def get_team(team_id: str, claims: dict = Depends(get_current_user)):
    row = select_one("org_teams", {"id": f"eq.{team_id}"})
    if not row:
        raise HTTPException(404, "Team not found")
    if claims["role"] not in (OrgRole.CEO, OrgRole.ADMIN):
        member = select_one("org_team_members", {
            "team_id": f"eq.{team_id}", "user_id": f"eq.{claims['sub']}",
        })
        if not member:
            raise HTTPException(403, "Not a member of this team")
    return _enrich_team(row, claims)


@router.patch("/{team_id}", response_model=TeamOut)
def update_team(team_id: str, body: TeamUpdate, claims: dict = Depends(get_current_user)):
    row = select_one("org_teams", {"id": f"eq.{team_id}"})
    if not row:
        raise HTTPException(404, "Team not found")
    if not can_manage_team(claims, team_id):
        raise HTTPException(403, "Only team manager, ADMIN, or CEO can update team")
    updates = body.model_dump(exclude_none=True)
    if not updates:
        raise HTTPException(400, "No fields to update")
    rows = update("org_teams", {"id": f"eq.{team_id}"}, updates)
    return _enrich_team(rows[0], claims)


@router.delete("/{team_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_team(team_id: str, claims: dict = Depends(require_admin_or_above())):
    delete("org_teams", {"id": f"eq.{team_id}"})


# ── Members ────────────────────────────────────────────────────────────────────

@router.get("/{team_id}/members", response_model=list[MemberOut])
def list_members(team_id: str, claims: dict = Depends(get_current_user)):
    row = select_one("org_teams", {"id": f"eq.{team_id}"})
    if not row:
        raise HTTPException(404, "Team not found")
    if claims["role"] not in (OrgRole.CEO, OrgRole.ADMIN):
        member = select_one("org_team_members", {
            "team_id": f"eq.{team_id}", "user_id": f"eq.{claims['sub']}",
        })
        if not member:
            raise HTTPException(403, "Not authorised to view this team")

    members = select("org_team_members", {"team_id": f"eq.{team_id}"})
    result = []
    for m in members:
        user_row = select_one("org_users", {"id": f"eq.{m['user_id']}"})
        user_out = None
        if user_row:
            user_out = UserOut(
                id=user_row["id"], email=user_row["email"], name=user_row["name"],
                role=user_row["role"], org_id=user_row.get("org_id"),
                is_active=user_row.get("is_active", True), created_at=user_row["created_at"],
            )
        result.append(MemberOut(
            user_id=m["user_id"], team_id=team_id,
            role=m["role"], joined_at=m["joined_at"], user=user_out,
        ))
    return result


@router.post("/{team_id}/members", response_model=MemberOut, status_code=status.HTTP_201_CREATED)
def add_member(team_id: str, body: AddMemberRequest, claims: dict = Depends(get_current_user)):
    """Add an existing user to a team. CEO/ADMIN, or the team's MANAGER, may call."""
    team = select_one("org_teams", {"id": f"eq.{team_id}"})
    if not team:
        raise HTTPException(404, "Team not found")
    if not can_manage_team(claims, team_id):
        raise HTTPException(403, "Only team manager, ADMIN, or CEO can add members")
    if body.role not in (TeamRole.MANAGER, TeamRole.MEMBER, TeamRole.ASSOCIATE):
        raise HTTPException(400, f"Invalid team role: {body.role}")

    # Prevent duplicate
    existing = select_one("org_team_members", {
        "team_id": f"eq.{team_id}", "user_id": f"eq.{body.user_id}",
    })
    if existing:
        raise HTTPException(409, "User is already a member of this team")

    try:
        row = insert("org_team_members", {
            "team_id": team_id, "user_id": body.user_id, "role": body.role,
        })
    except DBError as e:
        raise HTTPException(500, str(e))

    # Wire hierarchy: if the added user is MANAGER, link them to CEO in closure table.
    # For MEMBER/ASSOCIATE, find their team's manager and create reporting links.
    _wire_hierarchy(team_id, body.user_id, body.role)

    return MemberOut(
        user_id=body.user_id, team_id=team_id,
        role=body.role, joined_at=row["joined_at"],
    )


@router.delete("/{team_id}/members/{user_id}", status_code=status.HTTP_204_NO_CONTENT)
def remove_member(team_id: str, user_id: str, claims: dict = Depends(get_current_user)):
    """Remove a member from a team. CEO/ADMIN, or the team's MANAGER, may call."""
    team = select_one("org_teams", {"id": f"eq.{team_id}"})
    if not team:
        raise HTTPException(404, "Team not found")
    if not can_manage_team(claims, team_id):
        raise HTTPException(403, "Only team manager, ADMIN, or CEO can remove members")
    delete("org_team_members", {"team_id": f"eq.{team_id}", "user_id": f"eq.{user_id}"})
    _unwire_hierarchy(user_id)


@router.post("/{team_id}/invite", response_model=TeamInviteOut, status_code=status.HTTP_201_CREATED)
def invite_member(team_id: str, body: TeamInviteCreate, claims: dict = Depends(get_current_user)):
    """CEO or team manager: send an email invite to join this team."""
    team = select_one("org_teams", {"id": f"eq.{team_id}"})
    if not team:
        raise HTTPException(404, "Team not found")
    if not can_manage_team(claims, team_id):
        raise HTTPException(403, "Only team manager, ADMIN, or CEO can invite members")
    if body.role not in (TeamRole.MANAGER, TeamRole.MEMBER, TeamRole.ASSOCIATE):
        raise HTTPException(400, f"Invalid team role: {body.role}")

    email = (body.email or "").strip()
    if not email:
        raise HTTPException(400, "email is required")

    # Lazy GC (no scheduler in this stack): drop expired pending invites first.
    try:
        now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
        delete("org_team_invitations", {"expires_at": f"lt.{now_iso}", "status": "eq.pending"})
    except Exception:
        pass

    # Supersede any live invite for this address+team. Without this, re-inviting
    # somebody (because the first mail was missed, or the hour lapsed while they
    # were away) left several codes valid at once — each an independent way into
    # the org, and each expiring at a different time.
    try:
        for stale in select("org_team_invitations", {
            "team_id": f"eq.{team_id}", "status": "eq.pending",
        }):
            if (stale.get("email") or "").strip().lower() == email.lower():
                delete("org_team_invitations", {"code": f"eq.{stale['code']}"})
    except Exception as exc:
        logger.warning("[invite] could not supersede previous invites: %s", exc)

    # 16 hex chars = 64 bits. The previous 6-char code was 24 bits (~16.7M),
    # which is brute-forceable against a public, unauthenticated lookup. The code
    # travels in a link, so its length costs the user nothing.
    code = secrets.token_hex(8).upper()
    expires_at = (
        datetime.datetime.now(datetime.timezone.utc) + INVITE_TTL
    ).isoformat()

    inviter = select_one("org_users", {"id": f"eq.{claims['sub']}"})
    inviter_name = inviter["name"] if inviter else "A team admin"

    try:
        row = insert("org_team_invitations", {
            "team_id": team_id,
            "email": email,
            "role": body.role,
            "code": code,
            "status": "pending",
            "inviter_id": claims["sub"],
            "expires_at": expires_at,
        })
    except DBError as e:
        raise HTTPException(500, str(e))

    try:
        from ..email import send_invite_email
        send_invite_email(email, team["name"], inviter_name, code, body.role)
    except Exception as exc:
        logger.warning("[invite] Email delivery failed (invite code still valid): %s", exc)

    # Case-insensitive, like every other org_users email lookup. An exact eq.
    # match reported an existing member as a new user whenever the invite was
    # typed in different casing than the stored address, which made the accept
    # page ask them for a name they already had.
    existing = find_by_text_ci("org_users", "email", email)
    return TeamInviteOut(
        id=row["id"],
        team_id=team_id,
        email=email,
        role=body.role,
        code=code,
        status="pending",
        inviter_id=claims["sub"],
        created_at=row["created_at"],
        expires_at=expires_at,
        team_name=team["name"],
        user_exists=existing is not None,
    )


def _unwire_hierarchy(user_id: str) -> None:
    """Detach a user from the reporting tree after a team removal.

    Only drops the edges that the removal actually invalidated:

    - reporting edges ABOVE them (depth > 0) — they no longer report to that
      team's manager. Their depth-0 self-loop is preserved, because deleting it
      erases the person from the org chart entirely rather than just detaching
      them, and it is what every hierarchy query anchors on.
    - reporting edges BELOW them (depth > 0) — they no longer manage anyone.

    If the user is still on ANOTHER team, they are re-wired underneath that
    team's manager instead of being left dangling. The old code deleted every
    edge unconditionally, so removing someone from one of two teams erased their
    position in the org chart completely.
    """
    remaining = [
        m for m in select("org_team_members", {"user_id": f"eq.{user_id}"})
        if m.get("team_id")
    ]

    # Ancestors above them, and reports below them — but never the self-loop.
    delete("org_reporting_hierarchy", {"descendant_id": f"eq.{user_id}", "depth": "gt.0"})
    delete("org_reporting_hierarchy", {"ancestor_id": f"eq.{user_id}", "depth": "gt.0"})

    # Still on another team → re-attach under that team's manager so they keep a
    # place in the chart.
    for m in remaining:
        _wire_hierarchy(m["team_id"], user_id, m.get("role") or TeamRole.MEMBER)
        break


def _team_manager_id(team_id: str) -> str | None:
    """Return the user_id of the team's MANAGER, or None if the team has none."""
    row = select_one("org_team_members", {
        "team_id": f"eq.{team_id}", "role": f"eq.{TeamRole.MANAGER}",
    })
    return row["user_id"] if row else None


def _link_member_to_manager(user_id: str, manager_id: str) -> None:
    """Point one report at their manager: a depth-1 edge, plus a copy of the
    manager's own ancestor chain at depth+1 so the member also rolls up to the CEO.

    Idempotent — a duplicate (ancestor, descendant) pair violates the closure
    table's primary key and raises DBError, which we swallow.
    """
    if not manager_id or user_id == manager_id:
        return
    try:
        insert("org_reporting_hierarchy", {
            "ancestor_id": manager_id, "descendant_id": user_id, "depth": 1,
        })
    except DBError:
        pass
    ancestors = select("org_reporting_hierarchy", {
        "descendant_id": f"eq.{manager_id}", "depth": "gt.0",
    })
    for anc in ancestors:
        try:
            insert("org_reporting_hierarchy", {
                "ancestor_id": anc["ancestor_id"],
                "descendant_id": user_id,
                "depth": anc["depth"] + 1,
            })
        except DBError:
            pass


def _wire_existing_members_under_manager(team_id: str, manager_id: str) -> None:
    """Retro-wire every current MEMBER/ASSOCIATE of a team under its manager.

    Called when a MANAGER is added to (or backfilled onto) a team: members who
    were added *before* the manager existed had no reporting edge and were left
    orphaned. Additive and idempotent — it never removes an existing edge, so a
    member already reporting to this manager is untouched.
    """
    for m in select("org_team_members", {"team_id": f"eq.{team_id}"}):
        if m.get("role") == TeamRole.MANAGER or m["user_id"] == manager_id:
            continue
        _link_member_to_manager(m["user_id"], manager_id)


def _wire_hierarchy(team_id: str, user_id: str, team_role: str) -> None:
    """Insert closure-table rows so a newly added member appears in the right
    sub-tree.

    - MANAGER → linked under the CEO, then **adopts every existing member of the
      team** so people added before the manager report to them (not nobody).
    - MEMBER/ASSOCIATE → linked under the team's current manager (if any).
    """
    try:
        # Always add self-loop
        insert("org_reporting_hierarchy", {
            "ancestor_id": user_id, "descendant_id": user_id, "depth": 0,
        })
    except DBError:
        pass  # already exists (re-added after removal)

    if team_role == TeamRole.MANAGER:
        # Link MANAGER → CEO at depth 1
        ceo = select_one("org_users", {"role": f"eq.{OrgRole.CEO}"})
        if ceo and ceo["id"] != user_id:
            try:
                insert("org_reporting_hierarchy", {
                    "ancestor_id": ceo["id"], "descendant_id": user_id, "depth": 1,
                })
            except DBError:
                pass
        # A manager assigned after members already exist must adopt them, else
        # those members report to no one — the "everyone reports to the manager" fix.
        _wire_existing_members_under_manager(team_id, user_id)
    else:
        _link_member_to_manager(user_id, _team_manager_id(team_id))
