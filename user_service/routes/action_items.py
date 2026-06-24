"""Action-item workflow routes.

Lifecycle: open -> (owner submits) submitted -> (manager closes) closed.
Managers/ADMIN/CEO can also reopen or reassign. Rows live in `meeting_action_items`
(seeded by the my-agent background extractor); these routes drive the human workflow.
"""

from __future__ import annotations

import datetime as dt
import logging

from fastapi import APIRouter, Depends, HTTPException

from ..auth import get_current_user
from ..database import select, select_one, insert, update, delete, DBError
from ..email import send_action_items_email
from ..models import OrgRole

logger = logging.getLogger(__name__)

router = APIRouter(tags=["action-items"])


# ── Helpers ──────────────────────────────────────────────────────────────────

def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _is_team_manager(team_id: str | None, user_id: str) -> bool:
    if not team_id:
        return False
    return select_one("org_team_members", {
        "team_id": f"eq.{team_id}", "user_id": f"eq.{user_id}", "role": "eq.MANAGER",
    }) is not None


def _is_team_member(team_id: str | None, user_id: str) -> bool:
    if not team_id:
        return False
    return select_one("org_team_members", {
        "team_id": f"eq.{team_id}", "user_id": f"eq.{user_id}",
    }) is not None


def _team_in_org(team_id: str | None, org_id: str | None) -> bool:
    if not team_id or not org_id:
        return False
    t = select_one("org_teams", {"id": f"eq.{team_id}"})
    return bool(t and t.get("org_id") == org_id)


def _can_manage_item(item: dict, claims: dict) -> bool:
    role = claims.get("role")
    if role in (OrgRole.CEO, OrgRole.ADMIN):
        return _team_in_org(item.get("team_id"), claims.get("org_id"))
    return _is_team_manager(item.get("team_id"), claims["sub"])


def _user_map(user_ids: list[str]) -> dict[str, dict]:
    ids = [u for u in {*user_ids} if u]
    if not ids:
        return {}
    rows = select("org_users", {"id": f"in.({','.join(ids)})", "select": "id,name,email"})
    return {r["id"]: r for r in rows}


def _session_titles(session_ids: list[str]) -> dict[str, str]:
    ids = [s for s in {*session_ids} if s]
    if not ids:
        return {}
    rows = select("jarvis_sessions", {"session_id": f"in.({','.join(ids)})", "select": "session_id,summary"})
    out = {}
    for r in rows:
        summ = r.get("summary") or {}
        out[r["session_id"]] = (summ.get("title") if isinstance(summ, dict) else None) or f"Meeting {r['session_id'][:8]}"
    return out


def _enrich(items: list[dict], titles: dict[str, str] | None = None) -> list[dict]:
    umap = _user_map([i.get("owner_user_id") for i in items] + [i.get("reviewed_by") for i in items])
    for i in items:
        owner = umap.get(i.get("owner_user_id"))
        i["owner_display"] = (owner["name"] if owner else None) or i.get("owner_name")
        rev = umap.get(i.get("reviewed_by"))
        i["reviewed_by_name"] = rev["name"] if rev else None
        if titles is not None:
            i["meeting_title"] = titles.get(i.get("session_id"))
    return items


# ── Routes ───────────────────────────────────────────────────────────────────

@router.get("/me/action-items")
def my_action_items(claims: dict = Depends(get_current_user)):
    """The caller's own assigned action items across all meetings (newest first)."""
    items = select("meeting_action_items", {
        "owner_user_id": f"eq.{claims['sub']}", "order": "updated_at.desc",
    })
    titles = _session_titles([i.get("session_id") for i in items])
    return _enrich(items, titles)


@router.get("/me/action-items/review")
def items_to_review(claims: dict = Depends(get_current_user)):
    """Submitted items awaiting review for teams the caller manages (or all, for ADMIN/CEO)."""
    if claims.get("role") in (OrgRole.CEO, OrgRole.ADMIN):
        teams = select("org_teams", {"org_id": f"eq.{claims.get('org_id')}", "select": "id"})
        team_ids = [t["id"] for t in teams]
    else:
        mships = select("org_team_members", {
            "user_id": f"eq.{claims['sub']}", "role": "eq.MANAGER", "select": "team_id",
        })
        team_ids = [m["team_id"] for m in mships]
    if not team_ids:
        return []
    items = select("meeting_action_items", {
        "team_id": f"in.({','.join(team_ids)})", "status": "eq.submitted", "order": "updated_at.desc",
    })
    titles = _session_titles([i.get("session_id") for i in items])
    return _enrich(items, titles)


@router.get("/meetings/{session_id}/action-items")
def meeting_action_items(session_id: str, claims: dict = Depends(get_current_user)):
    """All action items for a meeting + the team roster (for assignment)."""
    s = select_one("jarvis_sessions", {"session_id": f"eq.{session_id}"})
    if not s:
        raise HTTPException(404, "Meeting not found")
    team_id = s.get("team_id")
    is_admin = claims.get("role") in (OrgRole.CEO, OrgRole.ADMIN)
    if not (is_admin and _team_in_org(team_id, claims.get("org_id"))) and not _is_team_member(team_id, claims["sub"]):
        raise HTTPException(403, "Not authorised to view this meeting's action items")

    items = _enrich(select("meeting_action_items", {"session_id": f"eq.{session_id}", "order": "created_at.asc"}))
    members = []
    if team_id:
        mids = [m["user_id"] for m in select("org_team_members", {"team_id": f"eq.{team_id}", "select": "user_id"})]
        umap = _user_map(mids)
        members = [{"user_id": uid, "name": umap[uid]["name"], "email": umap[uid].get("email")} for uid in mids if uid in umap]
    can_manage = is_admin or _is_team_manager(team_id, claims["sub"])
    return {"session_id": session_id, "team_id": team_id, "can_manage": can_manage, "members": members, "items": items}


@router.post("/meetings/{session_id}/action-items", status_code=201)
def add_action_item(session_id: str, body: dict, claims: dict = Depends(get_current_user)):
    """Manager/ADMIN/CEO: add a manual action item to a meeting."""
    s = select_one("jarvis_sessions", {"session_id": f"eq.{session_id}"})
    if not s:
        raise HTTPException(404, "Meeting not found")
    team_id = s.get("team_id")
    if not _can_manage_item({"team_id": team_id}, claims):
        raise HTTPException(403, "Only a team manager, ADMIN or CEO can add items")
    desc = (body.get("description") or "").strip()
    if not desc:
        raise HTTPException(400, "description is required")
    try:
        row = insert("meeting_action_items", {
            "session_id": session_id, "team_id": team_id, "org_id": claims.get("org_id"),
            "description": desc, "owner_user_id": body.get("owner_user_id"),
            "due": body.get("due"), "status": "open", "source": "manual",
        })
    except DBError as e:
        raise HTTPException(500, str(e))
    return _enrich([row])[0]


def _load_item(item_id: str) -> dict:
    item = select_one("meeting_action_items", {"id": f"eq.{item_id}"})
    if not item:
        raise HTTPException(404, "Action item not found")
    return item


@router.patch("/action-items/{item_id}/submit")
def submit_item(item_id: str, body: dict, claims: dict = Depends(get_current_user)):
    """Owner marks their item done (-> submitted)."""
    item = _load_item(item_id)
    if item.get("owner_user_id") != claims["sub"]:
        raise HTTPException(403, "Only the assigned owner can submit this item")
    rows = update("meeting_action_items", {"id": f"eq.{item_id}"}, {
        "status": "submitted", "member_note": body.get("member_note"), "updated_at": _now(),
    })
    return _enrich([rows[0]])[0]


@router.patch("/action-items/{item_id}/close")
def close_item(item_id: str, claims: dict = Depends(get_current_user)):
    """Manager/ADMIN/CEO reviews and closes a submitted item."""
    item = _load_item(item_id)
    if not _can_manage_item(item, claims):
        raise HTTPException(403, "Only a team manager, ADMIN or CEO can close items")
    rows = update("meeting_action_items", {"id": f"eq.{item_id}"}, {
        "status": "closed", "reviewed_by": claims["sub"], "reviewed_at": _now(), "updated_at": _now(),
    })
    return _enrich([rows[0]])[0]


@router.patch("/action-items/{item_id}/cancel")
def cancel_item(item_id: str, claims: dict = Depends(get_current_user)):
    """Manager/ADMIN/CEO cancels an action no longer needed (-> cancelled)."""
    item = _load_item(item_id)
    if not _can_manage_item(item, claims):
        raise HTTPException(403, "Only a team manager, ADMIN or CEO can cancel items")
    rows = update("meeting_action_items", {"id": f"eq.{item_id}"}, {
        "status": "cancelled", "reviewed_by": claims["sub"], "reviewed_at": _now(), "updated_at": _now(),
    })
    return _enrich([rows[0]])[0]


@router.patch("/action-items/{item_id}/reopen")
def reopen_item(item_id: str, claims: dict = Depends(get_current_user)):
    """Manager/ADMIN/CEO reopens an item (-> open)."""
    item = _load_item(item_id)
    if not _can_manage_item(item, claims):
        raise HTTPException(403, "Only a team manager, ADMIN or CEO can reopen items")
    rows = update("meeting_action_items", {"id": f"eq.{item_id}"}, {
        "status": "open", "reviewed_by": None, "reviewed_at": None, "updated_at": _now(),
    })
    return _enrich([rows[0]])[0]


@router.patch("/action-items/{item_id}")
def edit_item(item_id: str, body: dict, claims: dict = Depends(get_current_user)):
    """Manager/ADMIN/CEO: reassign owner or edit description/due."""
    item = _load_item(item_id)
    if not _can_manage_item(item, claims):
        raise HTTPException(403, "Only a team manager, ADMIN or CEO can edit items")
    patch = {k: body[k] for k in ("owner_user_id", "description", "due") if k in body}
    if not patch:
        raise HTTPException(400, "Nothing to update")
    if "owner_user_id" in patch:
        patch["owner_name"] = None  # clear raw name once a real user is assigned
    patch["updated_at"] = _now()
    rows = update("meeting_action_items", {"id": f"eq.{item_id}"}, patch)
    return _enrich([rows[0]])[0]


@router.delete("/action-items/{item_id}", status_code=204)
def delete_item(item_id: str, claims: dict = Depends(get_current_user)):
    item = _load_item(item_id)
    if not _can_manage_item(item, claims):
        raise HTTPException(403, "Only a team manager, ADMIN or CEO can delete items")
    delete("meeting_action_items", {"id": f"eq.{item_id}"})


@router.post("/meetings/{session_id}/action-items/email")
def email_action_items(session_id: str, claims: dict = Depends(get_current_user)):
    """Manager/ADMIN/CEO: email each assigned owner their open items for this meeting."""
    s = select_one("jarvis_sessions", {"session_id": f"eq.{session_id}"})
    if not s:
        raise HTTPException(404, "Meeting not found")
    if not _can_manage_item({"team_id": s.get("team_id")}, claims):
        raise HTTPException(403, "Only a team manager, ADMIN or CEO can email owners")
    items = select("meeting_action_items", {"session_id": f"eq.{session_id}", "status": "eq.open"})
    by_owner: dict[str, list[dict]] = {}
    for i in items:
        oid = i.get("owner_user_id")
        if oid:
            by_owner.setdefault(oid, []).append(i)
    if not by_owner:
        return {"ok": True, "emailed": 0}
    summ = (s.get("summary") or {})
    title = summ.get("title") if isinstance(summ, dict) else None
    umap = _user_map(list(by_owner.keys()))
    sent = 0
    for oid, its in by_owner.items():
        u = umap.get(oid)
        if u and u.get("email"):
            try:
                send_action_items_email(u["email"], u.get("name", "there"), its, title)
                sent += 1
            except Exception as exc:
                logger.warning("email_action_items: failed for %s: %s", oid, exc)
    return {"ok": True, "emailed": sent}
