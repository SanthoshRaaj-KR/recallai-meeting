"""Org hierarchy and reporting-tree routes."""

from fastapi import APIRouter, Depends, HTTPException

from ..auth import get_current_user
from ..database import select, select_one
from ..models import UserOut, HierarchyNode, PersonalHierarchy, OrgRole
from ..rbac import require_admin_or_above

router = APIRouter(prefix="/org", tags=["hierarchy"])


def _to_user_out(row: dict) -> UserOut:
    return UserOut(
        id=row["id"], email=row["email"], name=row["name"],
        role=row["role"], org_id=row.get("org_id"),
        is_active=row.get("is_active", True), created_at=row["created_at"],
    )


def _build_tree(user_id: str, all_users: dict[str, dict], hierarchy_rows: list[dict]) -> HierarchyNode:
    direct_report_ids = [
        r["descendant_id"] for r in hierarchy_rows
        if r["ancestor_id"] == user_id and r["depth"] == 1
    ]
    children = [
        _build_tree(uid, all_users, hierarchy_rows)
        for uid in direct_report_ids
        if uid in all_users
    ]
    return HierarchyNode(user=_to_user_out(all_users[user_id]), direct_reports=children)


@router.get("/hierarchy", response_model=HierarchyNode)
def full_org_hierarchy(claims: dict = Depends(require_admin_or_above())):
    """CEO or ADMIN: returns the full org tree rooted at the CEO."""
    org_id = claims.get("org_id")
    users_list = select("org_users", {"org_id": f"eq.{org_id}"})
    if not users_list:
        raise HTTPException(404, "No users found for this organisation")

    all_users = {u["id"]: u for u in users_list}
    hierarchy_rows = select("org_reporting_hierarchy", {})

    # Find the CEO
    ceo = next((u for u in users_list if u["role"] == OrgRole.CEO), None)
    if not ceo:
        raise HTTPException(404, "No CEO found in this organisation")

    return _build_tree(ceo["id"], all_users, hierarchy_rows)


@router.get("/hierarchy/me", response_model=PersonalHierarchy)
def my_hierarchy(claims: dict = Depends(get_current_user)):
    """Any user: their own reporting view — the manager chain above them and the
    subtree of people who report to them. Powers the member/manager org chart."""
    uid = claims["sub"]
    me = select_one("org_users", {"id": f"eq.{uid}"})
    if not me:
        raise HTTPException(404, "User not found")

    # Manager chain: ancestors at depth>0, nearest manager first.
    anc_rows = select("org_reporting_hierarchy", {
        "descendant_id": f"eq.{uid}", "depth": "gt.0", "order": "depth.asc",
    })
    manager_chain: list[UserOut] = []
    for r in anc_rows:
        u = select_one("org_users", {"id": f"eq.{r['ancestor_id']}"})
        if u:
            manager_chain.append(_to_user_out(u))

    # Reports subtree: reuse _build_tree rooted at this user over the org's rows.
    org_id = me.get("org_id")
    users_list = select("org_users", {"org_id": f"eq.{org_id}"}) if org_id else []
    all_users = {u["id"]: u for u in users_list}
    all_users.setdefault(uid, me)
    hierarchy_rows = select("org_reporting_hierarchy", {})
    subtree = _build_tree(uid, all_users, hierarchy_rows)

    return PersonalHierarchy(
        me=_to_user_out(me),
        manager_chain=manager_chain,
        reports=subtree.direct_reports,
    )
