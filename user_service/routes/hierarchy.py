"""Org hierarchy and reporting-tree routes."""

from fastapi import APIRouter, Depends, HTTPException

from ..auth import get_current_user
from ..database import select, select_one
from ..models import UserOut, HierarchyNode, OrgRole
from ..rbac import require_ceo

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
def full_org_hierarchy(claims: dict = Depends(require_ceo())):
    """CEO only: returns the full org tree rooted at the CEO."""
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
