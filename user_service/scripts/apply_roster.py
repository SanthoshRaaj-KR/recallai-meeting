"""
Apply the real roster to the org (idempotent).

Target state:
  * genreal.ai@gmail.com         -> CEO  (org owner, unchanged)
  * santhoshraajkr.17@gmail.com  -> ADMIN (was MANAGER; removed as Engineering manager)
  * santhoshraaj1710@gmail.com   -> MANAGER of Engineering (created if absent, Google-auth, no password)
  * Akshath / Vishwa             -> left as-is

Usage (from Confluence/):
    python -m user_service.scripts.apply_roster            # DRY RUN
    python -m user_service.scripts.apply_roster --apply    # execute
"""
from __future__ import annotations

import os
import sys

import requests
from dotenv import load_dotenv

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))

URL = os.getenv("SUPABASE_URL", "").rstrip("/")
KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_ANON_KEY") or ""

CEO_EMAIL = "genreal.ai@gmail.com"
ADMIN_EMAIL = "santhoshraajkr.17@gmail.com"
NEW_MGR_EMAIL = "santhoshraaj1710@gmail.com"
NEW_MGR_NAME = "Santhosh Raaj"
ENG_TEAM = "Engineering"

APPLY = "--apply" in sys.argv


def _h(extra: dict | None = None) -> dict:
    h = {"apikey": KEY, "Authorization": f"Bearer {KEY}", "Content-Type": "application/json"}
    if extra:
        h.update(extra)
    return h


def get_one(table: str, params: dict) -> dict | None:
    r = requests.get(f"{URL}/rest/v1/{table}", headers=_h(), params={**params, "limit": "1"}, timeout=15)
    r.raise_for_status()
    rows = r.json()
    return rows[0] if rows else None


def patch(table: str, params: dict, data: dict) -> None:
    if not APPLY:
        print(f"   [dry] PATCH {table} {params} <- {data}"); return
    r = requests.patch(f"{URL}/rest/v1/{table}", headers=_h({"Prefer": "return=representation"}),
                       params=params, json=data, timeout=15)
    r.raise_for_status()
    print(f"   OK PATCH {table} {params}")


def insert(table: str, data: dict) -> dict:
    if not APPLY:
        print(f"   [dry] INSERT {table} <- {data}"); return {**data, "id": "<new>"}
    r = requests.post(f"{URL}/rest/v1/{table}", headers=_h({"Prefer": "return=representation"}),
                      json=data, timeout=15)
    r.raise_for_status()
    rows = r.json()
    print(f"   OK INSERT {table}")
    return rows[0] if rows else data


def insert_ignore(table: str, data: dict) -> None:
    if not APPLY:
        print(f"   [dry] INSERT(ignore) {table} <- {data}"); return
    r = requests.post(f"{URL}/rest/v1/{table}", headers=_h({"Prefer": "resolution=ignore-duplicates"}),
                      json=data, timeout=15)
    if not r.ok and r.status_code != 409:
        print(f"   FAIL INSERT {table}: {r.status_code} {r.text[:120]}")
    else:
        print(f"   OK INSERT(ignore) {table}")


def delete(table: str, params: dict) -> None:
    if not APPLY:
        print(f"   [dry] DELETE {table} {params}"); return
    r = requests.delete(f"{URL}/rest/v1/{table}", headers=_h(), params=params, timeout=15)
    if not r.ok and r.status_code != 404:
        print(f"   FAIL DELETE {table}: {r.status_code} {r.text[:120]}")
    else:
        print(f"   OK DELETE {table} {params}")


def main() -> None:
    if not URL or not KEY:
        sys.exit("SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY must be set in user_service/.env")
    print(f"\n=== Apply roster ({'APPLY' if APPLY else 'DRY RUN'}) ===\n")

    ceo = get_one("org_users", {"email": f"eq.{CEO_EMAIL}"})
    if not ceo:
        sys.exit(f"CEO {CEO_EMAIL} not found — cannot proceed.")
    org_id = ceo["org_id"]
    ceo_id = ceo["id"]
    print(f"Org owner: {CEO_EMAIL}  role={ceo['role']}  org_id={org_id}")
    if ceo["role"] != "CEO":
        patch("org_users", {"id": f"eq.{ceo_id}"}, {"role": "CEO"})
    else:
        print("   (already CEO — no change)")

    eng = get_one("org_teams", {"name": f"eq.{ENG_TEAM}", "org_id": f"eq.{org_id}"})
    if not eng:
        sys.exit(f"Team '{ENG_TEAM}' not found.")
    eng_id = eng["id"]
    print(f"Engineering team: {eng_id}")

    # 1) santhoshraajkr.17 -> ADMIN, drop Engineering manager membership
    print(f"\n1. {ADMIN_EMAIL} -> ADMIN")
    admin_u = get_one("org_users", {"email": f"eq.{ADMIN_EMAIL}"})
    if not admin_u:
        print("   not found — skipping (expected to exist from seed)")
    else:
        if admin_u["role"] != "ADMIN":
            try:
                patch("org_users", {"id": f"eq.{admin_u['id']}"}, {"role": "ADMIN"})
            except requests.exceptions.HTTPError as exc:
                body = exc.response.text if exc.response is not None else ""
                if "23514" in body or "role_check" in body:
                    sys.exit(
                        "\n  ✗ The DB rejects role='ADMIN' (CHECK constraint).\n"
                        "    Run migrations/004_add_admin_role.sql in the Supabase SQL editor first,\n"
                        "    then re-run this script. No changes were made."
                    )
                raise
        else:
            print("   (already ADMIN)")
        # remove as Engineering team member (1710 takes over as manager)
        delete("org_team_members", {"team_id": f"eq.{eng_id}", "user_id": f"eq.{admin_u['id']}"})

    # 2) santhoshraaj1710 -> create (if absent) + MANAGER of Engineering
    print(f"\n2. {NEW_MGR_EMAIL} -> MANAGER of {ENG_TEAM}")
    mgr = get_one("org_users", {"email": f"eq.{NEW_MGR_EMAIL}"})
    if not mgr:
        mgr = insert("org_users", {
            "email": NEW_MGR_EMAIL, "name": NEW_MGR_NAME, "role": "MANAGER",
            "org_id": org_id, "is_active": True,
        })
    else:
        print(f"   exists (id={mgr['id']}) — ensuring role=MANAGER")
        if mgr.get("role") != "MANAGER":
            patch("org_users", {"id": f"eq.{mgr['id']}"}, {"role": "MANAGER"})
    mgr_id = mgr["id"]
    insert_ignore("org_team_members", {"team_id": eng_id, "user_id": mgr_id, "role": "MANAGER"})
    # hierarchy: self-loop + CEO -> manager (depth 1)
    insert_ignore("org_reporting_hierarchy", {"ancestor_id": mgr_id, "descendant_id": mgr_id, "depth": 0})
    if ceo_id and mgr_id != "<new>":
        insert_ignore("org_reporting_hierarchy", {"ancestor_id": ceo_id, "descendant_id": mgr_id, "depth": 1})

    print("\nDone." if APPLY else "\nDRY RUN complete — re-run with --apply.")


if __name__ == "__main__":
    main()
