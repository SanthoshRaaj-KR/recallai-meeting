"""
Full seed script — creates/upserts CEO + 3 managers with rich random data,
creates an org, and wires up the hierarchy.

Run from Confluence/:
    python -m user_service.scripts.seed_managers

Required env vars: SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY
Optional:  ADMIN_PASSWORD (default: Jarvis@2024!)
"""
from __future__ import annotations

import os
import sys
import secrets
import requests
from dotenv import load_dotenv

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))

SUPABASE_URL = os.getenv("SUPABASE_URL", "").rstrip("/")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_ANON_KEY") or ""
DEFAULT_PASSWORD = os.getenv("ADMIN_PASSWORD", "Jarvis@2024!")

# ── Seed data ──────────────────────────────────────────────────────────────────

ORG_NAME = "GenReal AI"

CEO = {
    "email": "genreal.ai@gmail.com",
    "name": "GenReal AI",
    "role": "CEO",
    "job_title": "Chief Executive Officer",
    "department": "Executive",
    "phone": "+91-9000000000",
    "bio": "CEO of GenReal AI — building the future of meeting intelligence.",
    "avatar_url": None,
}

MANAGERS = [
    {
        "email": "santhoshraajkr.17@gmail.com",
        "name": "Santhosh Raaj K R",
        "role": "MANAGER",
        "job_title": "Lead Software Engineer",
        "department": "Engineering",
        "phone": "+91-9876543210",
        "bio": "Full-stack engineer passionate about AI systems and developer tooling.",
        "avatar_url": None,
    },
    {
        "email": "akshath.r333@gmail.com",
        "name": "Akshath R",
        "role": "MANAGER",
        "job_title": "Senior Product Manager",
        "department": "Product",
        "phone": "+91-9123456780",
        "bio": "Product manager focused on user experience and AI product strategy.",
        "avatar_url": None,
    },
    {
        "email": "vishy6400@gmail.com",
        "name": "Vishwa V",
        "role": "MANAGER",
        "job_title": "Design Lead",
        "department": "Design",
        "phone": "+91-9988776655",
        "bio": "Design lead crafting intuitive interfaces for enterprise AI tools.",
        "avatar_url": None,
    },
]

# ── HTTP helpers ───────────────────────────────────────────────────────────────

def _headers(return_repr: bool = False) -> dict:
    h = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json",
    }
    if return_repr:
        h["Prefer"] = "return=representation"
    return h


def _get(table: str, params: dict) -> list[dict]:
    r = requests.get(f"{SUPABASE_URL}/rest/v1/{table}", headers=_headers(), params=params, timeout=10)
    if not r.ok:
        sys.exit(f"GET {table} failed: {r.text[:200]}")
    return r.json()


def _upsert(table: str, data: dict, on_conflict: str) -> dict:
    r = requests.post(
        f"{SUPABASE_URL}/rest/v1/{table}",
        headers={**_headers(), "Prefer": f"resolution=merge-duplicates,return=representation"},
        params={"on_conflict": on_conflict},
        json=data,
        timeout=10,
    )
    if not r.ok:
        print(f"  ✗  UPSERT {table} failed: {r.text[:200]}")
        return {}
    rows = r.json()
    return rows[0] if rows else data


def _insert_ignore(table: str, data: dict) -> dict:
    r = requests.post(
        f"{SUPABASE_URL}/rest/v1/{table}",
        headers={**_headers(), "Prefer": "resolution=ignore-duplicates,return=representation"},
        json=data,
        timeout=10,
    )
    if not r.ok and r.status_code != 409:
        print(f"  ✗  INSERT {table} failed ({r.status_code}): {r.text[:200]}")
        return {}
    rows = r.json()
    return rows[0] if (r.ok and rows) else data


# ── Password hashing (same as auth.py) ────────────────────────────────────────

def _hash_password(plain: str) -> str:
    import bcrypt
    return bcrypt.hashpw(plain.encode(), bcrypt.gensalt(rounds=12)).decode()


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    if not SUPABASE_URL or not SUPABASE_KEY:
        sys.exit("SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY must be set in user_service/.env")

    print(f"\n=== Jarvis Org Seeder ===\n")
    print(f"Password for all seeded accounts: {DEFAULT_PASSWORD}\n")

    # 1. Create / get org
    print(f"1. Org: {ORG_NAME}")
    existing_orgs = _get("organizations", {"name": f"eq.{ORG_NAME}"})
    if existing_orgs:
        org = existing_orgs[0]
        print(f"   ↳ existing org_id = {org['id']}")
    else:
        r = requests.post(
            f"{SUPABASE_URL}/rest/v1/organizations",
            headers={**_headers(), "Prefer": "return=representation"},
            json={"name": ORG_NAME},
            timeout=10,
        )
        if not r.ok:
            sys.exit(f"Could not create org: {r.text}")
        org = r.json()[0]
        print(f"   ✓  created org_id = {org['id']}")
    org_id = org["id"]

    pw_hash = _hash_password(DEFAULT_PASSWORD)

    # 2. Upsert CEO
    print(f"\n2. CEO: {CEO['email']}")
    ceo_row = _upsert("org_users", {
        **CEO,
        "password_hash": pw_hash,
        "org_id": org_id,
        "is_active": True,
    }, "email")
    ceo_id = ceo_row.get("id")
    print(f"   ✓  id = {ceo_id}")

    # CEO self-loop
    if ceo_id:
        _insert_ignore("org_reporting_hierarchy", {
            "ancestor_id": ceo_id, "descendant_id": ceo_id, "depth": 0,
        })

    # 3. Upsert managers
    print(f"\n3. Managers:")
    manager_ids: list[str] = []
    for mgr in MANAGERS:
        row = _upsert("org_users", {
            **mgr,
            "password_hash": pw_hash,
            "org_id": org_id,
            "is_active": True,
        }, "email")
        mgr_id = row.get("id")
        manager_ids.append(mgr_id)
        print(f"   ✓  {mgr['email']}  id = {mgr_id}")

        if mgr_id and ceo_id:
            # Self-loop
            _insert_ignore("org_reporting_hierarchy", {
                "ancestor_id": mgr_id, "descendant_id": mgr_id, "depth": 0,
            })
            # Manager → CEO at depth 1
            _insert_ignore("org_reporting_hierarchy", {
                "ancestor_id": ceo_id, "descendant_id": mgr_id, "depth": 1,
            })

    # 4. Create sample teams (one per manager)
    print(f"\n4. Sample teams:")
    team_defs = [
        {"name": "Engineering", "description": "Builds and maintains all product infrastructure, APIs, and AI systems.", "manager_idx": 0},
        {"name": "Product",     "description": "Defines product vision, roadmap, and user experience strategy.",       "manager_idx": 1},
        {"name": "Design",      "description": "Crafts user interfaces, brand identity, and design systems.",          "manager_idx": 2},
    ]
    for td in team_defs:
        existing = _get("org_teams", {"name": f"eq.{td['name']}", "org_id": f"eq.{org_id}"})
        if existing:
            team = existing[0]
            print(f"   ↳ team '{td['name']}' already exists — skipping")
        else:
            r = requests.post(
                f"{SUPABASE_URL}/rest/v1/org_teams",
                headers={**_headers(), "Prefer": "return=representation"},
                json={"name": td["name"], "org_id": org_id, "description": td["description"]},
                timeout=10,
            )
            if not r.ok:
                print(f"   ✗  Could not create team '{td['name']}': {r.text[:100]}")
                continue
            team = r.json()[0]
            print(f"   ✓  '{td['name']}' team_id = {team['id']}")

        # Add manager to their team
        mgr_id = manager_ids[td["manager_idx"]] if td["manager_idx"] < len(manager_ids) else None
        if mgr_id:
            _insert_ignore("org_team_members", {
                "team_id": team["id"], "user_id": mgr_id, "role": "MANAGER",
            })

    print(f"\n✓ Seeding complete. Org ID: {org_id}\n")
    print("All 3 managers + CEO created with password:", DEFAULT_PASSWORD)
    print("Sign in at /login with any of these emails.\n")


if __name__ == "__main__":
    main()
