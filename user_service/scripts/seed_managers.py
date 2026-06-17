"""
Full seed script — creates/upserts CEO + 3 managers + 8 members, creates teams,
wires up hierarchy, and seeds 15 dummy bot sessions with attendance records.

Run from Confluence/:
    python -m user_service.scripts.seed_managers

Required env vars: SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY
Optional:  ADMIN_PASSWORD (default: Jarvis@2024!)
"""
from __future__ import annotations

import os
import sys
import uuid
import random
import datetime
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
        "_team": "Engineering",
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
        "_team": "Product",
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
        "_team": "Design",
    },
]

# Dummy members per team (name, email, job_title, role_in_org)
MEMBERS_BY_TEAM: dict[str, list[dict]] = {
    "Engineering": [
        {
            "email": "alice.chen@genreal.ai",
            "name": "Alice Chen",
            "role": "MEMBER",
            "job_title": "Software Engineer",
            "department": "Engineering",
            "phone": "+91-9111111101",
            "bio": "Backend engineer specialising in distributed systems and APIs.",
            "avatar_url": None,
        },
        {
            "email": "bob.kumar@genreal.ai",
            "name": "Bob Kumar",
            "role": "MEMBER",
            "job_title": "Frontend Developer",
            "department": "Engineering",
            "phone": "+91-9111111102",
            "bio": "React & TypeScript developer building delightful UIs.",
            "avatar_url": None,
        },
        {
            "email": "carol.white@genreal.ai",
            "name": "Carol White",
            "role": "ASSOCIATE",
            "job_title": "DevOps Engineer",
            "department": "Engineering",
            "phone": "+91-9111111103",
            "bio": "Keeps the CI/CD pipelines green and infra costs low.",
            "avatar_url": None,
        },
    ],
    "Product": [
        {
            "email": "david.park@genreal.ai",
            "name": "David Park",
            "role": "MEMBER",
            "job_title": "Product Analyst",
            "department": "Product",
            "phone": "+91-9222222201",
            "bio": "Data-driven analyst who turns user research into product insights.",
            "avatar_url": None,
        },
        {
            "email": "emma.rodriguez@genreal.ai",
            "name": "Emma Rodriguez",
            "role": "MEMBER",
            "job_title": "UX Researcher",
            "department": "Product",
            "phone": "+91-9222222202",
            "bio": "Qualitative and quantitative researcher helping teams stay user-centred.",
            "avatar_url": None,
        },
        {
            "email": "frank.liu@genreal.ai",
            "name": "Frank Liu",
            "role": "ASSOCIATE",
            "job_title": "Growth Specialist",
            "department": "Product",
            "phone": "+91-9222222203",
            "bio": "Runs experiments and growth loops to accelerate adoption.",
            "avatar_url": None,
        },
    ],
    "Design": [
        {
            "email": "grace.kim@genreal.ai",
            "name": "Grace Kim",
            "role": "MEMBER",
            "job_title": "UI Designer",
            "department": "Design",
            "phone": "+91-9333333301",
            "bio": "Pixel-perfect designer obsessed with accessibility and clarity.",
            "avatar_url": None,
        },
        {
            "email": "henry.shah@genreal.ai",
            "name": "Henry Shah",
            "role": "ASSOCIATE",
            "job_title": "Brand Designer",
            "department": "Design",
            "phone": "+91-9333333302",
            "bio": "Brand storyteller who ensures every touchpoint feels intentional.",
            "avatar_url": None,
        },
    ],
}

TEAM_DEFS = [
    {
        "name": "Engineering",
        "description": "Builds and maintains all product infrastructure, APIs, and AI systems.",
    },
    {
        "name": "Product",
        "description": "Defines product vision, roadmap, and user experience strategy.",
    },
    {
        "name": "Design",
        "description": "Crafts user interfaces, brand identity, and design systems.",
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
        headers={**_headers(), "Prefer": "resolution=merge-duplicates,return=representation"},
        params={"on_conflict": on_conflict},
        json=data,
        timeout=10,
    )
    if not r.ok:
        print(f"  FAIL  UPSERT {table} failed: {r.text[:200]}")
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
        print(f"  FAIL  INSERT {table} failed ({r.status_code}): {r.text[:200]}")
        return {}
    rows = r.json()
    return rows[0] if (r.ok and rows) else data


def _insert(table: str, data: dict) -> dict:
    r = requests.post(
        f"{SUPABASE_URL}/rest/v1/{table}",
        headers={**_headers(), "Prefer": "return=representation"},
        json=data,
        timeout=10,
    )
    if not r.ok:
        print(f"  FAIL  INSERT {table} failed ({r.status_code}): {r.text[:100]}")
        return {}
    rows = r.json()
    return rows[0] if rows else data


# ── Password hashing ───────────────────────────────────────────────────────────

def _hash_password(plain: str) -> str:
    import bcrypt
    return bcrypt.hashpw(plain.encode(), bcrypt.gensalt(rounds=12)).decode()


# ── Hierarchy helpers ──────────────────────────────────────────────────────────

def _wire_member_hierarchy(team_id: str, user_id: str, ceo_id: str) -> None:
    """Self-loop + link MEMBER → MANAGER → CEO via closure table."""
    _insert_ignore("org_reporting_hierarchy", {
        "ancestor_id": user_id, "descendant_id": user_id, "depth": 0,
    })
    manager_row = _get("org_team_members", {"team_id": f"eq.{team_id}", "role": "eq.MANAGER"})
    if not manager_row:
        return
    mgr_id = manager_row[0]["user_id"]
    _insert_ignore("org_reporting_hierarchy", {
        "ancestor_id": mgr_id, "descendant_id": user_id, "depth": 1,
    })
    _insert_ignore("org_reporting_hierarchy", {
        "ancestor_id": ceo_id, "descendant_id": user_id, "depth": 2,
    })


# ── Session seeding ────────────────────────────────────────────────────────────

def _seed_sessions(team_id: str, team_members: list[str], count: int = 5) -> None:
    """Seed `count` fake ended bot sessions for a team + attendance records."""
    now = datetime.datetime.now(datetime.timezone.utc)

    for i in range(count):
        days_ago = random.randint(0, 30)
        hour = random.randint(9, 17)
        start = (now - datetime.timedelta(days=days_ago)).replace(
            hour=hour, minute=random.randint(0, 59), second=0, microsecond=0,
        )
        duration_mins = random.randint(20, 90)
        end = start + datetime.timedelta(minutes=duration_mins)

        session_id = f"seed-{uuid.uuid4().hex[:16]}"
        row = _insert("jarvis_sessions", {
            "session_id": session_id,
            "team_id": team_id,
            "status": "ended",
            "meeting_url": f"https://meet.google.com/seed-{session_id[:8]}",
            "started_at": start.isoformat(),
            "ended_at": end.isoformat(),
            "updated_at": end.isoformat(),
            "changes": [],
            "transcript": [],
            "transcript_memory_text": "",
            "pipeline_diagnostics": [],
        })
        if not row:
            continue

        # Randomly pick 60–100% of team members as attendees
        attendees = [m for m in team_members if random.random() > 0.2]
        if not attendees:
            attendees = team_members[:1]

        for uid in attendees:
            member_duration = round(duration_mins * random.uniform(0.7, 1.0), 1)
            _insert_ignore("user_meeting_activity", {
                "user_id": uid,
                "session_id": session_id,
                "team_id": team_id,
                "joined_at": start.isoformat(),
                "duration_mins": member_duration,
            })

    print(f"   OK  {count} sessions seeded")


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
        print(f"   -> existing org_id = {org['id']}")
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
        print(f"   OK created org_id = {org['id']}")
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
    print(f"   OK  id = {ceo_id}")

    if ceo_id:
        _insert_ignore("org_reporting_hierarchy", {
            "ancestor_id": ceo_id, "descendant_id": ceo_id, "depth": 0,
        })

    # 3. Upsert managers
    print(f"\n3. Managers:")
    manager_ids: list[str] = []
    manager_team_map: dict[str, str] = {}  # manager_id → team_name
    for mgr in MANAGERS:
        team_name = mgr.pop("_team", "")
        row = _upsert("org_users", {
            **mgr,
            "password_hash": pw_hash,
            "org_id": org_id,
            "is_active": True,
        }, "email")
        mgr_id = row.get("id")
        manager_ids.append(mgr_id)
        manager_team_map[mgr_id] = team_name
        print(f"   OK  {mgr['email']}  id = {mgr_id}")

        if mgr_id and ceo_id:
            _insert_ignore("org_reporting_hierarchy", {
                "ancestor_id": mgr_id, "descendant_id": mgr_id, "depth": 0,
            })
            _insert_ignore("org_reporting_hierarchy", {
                "ancestor_id": ceo_id, "descendant_id": mgr_id, "depth": 1,
            })

    # 4. Create teams and assign managers
    print(f"\n4. Teams:")
    team_id_map: dict[str, str] = {}  # team_name → team_id
    for i, td in enumerate(TEAM_DEFS):
        existing = _get("org_teams", {"name": f"eq.{td['name']}", "org_id": f"eq.{org_id}"})
        if existing:
            team = existing[0]
            print(f"   -> '{td['name']}' already exists — team_id = {team['id']}")
        else:
            r = requests.post(
                f"{SUPABASE_URL}/rest/v1/org_teams",
                headers={**_headers(), "Prefer": "return=representation"},
                json={"name": td["name"], "org_id": org_id, "description": td["description"]},
                timeout=10,
            )
            if not r.ok:
                print(f"   FAIL  Could not create team '{td['name']}': {r.text[:100]}")
                continue
            team = r.json()[0]
            print(f"   OK  '{td['name']}' team_id = {team['id']}")

        team_id_map[td["name"]] = team["id"]

        # Add manager to their team
        if i < len(manager_ids) and manager_ids[i]:
            _insert_ignore("org_team_members", {
                "team_id": team["id"], "user_id": manager_ids[i], "role": "MANAGER",
            })

    # Real skeleton ends here. Demo members + fake sessions are OPT-IN so a
    # normal run never reintroduces dummy data (use --with-dummy for a demo org).
    if "--with-dummy" not in sys.argv:
        print(f"\nOK Real skeleton seeded. Org ID: {org_id}")
        print(f"  CEO + {len(manager_ids)} managers + {len(team_id_map)} teams.")
        print(f"  Add people via the app (create team → invite by email), or re-run")
        print(f"  with --with-dummy to populate demo members + sessions.\n")
        return

    # 5. Add dummy members to each team
    print(f"\n5. Dummy members:")
    team_member_ids: dict[str, list[str]] = {}  # team_id → [user_ids] (incl. manager)

    # Seed manager IDs into team_member_ids first
    for mgr_id in manager_ids:
        tname = manager_team_map.get(mgr_id, "")
        tid = team_id_map.get(tname)
        if tid:
            team_member_ids.setdefault(tid, []).append(mgr_id)

    for team_name, members in MEMBERS_BY_TEAM.items():
        team_id = team_id_map.get(team_name)
        if not team_id:
            print(f"   FAIL  Team '{team_name}' not found — skipping members")
            continue

        print(f"   Team: {team_name}")
        for m in members:
            row = _upsert("org_users", {
                **m,
                "password_hash": pw_hash,
                "org_id": org_id,
                "is_active": True,
            }, "email")
            uid = row.get("id")
            if not uid:
                print(f"      FAIL  failed to upsert {m['email']}")
                continue
            print(f"      OK  {m['email']}  id = {uid}")

            # Add to team
            team_role = m["role"]  # MEMBER or ASSOCIATE
            _insert_ignore("org_team_members", {
                "team_id": team_id, "user_id": uid, "role": team_role,
            })

            # Wire hierarchy
            if ceo_id:
                _wire_member_hierarchy(team_id, uid, ceo_id)

            team_member_ids.setdefault(team_id, []).append(uid)

    # 6. Seed dummy bot sessions (5 per team)
    print(f"\n6. Seeding dummy bot sessions (5 per team):")
    random.seed(42)
    for team_name, team_id in team_id_map.items():
        member_ids = team_member_ids.get(team_id, [])
        print(f"   {team_name} (team_id={team_id[:8]}…) — {len(member_ids)} members")
        _seed_sessions(team_id, member_ids, count=5)

    print(f"\nOK Seeding complete. Org ID: {org_id}")
    print(f"  CEO + 3 managers + 8 members created with password: {DEFAULT_PASSWORD}")
    print(f"  15 bot sessions + attendance records seeded across 3 teams\n")


if __name__ == "__main__":
    main()
