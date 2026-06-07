#!/usr/bin/env python3
"""
Seed script: create initial org, CEO, ADMIN, two teams, managers, and dummy members.

Run from the Confluence/ directory:
    python seed_data.py
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from dotenv import load_dotenv
load_dotenv(os.path.join(_HERE, "user_service", ".env"))

from user_service.auth import hash_password
from user_service.database import select_one, select, insert, DBError


def _upsert_user(email: str, name: str, password: str, role: str, org_id: str) -> dict:
    existing = select_one("org_users", {"email": f"eq.{email}"})
    if existing:
        print(f"  [skip] {email} already exists (role={existing['role']})")
        return existing
    user = insert("org_users", {
        "email": email,
        "name": name,
        "password_hash": hash_password(password),
        "role": role,
        "org_id": org_id,
        "is_active": True,
    })
    print(f"  [ok]   Created {email} (role={role})")
    return user


def _upsert_team(name: str, org_id: str, description: str = "") -> dict:
    existing = select_one("org_teams", {"name": f"eq.{name}", "org_id": f"eq.{org_id}"})
    if existing:
        print(f"  [skip] Team '{name}' already exists")
        return existing
    data: dict = {"name": name, "org_id": org_id}
    if description:
        data["description"] = description
    team = insert("org_teams", data)
    print(f"  [ok]   Created team '{name}'")
    return team


def _add_member(team_id: str, user_id: str, team_role: str) -> None:
    existing = select_one("org_team_members", {
        "team_id": f"eq.{team_id}", "user_id": f"eq.{user_id}",
    })
    if existing:
        return
    insert("org_team_members", {"team_id": team_id, "user_id": user_id, "role": team_role})
    print(f"       → added user {user_id[:8]}… as {team_role}")


def _wire(team_id: str, user_id: str, team_role: str, ceo_id: str) -> None:
    try:
        insert("org_reporting_hierarchy", {
            "ancestor_id": user_id, "descendant_id": user_id, "depth": 0,
        })
    except DBError:
        pass

    if team_role == "MANAGER":
        try:
            insert("org_reporting_hierarchy", {
                "ancestor_id": ceo_id, "descendant_id": user_id, "depth": 1,
            })
        except DBError:
            pass
    else:
        mgr = select_one("org_team_members", {
            "team_id": f"eq.{team_id}", "role": "eq.MANAGER",
        })
        if mgr:
            mgr_id = mgr["user_id"]
            try:
                insert("org_reporting_hierarchy", {
                    "ancestor_id": mgr_id, "descendant_id": user_id, "depth": 1,
                })
            except DBError:
                pass
            ancestors = select("org_reporting_hierarchy", {
                "descendant_id": f"eq.{mgr_id}", "depth": "gt.0",
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


def main() -> None:
    print("=== Seeding GenReal.ai org data ===\n")

    # Organisation
    org = select_one("organizations", {"name": "eq.GenReal.ai"})
    if not org:
        org = insert("organizations", {"name": "GenReal.ai"})
        print(f"[ok]   Created org 'GenReal.ai' (id={org['id']})")
    else:
        print(f"[skip] Org 'GenReal.ai' already exists")
    org_id = org["id"]

    # ── Users ──────────────────────────────────────────────────────────
    print("\n--- Users ---")

    ceo = _upsert_user("genreal.ai@gmail.com", "GenReal CEO", "Admin@123", "CEO", org_id)
    ceo_id = ceo["id"]
    try:
        insert("org_reporting_hierarchy", {
            "ancestor_id": ceo_id, "descendant_id": ceo_id, "depth": 0,
        })
    except DBError:
        pass

    admin = _upsert_user(
        "santhoshraajkr.17@gmail.com", "Santhosh Raaj", "Admin@123", "ADMIN", org_id,
    )
    akshath = _upsert_user("akshath.r333@gmail.com", "Akshath R", "Test@123", "MANAGER", org_id)
    vishy = _upsert_user("vishy6400@gmail.com", "Vishy S", "Test@123", "MANAGER", org_id)

    dummy1 = _upsert_user("alice.dev@genreal.ai", "Alice Dev", "Test@123", "MEMBER", org_id)
    dummy2 = _upsert_user("bob.dev@genreal.ai", "Bob Dev", "Test@123", "MEMBER", org_id)
    dummy3 = _upsert_user("carol.prod@genreal.ai", "Carol Prod", "Test@123", "MEMBER", org_id)
    dummy4 = _upsert_user("dave.prod@genreal.ai", "Dave Prod", "Test@123", "MEMBER", org_id)

    # ── Teams ──────────────────────────────────────────────────────────
    print("\n--- Teams ---")
    team1 = _upsert_team("Engineering", org_id, "Core engineering team")
    team2 = _upsert_team("Product", org_id, "Product and design team")

    # ── Memberships ────────────────────────────────────────────────────
    print("\n--- Team memberships ---")
    print("  Engineering:")
    for uid, role in [
        (akshath["id"], "MANAGER"),
        (admin["id"], "MANAGER"),   # santhosh: org=ADMIN, team=MANAGER
        (dummy1["id"], "MEMBER"),
        (dummy2["id"], "MEMBER"),
    ]:
        _add_member(team1["id"], uid, role)
        _wire(team1["id"], uid, role, ceo_id)

    print("  Product:")
    for uid, role in [
        (vishy["id"], "MANAGER"),
        (dummy3["id"], "MEMBER"),
        (dummy4["id"], "MEMBER"),
    ]:
        _add_member(team2["id"], uid, role)
        _wire(team2["id"], uid, role, ceo_id)

    # ── Summary ────────────────────────────────────────────────────────
    print("\n=== Done ===")
    print(f"\nOrg ID : {org_id}")
    print(f"CEO    : genreal.ai@gmail.com  /  Admin@123")
    print(f"ADMIN  : santhoshraajkr.17@gmail.com  /  Admin@123")
    print(f"\nTeam managers:")
    print(f"  Engineering → akshath.r333@gmail.com  /  Test@123")
    print(f"  Product     → vishy6400@gmail.com  /  Test@123")
    print(f"\nDummy members (password: Test@123):")
    print(f"  alice.dev@genreal.ai  (Engineering)")
    print(f"  bob.dev@genreal.ai    (Engineering)")
    print(f"  carol.prod@genreal.ai (Product)")
    print(f"  dave.prod@genreal.ai  (Product)")


if __name__ == "__main__":
    main()
