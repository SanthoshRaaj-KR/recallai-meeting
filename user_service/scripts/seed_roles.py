"""
Bootstrap org roles — sets roles directly in Supabase, no service needed.
Run from Confluence/:  python -m user_service.scripts.seed_roles
"""

import os
import sys
import requests
from dotenv import load_dotenv

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))

SUPABASE_URL = os.getenv("SUPABASE_URL", "").rstrip("/")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_ANON_KEY") or ""

# ── Configure these ────────────────────────────────────────────────────────────

MANAGER_EMAILS = [
    "vishy6400@gmail.com",
    "santhoshraajkr.17@gmail.com",
    "akshath.r333@gmail.com",
]

USER_EMAIL = "genreal.ai@gmail.com"  # CEO email

# ──────────────────────────────────────────────────────────────────────────────


def _headers() -> dict:
    return {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json",
        "Prefer": "return=representation",
    }


def _get_all_users() -> dict[str, dict]:
    r = requests.get(
        f"{SUPABASE_URL}/rest/v1/org_users",
        headers=_headers(),
        timeout=10,
    )
    if not r.ok:
        sys.exit(f"Could not fetch users: {r.text}")
    return {u["email"]: u for u in r.json()}


def _set_role(user: dict, role: str) -> None:
    r = requests.patch(
        f"{SUPABASE_URL}/rest/v1/org_users",
        headers=_headers(),
        params={"email": f"eq.{user['email']}"},
        json={"role": role},
        timeout=10,
    )
    icon = "✓" if r.ok else "✗"
    detail = "" if r.ok else f"  → {r.text[:120]}"
    print(f"  {icon}  {user['email']}  ({user['role']} → {role}){detail}")


def main() -> None:
    if not SUPABASE_URL or not SUPABASE_KEY:
        sys.exit("SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY must be set in user_service/.env")

    print("\n=== Jarvis Org Role Seeder ===\n")
    print("Fetching all org users…")
    by_email = _get_all_users()
    print(f"  Found {len(by_email)} user(s).\n")

    print(f"Setting CEO role for {USER_EMAIL}:")
    if USER_EMAIL in by_email:
        _set_role(by_email[USER_EMAIL], "CEO")
    else:
        print(f"  ✗  {USER_EMAIL} — not found (run reset_password.py first)")

    print(f"\nSetting MANAGER role for {len(MANAGER_EMAILS)} email(s):")
    for email in MANAGER_EMAILS:
        if email in by_email:
            _set_role(by_email[email], "MANAGER")
        else:
            print(f"  ✗  {email} — not found in org")

    print("\nDone.")


if __name__ == "__main__":
    main()
