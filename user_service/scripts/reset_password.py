"""
One-shot org setup: upserts CEO and managers directly into Supabase.
Run from Confluence/:  python -m user_service.scripts.reset_password
"""

import os
import sys
import bcrypt
import requests
from dotenv import load_dotenv

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))

SUPABASE_URL = os.getenv("SUPABASE_URL", "").rstrip("/")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_ANON_KEY") or ""

# ── Hardcoded users ────────────────────────────────────────────────────────────

CEO = {"email": "genreal.ai@gmail.com", "name": "GenReal AI", "password": "ceo123", "role": "CEO"}

MANAGERS = [
    {"email": "vishy6400@gmail.com",         "name": "Vishy",    "password": "manager123", "role": "MANAGER"},
    {"email": "santhoshraajkr.17@gmail.com", "name": "Santhosh", "password": "manager123", "role": "MANAGER"},
    {"email": "akshath.r333@gmail.com",      "name": "Akshath",  "password": "manager123", "role": "MANAGER"},
]

# ──────────────────────────────────────────────────────────────────────────────

def _hash(plain: str) -> str:
    return bcrypt.hashpw(plain.encode(), bcrypt.gensalt(rounds=12)).decode()


def _headers() -> dict:
    return {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json",
        "Prefer": "resolution=merge-duplicates,return=representation",
    }


def _upsert(user: dict) -> None:
    url = f"{SUPABASE_URL}/rest/v1/org_users"
    payload = {
        "email": user["email"],
        "name": user["name"],
        "password_hash": _hash(user["password"]),
        "role": user["role"],
        "is_active": True,
    }
    r = requests.post(
        url,
        headers=_headers(),
        params={"on_conflict": "email"},
        json=payload,
        timeout=10,
    )
    if r.ok:
        print(f"  ✓  {user['email']}  ({user['role']})")
    else:
        print(f"  ✗  {user['email']}  → {r.text[:120]}")


def main() -> None:
    if not SUPABASE_URL or not SUPABASE_KEY:
        sys.exit("SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY must be set in user_service/.env")

    print("\n=== Org Setup ===\n")
    print("Creating CEO…")
    _upsert(CEO)

    print("\nCreating managers…")
    for mgr in MANAGERS:
        _upsert(mgr)

    print("\nDone. CEO password: ceo123 | Manager password: manager123")


if __name__ == "__main__":
    main()
