"""
Remove the dummy data seeded by seed_managers.py, while KEEPING the real org
skeleton (organization, CEO, the 3 real managers, the 3 teams).

Deletes:
  * user_meeting_activity rows for seeded sessions  (session_id LIKE 'seed-%')
  * jarvis_sessions rows that are seeded            (session_id LIKE 'seed-%')
  * the 8 fake @genreal.ai member accounts + their team memberships + hierarchy

Usage (from Confluence/):
    python -m user_service.scripts.cleanup_dummy            # DRY RUN (counts only)
    python -m user_service.scripts.cleanup_dummy --apply    # actually delete

Reads SUPABASE_URL + SUPABASE_SERVICE_ROLE_KEY from user_service/.env.
"""
from __future__ import annotations

import os
import sys

import requests
from dotenv import load_dotenv

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))

URL = os.getenv("SUPABASE_URL", "").rstrip("/")
KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_ANON_KEY") or ""

# The exact fake members seed_managers.py creates (real people use @gmail.com).
FAKE_EMAILS = [
    "alice.chen@genreal.ai", "bob.kumar@genreal.ai", "carol.white@genreal.ai",
    "david.park@genreal.ai", "emma.rodriguez@genreal.ai", "frank.liu@genreal.ai",
    "grace.kim@genreal.ai", "henry.shah@genreal.ai",
]


def _h(extra: dict | None = None) -> dict:
    h = {"apikey": KEY, "Authorization": f"Bearer {KEY}", "Content-Type": "application/json"}
    if extra:
        h.update(extra)
    return h


def _count(table: str, params: dict) -> int:
    r = requests.get(
        f"{URL}/rest/v1/{table}",
        headers=_h({"Prefer": "count=exact"}),
        params={**params, "select": "*", "limit": "1"},
        timeout=15,
    )
    r.raise_for_status()
    cr = r.headers.get("content-range", "*/0")
    return int(cr.split("/")[-1]) if "/" in cr else 0


def _delete(table: str, params: dict) -> None:
    r = requests.delete(f"{URL}/rest/v1/{table}", headers=_h(), params=params, timeout=30)
    if not r.ok and r.status_code != 404:
        print(f"  FAIL delete {table} {params}: {r.status_code} {r.text[:150]}")
    else:
        print(f"  OK   deleted from {table} where {params}")


def _fake_user_ids() -> list[str]:
    emails = ",".join(FAKE_EMAILS)
    r = requests.get(
        f"{URL}/rest/v1/org_users",
        headers=_h(), params={"email": f"in.({emails})", "select": "id,email"}, timeout=15,
    )
    r.raise_for_status()
    rows = r.json()
    for row in rows:
        print(f"    - {row['email']}  ({row['id']})")
    return [row["id"] for row in rows]


def main() -> None:
    if not URL or not KEY:
        sys.exit("SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY must be set in user_service/.env")
    apply = "--apply" in sys.argv

    print(f"\n=== Dummy-data cleanup ({'APPLY' if apply else 'DRY RUN'}) ===\n")

    seed_act = _count("user_meeting_activity", {"session_id": "like.seed-*"})
    seed_sess = _count("jarvis_sessions", {"session_id": "like.seed-*"})
    print("Will remove:")
    print(f"  user_meeting_activity (seeded sessions): {seed_act}")
    print(f"  jarvis_sessions       (seeded sessions): {seed_sess}")
    print(f"  fake member accounts (@genreal.ai):")
    fake_ids = _fake_user_ids()
    print(f"  -> {len(fake_ids)} fake users\n")

    if not apply:
        print("DRY RUN — nothing deleted. Re-run with --apply to execute.")
        return

    print("Applying deletes…")
    _delete("user_meeting_activity", {"session_id": "like.seed-*"})
    _delete("jarvis_sessions", {"session_id": "like.seed-*"})
    for uid in fake_ids:
        # Explicit child deletes first (in case FKs aren't ON DELETE CASCADE).
        _delete("user_meeting_activity", {"user_id": f"eq.{uid}"})
        _delete("org_team_members", {"user_id": f"eq.{uid}"})
        _delete("org_reporting_hierarchy", {"descendant_id": f"eq.{uid}"})
        _delete("org_reporting_hierarchy", {"ancestor_id": f"eq.{uid}"})
        _delete("org_users", {"id": f"eq.{uid}"})
    print("\nDone. Real org (CEO + 3 managers + 3 teams) is untouched.")


if __name__ == "__main__":
    main()
