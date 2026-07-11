"""One-time backfill: ensure every team member reports to their team's manager.

Members added to a team *before* it had a manager were never wired into the
reporting hierarchy, so they report to nobody. This walks every team, and for
each team that has a MANAGER, links all of its MEMBER/ASSOCIATE members under
that manager (and rolls them up to the CEO).

Idempotent and additive — safe to run repeatedly; it only inserts missing
closure-table edges and never deletes.

Run from the Confluence/ directory:
    python -m user_service.scripts.backfill_team_hierarchy --dry-run   # preview only
    python -m user_service.scripts.backfill_team_hierarchy             # apply

Required env: SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY (loaded from user_service/.env)
"""
from __future__ import annotations

import os
import sys

from dotenv import load_dotenv

# Load env BEFORE importing the DB layer (database.py reads SUPABASE_* at import).
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))


def main(dry_run: bool = False) -> None:
    from user_service.database import select, select_one
    from user_service.models import TeamRole
    from user_service.routes.teams import _wire_hierarchy, _wire_existing_members_under_manager

    teams = select("org_teams", {})
    banner = "DRY RUN — no writes" if dry_run else "APPLYING changes"
    print(f"[{banner}] {len(teams)} team(s)\n")

    total = 0
    for team in teams:
        team_id = team["id"]
        name = team.get("name", team_id)
        members = select("org_team_members", {"team_id": f"eq.{team_id}"})
        manager = next((m for m in members if m.get("role") == TeamRole.MANAGER), None)
        if not manager:
            print(f"  · {name}: no manager assigned — skipped ({len(members)} member(s))")
            continue

        mgr_id = manager["user_id"]
        reports = [m for m in members if m.get("role") != TeamRole.MANAGER and m["user_id"] != mgr_id]

        if dry_run:
            missing = 0
            for m in reports:
                edge = select_one("org_reporting_hierarchy", {
                    "ancestor_id": f"eq.{mgr_id}", "descendant_id": f"eq.{m['user_id']}",
                })
                if not edge:
                    missing += 1
            total += missing
            print(f"  · {name}: {len(reports)} member(s) — {missing} missing manager link(s) would be created")
        else:
            # Ensure the manager's own edges (self-loop + manager→CEO) exist, then
            # adopt every non-manager member of the team.
            _wire_hierarchy(team_id, mgr_id, TeamRole.MANAGER)
            _wire_existing_members_under_manager(team_id, mgr_id)
            total += len(reports)
            print(f"  · {name}: ensured {len(reports)} member(s) report to the manager")

    verb = "Would create" if dry_run else "Done. Ensured"
    print(f"\n{verb} {total} member link(s) across {len(teams)} team(s).")


if __name__ == "__main__":
    main(dry_run="--dry-run" in sys.argv)
