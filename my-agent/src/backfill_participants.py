"""One-off backfill: derive real-ish attendance for HISTORICAL meetings.

For every ended session that has no meeting_participants rows yet, gather the
distinct diarized speaker names from its stored transcript and write best-effort
attendance rows (source='backfill', whole-meeting duration, no join/leave — that
data doesn't exist for past meetings).

Idempotent: sessions that already have presence/attendance rows are skipped.

Run once from my-agent/:
    uv run --no-sync python src/backfill_participants.py          # dry run (counts only)
    uv run --no-sync python src/backfill_participants.py --write  # actually write
"""

import sys

try:
    from . import org_activity, session_store
except ImportError:
    import org_activity
    import session_store


def _distinct_speakers(session_id: str, session: dict) -> list[str]:
    """Recall-diarized speaker names for a session — prefer the turns table, fall
    back to the inline transcript blob for very old sessions."""
    names: list[str] = []
    for e in session_store.get_transcript_turns(session_id):
        n = e.get("participant")
        if n:
            names.append(n)
    if not names:
        for e in session.get("transcript") or []:
            if isinstance(e, dict):
                n = e.get("participant") or e.get("speaker")
                if n:
                    names.append(n)
    return names


def main(write: bool) -> None:
    sessions = session_store.list_all()
    ended = [s for s in sessions if s.get("status") == "ended" and s.get("team_id")]
    print(f"Scanning {len(ended)} ended team-scoped sessions "
          f"(of {len(sessions)} total)…  mode={'WRITE' if write else 'DRY-RUN'}")

    total_rows = 0
    touched = 0
    for s in ended:
        sid = s.get("session_id")
        speakers = _distinct_speakers(sid, s)
        if not speakers:
            continue
        if not write:
            # Dry run: report distinct speaker count without touching the DB.
            distinct = {org_activity._normalize_name(n) for n in speakers}
            distinct.discard("")
            distinct -= {"jarvis", "meeting", "meeting assistant"}
            if distinct:
                touched += 1
                total_rows += len(distinct)
                print(f"  {sid}: {len(distinct)} distinct speakers → {sorted(distinct)}")
            continue
        n = org_activity.backfill_participants_from_transcript(s, speakers)
        if n:
            touched += 1
            total_rows += n
            print(f"  {sid}: wrote {n} participant rows")

    print(f"\nDone. sessions_touched={touched} rows={total_rows} "
          f"({'written' if write else 'would write'}).")


if __name__ == "__main__":
    main(write="--write" in sys.argv)
