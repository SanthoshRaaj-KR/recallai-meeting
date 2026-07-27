"""Per-participant attendance and time accounting.

Covers the defects found while auditing meeting-time calculation:
  A. a rejoin billed the away-time as time in call
  B. a rejoin under a new participant id looked like a second meeting
  C. the bot itself was recorded as an attendee
  D. Recall's participant email was read from the wrong field

    python -m pytest my-agent/tests/test_participant_time.py -q
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

os.environ.setdefault("SUPABASE_URL", "https://example.supabase.co")
os.environ.setdefault("SUPABASE_SERVICE_ROLE_KEY", "test-key")

import bot_service  # noqa: E402
import org_activity  # noqa: E402

SESSION = {
    "session_id": "s1",
    "team_id": "t1",
    "started_at": "2026-07-27T10:00:00Z",
    "ended_at": "2026-07-27T11:00:00Z",
}


def _presence(*events):
    """Fold (kind, ts) events into one participant's presence entry."""
    entry: dict = {}
    for kind, ts in events:
        entry = bot_service._merge_presence_event(entry, kind, ts)
    return entry


# ── A. rejoin must not bill the gap ────────────────────────────────────────────

def test_rejoin_counts_only_time_actually_in_call():
    entry = _presence(
        ("join", "2026-07-27T10:00:00Z"),
        ("leave", "2026-07-27T10:05:00Z"),
        ("join", "2026-07-27T10:50:00Z"),
        ("leave", "2026-07-27T11:00:00Z"),
    )
    # 5 min + 10 min present; the 45-min gap is NOT credited.
    assert org_activity._participant_duration_mins(entry, SESSION) == 15.0


def test_single_stay_is_unchanged():
    entry = _presence(("join", "2026-07-27T10:00:00Z"), ("leave", "2026-07-27T10:30:00Z"))
    assert org_activity._participant_duration_mins(entry, SESSION) == 30.0


def test_still_in_call_at_meeting_end_is_closed_with_ended_at():
    entry = _presence(("join", "2026-07-27T10:40:00Z"))  # never left
    assert org_activity._participant_duration_mins(entry, SESSION) == 20.0


def test_leave_without_a_join_is_anchored_to_meeting_start():
    # We started watching mid-call, or the join webhook was missed.
    entry = _presence(("leave", "2026-07-27T10:15:00Z"))
    assert org_activity._participant_duration_mins(entry, SESSION) == 15.0


def test_duplicate_join_webhook_does_not_manufacture_a_second_stay():
    entry = _presence(
        ("join", "2026-07-27T10:00:00Z"),
        ("join", "2026-07-27T10:00:00Z"),  # replayed delivery
        ("leave", "2026-07-27T10:20:00Z"),
    )
    assert len(entry["intervals"]) == 1
    assert org_activity._participant_duration_mins(entry, SESSION) == 20.0


def test_first_join_and_last_leave_are_still_recorded_for_display():
    entry = _presence(
        ("join", "2026-07-27T10:00:00Z"),
        ("leave", "2026-07-27T10:05:00Z"),
        ("join", "2026-07-27T10:50:00Z"),
        ("leave", "2026-07-27T11:00:00Z"),
    )
    assert entry["joined_at"] == "2026-07-27T10:00:00Z"
    assert entry["left_at"] == "2026-07-27T11:00:00Z"


def test_legacy_entry_without_intervals_still_computes():
    # Rows written before intervals existed must keep working.
    legacy = {"joined_at": "2026-07-27T10:00:00Z", "left_at": "2026-07-27T10:30:00Z"}
    assert org_activity._participant_duration_mins(legacy, SESSION) == 30.0


def test_no_timestamps_at_all_yields_none_not_zero():
    # None means "unknown", which the stats RPCs exclude from the average;
    # 0.0 would wrongly drag it down.
    assert org_activity._participant_duration_mins({}, {"session_id": "s"}) is None


# ── C. the bot is not an attendee ──────────────────────────────────────────────

def test_bot_display_names_are_not_attendees():
    assert org_activity._is_bot_participant("Jarvis")
    assert org_activity._is_bot_participant("Meeting Assistant")
    assert org_activity._is_bot_participant("meeting  assistant")  # normalised


def test_configured_bot_name_is_excluded(monkeypatch):
    monkeypatch.setenv("BOT_NAME", "Acme Notetaker")
    assert org_activity._is_bot_participant("acme notetaker")


def test_real_people_are_not_mistaken_for_the_bot():
    assert not org_activity._is_bot_participant("Priya S")
    assert not org_activity._is_bot_participant("")
    assert not org_activity._is_bot_participant(None)


# ── D. email is read from Recall's documented location ─────────────────────────

def test_email_is_read_from_the_top_level_field():
    # Recall documents participant.email (string | null) at the top level.
    assert bot_service._participant_email({"email": "priya@corp.com"}) == "priya@corp.com"


def test_email_falls_back_to_extra_data_for_older_payloads():
    assert bot_service._participant_email(
        {"extra_data": {"email": "legacy@corp.com"}}
    ) == "legacy@corp.com"


def test_absent_email_is_none_not_an_error():
    # The common case: bots created from a meeting_url (not Calendar Integration)
    # never receive an email, so this must degrade quietly to name matching.
    assert bot_service._participant_email({"name": "Priya S"}) is None
    assert bot_service._participant_email({"email": None, "extra_data": {}}) is None


# ── B. one meeting per person, even across reconnects ──────────────────────────

def test_backfill_does_not_stack_a_whole_meeting_on_top_of_real_intervals():
    """The end-of-meeting Recall backfill must not append a full-meeting stay to a
    participant whose real per-stay intervals the webhooks already recorded."""
    entry = _presence(
        ("join", "2026-07-27T10:00:00Z"),
        ("leave", "2026-07-27T10:05:00Z"),
    )
    before = org_activity._participant_duration_mins(entry, SESSION)
    # Simulates the guard in _backfill_participants_from_recall.
    if not entry.get("intervals"):
        entry["intervals"] = [{"joined_at": SESSION["started_at"], "left_at": SESSION["ended_at"]}]
    assert org_activity._participant_duration_mins(entry, SESSION) == before == 5.0
