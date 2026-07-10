"""Integration check that org_activity._resolve_participant delegates to the
token-aware matcher — the p/n case must resolve correctly through the real code path.

    python -m pytest my-agent/tests/test_org_activity_resolve.py -q
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import org_activity  # noqa: E402


ROSTER = [
    {"user_id": "u-prakash", "name": "Vishwajith Prakash", "aliases": ["vishu"]},
    {"user_id": "u-nair", "name": "Vishwajith Nair", "aliases": []},
]


def test_initial_resolves_to_correct_surname():
    assert org_activity._resolve_participant("vishwajith p", ROSTER) == ("u-prakash", "high")
    assert org_activity._resolve_participant("Vishwajith N", ROSTER)[0] == "u-nair"


def test_ambiguous_first_name_is_guest():
    # No clear winner → (None, 'none') → caller stores a guest row (is_guest).
    assert org_activity._resolve_participant("Vishwajith", ROSTER) == (None, "none")


def test_alias_and_device():
    assert org_activity._resolve_participant("vishu", ROSTER) == ("u-prakash", "high")
    assert org_activity._resolve_participant("iPhone", ROSTER) == (None, "none")
