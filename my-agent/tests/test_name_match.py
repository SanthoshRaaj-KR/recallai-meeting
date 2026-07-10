"""Unit tests for name_match.resolve — participant name → org user attribution.

Pure stdlib module, so this runs without the agent runtime:
    python -m pytest my-agent/tests/test_name_match.py -q
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import name_match  # noqa: E402


# Roster with two people who differ only by surname initial — the hard case.
PRAKASH = {"user_id": "u-prakash", "name": "Vishwajith Prakash", "aliases": ["vishu"]}
NAIR = {"user_id": "u-nair", "name": "Vishwajith Nair", "aliases": []}
ALICE = {"user_id": "u-alice", "name": "Alice Anderson", "aliases": []}
BOTH = [PRAKASH, NAIR, ALICE]


def _r(name, candidates=BOTH):
    return name_match.resolve(name, candidates)


# ── Exact ────────────────────────────────────────────────────────────────────

def test_exact_full_name_high():
    assert _r("Vishwajith Prakash") == ("u-prakash", "high")


def test_exact_is_case_and_punctuation_insensitive():
    assert _r("vishwajith  prakash!") == ("u-prakash", "high")


def test_alias_exact_high():
    assert _r("Vishu") == ("u-prakash", "high")


# ── The surname-initial disambiguation (the whole point) ─────────────────────

def test_initial_p_matches_prakash_not_nair():
    uid, conf = _r("vishwajith p")
    assert uid == "u-prakash"
    assert conf in ("high", "medium")


def test_initial_n_matches_nair_not_prakash():
    uid, conf = _r("Vishwajith N")
    assert uid == "u-nair"


def test_initial_conflict_never_crosses():
    # 'p' must never resolve to Nair, 'n' never to Prakash.
    assert _r("vishwajith p")[0] != "u-nair"
    assert _r("vishwajith n")[0] != "u-prakash"


# ── Ambiguity → unattributed guest ───────────────────────────────────────────

def test_bare_first_name_ambiguous_returns_none():
    # Two Vishwajiths, no surname → don't guess.
    assert _r("Vishwajith") == (None, "none")


def test_bare_first_name_unique_matches():
    # Only one Vishwajith in the roster → safe to attribute.
    assert name_match.resolve("Vishwajith", [PRAKASH, ALICE]) == ("u-prakash", "medium")


def test_initial_ambiguous_between_two_p_surnames():
    prasad = {"user_id": "u-prasad", "name": "Vishwajith Prasad", "aliases": []}
    # 'p' matches both Prakash and Prasad → ambiguous.
    assert name_match.resolve("vishwajith p", [PRAKASH, prasad]) == (None, "none")


# ── Typos ────────────────────────────────────────────────────────────────────

def test_typo_in_given_name_still_matches():
    uid, _ = _r("vishwajit prakash")   # missing 'h'
    assert uid == "u-prakash"


def test_full_surname_beats_close_surname():
    prasad = {"user_id": "u-prasad", "name": "Vishwajith Prasad", "aliases": []}
    # Full surname 'prakash' should win clearly over the similar 'prasad'.
    assert name_match.resolve("Vishwajith Prakash", [PRAKASH, prasad])[0] == "u-prakash"


# ── Non-matches ──────────────────────────────────────────────────────────────

def test_device_label_no_match():
    assert _r("iPhone") == (None, "none")
    assert _r("Meeting Room 2") == (None, "none")


def test_empty_no_match():
    assert _r("") == (None, "none")
    assert _r(None) == (None, "none")


def test_different_person_no_match():
    assert _r("Bob Martin") == (None, "none")


def test_extra_middle_name_still_matches():
    assert _r("Vishwajith Kumar Prakash")[0] == "u-prakash"
