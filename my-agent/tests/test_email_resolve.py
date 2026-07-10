"""Email-first participant resolution: the work login email is authoritative and
resolves an org user org-wide (any team); no email / not-in-org → falls through.

    python -m pytest my-agent/tests/test_email_resolve.py -q
"""

import os
import sys
import types

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import org_activity  # noqa: E402


class _Resp:
    def __init__(self, ok, data):
        self.ok, self._data = ok, data

    def json(self):
        return self._data


def _fake_requests(rows_for_email):
    """Fake requests.get that returns `rows_for_email` keyed by the lowercased
    email in the query params."""
    def get(url, headers=None, params=None, timeout=None):
        email = (params or {}).get("email", "").replace("eq.", "")
        return _Resp(True, rows_for_email.get(email, []))
    return types.SimpleNamespace(get=get)


def test_email_matches_org_user(monkeypatch):
    monkeypatch.setattr(org_activity, "requests",
                        _fake_requests({"akshath.p@company.com": [{"id": "u-akshath-p"}]}))
    # Case-insensitive: Recall may give mixed case.
    assert org_activity._resolve_by_email("Akshath.P@Company.com", "org1") == "u-akshath-p"


def test_two_akshaths_resolve_by_distinct_emails(monkeypatch):
    monkeypatch.setattr(org_activity, "requests", _fake_requests({
        "akshath.p@company.com": [{"id": "u-p"}],
        "akshath.n@company.com": [{"id": "u-n"}],
    }))
    assert org_activity._resolve_by_email("akshath.p@company.com", "org1") == "u-p"
    assert org_activity._resolve_by_email("akshath.n@company.com", "org1") == "u-n"


def test_email_not_in_org_returns_none(monkeypatch):
    monkeypatch.setattr(org_activity, "requests", _fake_requests({}))
    assert org_activity._resolve_by_email("stranger@other.com", "org1") is None


def test_missing_email_or_org_returns_none(monkeypatch):
    # Must not even hit the network.
    monkeypatch.setattr(org_activity, "requests", None)
    assert org_activity._resolve_by_email(None, "org1") is None
    assert org_activity._resolve_by_email("x@y.com", None) is None
