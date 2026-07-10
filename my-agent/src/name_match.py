"""Token-aware matching of a meeting participant's display name to one org user.

Recall gives display names ("Vishwajith P", "vishwajit prakash", "iPhone"); we must
attribute each attendee's in-call time to the right org user. A whole-string fuzzy
ratio conflates people who differ only by a surname initial ("vishwajith p" vs
"vishwajith n" ≈ 0.92), so this matches by tokens instead:

- given name (first token) must match (with typo tolerance),
- a trailing single letter is treated as a **surname initial** and must agree with the
  candidate's surname first letter — "vishwajith p" matches "Vishwajith **P**rakash" and
  is rejected for "Vishwajith **N**air",
- full surnames fuzzy-match (typos allowed),
- if the best candidate isn't a clear, unique winner the name is **ambiguous** and we
  return no match (the caller records a guest; an admin can map it later).

Pure stdlib (re, difflib) so it's trivially unit-testable without the agent runtime.
"""

from __future__ import annotations

import difflib
import re

# Tunables.
_GIVEN_MIN = 0.8      # min similarity for the given (first) name to consider a candidate
_SURNAME_MIN = 0.6    # below this a surname/initial is a CONFLICT → reject the candidate
_MARGIN = 0.08        # top must beat the runner-up by this much, else ambiguous → no match


def normalize(name: str | None) -> str:
    """Lowercase, drop punctuation, collapse whitespace. Device labels keep letters
    (e.g. 'iPhone' → 'iphone') and simply won't match a real name."""
    if not name:
        return ""
    n = re.sub(r"[^a-z0-9 ]+", "", name.strip().lower())
    return re.sub(r"\s+", " ", n).strip()


def _tokens(name: str | None) -> list[str]:
    norm = normalize(name)
    return norm.split() if norm else []


def _token_match(a: str, b: str) -> float:
    """Similarity of two name tokens. A single-letter token is an initial: it matches
    iff it shares the other token's first letter (else 0 — a hard conflict)."""
    if a == b:
        return 1.0
    if len(a) == 1 or len(b) == 1:
        return 1.0 if a[0] == b[0] else 0.0
    return difflib.SequenceMatcher(None, a, b).ratio()


def _score(p: list[str], c: list[str]) -> tuple[float, str] | None:
    """Score participant tokens `p` against candidate tokens `c`.

    Returns (score, confidence) or None when the candidate is a different person
    (given name too far, or a surname/initial conflict)."""
    if not p or not c:
        return None

    # Given name: the participant's first token vs the candidate's best-matching token.
    gn = max(_token_match(p[0], ct) for ct in c)
    if gn < _GIVEN_MIN:
        return None

    p_sur, c_sur = p[1:], c[1:]

    if p_sur and c_sur:
        # Best agreement between any participant surname token and any candidate one.
        # A conflict (initial 'p' vs surname 'nair' → 0.0) drops below _SURNAME_MIN.
        best = max(_token_match(ps, cs) for ps in p_sur for cs in c_sur)
        if best < _SURNAME_MIN:
            return None
        return (gn + best) / 2.0, "high"

    if p_sur and not c_sur:
        # Candidate has only a first name — surname can't be verified.
        return gn * 0.9, "medium"

    # Participant is first-name-only.
    return gn * 0.85, "medium"


def resolve(rec_name: str | None, candidates: list[dict]) -> tuple[str | None, str]:
    """Match a Recall display name to one org user.

    candidates: [{"user_id", "name", "aliases": [str, ...]}, ...]
    Returns (user_id | None, "high" | "medium" | "none").
    """
    key = normalize(rec_name)
    if not key:
        return None, "none"

    # 1) Exact full-name or alias hit — only if it maps to exactly one user.
    exact = [
        c for c in candidates
        if normalize(c.get("name")) == key
        or any(normalize(a) == key for a in (c.get("aliases") or []))
    ]
    if len(exact) == 1:
        return exact[0]["user_id"], "high"
    if len(exact) > 1:
        return None, "none"

    # 2) Token-aware scoring across each candidate's name + aliases.
    p = key.split()
    scored: list[tuple[float, str, str]] = []  # (score, confidence, user_id)
    for c in candidates:
        forms = [c.get("name"), *(c.get("aliases") or [])]
        best: tuple[float, str] | None = None
        for form in forms:
            res = _score(p, _tokens(form))
            if res is not None and (best is None or res[0] > best[0]):
                best = res
        if best is not None:
            scored.append((best[0], best[1], c["user_id"]))

    if not scored:
        return None, "none"

    scored.sort(key=lambda x: x[0], reverse=True)
    top = scored[0]
    # Ambiguous when a runner-up is within the margin — don't guess.
    if len(scored) > 1 and (top[0] - scored[1][0]) < _MARGIN:
        return None, "none"
    return top[2], top[1]
