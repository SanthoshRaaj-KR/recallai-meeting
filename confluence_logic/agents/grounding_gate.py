"""Deterministic post-verifier hallucination gate (PROP-V2-01, D-04).

Page existence + per-op token grounding. No LLM calls.

This module implements the hard hallucination gate that runs after the
verifier and BEFORE persist. Two gates:

1. Page existence (``check_page_existence``):
   - ``card.page_id`` is a node in the user's ``confluence_page_graph``
     (Neo4j, scoped by ``user_id`` via an EXPLICIT parameter — never via
     ContextVar, per Pitfall 4) OR
   - has been verified via a live REST ``GET /content/{id}`` returning a
     payload with an ``id`` key. Otherwise the card is DROPPED.

2. Per-operation token grounding (``check_grounding``):
   - For ``replace.old_text`` / ``delete``: every content-bearing token of
     ``before_content`` must appear in ``current_page_content``.
   - For ``replace.new_text`` / ``insert`` / ``add`` / ``create``: every
     content-bearing token of ``after_content`` must appear in
     ``{transcript ∪ current_page_content}``.
   - For ``reorder``: no new content-bearing token may be introduced
     beyond ``before_content``.

Cards failing either check are dropped with a ``logger.warning`` naming
the offending tokens and the source check. ``card['grounding_failures']``
is populated by the orchestrator for SSE diagnostic emission.

Design notes:
   - Stopword set extends ``page_qualifier._STOP`` (sourced) with
     additional Confluence-noise words.
   - ``ALWAYS_CONTENT_RE`` force-keeps numbers/dates, UPPER acronyms,
     snake_case and camelCase identifiers — even if they happen to be
     short or look stopword-ish lowercased.
   - Tokenizer keeps hyphen/dot/slash-joined identifiers as single tokens
     (e.g., compound model identifiers, ``2024-05-30`` dates,
     ``v1.2.3`` semver tags).
   - Pure stdlib + asyncio; zero new package deps.
"""
from __future__ import annotations

import asyncio
import logging
import os
import re
from typing import Any, Dict, List, Optional, Set

from confluence_logic.agents.page_qualifier import _STOP as _BASE_STOP

logger = logging.getLogger(__name__)


# Kill-switch for safety per RESEARCH.md line 862.
# When False, ``check_grounding`` returns ok=True with reason="disabled".
JARVIS_GROUNDING_GATE_ENABLED: bool = (
    os.getenv("JARVIS_GROUNDING_GATE_ENABLED", "1") == "1"
)


# Stopwords: base set from page_qualifier extended with Confluence-noise words.
# Keep this conservative — over-stripping risks false-positive "missing"
# tokens that are actually present.
STOPWORDS = _BASE_STOP | {
    "the", "a", "an", "and", "or", "of", "for", "to", "in", "on", "with",
    "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did",
    "will", "would", "should", "could", "may", "might",
    "this", "that", "these", "those", "it", "its",
    "as", "at", "by", "from",
    "page", "section", "doc", "documentation", "confluence",
    "but", "let", "lets", "just", "also", "so", "then",
    "instead", "rather", "than",
}


# Always content-bearing regardless of the stopword list. Tokens matching this
# regex are kept even if they would otherwise be dropped.
#   ^\d+([.,/\-]\d+)*[a-z%]*$   numbers, dates, ratios — "30", "99.9%", "2024-05-30"
#   ^[A-Z]{2,}$                  UPPER acronyms — "API", "SLA"
#   ^[a-z]+(_[a-z]+)+$           snake_case identifiers — "foo_bar"
#   ^[a-z]+[A-Z][a-zA-Z]*$       camelCase identifiers — "fooBar", "camelCaseName"
ALWAYS_CONTENT_RE: re.Pattern = re.compile(
    r"^\d+([.,/\-]\d+)*[a-z%]*$"
    r"|^[A-Z]{2,}$"
    r"|^[a-z]+(_[a-z]+)+$"
    r"|^[a-z]+[A-Z][a-zA-Z]*$"
)


# Tokenizer: keep hyphen/dot/slash-joined runs as single tokens so technical
# identifiers like compound model names, "v1.2.3" semver tags and
# "2024-05-30" dates stay atomic. The base atom is [A-Za-z0-9_]+ so
# simple identifiers also match.
_TOKEN_RE: re.Pattern = re.compile(r"[A-Za-z0-9_]+(?:[.\-/][A-Za-z0-9_]+)*")


def content_bearing_tokens(text: str) -> List[str]:
    """Tokenize, lowercase, strip stopwords, keep numbers/identifiers/dates.

    Returns content-bearing tokens (lowercased). Stopwords and length<2
    tokens are dropped, UNLESS the raw or lowercased token matches
    ``ALWAYS_CONTENT_RE`` (numbers, dates, UPPER acronyms, snake_case,
    camelCase) — those are always kept.

    Order is preserved (callers may use as list or set).
    """
    tokens: List[str] = []
    for raw in _TOKEN_RE.findall(text or ""):
        low = raw.lower()
        # Always-content takes precedence over stopword + length filters.
        if ALWAYS_CONTENT_RE.match(raw) or ALWAYS_CONTENT_RE.match(low):
            tokens.append(low)
            continue
        if low in STOPWORDS or len(low) < 2:
            continue
        tokens.append(low)
    return tokens


def tokens_subset(needles: List[str], haystack_text: str) -> List[str]:
    """Return the list of ``needles`` tokens MISSING from ``haystack_text``.

    Empty return value ⇒ every needle present (subset holds).
    """
    haystack_set: Set[str] = set(content_bearing_tokens(haystack_text))
    return [t for t in needles if t not in haystack_set]


async def check_grounding(
    card: Dict[str, Any],
    transcript_text: str,
    current_page_content: str,
) -> Dict[str, Any]:
    """Per-op token grounding check (D-04 gate 2).

    Returns ``{ok: bool, failures: [token, ...], reason: str}``. Caller is
    responsible for dropping the card on ``ok=False`` and writing
    ``card['grounding_failures']`` for SSE diagnostic emission.

    Branches by operation:
      - replace.old_text / delete  → before_tokens ⊆ current_page_content.
      - reorder                    → after_tokens ⊆ before_tokens.
      - additive (insert/add/create/replace.new) →
          after_tokens ⊆ {transcript ∪ current_page_content}.

    Kill-switch: when ``JARVIS_GROUNDING_GATE_ENABLED`` is False, returns
    ok=True with reason="disabled" and logs a warning (operator-visible).
    """
    if not JARVIS_GROUNDING_GATE_ENABLED:
        logger.warning(
            "GroundingGate disabled via env — running in unsafe mode"
        )
        return {"ok": True, "failures": [], "reason": "disabled"}

    action = (card.get("change_type") or "edit").lower()
    edit_mode = (card.get("edit_mode") or "").lower()
    op_type = card.get("operation_type")

    after = card.get("after_content") or ""
    before = card.get("before_content") or ""
    after_tokens = content_bearing_tokens(after)
    before_tokens = content_bearing_tokens(before)

    page_id = card.get("page_id")

    # Branch 1 — replace.old / delete: before tokens must all be on the page.
    if (action == "edit" and edit_mode == "replace") or action == "delete":
        missing = tokens_subset(before_tokens, current_page_content)
        if missing:
            reason = "old_text tokens missing from current_page_content"
            logger.warning(
                "GroundingGate drop page=%s reason=%s failures=%s",
                page_id, reason, missing,
            )
            return {"ok": False, "failures": missing, "reason": reason}
        # Fall through to additive check for the new_text side of a replace.

    # Branch 2 — reorder: after tokens must be a subset of before tokens.
    if op_type == "reorder":
        before_set = set(before_tokens)
        introduced = [t for t in after_tokens if t not in before_set]
        if introduced:
            reason = "reorder introduced new tokens"
            logger.warning(
                "GroundingGate drop page=%s reason=%s failures=%s",
                page_id, reason, introduced,
            )
            return {"ok": False, "failures": introduced, "reason": reason}
        return {"ok": True, "failures": [], "reason": ""}

    # Branch 3 — additive (insert / add / create / append / replace.new):
    # every after token must be in {transcript ∪ current_page_content}.
    allowed = (transcript_text or "") + "\n" + (current_page_content or "")
    missing_after = tokens_subset(after_tokens, allowed)
    if missing_after:
        reason = "after_content tokens missing from {transcript U page}"
        logger.warning(
            "GroundingGate drop page=%s reason=%s failures=%s",
            page_id, reason, missing_after,
        )
        return {"ok": False, "failures": missing_after, "reason": reason}

    return {"ok": True, "failures": [], "reason": ""}


async def check_page_existence(
    page_id: Optional[str],
    user_id: str,
    connector: Optional[Any] = None,
) -> bool:
    """Page-existence check (D-04 gate 1).

    Returns True iff:
      (a) ``page_id`` appears in the user-scoped Neo4j
          ``confluence_page_graph`` (via ``list_user_confluence_pages``), OR
      (b) a live Confluence REST ``GET /content/{page_id}`` returns a
          payload containing an ``id`` key.

    ``user_id`` MUST be passed as an EXPLICIT parameter — never read from
    a ContextVar inside this module (Pitfall 4: cross-user contamination
    in async pipelines).
    """
    if not page_id:
        return False

    # 1) Graph membership (already user-scoped). Deferred import avoids
    # circulars between agents/* and confluence_page_graph.
    try:
        from confluence_logic import confluence_page_graph as cpg

        pages = await cpg.list_user_confluence_pages(user_id, limit=2000)
        if any(p.get("page_id") == page_id for p in pages or []):
            return True
    except Exception as exc:
        # Graph errors are non-fatal — fall through to REST check.
        logger.warning(
            "GroundingGate graph lookup failed for page_id=%s user_id=%s: %s",
            page_id, user_id, exc,
        )

    # 2) Live REST fallback. Lazy-construct a connector if one was not
    # injected by the caller (tests typically inject an AsyncMock or a
    # MagicMock with a fake .get_page_metadata).
    if connector is None:
        try:
            from confluence_logic.connectors.confluence import ConfluenceConnector
            connector = ConfluenceConnector()
        except Exception as exc:
            logger.warning(
                "GroundingGate REST fallback connector init failed for page_id=%s: %s",
                page_id, exc,
            )
            return False

    try:
        meta = await asyncio.to_thread(connector.get_page_metadata, page_id)
        return bool(meta and meta.get("id"))
    except Exception as exc:
        logger.warning(
            "GroundingGate REST GET failed for page_id=%s: %s",
            page_id, exc,
        )
        return False
