"""Phase 10 PageRouter — three-signal merge for ChangeIntent → candidate pages.

PROP-V2-03 / D-05. Deterministic; no LLM in the default path.

This stage runs BETWEEN FactExtraction and PageQualifier. For each
ChangeIntent the pipeline produces, ``route_intent()`` returns a ranked
list of up to ``top_n`` candidate pages by merging three deterministic
signals:

    1. Pinecone semantic match — vector search on
       ``intent.subject + intent.target_hint + intent.instruction``.
    2. Neo4j heading-aware graph lookup — ``query_user_confluence_graph``
       already weights ``page_score + section_score * 2`` inside Cypher,
       so a section-heading hit naturally outranks a title-only hit.
    3. Explicit-token verbatim gate — if any content-bearing token of
       ``intent.subject`` appears verbatim (case-insensitive) in a page's
       title OR any of its H1/H2 headings, that page is force-promoted
       to the top. Heading matches rank above title-only matches.

The router NEVER reads a ContextVar for the user id (Pitfall 4 — those
do not survive ``asyncio.to_thread`` hops or parallel pipeline waves);
``graph_user_id`` MUST be passed explicitly by the caller.

Pinecone matches are normalized through ``_normalize_pinecone_match`` so
the chunk id (e.g. ``"pageX_3"``) is never confused with the real
``metadata.page_id`` (Pitfall 2 — fixed in Phase 8 and pinned here too).
"""
from __future__ import annotations

import asyncio
import logging
import re
from typing import Any, Dict, List, Optional, Set

from confluence_logic.confluence_page_graph import (
    list_user_confluence_pages,
    query_user_confluence_graph,
)
from confluence_logic.db.vector_store import PineconeStore

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Module-level Pinecone singleton (mirrors fact_extraction_agent._get_store)
# ---------------------------------------------------------------------------

_store: Optional[PineconeStore] = None


def _get_store() -> PineconeStore:
    """Lazily construct a module-level PineconeStore singleton."""
    global _store
    if _store is None:
        _store = PineconeStore()
    return _store


# ---------------------------------------------------------------------------
# Signal weights — explicit-token always outranks graph which always outranks
# semantic. Tuned so a heading-token hit beats a title-token hit beats a graph
# hit beats a semantic hit, regardless of raw Pinecone score magnitudes.
# ---------------------------------------------------------------------------

_W_TOKEN_HEADING = 20.0
_W_TOKEN_TITLE = 10.0
_W_GRAPH = 2.0
_W_SEMANTIC = 1.0

# Workspace pages scanned per route_intent call for the explicit-token gate.
_MAX_WORKSPACE_PAGES = 2000

# Pinecone + graph fetch fan-out — generous so the merge has room to rank.
_PER_SIGNAL_LIMIT = 10


# ---------------------------------------------------------------------------
# Pinecone match normalization
# COPIED FROM confluence_logic/agents/fact_extraction_agent.py (kept in sync —
# Pitfall 2: iterate metadata.page_id, never match.id which is the chunk id).
# ---------------------------------------------------------------------------


def _normalize_pinecone_match(match: Any) -> Optional[Dict[str, Any]]:
    """Flatten a raw Pinecone match into the shared {page_id, title, ...} shape.

    Pinecone returns matches where the chunk id is at the top level (e.g.
    ``"pageX_3"``) and the real page_id, title, heading, content all live
    under ``metadata``. Without this normalization, downstream code groups
    by chunk-id instead of page-id — causing duplicates and wrong-page
    selection.

    Returns None if the match has no real ``metadata.page_id``.
    """
    if match is None:
        return None
    if isinstance(match, dict):
        metadata = match.get("metadata") or {}
        score = match.get("score")
        chunk_id = match.get("id") or ""
    else:
        metadata = getattr(match, "metadata", None) or {}
        score = getattr(match, "score", None)
        chunk_id = getattr(match, "id", "") or ""

    if not isinstance(metadata, dict):
        return None

    page_id = (metadata.get("page_id") or "").strip()
    if not page_id:
        return None

    return {
        "page_id": page_id,
        "title": metadata.get("title") or "",
        "space_key": metadata.get("space_key") or "",
        "heading": metadata.get("heading") or None,
        "relevant_content": (
            metadata.get("markdown_content")
            or metadata.get("text_summary")
            or ""
        ),
        "score": float(score) if score is not None else 0.0,
        "source": "pinecone_rag",
        "_chunk_id": chunk_id,
    }


# ---------------------------------------------------------------------------
# Token utilities — small local stopword set + word-tokenizer. The grounding
# gate's ``content_bearing_tokens`` is intentionally not imported here to
# avoid Wave-1 ordering coupling with Plan 03; this fallback is enough for
# the explicit-token verbatim gate.
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_+-]{1,}")

_STOPWORDS: Set[str] = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has",
    "have", "in", "into", "is", "it", "of", "on", "or", "our", "page", "pages",
    "section", "sections", "that", "the", "their", "this", "to", "was", "we",
    "were", "with", "you", "your", "doc", "docs", "documentation",
}


def _content_tokens(text: str) -> Set[str]:
    """Lowercase content-bearing tokens from ``text``. Stopwords are dropped."""
    if not text:
        return set()
    tokens = {match.group(0).lower() for match in _TOKEN_RE.finditer(text)}
    return {t for t in tokens if t not in _STOPWORDS and len(t) >= 3}


# ---------------------------------------------------------------------------
# Signal 3 — explicit-token verbatim gate
# ---------------------------------------------------------------------------


async def _verbatim_token_promote(
    graph_user_id: str,
    subject: str,
) -> Dict[str, Dict[str, Any]]:
    """Scan workspace pages for verbatim subject-token matches in title/headings.

    Returns ``{page_id: page_row}`` where each row carries a transient
    ``_token_weight`` (heading match = ``_W_TOKEN_HEADING``, title-only
    match = ``_W_TOKEN_TITLE``).
    """
    if not graph_user_id or not subject:
        return {}

    subject_tokens = _content_tokens(subject)
    if not subject_tokens:
        return {}

    try:
        workspace_pages = await list_user_confluence_pages(
            graph_user_id, limit=_MAX_WORKSPACE_PAGES
        )
    except Exception as exc:  # noqa: BLE001 — graph backend may be offline
        logger.warning(
            "PageRouter: list_user_confluence_pages failed user=%s: %s",
            graph_user_id, exc,
        )
        return {}

    promoted: Dict[str, Dict[str, Any]] = {}
    for page in workspace_pages or []:
        page_id = page.get("page_id")
        if not page_id:
            continue

        title = (page.get("title") or "")
        title_tokens = _content_tokens(title)
        headings = page.get("headings") or []
        if not isinstance(headings, (list, tuple)):
            headings = []
        heading_tokens: Set[str] = set()
        for heading in headings:
            heading_tokens.update(_content_tokens(str(heading or "")))

        heading_hit = bool(subject_tokens & heading_tokens)
        title_hit = bool(subject_tokens & title_tokens)
        if not (heading_hit or title_hit):
            continue

        weight = _W_TOKEN_HEADING if heading_hit else _W_TOKEN_TITLE
        promoted[page_id] = {
            "page_id": page_id,
            "title": title,
            "space_key": page.get("space_key") or "",
            "version": page.get("version"),
            "headings": list(headings),
            "source": "explicit_token_gate",
            "_token_weight": weight,
            "_token_match_kind": "heading" if heading_hit else "title",
        }

    return promoted


# ---------------------------------------------------------------------------
# route_intent — three-signal merge entrypoint
# ---------------------------------------------------------------------------


async def route_intent(
    intent: Any,  # ChangeIntent
    graph_user_id: str,
    *,
    top_n: int = 5,
    connector: Optional[Any] = None,  # reserved for live-fallback wiring
) -> List[Dict[str, Any]]:
    """Return up to ``top_n`` ranked candidate pages for one ChangeIntent.

    Args:
        intent: A ``ChangeIntent`` (duck-typed — uses subject / target_hint /
            instruction attributes).
        graph_user_id: Required explicit user scope for the Neo4j +
            workspace-pages lookups. Never read from ContextVar (Pitfall 4).
        top_n: Cap on returned candidates.
        connector: Reserved for live-Confluence fallback in later plans.

    Returns:
        Ranked list of page dicts ``{page_id, title, space_key, signals,
        forced, score, ...}`` — empty list when all three signals miss.
    """
    subject = (getattr(intent, "subject", "") or "").strip()
    target_hint = (getattr(intent, "target_hint", "") or "").strip()
    instruction = (getattr(intent, "instruction", "") or "").strip()

    if not subject:
        return []

    query = " ".join(filter(None, [subject, target_hint, instruction]))

    # ---- Signal 1: Pinecone semantic (sync — wrap in to_thread) ------------
    sem_pages: Dict[str, Dict[str, Any]] = {}
    try:
        sem_hits = await asyncio.to_thread(
            _get_store().search, query, _PER_SIGNAL_LIMIT
        )
    except Exception as exc:  # noqa: BLE001 — Pinecone may be down/unconfigured
        logger.warning("PageRouter: Pinecone search failed: %s", exc)
        sem_hits = []

    for raw in sem_hits or []:
        normalized = _normalize_pinecone_match(raw)
        if not normalized:
            continue
        pid = normalized["page_id"]
        # First chunk per page wins (highest score by Pinecone ranking).
        if pid not in sem_pages:
            normalized["signals"] = {"semantic": normalized.get("score", 0.0)}
            sem_pages[pid] = normalized

    # ---- Signal 2: Neo4j heading-aware graph -------------------------------
    try:
        graph_hits = await query_user_confluence_graph(
            graph_user_id, subject, limit=_PER_SIGNAL_LIMIT
        )
    except Exception as exc:  # noqa: BLE001 — Neo4j may be down
        logger.warning("PageRouter: Neo4j graph lookup failed: %s", exc)
        graph_hits = []

    graph_pages: Dict[str, Dict[str, Any]] = {}
    for hit in graph_hits or []:
        pid = hit.get("page_id")
        if not pid:
            continue
        if pid not in graph_pages:
            graph_pages[pid] = {
                **hit,
                "signals": {"graph": float(hit.get("score") or 0)},
            }

    # ---- Signal 3: Explicit-token verbatim gate ----------------------------
    forced = await _verbatim_token_promote(graph_user_id, subject)

    # ---- Merge -------------------------------------------------------------
    merged: Dict[str, Dict[str, Any]] = {}

    # Start with semantic.
    for pid, page in sem_pages.items():
        merged[pid] = dict(page)
        merged[pid].setdefault("signals", {})

    # Layer graph (combine signals, keep richer fields).
    for pid, page in graph_pages.items():
        if pid in merged:
            existing_signals = dict(merged[pid].get("signals") or {})
            existing_signals.update(page.get("signals") or {})
            merged[pid]["signals"] = existing_signals
            # Prefer non-empty heading + relevant_content from graph
            if page.get("heading") and not merged[pid].get("heading"):
                merged[pid]["heading"] = page["heading"]
            if page.get("relevant_content") and not merged[pid].get("relevant_content"):
                merged[pid]["relevant_content"] = page["relevant_content"]
            if page.get("title") and not merged[pid].get("title"):
                merged[pid]["title"] = page["title"]
        else:
            merged[pid] = dict(page)

    # Layer forced (over-stamps with forced=True + explicit_token signal).
    for pid, page in forced.items():
        if pid not in merged:
            merged[pid] = {
                "page_id": pid,
                "title": page.get("title") or "",
                "space_key": page.get("space_key") or "",
                "heading": None,
                "relevant_content": "",
                "source": "explicit_token_gate",
                "signals": {},
            }
        merged[pid]["forced"] = True
        signals = dict(merged[pid].get("signals") or {})
        signals["explicit_token"] = float(page.get("_token_weight", _W_TOKEN_TITLE))
        merged[pid]["signals"] = signals
        merged[pid]["_token_match_kind"] = page.get("_token_match_kind")
        # Carry title/headings forward if missing.
        if page.get("title") and not merged[pid].get("title"):
            merged[pid]["title"] = page["title"]

    if not merged:
        logger.info(
            "route_intent subject=%r graph_user_id=%s sem=0 graph=0 forced=0 top=0",
            subject, graph_user_id,
        )
        return []

    # ---- Rank --------------------------------------------------------------
    def _weighted_score(page: Dict[str, Any]) -> float:
        sig = page.get("signals") or {}
        return (
            float(sig.get("explicit_token", 0.0))
            + float(sig.get("graph", 0.0)) * _W_GRAPH
            + float(sig.get("semantic", 0.0)) * _W_SEMANTIC
        )

    def _rank_key(page: Dict[str, Any]):
        # Tuple: forced first (0 sorts before 1), then by descending weighted score.
        return (
            0 if page.get("forced") else 1,
            -_weighted_score(page),
            page.get("title") or "",
        )

    ranked = sorted(merged.values(), key=_rank_key)

    # Attach final weighted score for downstream observability.
    for page in ranked:
        page["score"] = _weighted_score(page)

    result = ranked[:top_n]
    logger.info(
        "route_intent subject=%r graph_user_id=%s sem=%d graph=%d forced=%d top=%d",
        subject, graph_user_id,
        len(sem_pages), len(graph_pages), len(forced), len(result),
    )
    return result
