"""Section corpus fetch and per-run cache (RETR-V3-01/02, ARCH-V3-01).

Fetches ALL Confluence sections for a user from the Neo4j graph (CfSection
nodes) and caches them on ``PipelineContext.section_corpus`` so the BM25
index build and contradiction fan-out share one Neo4j round-trip per run.

Design notes:
  - ``graph_user_id`` is an EXPLICIT parameter (Pitfall 1 / T-11-02).
    NEVER read the ``confluence_graph_user_id`` ContextVar inside this module —
    it does not survive ``asyncio.to_thread`` boundaries.
  - Degradation: Neo4j unavailable → returns an empty list and logs a warning
    at ``%``-style (CLAUDE.md convention).  Never raises into the orchestrator.
  - MAX_SECTION_CHARS truncation is applied to all ``text`` fields (mirrors
    ``confluence_page_graph._truncate`` so section text stays bounded).

SectionRow keys returned by ``fetch_section_corpus``:
  ``(page_id, section_id, heading)`` → carries ``text``, ``terms``,
  ``page_title``, ``space_key``, ``version``.

The return type is intentionally a plain list of ``SectionRow`` namedtuples
(no Pydantic) so the corpus is cheap to build and traverse deterministically
without Pydantic validation overhead on ≤40k rows.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, List, NamedTuple, Optional

if TYPE_CHECKING:
    from confluence_logic.pipeline.context import PipelineContext

logger = logging.getLogger(__name__)

# Mirror the truncation applied by confluence_page_graph._truncate.
MAX_SECTION_CHARS: int = 3500


# ---------------------------------------------------------------------------
# SectionRow — lightweight named tuple representing one corpus entry
# ---------------------------------------------------------------------------

class SectionRow(NamedTuple):
    """One section in the per-run corpus, keyed by (page_id, section_id, heading).

    ``text`` is truncated to MAX_SECTION_CHARS.
    ``terms`` is the list of lowercased content-bearing tokens stored in
    Neo4j (reused by the contradiction fan-out lexical scan).
    """

    page_id: str
    section_id: str
    heading: str
    text: str
    terms: List[str]
    page_title: str = ""
    space_key: str = ""
    version: int = 0


# ---------------------------------------------------------------------------
# Cypher query — returns all CfSection nodes scoped to graph_user_id
# ---------------------------------------------------------------------------

_CORPUS_CYPHER = """
MATCH (p:CfPage {user_id: $user_id, graph_kind: 'confluence_pages'})
      -[:HAS_SECTION]->(s:CfSection)
RETURN
    p.page_id   AS page_id,
    p.title     AS title,
    p.space_key AS space_key,
    p.version   AS version,
    s.section_id AS section_id,
    s.heading   AS heading,
    s.text      AS text,
    s.terms     AS terms
ORDER BY p.page_id, s.section_id
"""


def _truncate(text: str) -> str:
    """Truncate section text to MAX_SECTION_CHARS (mirrors _truncate in graph module)."""
    if text and len(text) > MAX_SECTION_CHARS:
        return text[:MAX_SECTION_CHARS]
    return text or ""


def _row_from_record(record: Any) -> Optional[SectionRow]:
    """Convert a neo4j record.data() dict to a SectionRow.

    Returns None if the record is missing a page_id or section_id.
    """
    if isinstance(record, dict):
        data = record
    else:
        try:
            data = record.data()
        except Exception:
            return None

    page_id = (data.get("page_id") or "").strip()
    section_id = (data.get("section_id") or "").strip()
    heading = (data.get("heading") or "").strip()
    if not page_id:
        return None
    if not section_id:
        # Fall back to a synthetic section_id from page_id + heading
        section_id = f"{page_id}::{heading}" if heading else page_id

    raw_text = data.get("text") or ""
    terms_raw = data.get("terms") or []
    if not isinstance(terms_raw, list):
        terms_raw = []

    return SectionRow(
        page_id=page_id,
        section_id=section_id,
        heading=heading,
        text=_truncate(str(raw_text)),
        terms=[str(t).lower() for t in terms_raw if t],
        page_title=(data.get("title") or "").strip(),
        space_key=(data.get("space_key") or "").strip(),
        version=int(data.get("version") or 0),
    )


async def fetch_section_corpus(
    graph_user_id: str,
    ctx: "PipelineContext",
) -> List[SectionRow]:
    """Fetch all Confluence sections for a user and cache on ``ctx.section_corpus``.

    On a second call in the same run, returns the cached result immediately
    without hitting Neo4j again.  This lets the BM25 index build (RETR-V3-01)
    and the contradiction fan-out (CON-V3-01) share one Neo4j fetch.

    Degradation: if the Neo4j driver is unavailable or any exception occurs,
    logs a warning and returns an empty list — never raises.

    Args:
        graph_user_id: The per-user Neo4j scoping key.  MUST be passed
            explicitly — never read the ContextVar inside this function.
        ctx: The current PipelineContext.  Its ``section_corpus`` slot is
            populated (and returned on cache hit) by this function.

    Returns:
        A list of SectionRow namedtuples covering every CfSection for the
        user.  Empty list on error or when no sections are indexed.
    """
    # Cache hit — return without re-hitting Neo4j.
    if ctx.section_corpus is not None:
        return ctx.section_corpus  # type: ignore[return-value]

    rows: List[SectionRow] = []
    try:
        # Import lazily so the module is importable without Neo4j credentials
        # (ARCH-V3-01: no live-service side effects at import time).
        import neo4j  # noqa: PLC0415
        from confluence_logic.confluence_page_graph import _driver  # type: ignore[attr-defined]  # noqa: PLC0415

        driver = _driver()
        if driver is None:
            logger.warning(
                "fetch_section_corpus: Neo4j driver not initialised for user %s — "
                "returning empty corpus (graceful degradation)",
                graph_user_id,
            )
            ctx.section_corpus = rows
            return rows

        records, _summary, _keys = await driver.execute_query(
            _CORPUS_CYPHER,
            {"user_id": graph_user_id},
            routing_=neo4j.RoutingControl.READ,
        )
        for rec in records:
            row = _row_from_record(rec)
            if row is not None:
                rows.append(row)

        logger.debug(
            "fetch_section_corpus: loaded %d sections for user %s",
            len(rows),
            graph_user_id,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "fetch_section_corpus: Neo4j error for user %s — returning empty corpus "
            "(non-fatal): %s",
            graph_user_id,
            exc,
        )
        rows = []

    ctx.section_corpus = rows
    return rows
