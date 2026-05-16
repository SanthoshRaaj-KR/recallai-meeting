"""Per-user Neo4j graph RAG over Confluence pages.

This graph is intentionally separate from the live meeting transcript graph in
graph_rag.py. It uses distinct labels and always scopes nodes by user_id.
"""
from __future__ import annotations

import asyncio
import logging
import os
import re
import time
from contextvars import ContextVar
from typing import Any, Dict, List, Optional

from bs4 import BeautifulSoup

from confluence_logic import graph_rag
from confluence_logic.connectors.confluence import ConfluenceConnector

logger = logging.getLogger(__name__)

GRAPH_KIND = "confluence_pages"
MAX_INDEX_PAGES = int(os.getenv("JARVIS_CONFLUENCE_GRAPH_MAX_PAGES", "500"))
GRAPH_TTL_SECONDS = int(os.getenv("JARVIS_CONFLUENCE_GRAPH_TTL_SECONDS", "7200"))
REFRESH_INTERVAL_SECONDS = int(os.getenv("JARVIS_CONFLUENCE_GRAPH_REFRESH_SECONDS", "7200"))
MAX_QUERY_RESULTS = int(os.getenv("JARVIS_CONFLUENCE_GRAPH_QUERY_RESULTS", "8"))
MAX_SECTION_CHARS = int(os.getenv("JARVIS_CONFLUENCE_GRAPH_SECTION_CHARS", "3500"))

_current_graph_user_id: ContextVar[str] = ContextVar("confluence_graph_user_id", default="")

_STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "have",
    "in", "into", "is", "it", "of", "on", "or", "our", "that", "the", "their",
    "this", "to", "was", "we", "were", "with", "you", "your",
}


def _driver():
    return graph_rag._get_driver()


def _is_connection_error(exc: Exception) -> bool:
    """Return True for Neo4j errors that mean the driver should be reset."""
    name = type(exc).__name__
    msg = str(exc).lower()
    return (
        name in {"ServiceUnavailable", "SessionExpired", "DriverError"}
        or "connectionreset" in msg
        or "routing" in msg
        or "defunct connection" in msg
    )


def _reset_driver_on_error(exc: Exception) -> None:
    if _is_connection_error(exc):
        logger.warning("Resetting Neo4j driver after connection error: %s", exc)
        graph_rag.reset_driver()


def set_current_graph_user_id(user_id: str):
    return _current_graph_user_id.set(user_id or "")


def reset_current_graph_user_id(token) -> None:
    _current_graph_user_id.reset(token)


def get_current_graph_user_id() -> str:
    return _current_graph_user_id.get()


def _clean_text(value: str) -> str:
    return re.sub(r"\s+", " ", value or "").strip()


def _terms(text: str, limit: int = 32) -> List[str]:
    words = re.findall(r"[A-Za-z][A-Za-z0-9_+-]{2,}", text.lower())
    seen: set[str] = set()
    terms: List[str] = []
    for word in words:
        if word in _STOP_WORDS or word in seen:
            continue
        seen.add(word)
        terms.append(word)
        if len(terms) >= limit:
            break
    return terms


def _truncate(value: str, max_chars: int = MAX_SECTION_CHARS) -> str:
    text = _clean_text(value)
    if len(text) <= max_chars:
        return text
    return f"{text[: max_chars // 2]} [... omitted ...] {text[-(max_chars // 2):]}"


def _sections_from_html(html: str) -> List[Dict[str, Any]]:
    soup = BeautifulSoup(html or "", "html.parser")
    headings = soup.find_all(re.compile("^h[1-6]$"))
    sections: List[Dict[str, Any]] = []

    root_parts: List[str] = []
    for child in soup.contents:
        if getattr(child, "name", None) and re.match(r"^h[1-6]$", child.name):
            break
        text = _clean_text(child.get_text(" ", strip=True) if hasattr(child, "get_text") else str(child))
        if text:
            root_parts.append(text)
    if root_parts:
        sections.append({"heading": "Root", "level": 0, "text": _truncate(" ".join(root_parts))})

    for heading in headings:
        level = int(heading.name[1])
        parts = []
        for sibling in heading.next_siblings:
            if getattr(sibling, "name", None) and re.match(r"^h[1-6]$", sibling.name):
                if int(sibling.name[1]) <= level:
                    break
            text = _clean_text(sibling.get_text(" ", strip=True) if hasattr(sibling, "get_text") else str(sibling))
            if text:
                parts.append(text)
        text = _truncate(" ".join(parts))
        sections.append(
            {
                "heading": _clean_text(heading.get_text(" ", strip=True)) or "Untitled section",
                "level": level,
                "text": text,
            }
        )

    if not sections:
        text = _truncate(soup.get_text(" ", strip=True))
        if text:
            sections.append({"heading": "Root", "level": 0, "text": text})
    return sections[:80]


_CONSTRAINTS_ENSURED: set[str] = set()


async def _ensure_constraints(driver) -> None:
    import neo4j

    constraints = [
        "CREATE CONSTRAINT cf_graph_user_scope IF NOT EXISTS FOR (g:CfGraphUser) REQUIRE (g.user_id, g.graph_kind) IS UNIQUE",
        "CREATE CONSTRAINT cf_page_scope IF NOT EXISTS FOR (p:CfPage) REQUIRE (p.user_id, p.page_id) IS UNIQUE",
        "CREATE CONSTRAINT cf_section_scope IF NOT EXISTS FOR (s:CfSection) REQUIRE (s.user_id, s.section_id) IS UNIQUE",
        "CREATE CONSTRAINT cf_term_scope IF NOT EXISTS FOR (t:CfTerm) REQUIRE (t.user_id, t.name) IS UNIQUE",
    ]
    for cypher in constraints:
        # Skip if already confirmed by a previous call this process lifetime — avoids
        # the repeated SCHEMA notification Neo4j sends when the constraint already exists.
        if cypher in _CONSTRAINTS_ENSURED:
            continue
        await driver.execute_query(cypher, routing_=neo4j.RoutingControl.WRITE)
        _CONSTRAINTS_ENSURED.add(cypher)


async def _is_fresh(driver, user_id: str) -> bool:
    import neo4j

    records, _, _ = await driver.execute_query(
        "MATCH (g:CfGraphUser {user_id: $user_id, graph_kind: $graph_kind}) "
        "RETURN g.indexed_at AS indexed_at LIMIT 1",
        {"user_id": user_id, "graph_kind": GRAPH_KIND},
        routing_=neo4j.RoutingControl.READ,
    )
    if not records:
        return False
    indexed_at = (records[0].data() if hasattr(records[0], "data") else records[0]).get("indexed_at") or 0
    return (time.time() * 1000) - float(indexed_at) < GRAPH_TTL_SECONDS * 1000


async def ensure_user_confluence_graph(user_id: str, force: bool = False) -> bool:
    """Build or incrementally refresh the isolated Confluence page graph for one user."""
    if not user_id:
        return False
    driver = _driver()
    if driver is None:
        return False

    try:
        await _ensure_constraints(driver)
        if not force and await _is_fresh(driver, user_id):
            return True

        connector = ConfluenceConnector()
        pages = await asyncio.to_thread(connector.list_pages, MAX_INDEX_PAGES)
        await _write_pages_incremental(driver, user_id, connector, pages)
        return True
    except Exception as exc:
        _reset_driver_on_error(exc)
        logger.warning("Confluence page graph build failed for user %s: %s", user_id, exc)
        return False


async def _existing_page_versions(driver, user_id: str) -> Dict[str, Any]:
    import neo4j

    records, _, _ = await driver.execute_query(
        "MATCH (:CfGraphUser {user_id: $user_id, graph_kind: $graph_kind})-[:OWNS_CONFLUENCE_PAGE]->(p:CfPage) "
        "RETURN p.page_id AS page_id, p.version AS version",
        {"user_id": user_id, "graph_kind": GRAPH_KIND},
        routing_=neo4j.RoutingControl.READ,
    )
    versions: Dict[str, Any] = {}
    for record in records:
        data = record.data() if hasattr(record, "data") else dict(record)
        if data.get("page_id"):
            versions[data["page_id"]] = data.get("version")
    return versions


async def _delete_missing_pages(driver, user_id: str, live_page_ids: List[str]) -> None:
    import neo4j

    await driver.execute_query(
        "MATCH (:CfGraphUser {user_id: $user_id, graph_kind: $graph_kind})-[:OWNS_CONFLUENCE_PAGE]->(p:CfPage) "
        "WHERE NOT p.page_id IN $live_page_ids "
        "DETACH DELETE p",
        {"user_id": user_id, "graph_kind": GRAPH_KIND, "live_page_ids": live_page_ids},
        routing_=neo4j.RoutingControl.WRITE,
    )


async def _write_pages_incremental(driver, user_id: str, connector: ConfluenceConnector, pages: List[Dict[str, Any]]) -> None:
    import neo4j

    existing_versions = await _existing_page_versions(driver, user_id)
    live_page_ids = [page.get("page_id") for page in pages if page.get("page_id")]
    changed_pages = [
        page for page in pages
        if page.get("page_id") and existing_versions.get(page.get("page_id")) != page.get("version")
    ]

    await driver.execute_query(
        "MERGE (g:CfGraphUser {user_id: $user_id, graph_kind: $graph_kind}) "
        "SET g.indexed_at = timestamp(), g.page_count = $page_count, "
        "g.changed_page_count = $changed_page_count, g.last_incremental = true",
        {
            "user_id": user_id,
            "graph_kind": GRAPH_KIND,
            "page_count": len(pages),
            "changed_page_count": len(changed_pages),
        },
        routing_=neo4j.RoutingControl.WRITE,
    )
    await _delete_missing_pages(driver, user_id, live_page_ids)

    for page in changed_pages:
        await _write_single_page(driver, user_id, connector, page)


async def _write_single_page(driver, user_id: str, connector: ConfluenceConnector, page: Dict[str, Any]) -> None:
    import neo4j

    page_id = page.get("page_id")
    if not page_id:
        return
    try:
        html = await asyncio.to_thread(connector.fetch_page_html, page_id)
        sections = _sections_from_html(html)
    except Exception as exc:
        logger.warning("Skipping Confluence graph page %s: %s", page_id, exc)
        sections = []

    page_terms = _terms(f"{page.get('title', '')} {page.get('excerpt', '')}")
    section_payload = []
    for index, section in enumerate(sections):
        page_id = page.get("page_id")
        section_id = f"{page_id}::{index}"
        section_payload.append(
            {
                "section_id": section_id,
                "heading": section.get("heading") or "Root",
                "level": section.get("level") or 0,
                "text": section.get("text") or "",
                "terms": _terms(f"{section.get('heading', '')} {section.get('text', '')}"),
            }
        )

    await driver.execute_query(
        "MATCH (g:CfGraphUser {user_id: $user_id, graph_kind: $graph_kind}) "
        "MERGE (p:CfPage {user_id: $user_id, page_id: $page_id}) "
        "SET p.title = $title, p.space_key = $space_key, p.version = $version, "
        "p.excerpt = $excerpt, p.terms = $terms, p.updated_at = timestamp() "
        "MERGE (g)-[:OWNS_CONFLUENCE_PAGE]->(p) "
        "WITH p "
        "OPTIONAL MATCH (p)-[:HAS_SECTION]->(old:CfSection) "
        "DETACH DELETE old",
        {
            "user_id": user_id,
            "graph_kind": GRAPH_KIND,
            "page_id": page_id,
            "title": page.get("title", ""),
            "space_key": page.get("space_key", ""),
            "version": page.get("version"),
            "excerpt": page.get("excerpt", ""),
            "terms": page_terms,
        },
        routing_=neo4j.RoutingControl.WRITE,
    )
    if not section_payload:
        return

    await driver.execute_query(
        "MATCH (p:CfPage {user_id: $user_id, page_id: $page_id}) "
        "UNWIND $sections AS section "
        "MERGE (s:CfSection {user_id: $user_id, section_id: section.section_id}) "
        "SET s.heading = section.heading, s.level = section.level, s.text = section.text, s.terms = section.terms "
        "MERGE (p)-[:HAS_SECTION]->(s)",
        {"user_id": user_id, "page_id": page_id, "sections": section_payload},
        routing_=neo4j.RoutingControl.WRITE,
    )
    all_terms = list(dict.fromkeys(page_terms + [term for section in section_payload for term in section["terms"]]))[:120]
    if all_terms:
        await driver.execute_query(
            "UNWIND $terms AS term "
            "MERGE (t:CfTerm {user_id: $user_id, name: term}) "
            "WITH t "
            "MATCH (p:CfPage {user_id: $user_id, page_id: $page_id}) "
            "MERGE (p)-[:MENTIONS]->(t)",
            {"user_id": user_id, "page_id": page_id, "terms": all_terms},
            routing_=neo4j.RoutingControl.WRITE,
        )


async def query_user_confluence_graph(user_id: str, query: str, limit: int = MAX_QUERY_RESULTS) -> List[Dict[str, Any]]:
    """Return relevant page/section contexts for one user's Confluence graph only."""
    if not user_id or not query:
        return []
    driver = _driver()
    if driver is None:
        return []

    import neo4j

    query_terms = _terms(query, limit=12)
    try:
        records, _, _ = await driver.execute_query(
            "MATCH (:CfGraphUser {user_id: $user_id, graph_kind: $graph_kind})-[:OWNS_CONFLUENCE_PAGE]->(p:CfPage) "
            "OPTIONAL MATCH (p)-[:HAS_SECTION]->(s:CfSection) "
            "WITH p, s, "
            "size([term IN $terms WHERE term IN coalesce(p.terms, [])]) AS page_score, "
            "size([term IN $terms WHERE term IN coalesce(s.terms, [])]) AS section_score "
            "WHERE page_score > 0 OR section_score > 0 "
            "RETURN p.page_id AS page_id, p.title AS title, p.space_key AS space_key, p.version AS version, "
            "s.heading AS heading, s.text AS text, page_score + section_score * 2 AS score "
            "ORDER BY score DESC, title ASC "
            "LIMIT $limit",
            {
                "user_id": user_id,
                "graph_kind": GRAPH_KIND,
                "terms": query_terms,
                "limit": max(1, limit),
            },
            routing_=neo4j.RoutingControl.READ,
        )
        results = []
        for record in records:
            data = record.data() if hasattr(record, "data") else dict(record)
            results.append(
                {
                    "page_id": data.get("page_id"),
                    "title": data.get("title") or "",
                    "space_key": data.get("space_key") or "",
                    "version": data.get("version"),
                    "heading": data.get("heading"),
                    "relevant_content": data.get("text") or "",
                    "score": data.get("score") or 0,
                    "source": "neo4j_confluence_graph",
                }
            )
        return results
    except Exception as exc:
        _reset_driver_on_error(exc)
        logger.warning("Confluence page graph query failed for user %s: %s", user_id, exc)
        return []


async def list_user_confluence_pages(user_id: str, limit: int = MAX_QUERY_RESULTS) -> List[Dict[str, Any]]:
    if not user_id:
        return []
    driver = _driver()
    if driver is None:
        return []

    import neo4j

    try:
        records, _, _ = await driver.execute_query(
            "MATCH (:CfGraphUser {user_id: $user_id, graph_kind: $graph_kind})-[:OWNS_CONFLUENCE_PAGE]->(p:CfPage) "
            "RETURN p.page_id AS page_id, p.title AS title, p.space_key AS space_key, p.version AS version "
            "ORDER BY p.updated_at DESC, p.title ASC "
            "LIMIT $limit",
            {
                "user_id": user_id,
                "graph_kind": GRAPH_KIND,
                "limit": max(1, limit),
            },
            routing_=neo4j.RoutingControl.READ,
        )
        return [
            {
                "page_id": (record.data() if hasattr(record, "data") else dict(record)).get("page_id"),
                "title": (record.data() if hasattr(record, "data") else dict(record)).get("title") or "",
                "space_key": (record.data() if hasattr(record, "data") else dict(record)).get("space_key") or "",
                "version": (record.data() if hasattr(record, "data") else dict(record)).get("version"),
            }
            for record in records
        ]
    except Exception as exc:
        _reset_driver_on_error(exc)
        logger.warning("Could not list Confluence graph pages for user %s: %s", user_id, exc)
        return []


async def list_graph_user_ids() -> List[str]:
    driver = _driver()
    if driver is None:
        return []
    import neo4j

    try:
        records, _, _ = await driver.execute_query(
            "MATCH (g:CfGraphUser {graph_kind: $graph_kind}) RETURN g.user_id AS user_id",
            {"graph_kind": GRAPH_KIND},
            routing_=neo4j.RoutingControl.READ,
        )
        return [
            (record.data() if hasattr(record, "data") else dict(record)).get("user_id")
            for record in records
            if (record.data() if hasattr(record, "data") else dict(record)).get("user_id")
        ]
    except Exception as exc:
        logger.warning("Could not list Confluence graph users: %s", exc)
        return []


async def refresh_all_known_user_graphs() -> None:
    for user_id in await list_graph_user_ids():
        await ensure_user_confluence_graph(user_id, force=True)


async def refresh_known_user_graphs_forever(stop_event: asyncio.Event) -> None:
    while not stop_event.is_set():
        try:
            await refresh_all_known_user_graphs()
        except Exception as exc:
            logger.warning("Periodic Confluence graph refresh failed: %s", exc)
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=REFRESH_INTERVAL_SECONDS)
        except asyncio.TimeoutError:
            continue


async def refresh_page_in_graph(user_id: str, page_id: str) -> bool:
    """Delete and re-create the Neo4j graph nodes for one Confluence page (APPLY-03).

    Lightweight alternative to `ensure_user_confluence_graph(force=True)` — only
    touches the single committed page rather than re-building the entire user graph.
    Swallows all exceptions and logs at WARNING so callers can use fire-and-forget.
    """
    if not user_id or not page_id:
        return False
    driver = _driver()
    if driver is None:
        logger.warning("refresh_page_in_graph: Neo4j driver unavailable for user %s page %s", user_id, page_id)
        return False

    import neo4j

    try:
        # 1. Detach-delete the existing CfPage node (cascades to CfSection children via DETACH DELETE)
        await driver.execute_query(
            "MATCH (p:CfPage {user_id: $user_id, page_id: $page_id, graph_kind: $graph_kind}) "
            "DETACH DELETE p",
            {"user_id": user_id, "page_id": page_id, "graph_kind": GRAPH_KIND},
            routing_=neo4j.RoutingControl.WRITE,
        )

        # 2. Fetch fresh metadata from Confluence to build the page dict _write_single_page expects
        connector = ConfluenceConnector()
        try:
            meta = await asyncio.to_thread(connector.get_page_metadata, page_id)
        except Exception as exc:
            logger.warning("refresh_page_in_graph: could not fetch metadata for page %s: %s", page_id, exc)
            return False

        page_dict = {
            "page_id": page_id,
            "title": meta.get("title", ""),
            "space_key": (meta.get("space") or {}).get("key", ""),
            "version": (meta.get("version") or {}).get("number"),
            "excerpt": "",
        }

        # 3. Re-create via the existing single-page writer
        await _write_single_page(driver, user_id, connector, page_dict)
        logger.info("refresh_page_in_graph: re-indexed page %s for user %s", page_id, user_id)
        return True

    except Exception as exc:
        _reset_driver_on_error(exc)
        logger.warning("refresh_page_in_graph failed for user %s page %s: %s", user_id, page_id, exc)
        return False
