"""
LiveKit Agent <-> Jarvis responder bridge (Phase 03 / REQ-15).

Exposes the existing meeting_responder + general_responder + classifier functions as
`@function_tool` callables that the agent_worker.AgentSession LLM can invoke by name.

Design contract: this module is PURE DELEGATION. No LLM calls live here. No prompt
strings live here. Every tool body is a one-liner `await existing_function(...)` so
the canonical Jarvis logic in meeting_responder.py / general_responder.py remains the
single source of truth.

Cross-process note (Pitfall 3 in RESEARCH): the agent worker runs in a separate
process from jarvis_agentic.py's FastAPI server. Tools cannot touch the live
`_meeting_sessions` dict directly — they receive `session_id` via RunContext metadata
and look up transcript data via the IPC-friendly `get_transcript_log_for_session`
helper, which Plan 05 backs with either an HTTP fetch or shared file mechanism.
For Wave 1 this helper returns an in-process lookup; Plan 05 swaps it for IPC.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from datetime import datetime
from typing import Any, Dict, List

from livekit.agents import RunContext
from livekit.agents.llm import function_tool

# Underlying responders — these are the source of truth for Jarvis voice answers.
from confluence_logic.meeting_responder import (
    summarize_meeting,
    generate_opinion,
    extract_action_items,
    summarize_speaker,
)
from confluence_logic.general_responder import answer_general_question
from confluence_logic.classifier import classify_intent  # re-exported for Plan 04 routing

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Transcript-log access — HTTP fetch to the FastAPI server (cross-process safe).
# The agent worker runs in a separate process; in-process dict access would always
# return [] because the worker has its own empty _meeting_sessions copy.
# ---------------------------------------------------------------------------
_JARVIS_API_BASE: str = os.getenv("WEBHOOK_URL", "").rstrip("/")


async def get_transcript_log_for_session(session_id: str) -> List[Dict[str, Any]]:
    """Return the transcript log for session_id by fetching /transcript/{session_id}.

    Falls back to in-process lookup when WEBHOOK_URL is not set (dev / test mode).
    """
    if not session_id:
        return []

    if _JARVIS_API_BASE:
        try:
            import requests as _req  # noqa: PLC0415
            resp = await asyncio.to_thread(
                _req.get,
                f"{_JARVIS_API_BASE}/transcript/{session_id}",
                timeout=3,
            )
            resp.raise_for_status()
            data = resp.json()
            log = data.get("transcript_log", [])
            logger.debug("transcript fetch: %d entries for session=%s", len(log), session_id)
            return log
        except Exception as exc:
            logger.warning("get_transcript_log_for_session HTTP failed (session=%s): %s", session_id, exc)
            return []

    # Dev fallback: same-process import (only works when not using a separate worker process)
    try:
        from confluence_logic.jarvis_agentic import _meeting_sessions  # noqa: PLC0415
    except Exception as exc:
        logger.debug("agent_bridge: jarvis_agentic not importable: %s", exc)
        return []
    state = _meeting_sessions.get(session_id)
    if not state:
        return []
    return list(state.get("transcript_log") or [])


def _session_id_from_context(context: RunContext) -> str:
    """Extract session_id from RunContext.

    In-process path (Phase 03): reads from context.session.userdata["session_id"]
    set by _start_in_process_agent_session in jarvis_agentic.py.
    Agent worker path (fallback): reads from context.job.metadata JSON.
    """
    # In-process path: userdata set on AgentSession constructor
    try:
        userdata = context.session.userdata
        if isinstance(userdata, dict):
            sid = userdata.get("session_id")
            if sid:
                return str(sid)
    except Exception:
        pass
    # Agent worker fallback: session_id in job metadata JSON
    try:
        meta = json.loads(context.job.metadata or "{}")
        return str(meta.get("session_id") or "")
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# Tool wrappers (REQ-15)
# ---------------------------------------------------------------------------
@function_tool
async def summarize_meeting_tool(context: RunContext, detail_level: str = "brief") -> str:
    """Summarize what has been said in the current meeting so far.

    Args:
        detail_level: 'brief' for a 2-3 sentence recap, 'full' for a longer summary.
    """
    sid = _session_id_from_context(context)
    transcript = await get_transcript_log_for_session(sid)
    t0 = time.perf_counter()
    result = await summarize_meeting(transcript, detail_level=detail_level)
    logger.info("⏱️  summarize_meeting_tool: %.0fms (%d transcript entries)", (time.perf_counter() - t0) * 1000, len(transcript))
    return result


@function_tool
async def generate_opinion_tool(context: RunContext, query: str = "") -> str:
    """Give Jarvis's opinion or recommendation on what is being discussed.

    Args:
        query: The specific question the user asked (e.g. 'which option is better?').
    """
    sid = _session_id_from_context(context)
    transcript = await get_transcript_log_for_session(sid)
    t0 = time.perf_counter()
    result = await generate_opinion(transcript, query=query)
    logger.info("⏱️  generate_opinion_tool: %.0fms (query=%.40r)", (time.perf_counter() - t0) * 1000, query)
    return result


@function_tool
async def extract_action_items_tool(context: RunContext) -> str:
    """Extract action items, commitments, and next steps from the meeting transcript."""
    sid = _session_id_from_context(context)
    transcript = await get_transcript_log_for_session(sid)
    t0 = time.perf_counter()
    result = await extract_action_items(transcript)
    logger.info("⏱️  extract_action_items_tool: %.0fms (%d transcript entries)", (time.perf_counter() - t0) * 1000, len(transcript))
    return result


@function_tool
async def summarize_speaker_tool(context: RunContext, speaker_name: str) -> str:
    """Summarize what a specific participant said in the meeting.

    Args:
        speaker_name: The participant's display name as it appears in the transcript.
    """
    sid = _session_id_from_context(context)
    transcript = await get_transcript_log_for_session(sid)
    t0 = time.perf_counter()
    result = await summarize_speaker(transcript, speaker_name=speaker_name)
    logger.info("⏱️  summarize_speaker_tool: %.0fms (speaker=%.30r)", (time.perf_counter() - t0) * 1000, speaker_name)
    return result


@function_tool
async def get_current_datetime(context: RunContext) -> str:
    """Return the current date and time. Call this whenever the user asks about the current date, time, or day."""
    return datetime.now().strftime("%A, %B %d %Y — %I:%M %p")


@function_tool
async def answer_general_question_tool(
    context: RunContext,
    question: str,
    force_web_search: bool = False,
) -> str:
    """Answer a general (non-Confluence, non-meeting) question.

    Args:
        question: The user's question text.
        force_web_search: True to force Tavily web search (use for current events / live data).
    """
    t0 = time.perf_counter()
    result = await answer_general_question(
        question,
        conversation_history="",
        graph_context="",
        speech_rewrite_enabled=False,
        multiturn_reference=False,
        force_web_search=force_web_search,
    )
    logger.info("⏱️  answer_general_question_tool: %.0fms (web=%s, q=%.50r)", (time.perf_counter() - t0) * 1000, force_web_search, question)
    return result


# ---------------------------------------------------------------------------
# Confluence page tools (LiveKit-native wrappers over ConfluenceConnector)
# ---------------------------------------------------------------------------
def _confluence_connector():
    from confluence_logic.connectors.confluence import ConfluenceConnector
    return ConfluenceConnector()


@function_tool
async def search_confluence_pages(context: RunContext, query: str) -> str:
    """Search Confluence for pages relevant to a topic or keyword.

    Args:
        query: The topic or keywords to search for.
    """
    def _run():
        try:
            connector = _confluence_connector()
            results = connector.search_pages(query, limit=5)
            if not results:
                return "No Confluence pages found for that query."
            lines = [f"- {r.get('title', 'Untitled')} (id={r.get('page_id', '?')})" for r in results]
            return "Found pages:\n" + "\n".join(lines)
        except Exception as exc:
            return f"Search failed: {exc}"
    return await asyncio.to_thread(_run)


@function_tool
async def list_confluence_pages(context: RunContext, limit: int = 20) -> str:
    """List recent Confluence pages in the workspace.

    Args:
        limit: Maximum number of pages to return (default 20).
    """
    def _run():
        try:
            connector = _confluence_connector()
            pages = connector.list_pages(limit=limit)
            if not pages:
                return "No pages found in the workspace."
            lines = [f"- {p.get('title', 'Untitled')} (id={p.get('page_id', '?')})" for p in pages]
            return f"{len(lines)} pages:\n" + "\n".join(lines)
        except Exception as exc:
            return f"Failed to list pages: {exc}"
    return await asyncio.to_thread(_run)


@function_tool
async def fetch_confluence_page(context: RunContext, page_id: str, heading: str = "") -> str:
    """Fetch a Confluence page's content, optionally scoped to a section heading.

    Args:
        page_id: The Confluence page ID to fetch.
        heading: Optional section heading to isolate (returns only that section).
    """
    def _run():
        try:
            from confluence_logic.utils.html_parser import extract_headings, get_section_html
            connector = _confluence_connector()
            html = connector.fetch_page_html(page_id)
            headings = extract_headings(html)
            if heading:
                section = get_section_html(html, heading)
                return (
                    f"Section '{heading}':\n{section or '(section not found)'}\n\n"
                    f"Available headings: {', '.join(headings)}"
                )
            return f"Available headings: {', '.join(headings) or '(none)'}\n\nContent (first 2000 chars):\n{html[:2000]}"
        except Exception as exc:
            return f"Failed to fetch page {page_id}: {exc}"
    return await asyncio.to_thread(_run)


@function_tool
async def create_confluence_page(
    context: RunContext,
    title: str,
    space_key: str,
    body_text: str = "",
    parent_page_id: str = "",
) -> str:
    """Create a new Confluence page with the given title and content.

    Args:
        title: The page title.
        space_key: The Confluence space key (e.g. 'ENG', 'DOCS').
        body_text: Plain text or markdown body content.
        parent_page_id: Optional parent page ID to nest the new page under.
    """
    def _run():
        try:
            from confluence_logic.utils.html_builder import build_page_html
            connector = _confluence_connector()
            html = build_page_html(title=title, body_text=body_text)
            result = connector.create_page(
                space_key=space_key or None,
                title=title,
                content=html,
                parent_page_id=parent_page_id or None,
            )
            page_id = result.get("id")
            if page_id:
                return f"Page '{title}' created. Page ID: {page_id}"
            return "Page creation returned no ID — check Confluence."
        except Exception as exc:
            return f"Failed to create page: {exc}"
    return await asyncio.to_thread(_run)


@function_tool
async def edit_confluence_section(
    context: RunContext,
    page_id: str,
    heading: str,
    new_content: str,
    append: bool = False,
) -> str:
    """Edit or append to a section of a Confluence page.

    Args:
        page_id: The Confluence page ID to edit.
        heading: The section heading to target.
        new_content: Markdown content to place in the section.
        append: If True, appends new_content instead of replacing the section.
    """
    def _run():
        try:
            from confluence_logic.utils.html_parser import get_section_html, edit_block_in_section
            from confluence_logic.utils.html_builder import markdown_to_html
            connector = _confluence_connector()
            html = connector.fetch_page_html(page_id)
            meta = connector.get_page_metadata(page_id)
            version = meta.get("version", {}).get("number", 1)
            new_html = markdown_to_html(new_content)
            current = get_section_html(html, heading)
            if append:
                merged = (current.rstrip() + "\n" + new_html) if current.strip() else new_html
                updated = edit_block_in_section(html, heading, current, merged)
            else:
                updated = edit_block_in_section(html, heading, current, new_html)
            success = connector.push_update(page_id, updated, expected_version=version)
            if success:
                return f"Section '{heading}' on page {page_id} updated successfully."
            return f"Update push failed for page {page_id}."
        except Exception as exc:
            return f"Failed to edit page {page_id}: {exc}"
    return await asyncio.to_thread(_run)


@function_tool
async def delete_confluence_section(
    context: RunContext,
    page_id: str,
    heading: str,
    delete_entire_section: bool = False,
) -> str:
    """Delete content within a section of a Confluence page.

    Args:
        page_id: The Confluence page ID.
        heading: The section heading to target.
        delete_entire_section: If True, removes the entire section including its heading.
    """
    def _run():
        try:
            from confluence_logic.utils.html_parser import delete_content_in_section
            connector = _confluence_connector()
            html = connector.fetch_page_html(page_id)
            meta = connector.get_page_metadata(page_id)
            version = meta.get("version", {}).get("number", 1)
            updated = delete_content_in_section(html, heading, delete_entire_section=delete_entire_section)
            success = connector.push_update(page_id, updated, expected_version=version)
            if success:
                return f"Section '{heading}' deleted from page {page_id}."
            return f"Delete push failed for page {page_id}."
        except Exception as exc:
            return f"Failed to delete section: {exc}"
    return await asyncio.to_thread(_run)


@function_tool
async def delete_confluence_page(context: RunContext, page_id: str) -> str:
    """Permanently delete an entire Confluence page.

    Args:
        page_id: The Confluence page ID to delete.
    """
    def _run():
        try:
            connector = _confluence_connector()
            success = connector.delete_page(page_id)
            if success:
                return f"Page {page_id} permanently deleted."
            return f"Delete returned unexpected status for page {page_id}."
        except Exception as exc:
            return f"Failed to delete page {page_id}: {exc}"
    return await asyncio.to_thread(_run)


# ---------------------------------------------------------------------------
# Advanced Confluence tools — rich wrappers using Graph RAG, Pinecone, and
# version-conflict retry logic.  These replace / complement the simpler tools
# above and expose the full capability of the knowledge pipeline to the
# LiveKit voice agent.
# ---------------------------------------------------------------------------

def _retry_connector():
    from confluence_logic.connectors.confluence import ConfluenceConnector
    return ConfluenceConnector()


def _retry_store():
    from confluence_logic.db.vector_store import PineconeStore
    return PineconeStore()


async def _commit_with_retry_async(
    page_id: str,
    apply_fn,
    expected_version: int,
    title_override: str | None = None,
):
    """Fetch-transform-push with 3-attempt version-conflict backoff retry."""
    _max = 3

    def _run():
        nonlocal expected_version
        connector = _retry_connector()
        for attempt in range(_max):
            try:
                live_html = connector.fetch_page_html(page_id)
                new_html = apply_fn(live_html)
                ok = connector.push_update(
                    page_id, new_html,
                    expected_version=expected_version,
                    title_override=title_override,
                )
                return ok, (expected_version + 1) if ok else None
            except ValueError as ve:
                if "Version Conflict" not in str(ve) or attempt == _max - 1:
                    raise
                import time
                meta = connector.get_page_metadata(page_id)
                expected_version = meta.get("version", {}).get("number", expected_version)
                time.sleep(0.5 * (2 ** attempt))
        return False, None

    return await asyncio.to_thread(_run)


@function_tool
async def search_workspace_knowledge_tool(context: RunContext, query: str) -> str:
    """Search all Confluence pages using Pinecone vector search and live Confluence search.

    More powerful than search_confluence_pages — use this for complex knowledge lookups
    that benefit from semantic similarity matching across all indexed content.

    Args:
        query: Topic, keyword, or natural-language description of what to find.
    """
    def _run():
        import concurrent.futures, difflib

        connector = _retry_connector()

        def _live():
            try:
                return connector.search_pages(query, limit=8)
            except Exception as exc:
                logger.warning("Live Confluence search failed: %s", exc)
                return []

        def _vec():
            try:
                return _retry_store().search(query, top_k=5)
            except Exception as exc:
                logger.warning("Pinecone search unavailable: %s", exc)
                return []

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
            live_results = ex.submit(_live).result()
            pinecone_results = ex.submit(_vec).result()

        def _sim(title: str) -> float:
            return difflib.SequenceMatcher(None, query.lower(), title.lower()).ratio()

        seen: dict[str, tuple[dict, float]] = {}
        for item in live_results:
            pid = item.get("page_id", "")
            if pid:
                seen[pid] = (item, _sim(item.get("title", "")))
        for match in pinecone_results:
            meta = match.get("metadata", {})
            pid = meta.get("page_id", "")
            if pid and pid not in seen:
                seen[pid] = (meta, _sim(meta.get("title", "")))

        if not seen:
            return "No Confluence pages found for that query."

        ranked = sorted(seen.values(), key=lambda t: -t[1])[:5]
        lines = [f"- {v.get('title', 'Untitled')} (id={pid})"
                 for (v, _), pid in zip(ranked, seen)]
        return "Found pages:\n" + "\n".join(lines)

    return await asyncio.to_thread(_run)


@function_tool
async def fetch_live_page_tool(context: RunContext, page_id: str, heading: str = "") -> str:
    """Fetch a Confluence page with its current version number.

    Always call this before commit_document_edit_tool or commit_delete_tool to
    get the correct expected_version for the page you are about to modify.

    Args:
        page_id: Confluence page ID.
        heading: Optional section heading to fetch content for.
    """
    def _run():
        try:
            from confluence_logic.utils.html_parser import extract_headings, get_section_html
            connector = _retry_connector()
            html = connector.fetch_page_html(page_id)
            meta = connector.get_page_metadata(page_id)
            version = meta.get("version", {}).get("number", 1)
            headings = extract_headings(html)
            result = f"Page {page_id} | version={version}\nHeadings: {', '.join(headings) or '(none)'}"
            if heading:
                section = get_section_html(html, heading)
                result += f"\n\nSection '{heading}':\n{section[:2000] if section else '(section not found)'}"
            return result
        except Exception as exc:
            return f"Failed to fetch page {page_id}: {exc}"

    return await asyncio.to_thread(_run)


@function_tool
async def commit_document_edit_tool(
    context: RunContext,
    page_id: str,
    expected_version: int,
    heading: str,
    new_content: str,
    append: bool = False,
) -> str:
    """Edit a section of a Confluence page with automatic version-conflict retry.

    First call fetch_live_page_tool to get the expected_version for the page.

    Args:
        page_id: Confluence page ID.
        expected_version: Current page version from fetch_live_page_tool.
        heading: Section heading to edit.
        new_content: New markdown content for the section.
        append: If True, appends new_content instead of replacing the section.
    """
    try:
        from confluence_logic.utils.html_parser import get_section_html, edit_block_in_section
        from confluence_logic.utils.html_builder import markdown_to_html
        new_html = markdown_to_html(new_content) if new_content else ""

        if append:
            def apply_fn(live_html: str) -> str:
                cur = get_section_html(live_html, heading)
                merged = (cur.rstrip() + "\n" + new_html) if cur.strip() else new_html
                return edit_block_in_section(live_html, heading, cur, merged)
        else:
            def apply_fn(live_html: str) -> str:
                old = get_section_html(live_html, heading)
                return edit_block_in_section(live_html, heading, old, new_html)

        ok, _ = await _commit_with_retry_async(page_id, apply_fn, expected_version)
        if ok:
            return f"Section '{heading}' on page {page_id} updated successfully."
        return f"Failed to update section '{heading}' on page {page_id}."
    except Exception as exc:
        return f"Edit failed: {exc}"


@function_tool
async def commit_delete_tool(
    context: RunContext,
    page_id: str,
    expected_version: int,
    heading: str,
    delete_entire_section: bool = False,
) -> str:
    """Delete content in a Confluence page section with automatic version-conflict retry.

    First call fetch_live_page_tool to get the expected_version for the page.

    Args:
        page_id: Confluence page ID.
        expected_version: Current page version from fetch_live_page_tool.
        heading: Section heading to target.
        delete_entire_section: If True, removes the entire section including its heading.
    """
    try:
        from confluence_logic.utils.html_parser import delete_content_in_section

        def apply_fn(live_html: str) -> str:
            return delete_content_in_section(
                live_html, heading,
                delete_entire_section=delete_entire_section,
            )

        ok, _ = await _commit_with_retry_async(page_id, apply_fn, expected_version)
        if ok:
            return f"Section '{heading}' deleted from page {page_id}."
        return f"Failed to delete section '{heading}' from page {page_id}."
    except Exception as exc:
        return f"Delete failed: {exc}"


@function_tool
async def update_page_title_tool(
    context: RunContext,
    page_id: str,
    expected_version: int,
    new_title: str,
) -> str:
    """Rename a Confluence page.

    First call fetch_live_page_tool to get the expected_version for the page.

    Args:
        page_id: Confluence page ID.
        expected_version: Current page version from fetch_live_page_tool.
        new_title: New title for the page.
    """
    try:
        ok, _ = await _commit_with_retry_async(
            page_id, lambda html: html, expected_version, title_override=new_title,
        )
        if ok:
            return f"Page {page_id} renamed to '{new_title}'."
        return f"Failed to rename page {page_id}."
    except Exception as exc:
        return f"Title update failed: {exc}"


# Export — JarvisAgent in agent_worker.py will pass this list to Agent(tools=JARVIS_TOOLS, ...)
JARVIS_TOOLS = [
    # General utility
    get_current_datetime,
    # Meeting intelligence tools
    summarize_meeting_tool,
    generate_opinion_tool,
    extract_action_items_tool,
    summarize_speaker_tool,
    answer_general_question_tool,
    # Confluence page tools — basic
    search_confluence_pages,
    list_confluence_pages,
    fetch_confluence_page,
    create_confluence_page,
    edit_confluence_section,
    delete_confluence_section,
    delete_confluence_page,
    # Confluence page tools — advanced (Graph RAG + Pinecone + version-retry)
    search_workspace_knowledge_tool,
    fetch_live_page_tool,
    commit_document_edit_tool,
    commit_delete_tool,
    update_page_title_tool,
]


def build_jarvis_tools() -> list:
    """Convenience factory in case Plan 04 needs to rebuild the list with extra tools."""
    return list(JARVIS_TOOLS)
