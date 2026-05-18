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
import time
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
# Transcript-log access (Plan 05 swaps the body for IPC; today it's in-process)
# ---------------------------------------------------------------------------
def get_transcript_log_for_session(session_id: str) -> List[Dict[str, Any]]:
    """Return the transcript log for a given session_id, or [] if unknown.

    Performs a LAZY import of jarvis_agentic to avoid a load-time circular import
    (jarvis_agentic.py will later import THIS module for IPC in Plan 05).
    """
    try:
        from confluence_logic.jarvis_agentic import _meeting_sessions  # noqa: PLC0415
    except Exception as exc:  # pragma: no cover — import failure is non-fatal
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
    transcript = get_transcript_log_for_session(sid)
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
    transcript = get_transcript_log_for_session(sid)
    t0 = time.perf_counter()
    result = await generate_opinion(transcript, query=query)
    logger.info("⏱️  generate_opinion_tool: %.0fms (query=%.40r)", (time.perf_counter() - t0) * 1000, query)
    return result


@function_tool
async def extract_action_items_tool(context: RunContext) -> str:
    """Extract action items, commitments, and next steps from the meeting transcript."""
    sid = _session_id_from_context(context)
    transcript = get_transcript_log_for_session(sid)
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
    transcript = get_transcript_log_for_session(sid)
    t0 = time.perf_counter()
    result = await summarize_speaker(transcript, speaker_name=speaker_name)
    logger.info("⏱️  summarize_speaker_tool: %.0fms (speaker=%.30r)", (time.perf_counter() - t0) * 1000, speaker_name)
    return result


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


# Export — JarvisAgent in agent_worker.py will pass this list to Agent(tools=JARVIS_TOOLS, ...)
JARVIS_TOOLS = [
    # Meeting intelligence tools
    summarize_meeting_tool,
    generate_opinion_tool,
    extract_action_items_tool,
    summarize_speaker_tool,
    answer_general_question_tool,
    # Confluence page tools
    search_confluence_pages,
    list_confluence_pages,
    fetch_confluence_page,
    create_confluence_page,
    edit_confluence_section,
    delete_confluence_section,
    delete_confluence_page,
]


def build_jarvis_tools() -> list:
    """Convenience factory in case Plan 04 needs to rebuild the list with extra tools."""
    return list(JARVIS_TOOLS)
