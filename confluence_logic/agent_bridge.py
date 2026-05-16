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

import json
import logging
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
    """Extract session_id from RunContext.job.metadata JSON (set by AgentDispatchService — Plan 05)."""
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
    return await summarize_meeting(transcript, detail_level=detail_level)


@function_tool
async def generate_opinion_tool(context: RunContext, query: str = "") -> str:
    """Give Jarvis's opinion or recommendation on what is being discussed.

    Args:
        query: The specific question the user asked (e.g. 'which option is better?').
    """
    sid = _session_id_from_context(context)
    transcript = get_transcript_log_for_session(sid)
    return await generate_opinion(transcript, query=query)


@function_tool
async def extract_action_items_tool(context: RunContext) -> str:
    """Extract action items, commitments, and next steps from the meeting transcript."""
    sid = _session_id_from_context(context)
    transcript = get_transcript_log_for_session(sid)
    return await extract_action_items(transcript)


@function_tool
async def summarize_speaker_tool(context: RunContext, speaker_name: str) -> str:
    """Summarize what a specific participant said in the meeting.

    Args:
        speaker_name: The participant's display name as it appears in the transcript.
    """
    sid = _session_id_from_context(context)
    transcript = get_transcript_log_for_session(sid)
    return await summarize_speaker(transcript, speaker_name=speaker_name)


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
    return await answer_general_question(
        question,
        conversation_history="",
        graph_context="",
        speech_rewrite_enabled=False,
        multiturn_reference=False,
        force_web_search=force_web_search,
    )


# Export — JarvisAgent in agent_worker.py will pass this list to Agent(tools=JARVIS_TOOLS, ...)
JARVIS_TOOLS = [
    summarize_meeting_tool,
    generate_opinion_tool,
    extract_action_items_tool,
    summarize_speaker_tool,
    answer_general_question_tool,
]


def build_jarvis_tools() -> list:
    """Convenience factory in case Plan 04 needs to rebuild the list with extra tools."""
    return list(JARVIS_TOOLS)
