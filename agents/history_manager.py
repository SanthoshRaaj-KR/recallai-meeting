"""
HistoryManagerAgent — reads the JSON meeting index, selects the best matching
meeting for a user's question via an LLM call, handles ambiguous matches via
a disambiguation flow, loads the .md file, and calls AnswerAgent.

Design notes:
- HistoryManagerAgent is a plain Python class (NOT an openai-agents Agent()
  instance) — consistent with OrchestratorAgent and RetrieverAgent pattern.
- _select_meeting() makes a single GPT call to identify the best matching
  meeting(s) from the JSON index metadata.
- Disambiguation reuses the same OrchestratorResult format as OrchestratorAgent
  (lines 224-247 of agents/orchestrator.py) — same field names, same shape.
- When no index match is found and a RetrieverAgent is injected, falls back
  to Pinecone semantic retrieval (D-09).
- ImportError for RetrieverAgent avoided by importing inside the method body
  to prevent circular imports.
"""

import datetime
import json
import logging
from typing import Optional, TYPE_CHECKING

from openai import AsyncOpenAI

from agents.answer_agent import AnswerAgent
from agents.date_resolver import DateResolutionAgent
from agents.meeting_writer import MeetingWriterAgent
from agents.retriever import RetrievalResult
from storage.models import MeetingIndexEntry

if TYPE_CHECKING:
    from agents.retriever import RetrieverAgent

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Date keyword pre-filter (mirrors OrchestratorAgent._extract_date_expression)
# ---------------------------------------------------------------------------

_DATE_KEYWORDS = frozenset([
    "last", "this", "next", "yesterday", "today", "tomorrow",
    "monday", "tuesday", "wednesday", "thursday", "friday",
    "saturday", "sunday", "week", "month", "ago", "before",
    "morning", "afternoon", "evening",
])


def _extract_date_expression(query: str) -> Optional[str]:
    """Return a date sub-expression from the query if one is found, else None."""
    words = query.lower().split()
    for i, word in enumerate(words):
        if word.strip(".,?!") in _DATE_KEYWORDS:
            original = query.split()
            start = max(0, i - 1)
            end = min(len(original), i + 2)
            return " ".join(original[start:end])
    return None


# ---------------------------------------------------------------------------
# Meeting selection prompt
# ---------------------------------------------------------------------------

_SELECTION_PROMPT = """You are a meeting index selector. The user asked a question about a past meeting.
You are given a JSON list of meeting index entries. Each entry has:
  meeting_id, title, date (YYYY-MM-DD), channel_name, overview, participants,
  goals, key_decisions, conclusions.

Use overview, goals, key_decisions, and conclusions to judge relevance — they capture
what the meeting was about. Use date and channel_name to narrow by time and context.

Your task: identify which meeting(s) best match the user's question.

Rules:
- If exactly one meeting clearly matches → return its meeting_id in the "selected" field.
- If two or more meetings are equally plausible → return all their meeting_ids in "candidates".
- If no meeting is relevant → return an empty "selected" and empty "candidates".
- You are provided the Current Date/Time. Use it to accurately resolve relative date references ("last Thursday", "yesterday") against the entry dates.
- Do not make up meetings. Only reference entries in the provided list.

Respond with JSON only:
{"selected": "<meeting_id or empty string>", "candidates": ["<id>", ...]}"""


# ---------------------------------------------------------------------------
# HistoryManagerAgent
# ---------------------------------------------------------------------------

class HistoryManagerAgent:
    """Selects the best matching past meeting and delegates to AnswerAgent.

    Reads the JSON meeting index, uses an LLM call to select the best
    matching meeting for the user's question, handles disambiguation when
    multiple meetings are equally relevant, loads the .md file for the
    selected meeting, and calls AnswerAgent with the full content.

    Optionally falls back to Pinecone semantic retrieval (via RetrieverAgent)
    when no index match is found and a retriever is injected (D-09).

    Usage:
        agent = HistoryManagerAgent(
            meeting_writer=writer,
            answer_agent=answer_agent,
            retriever=retriever_agent,  # optional
        )
        result = await agent.run(query="what did we decide?", user_id="U1", channel_id="C1")
    """

    def __init__(
        self,
        meeting_writer: MeetingWriterAgent,
        answer_agent: AnswerAgent,
        retriever: Optional["RetrieverAgent"] = None,
        model: str = "gpt-4o-mini",
    ):
        """
        Args:
            meeting_writer: MeetingWriterAgent instance for index and .md file reads.
            answer_agent: AnswerAgent instance for answer synthesis.
            retriever: Optional RetrieverAgent for Pinecone fallback (D-09).
            model: OpenAI model for _select_meeting(). Defaults to "gpt-4o-mini".
        """
        self._writer = meeting_writer
        self._answer = answer_agent
        self._retriever = retriever
        self._model = model
        self._openai = AsyncOpenAI()
        self._date_resolver = DateResolutionAgent()

    async def _select_meeting(
        self,
        query: str,
        entries: list[MeetingIndexEntry],
    ) -> dict:
        """Use an LLM call to identify which meeting(s) best match the query.

        Serializes entries to JSON, calls the LLM with the selection prompt,
        and parses the JSON response.

        Args:
            query: User's natural language question.
            entries: List of MeetingIndexEntry objects from the index.

        Returns:
            Dict with keys:
                "selected" (str): meeting_id of best match, or "" if none/ambiguous.
                "candidates" (list[str]): meeting_ids when multiple matches found.
        """
        index_json = json.dumps([e.model_dump() for e in entries])
        try:
            response = await self._openai.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": _SELECTION_PROMPT},
                    {
                        "role": "user",
                        "content": f"Current Date/Time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\nQuestion: {query}\n\nMeeting index:\n{index_json}",
                    },
                ],
                max_tokens=200,
                temperature=0,
                response_format={"type": "json_object"},
            )
            content = response.choices[0].message.content or ""
            parsed = json.loads(content)
            return {
                "selected": parsed.get("selected", ""),
                "candidates": parsed.get("candidates", []),
            }
        except Exception:
            logger.warning("HistoryManager: _select_meeting parse error — returning safe fallback")
            return {"selected": "", "candidates": []}

    def _md_to_retrieval_result(
        self,
        query: str,
        entry: MeetingIndexEntry,
        md_content: str,
    ) -> RetrievalResult:
        """Wrap a meeting .md file into a synthetic RetrievalResult for AnswerAgent.

        All four Pydantic-required fields must be populated to avoid ValidationError.

        Args:
            query: The user's original query string.
            entry: MeetingIndexEntry for the selected meeting.
            md_content: Full text content of the meeting's .md file.

        Returns:
            RetrievalResult with a single result item containing the .md content.
        """
        return RetrievalResult(
            query=query,
            results=[
                {
                    "id": entry.meeting_id,
                    "score": 1.0,
                    "metadata": {
                        "channel_name": entry.channel_name,
                        "start_ts": entry.start_ts,
                        "full_transcript": md_content,  # labelled correctly for AnswerAgent
                        "decisions": [],
                        "topics_covered": [],
                        "participants": entry.participants,
                        "action_items": [],
                    },
                }
            ],
            total_candidates=1,
            returned_count=1,
        )

    async def run(
        self,
        query: str,
        user_id: str,
        channel_id: str,
    ):
        """Select a meeting and synthesize an answer, or trigger disambiguation.

        Flow:
        1. Read meeting index via MeetingWriterAgent.read_index().
        2. If index is empty: return "no history" result.
        3. Call _select_meeting() to identify best matching meeting(s).
        4. Single match: read .md file, call AnswerAgent, return OrchestratorResult.
        5. Multiple candidates: return disambiguation OrchestratorResult.
        6. No match + retriever set: fall back to Pinecone via RetrieverAgent.
        7. No match + no retriever: return graceful "no match" result.

        Args:
            query: User's natural language question.
            user_id: Slack user ID (reserved for future personalization).
            channel_id: Slack channel ID (used by Pinecone fallback scope).

        Returns:
            OrchestratorResult with answer, source_meeting_ids, confidence,
            and optional disambiguation options.
        """
        # Import here to avoid circular dependency at module level
        from agents.orchestrator import OrchestratorResult

        all_entries = await self._writer.read_index()

        # Empty index: no history available
        if not all_entries:
            return OrchestratorResult(
                query=query,
                query_type="memory_query",
                answer="No meeting history found. No meetings have been recorded yet.",
                source_meeting_ids=[],
                confidence="low",
            )

        # --- Pre-filter: narrow candidates before the LLM call ---
        # This avoids sending the full index to the LLM when simple filters
        # can eliminate clearly irrelevant meetings.

        entries = all_entries

        # 1. Channel pre-filter: scope to the current channel when provided
        if channel_id:
            channel_filtered = [e for e in entries if e.channel_id == channel_id]
            if channel_filtered:
                entries = channel_filtered

        # 2. Date pre-filter: if the query mentions a date, narrow to a ±1-day window
        date_expr = _extract_date_expression(query)
        if date_expr:
            try:
                dr = self._date_resolver.resolve(date_expr)
                # Expand window by one day each side to handle timezone edge cases
                window_start = dr.start_ts - 86400
                window_end = dr.end_ts + 86400
                date_filtered = [
                    e for e in entries
                    if window_start <= e.start_ts <= window_end
                ]
                if date_filtered:
                    entries = date_filtered
            except ValueError:
                pass  # unparseable date expression — proceed with current set

        # 3. If pre-filtering uniquely identifies one meeting, skip the LLM entirely.
        #    Only skip when filters *actually narrowed the list* — if we started with
        #    1 entry and no filter applied, still call LLM to confirm relevance
        #    (the single meeting may not be relevant to the query at all).
        pre_filter_applied = len(entries) < len(all_entries)
        if pre_filter_applied and len(entries) == 1:
            logger.debug(
                "HistoryManager: pre-filter resolved single match %r — skipping LLM",
                entries[0].meeting_id,
            )
            entry = entries[0]
            md_content = await self._writer.read_md(entry.md_path)
            rr = self._md_to_retrieval_result(query, entry, md_content)
            answer_out = await self._answer.run(
                query=query,
                retrieval_result=rr,
                query_type="memory_query",
            )
            return OrchestratorResult(
                query=query,
                query_type="memory_query",
                answer=answer_out.answer,
                source_meeting_ids=[entry.meeting_id],
                confidence=answer_out.confidence,
            )

        # Use LLM to select from the (pre-filtered) candidate list
        selection = await self._select_meeting(query, entries)

        # Single clear match
        if selection["selected"]:
            entry = next(
                (e for e in entries if e.meeting_id == selection["selected"]),
                None,
            )
            if entry is None:
                # LLM hallucinated a meeting_id that doesn't exist in the index
                logger.warning(
                    "HistoryManager: LLM returned unknown meeting_id %r",
                    selection["selected"],
                )
                return OrchestratorResult(
                    query=query,
                    query_type="memory_query",
                    answer="I couldn't find a relevant meeting in the history. Try Pinecone search with /ask.",
                    source_meeting_ids=[],
                    confidence="low",
                )

            md_content = await self._writer.read_md(entry.md_path)
            rr = self._md_to_retrieval_result(query, entry, md_content)
            answer_out = await self._answer.run(
                query=query,
                retrieval_result=rr,
                query_type="memory_query",
            )
            return OrchestratorResult(
                query=query,
                query_type="memory_query",
                answer=answer_out.answer,
                source_meeting_ids=[entry.meeting_id],
                confidence=answer_out.confidence,
            )

        # Multiple candidates: disambiguation required (D-08)
        candidates = selection["candidates"]
        if len(candidates) >= 2:
            candidate_entries = [e for e in entries if e.meeting_id in candidates]
            options = [
                {
                    "index": i + 1,
                    "meeting_id": e.meeting_id,
                    "title": e.title,
                    "channel": e.channel_name,
                    "date": e.date,
                }
                for i, e in enumerate(candidate_entries)
            ]
            return OrchestratorResult(
                query=query,
                query_type="memory_query",
                answer="Multiple meetings matched. Please select one:",
                source_meeting_ids=[e.meeting_id for e in candidate_entries],
                confidence="low",
                needs_disambiguation=True,
                disambiguation_options=options,
            )

        # No match — attempt Pinecone fallback if retriever is available (D-09)
        if self._retriever is not None:
            logger.info("HistoryManager: no index match, falling back to Pinecone retrieval")
            retrieval_result = await self._retriever.retrieve(
                query_text=query,
                channel_id=channel_id or None,
            )
            if retrieval_result.results:
                answer_out = await self._answer.run(
                    query=query,
                    retrieval_result=retrieval_result,
                    query_type="memory_query",
                )
                return OrchestratorResult(
                    query=query,
                    query_type="memory_query",
                    answer=answer_out.answer,
                    source_meeting_ids=answer_out.source_meeting_ids,
                    confidence=answer_out.confidence,
                )

        # No match and no fallback results
        return OrchestratorResult(
            query=query,
            query_type="memory_query",
            answer="I couldn't find a relevant meeting in the history. Try Pinecone search with /ask.",
            source_meeting_ids=[],
            confidence="low",
        )
