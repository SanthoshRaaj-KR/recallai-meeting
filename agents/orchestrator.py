"""
OrchestratorAgent — the top-level orchestrator for the meeting memory system.

Classifies user queries into live_meeting | memory_query | action_item_query,
optionally resolves date expressions via DateResolutionAgent, delegates retrieval
to RetrieverAgent, checks for disambiguation when multiple meetings match a dated
query, and delegates answer synthesis to AnswerAgent.

Design notes:
- OrchestratorAgent is a plain Python class (NOT an openai-agents Agent() instance).
  This satisfies AGENT-01's spirit while remaining straightforward to test.
- classify() makes a single GPT call using AsyncOpenAI directly.
- All other operations are delegated to injected agents (testable via mocks).
- Disambiguation is triggered ONLY when (1) a date expression was in the query
  AND (2) the retriever returns more than one meeting.
"""

import datetime
from typing import Optional

from openai import AsyncOpenAI
from pydantic import BaseModel

from agents.date_resolver import DateResolutionAgent
from agents.retriever import RetrieverAgent, RetrievalResult
from agents.answer_agent import AnswerAgent


# ---------------------------------------------------------------------------
# Date expression detection
# ---------------------------------------------------------------------------

_DATE_KEYWORDS = frozenset([
    "last", "this", "next", "yesterday", "today", "tomorrow",
    "monday", "tuesday", "wednesday", "thursday", "friday",
    "saturday", "sunday", "week", "month", "ago", "before",
    "morning", "afternoon", "evening",
])


def _extract_date_expression(query: str) -> Optional[str]:
    """Return the date sub-expression if found, None otherwise.

    Strategy: scan query for any word in _DATE_KEYWORDS; if found, return
    a candidate sub-expression by extracting the surrounding 3-word window
    centered on the first matching keyword. This is a heuristic — good enough
    for 'last Monday', 'two weeks ago', 'yesterday afternoon'.

    Returns None if no date keyword found.
    """
    words = query.lower().split()
    for i, word in enumerate(words):
        clean = word.strip(".,?!")
        if clean in _DATE_KEYWORDS:
            # extract up to 3 words: [i-1, i, i+1] (original case)
            original_words = query.split()
            start = max(0, i - 1)
            end = min(len(original_words), i + 2)
            return " ".join(original_words[start:end])
    return None


# ---------------------------------------------------------------------------
# Classification prompt
# ---------------------------------------------------------------------------

_CLASSIFY_PROMPT = """You are a query classifier for a meeting memory system.
Classify the user query into exactly one category:

- "live_meeting": The user is asking about the current/live meeting ("what are we discussing?", "who spoke last?")
- "memory_query": The user is asking about past meetings ("what did we decide about X?", "what happened last Monday?", "summarize last week's standup")
- "action_item_query": The user is asking about their own or others' action items ("what did I commit to?", "my action items", "what tasks are pending?")

Respond with ONLY the category string. No explanation."""


# ---------------------------------------------------------------------------
# OrchestratorResult Pydantic model
# ---------------------------------------------------------------------------

class OrchestratorResult(BaseModel):
    """Structured result from OrchestratorAgent.run().

    Fields:
        query: The original user query.
        query_type: Classification — one of "live_meeting" | "memory_query" | "action_item_query".
        answer: Synthesized answer text (or disambiguation prompt).
        source_meeting_ids: Meeting IDs cited in the answer.
        confidence: Confidence rating — one of "high" | "medium" | "low".
        needs_disambiguation: True when multiple meetings match a dated query.
        disambiguation_options: List of option dicts when needs_disambiguation is True.
            Each dict has: index (int), meeting_id (str), title (str), channel (str), date (str).
    """

    query: str
    query_type: str                          # "live_meeting" | "memory_query" | "action_item_query"
    answer: str
    source_meeting_ids: list[str]
    confidence: str                          # "high" | "medium" | "low"
    needs_disambiguation: bool = False
    disambiguation_options: list[dict] = []  # [{"index": 1, "meeting_id": ..., "title": ..., "channel": ..., "date": ...}]


# ---------------------------------------------------------------------------
# OrchestratorAgent
# ---------------------------------------------------------------------------

class OrchestratorAgent:
    """Top-level orchestrator for the meeting memory answer path.

    Classifies queries, optionally resolves date expressions, retrieves meeting
    context, checks for disambiguation, and synthesizes answers.

    Usage:
        agent = OrchestratorAgent(
            retriever=retriever_agent,
            date_resolver=date_resolver_agent,
            answer_agent=answer_agent,
        )
        result = await agent.run(query="what did we decide?", user_id="U1", channel_id="C1")
    """

    def __init__(
        self,
        retriever: RetrieverAgent,
        date_resolver: DateResolutionAgent,
        answer_agent: AnswerAgent,
        model: str = "gpt-4o-mini",
    ):
        """
        Args:
            retriever: RetrieverAgent instance for hybrid RAG retrieval.
            date_resolver: DateResolutionAgent instance for NL date parsing.
            answer_agent: AnswerAgent instance for answer synthesis.
            model: OpenAI model for classify(). Defaults to "gpt-4o-mini".
        """
        self._retriever = retriever
        self._date_resolver = date_resolver
        self._answer_agent = answer_agent
        self._model = model
        self._openai = AsyncOpenAI()

    async def classify(self, query: str) -> str:
        """Classify query into live_meeting | memory_query | action_item_query.

        Makes a single GPT call with the classification prompt.
        Returns "memory_query" as safe default if response is unexpected.

        Args:
            query: Natural language user query.

        Returns:
            One of "live_meeting", "memory_query", "action_item_query".
        """
        response = await self._openai.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": _CLASSIFY_PROMPT},
                {"role": "user", "content": query},
            ],
            max_tokens=20,
            temperature=0,
        )
        raw = (response.choices[0].message.content or "").strip().lower()
        if raw in ("live_meeting", "memory_query", "action_item_query"):
            return raw
        return "memory_query"  # safe default

    async def run(
        self,
        query: str,
        user_id: str,
        channel_id: str,
    ) -> OrchestratorResult:
        """Full orchestration pipeline: classify → resolve date → retrieve → disambiguate → answer.

        Args:
            query: Natural language user question.
            user_id: Slack user ID (for future personalization).
            channel_id: Slack channel ID to scope retrieval.

        Returns:
            OrchestratorResult with answer, source meeting IDs, confidence,
            and optional disambiguation options.
        """
        query_type = await self.classify(query)

        # Live meeting: static response, no retrieval
        if query_type == "live_meeting":
            return OrchestratorResult(
                query=query,
                query_type=query_type,
                answer="I can only answer questions about past meetings recorded in memory.",
                source_meeting_ids=[],
                confidence="low",
            )

        # Determine query_type for AnswerAgent
        answer_type = "action_items" if query_type == "action_item_query" else "memory_query"

        # Date resolution: extract date expression if present
        date_expr = _extract_date_expression(query)
        has_date = date_expr is not None
        start_ts: Optional[int] = None
        end_ts: Optional[int] = None

        if has_date:
            try:
                date_result = self._date_resolver.resolve(date_expr)
                start_ts = date_result.start_ts
                end_ts = date_result.end_ts
            except ValueError:
                # Unparseable expression — proceed without date filter
                has_date = False

        # Retrieval
        retrieval_result = await self._retriever.retrieve(
            query_text=query,
            channel_id=channel_id if channel_id else None,
            start_ts=start_ts,
            end_ts=end_ts,
        )

        # Disambiguation check: only when a date was provided AND multiple meetings found
        if has_date and len(retrieval_result.results) > 1:
            options = []
            for idx, r in enumerate(retrieval_result.results, start=1):
                meta = r["metadata"]
                date_str = datetime.datetime.utcfromtimestamp(
                    meta.get("start_ts", 0)
                ).strftime("%Y-%m-%d")
                options.append({
                    "index": idx,
                    "meeting_id": r["id"],
                    "title": meta.get("series_name") or meta.get("summary_text", "")[:60],
                    "channel": meta.get("channel_name", "unknown"),
                    "date": date_str,
                })
            return OrchestratorResult(
                query=query,
                query_type=query_type,
                answer="Multiple meetings found. Please select one:",
                source_meeting_ids=[r["id"] for r in retrieval_result.results],
                confidence="low",
                needs_disambiguation=True,
                disambiguation_options=options,
            )

        # Determine better answer_type for memory queries
        if answer_type == "memory_query":
            ql = query.lower()
            if any(w in ql for w in ("decide", "decision", "agreed", "chose")):
                answer_type = "decision"
            elif any(w in ql for w in ("summarize", "summary", "recap", "happened", "went")):
                answer_type = "summary"
            elif any(w in ql for w in ("before", "history", "ever", "come up", "cross", "across")):
                answer_type = "cross_meeting"
            else:
                answer_type = "decision"  # default for ambiguous memory queries

        # Answer synthesis
        answer_output = await self._answer_agent.run(
            query=query,
            retrieval_result=retrieval_result,
            query_type=answer_type,
        )

        return OrchestratorResult(
            query=query,
            query_type=query_type,
            answer=answer_output.answer,
            source_meeting_ids=answer_output.source_meeting_ids,
            confidence=answer_output.confidence,
        )
