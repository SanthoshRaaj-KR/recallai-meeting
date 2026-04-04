"""
AnswerAgent — LLM-powered meeting answer synthesizer.

Takes retrieved meeting context (from RetrieverAgent) and synthesizes a
Slack-formatted answer with source attribution using the OpenAI Agents SDK.

Design notes:
- AnswerAgent wraps an openai-agents Agent(output_type=AnswerOutput) for
  structured output — no manual JSON parsing required.
- Runner.run() is awaited; this module must be called from async context.
- Source attribution format: "(Meeting: {channel_name}, {date})" where date
  is derived from the meeting's start_ts Unix timestamp.
- Context block embeds all retrieval result metadata in a structured prompt
  so the LLM can cite sources accurately.
"""

from agents import Agent, Runner
from agents.retriever import RetrievalResult
from pydantic import BaseModel

_INSTRUCTIONS = """You are a meeting memory assistant. Answer questions about past meetings using ONLY
the provided meeting context. Follow these rules exactly:

1. SOURCE ATTRIBUTION: Always end your answer with the source meeting in this exact
   format: "(Meeting: {channel_name}, {date})" where {date} is in YYYY-MM-DD format
   derived from the meeting's start_ts Unix timestamp. If multiple meetings, list
   each on a new line.

2. DECISION queries: Give a precise, direct answer. Quote the decision verbatim
   if available in the context.

3. SUMMARY queries: Provide a structured recap with sections:
   Topics, Decisions, Action Items, Participants.

4. CROSS_MEETING queries: Synthesize patterns across all provided meetings.
   Note when and how the topic evolved across meetings.

5. ACTION_ITEMS queries: List each action item as:
   "• {owner}: {task}"
   Group by meeting if multiple meetings. Include meeting attribution per group.

6. CONFIDENCE: Rate your answer as "high" (exact match found), "medium"
   (partial match), or "low" (inferred or no relevant context found).

7. If no relevant context is provided, say "I found no relevant meeting records
   for that query." and set confidence to "low"."""


class AnswerOutput(BaseModel):
    """Structured output from AnswerAgent.run().

    Fields:
        answer: Slack-formatted prose answer with source attribution in the form
            "(Meeting: {channel_name}, {date})".
        source_meeting_ids: List of meeting IDs cited in the answer.
        confidence: Confidence rating — one of "high", "medium", "low".
    """

    answer: str
    source_meeting_ids: list[str]
    confidence: str


class AnswerAgent:
    """Synthesizes meeting answers from retrieved context using the OpenAI Agents SDK.

    Wraps an Agent(output_type=AnswerOutput) instance so the LLM returns structured
    output directly. Builds a context block from RetrievalResult metadata and passes
    it as a formatted prompt to Runner.run().

    Usage:
        agent = AnswerAgent()
        result = await agent.run(
            query="what did we decide about the API?",
            retrieval_result=retrieval_result,
            query_type="decision",
        )
        print(result.answer)
    """

    def __init__(self, model: str = "gpt-4o-mini"):
        """
        Args:
            model: OpenAI model identifier. Defaults to "gpt-4o-mini".
        """
        self._model = model
        self._agent = Agent(
            name="meeting-answer-agent",
            instructions=_INSTRUCTIONS,
            model=self._model,
            output_type=AnswerOutput,
        )

    async def run(
        self,
        query: str,
        retrieval_result: RetrievalResult,
        query_type: str,
    ) -> AnswerOutput:
        """Synthesize an answer from retrieved meeting context.

        Builds a structured context block from retrieval_result.results, then
        calls Runner.run() with a formatted prompt. The Agent's output_type
        ensures the LLM returns a valid AnswerOutput directly.

        Args:
            query: Original user question.
            retrieval_result: Result from RetrieverAgent.retrieve().
            query_type: One of "decision" | "summary" | "cross_meeting" | "action_items"

        Returns:
            AnswerOutput with answer, source_meeting_ids, confidence.
        """
        if retrieval_result.results:
            context_parts = []
            for item in retrieval_result.results:
                metadata = item.get("metadata", {})
                part = (
                    f"--- Meeting: {metadata.get('channel_name', 'unknown')} "
                    f"| start_ts: {metadata.get('start_ts', 'N/A')} ---\n"
                    f"Summary: {metadata.get('summary_text', 'N/A')}\n"
                    f"Decisions: {metadata.get('decisions', [])}\n"
                    f"Topics: {metadata.get('topics_covered', [])}\n"
                    f"Participants: {metadata.get('participants', [])}\n"
                    f"Action Items: {metadata.get('action_items', [])}"
                )
                context_parts.append(part)
            context_block = "\n\n".join(context_parts)
        else:
            context_block = "No meeting records found."

        prompt = (
            f"Query type: {query_type}\n"
            f"User question: {query}\n\n"
            f"Meeting context:\n{context_block}"
        )

        result = await Runner.run(self._agent, input=prompt)
        return result.final_output
