"""EvaluationAgent — scores document section relevance for a given change intent.

Uses the OpenAI Agents SDK (agents.Agent + agents.Runner) with structured output
to return a float relevance score in [0.0, 1.0] for a (intent, chunk) pair.
Supports concurrent batch scoring via asyncio.gather.
"""

from __future__ import annotations

import asyncio
import logging
import os

from agents import Agent, Runner
from pydantic import BaseModel, Field

from models import ChunkRecord, LocalDocIntent

logger = logging.getLogger(__name__)

EVAL_INSTRUCTIONS = """\
You evaluate whether a document section is the right place to apply a proposed change.

Given a change intent and a candidate section from a document, score the relevance from 0.0 to 1.0:
- 0.9-1.0: The section IS the exact place to apply the change. The topic matches precisely.
- 0.7-0.89: The section is likely the right place; minor topic mismatch possible.
- 0.5-0.69: The section is plausibly related but not obviously the right target.
- 0.0-0.49: The section is unrelated to the intent; do not edit here.

Return a JSON object with:
  relevance_score: float (0.0-1.0)
  reasoning: str (one sentence explaining the score)
"""


class _EvalResult(BaseModel):
    relevance_score: float = Field(ge=0.0, le=1.0)
    reasoning: str


class EvaluationAgent:
    """Scores how relevant a document section is for a given change intent."""

    def __init__(self, model: str = None):
        self.model = model or os.getenv("LDOC_EVAL_MODEL", "gpt-4o-mini")
        self._agent = Agent(
            name="LocalDocEvaluator",
            model=self.model,
            instructions=EVAL_INSTRUCTIONS,
            output_type=_EvalResult,
        )

    async def score(self, intent: LocalDocIntent, chunk: ChunkRecord) -> float:
        """Score how relevant a document section is for a change intent.

        Returns a float in [0.0, 1.0]. Returns 0.0 on any exception.
        """
        prompt = (
            f"Intent:\n"
            f"  topic: {intent.affected_topic}\n"
            f"  old_value: {intent.old_value}\n"
            f"  new_value: {intent.new_value}\n\n"
            f"Section heading: {chunk.section_heading}\n"
            f"Section content:\n{chunk.content[:800]}"
        )
        try:
            result = await Runner.run(self._agent, prompt)
            return result.final_output.relevance_score
        except Exception:
            logger.warning(
                "EvaluationAgent.score() failed for topic=%s heading=%s",
                intent.affected_topic,
                chunk.section_heading,
                exc_info=True,
            )
            return 0.0

    async def score_batch(
        self,
        intent: LocalDocIntent,
        chunks: list[ChunkRecord],
    ) -> list[float]:
        """Score all chunks concurrently for the given intent.

        Returns a list of floats in the same order as the input chunks.
        """
        return list(
            await asyncio.gather(*[self.score(intent, c) for c in chunks])
        )
