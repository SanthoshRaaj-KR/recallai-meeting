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

from .llm_runtime import guarded_run
from .models import ChunkRecord, ConfluenceIntent

logger = logging.getLogger(__name__)

EVAL_INSTRUCTIONS = """\
You evaluate whether a document section is the right place to apply a proposed change.

Given a change intent and a candidate section from a document, score the relevance from 0.0 to 1.0:
- 0.9-1.0: The section IS the exact place to apply the change. The topic matches precisely.
- 0.7-0.89: The section is likely the right place; minor topic mismatch possible.
- 0.5-0.69: The section is plausibly related but not obviously the right target.
- 0.0-0.49: The section is unrelated to the intent; do not edit here.

Score on TOPIC match, not value match: if this section contains the field/parameter the
intent is about (same label/subject), it is the right place even if the section's current
value differs from the intent's old_value — the speaker may have misremembered the old
value. Do not lower the score just because the current number differs from old_value.

Judge by the section's BODY, not its heading. A section whose text actually states the
thing the intent changes is the right place even when its heading is generic or unrelated
(e.g. an intent about "metrics reported weekly" should score 0.85+ on a section whose body
says "Metrics for this area are reported every two weeks", even if that section is headed
"Postmortems" or "Overview"). A specific, on-topic heading is a bonus, not a requirement;
never penalise a section that contains the exact sentence the change targets.

DOCUMENT MATCH IS DECISIVE for disambiguation. If the speaker named a specific document,
page, company, or organization (see "Spoken context"/topic) AND the candidate's "Document"
CLEARLY belongs to a DIFFERENT company/organization, score 0.0-0.3 even if the topic,
heading, and values match perfectly — editing the right kind of section in the WRONG
document is exactly the failure to avoid (many documents share the same headings and
similar tables). BUT do NOT penalize when the document matches the named one, or when you
cannot confidently tell that it is a different document — in those cases score on topic
relevance as usual. Only a CLEAR wrong-document match is penalized; uncertainty is not.

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
            name="ConfluenceEvaluator",
            model=self.model,
            instructions=EVAL_INSTRUCTIONS,
            output_type=_EvalResult,
        )

    async def score(self, intent: ConfluenceIntent, chunk: ChunkRecord) -> float:
        """Score how relevant a document section is for a change intent.

        Returns a float in [0.0, 1.0]. Returns 0.0 on any exception.
        """
        # Show enough of the section that the relevant sentence is visible. The
        # old 800-char cap dropped the change target on long sections (a value to
        # edit often sits in a middle paragraph past 800 chars), making the
        # evaluator score the CORRECT section 0.0. Sections are bounded by the
        # chunker (~800 words), so ~4000 chars covers essentially all of them.
        content = chunk.content
        if len(content) > 4000:
            # Keep the head plus a window around any old_value match so the
            # decisive text is never truncated away on very long sections.
            content = content[:4000]
            if intent.old_value and intent.old_value not in content and intent.old_value in chunk.content:
                pos = chunk.content.find(intent.old_value)
                content = chunk.content[:2000] + "\n...\n" + chunk.content[max(0, pos - 500):pos + 500]
        spoken = " ".join(intent.verbatim_snippets or []).strip()
        doc_label = chunk.doc_title or os.path.basename(chunk.source_path)
        prompt = (
            f"Intent:\n"
            f"  topic: {intent.affected_topic}\n"
            f"  old_value: {intent.old_value}\n"
            f"  new_value: {intent.new_value}\n"
            f"  spoken context: {spoken or '(none)'}\n\n"
            f"Candidate section:\n"
            f"  Document (page): {doc_label}\n"
            f"  Section heading: {chunk.section_heading}\n"
            f"  Content:\n{content}"
        )
        try:
            result = await guarded_run(self._agent, prompt)
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
        intent: ConfluenceIntent,
        chunks: list[ChunkRecord],
    ) -> list[float]:
        """Score all chunks concurrently for the given intent.

        Returns a list of floats in the same order as the input chunks.
        """
        return list(
            await asyncio.gather(*[self.score(intent, c) for c in chunks])
        )
