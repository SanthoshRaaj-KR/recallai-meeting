"""IntentExtractionAgent — extracts actionable document-change intents from meeting transcripts.

Uses the OpenAI Agents SDK (agents.Agent + agents.Runner) with structured output
to return a list of LocalDocIntent objects. Filters out low-confidence intents
(confidence < 0.5) before returning.
"""

from __future__ import annotations

import logging
import os

from agents import Agent, AgentOutputSchema, Runner
from pydantic import BaseModel

from models import LocalDocIntent

logger = logging.getLogger(__name__)

INSTRUCTIONS = """\
You extract actionable document change intents from meeting transcripts.
A "document change intent" is a statement that implies a written company document (policy, process guide, handbook, onboarding doc, compliance doc) needs to be updated.

Intent types:
- policy_update: a policy rule, limit, or requirement is changing
- process_change: a workflow, procedure, or step is changing
- ownership_change: a role, owner, or responsible party is changing
- compliance_update: a regulatory or audit requirement is changing
- decision: a meeting decision that should be recorded in a document
- technology_migration: a tool, system, or platform change
- onboarding_update: a change to onboarding or training material
- other: any other document-worthy change

Completeness is critical — capture EVERY distinct change, not just the obvious ones:
- Extract one separate intent per distinct change. Never merge two unrelated changes into one intent.
- Removals / deletions count ("remove the last 4 security points", "no benefits anymore", "drop the carryover clause"). For a removal, set new_value to a short phrase describing what is removed (e.g. "remove the last 4 security sections"). affected_topic MUST name the specific document or subject area the removal applies to — carry forward the document or topic named earlier in the same discussion point. For example, if the speaker is discussing the support SOP and then says "trim the last two sections", affected_topic is "support SOP sections", NOT just "sections". When the removal sentence itself does not name the document, also include the earlier sentence that names it in verbatim_snippets.
- Renames / rebrands count ("we are renaming the company to X", "the team is now called Y"). For a rename, old_value is the current name if stated (else null) and new_value is the new name; affected_topic should say what is being renamed (e.g. "company name").
- Numeric and policy changes count even when phrased loosely ("only 3 days a month", "increased by 3 days", "extend retention to a year").
- Product / strategy pivots count ("we're switching the product to X", "web app first, then mobile").
- Ignore filler, hesitations, and side chatter that imply no document change.

Rules:
1. Only extract intents where a specific document update is clearly implied.
2. verbatim_snippets MUST contain exact quoted text from the transcript.
3. confidence: 0.9+ only when the transcript is unambiguous; 0.5-0.89 for probable; < 0.5 for speculative.
4. old_value: the current state BEFORE the change (null if unknown). Never guess a value the transcript does not give.
5. new_value: the intended new state AFTER the change (for a relative change like "increased by 3 days", describe the delta, e.g. "current sick days + 3").
6. If no actionable document change intents are found, return {"intents": []}.
"""


class _IntentList(BaseModel):
    intents: list[LocalDocIntent]


class IntentExtractionAgent:
    """Extracts document-change intents from a meeting transcript using GPT."""

    def __init__(self, model: str = None):
        self.model = model or os.getenv("LDOC_INTENT_MODEL", "gpt-4o-mini")
        # Use strict_json_schema=False because LocalDocIntent.metadata is an
        # untyped dict, which generates additionalProperties=True in JSON schema —
        # incompatible with the Agents SDK strict schema mode. (Rule 1 auto-fix)
        self._agent = Agent(
            name="LocalDocIntentExtractor",
            model=self.model,
            instructions=INSTRUCTIONS,
            output_type=AgentOutputSchema(_IntentList, strict_json_schema=False),
        )

    async def extract(self, transcript: str) -> list[LocalDocIntent]:
        """Extract document-change intents from a meeting transcript.

        Returns an empty list immediately for empty transcripts.
        Filters out intents with confidence < 0.5.
        """
        if not transcript.strip():
            return []
        try:
            result = await Runner.run(
                self._agent,
                f"Meeting transcript:\n\n{transcript}",
            )
            intent_list: _IntentList = result.final_output
            return [i for i in intent_list.intents if i.confidence >= 0.5]
        except Exception:
            logger.error("IntentExtractionAgent.extract() failed", exc_info=True)
            return []
