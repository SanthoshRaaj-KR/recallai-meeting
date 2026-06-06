"""VerifierAgent — verifies the quality of a proposed document edit.

Uses the OpenAI Agents SDK (agents.Agent + agents.Runner) with structured output
to return a VerifierResult with three dimension scores and a computed quality_score.

quality_score formula: factual_consistency * 0.4 + formatting_integrity * 0.2 + intent_fulfillment * 0.4
"""

from __future__ import annotations

import logging
import os

from agents import Agent, Runner
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

VERIFIER_INSTRUCTIONS = """\
You verify the quality of a proposed document edit.

Evaluate three dimensions:

factual_consistency (0.0-1.0):
  Does the after_content introduce any factually incorrect statements?
  1.0 = perfectly consistent; 0.0 = introduces false information.

formatting_integrity (0.0-1.0):
  Is the after_content formatted consistently with the before_content (same markdown style, spacing, list style)?
  1.0 = format matches exactly; 0.0 = formatting severely broken.

intent_fulfillment (0.0-1.0):
  Does the after_content fully implement what the intent description requires?
  1.0 = fully implemented; 0.0 = change not applied at all.

verifier_note: a single sentence summarising the verification result.
"""


class _VerifierRaw(BaseModel):
    factual_consistency: float = Field(ge=0.0, le=1.0)
    formatting_integrity: float = Field(ge=0.0, le=1.0)
    intent_fulfillment: float = Field(ge=0.0, le=1.0)
    verifier_note: str


class VerifierResult(BaseModel):
    """Result of a document edit verification, including computed quality score."""

    factual_consistency: float
    formatting_integrity: float
    intent_fulfillment: float
    quality_score: float
    verifier_note: str


class VerifierAgent:
    """Verifies the quality of a proposed before/after document edit."""

    def __init__(self, model: str = None):
        self.model = model or os.getenv("LDOC_VERIFIER_MODEL", "gpt-4o-mini")
        self._agent = Agent(
            name="LocalDocVerifier",
            model=self.model,
            instructions=VERIFIER_INSTRUCTIONS,
            output_type=_VerifierRaw,
        )

    async def verify(
        self,
        before_content: str,
        after_content: str,
        intent_description: str,
    ) -> VerifierResult:
        """Verify quality of a proposed document edit.

        Returns a VerifierResult with scores in [0,1] and a computed quality_score.
        On exception, returns a zero-score result with a failure note.
        """
        prompt = (
            f"Intent description:\n{intent_description}\n\n"
            f"Before:\n{before_content}\n\n"
            f"After:\n{after_content}"
        )
        try:
            result = await Runner.run(self._agent, prompt)
            raw: _VerifierRaw = result.final_output
            quality_score = (
                raw.factual_consistency * 0.4
                + raw.formatting_integrity * 0.2
                + raw.intent_fulfillment * 0.4
            )
            return VerifierResult(
                factual_consistency=raw.factual_consistency,
                formatting_integrity=raw.formatting_integrity,
                intent_fulfillment=raw.intent_fulfillment,
                quality_score=quality_score,
                verifier_note=raw.verifier_note,
            )
        except Exception:
            logger.error("VerifierAgent.verify() failed", exc_info=True)
            return VerifierResult(
                factual_consistency=0.0,
                formatting_integrity=0.0,
                intent_fulfillment=0.0,
                quality_score=0.0,
                verifier_note="Verification failed",
            )
