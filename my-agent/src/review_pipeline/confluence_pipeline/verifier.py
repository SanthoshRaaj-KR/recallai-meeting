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

from .llm_runtime import guarded_run

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

attribution_fit (0.0-1.0):
  Does the transcript actually attribute this value to the subject of THIS section?
  You are given the speaker's exact words and the document/section the edit lands on.
  Every value belongs to some party. A meeting routinely quotes OTHER parties' numbers —
  a competitor's fee, another firm's terms, a customer's headcount, an industry
  benchmark — and those must never be written into this organization's own document.
  1.0 = the quoted words state this value about the subject of this section.
  0.0 = the words state it about a DIFFERENT party while this section states the
        document owner's own equivalent figure.
  Judge ATTRIBUTION ONLY, independently of the other three dimensions. A perfectly
  clean, well-formed, factually coherent edit still scores 0.0 here if the value it
  writes was never said about this section's subject — that combination (immaculate
  diff, wrong subject) is exactly what this dimension exists to catch.
  Where the burden of proof sits depends on the attribution scope you are given:
  - scope "internal", or words plainly about this organization's own affairs: score
    0.8-1.0. This is the normal case and must not be penalised.
  - scope "third_party": score 0.8-1.0 ONLY if the section being edited is itself about
    that party — its case file, profile, directory row, or comparison entry. If the
    section holds the document owner's own equivalent figure, OR you cannot tell whose
    figure the section states, score 0.0-0.3. An organization's own handbook rarely
    restates whose numbers it lists, so an unlabelled row in it belongs to the OWNER;
    "it doesn't say" is a reason to reject a third party's value, never to accept it.

verifier_note: a single sentence summarising the verification result.
"""


class _VerifierRaw(BaseModel):
    factual_consistency: float = Field(ge=0.0, le=1.0)
    formatting_integrity: float = Field(ge=0.0, le=1.0)
    intent_fulfillment: float = Field(ge=0.0, le=1.0)
    attribution_fit: float = Field(default=1.0, ge=0.0, le=1.0)
    verifier_note: str


class VerifierResult(BaseModel):
    """Result of a document edit verification, including computed quality score."""

    factual_consistency: float
    formatting_integrity: float
    intent_fulfillment: float
    # Deliberately NOT folded into quality_score: attribution is a hard gate, not a
    # quality dimension to be averaged away. A mis-attributed edit is perfect on the
    # other three axes, so any weighted blend would still clear the quality bar.
    attribution_fit: float = 1.0
    quality_score: float
    verifier_note: str


class VerifierAgent:
    """Verifies the quality of a proposed before/after document edit."""

    def __init__(self, model: str = None):
        # The verifier is the last precision gate before a card reaches review, so
        # it must reliably reject edits that don't actually fulfill the intent.
        # gpt-4o-mini waved some of those through; this runs only on the handful of
        # qualified drafts (not fanned out like eval), so the stronger model is cheap.
        self.model = model or os.getenv("LDOC_VERIFIER_MODEL", "gpt-5.4-mini")
        self._agent = Agent(
            name="ConfluenceVerifier",
            model=self.model,
            instructions=VERIFIER_INSTRUCTIONS,
            output_type=_VerifierRaw,
        )

    async def verify(
        self,
        before_content: str,
        after_content: str,
        intent_description: str,
        *,
        spoken_context: str = "",
        subject_entity: str | None = None,
        subject_scope: str = "unspecified",
        doc_title: str = "",
        section_heading: str = "",
        owner: str = "",
    ) -> VerifierResult:
        """Verify quality of a proposed document edit.

        The attribution context (the speaker's exact words, the recorded subject, and
        the document/section being edited) is keyword-only and defaults to empty, so
        existing callers keep working; without it the verifier simply cannot see a
        wrong-subject edit, since a mis-attributed change is flawless on the other
        three dimensions.

        Returns a VerifierResult with scores in [0,1] and a computed quality_score.
        On exception, returns a zero-score result with a failure note.
        """
        target = " — ".join(p for p in (doc_title, section_heading) if p)
        # Naming the owner matters most exactly where this backstop earns its keep:
        # when the extractor MISLABELLED an outsider's value as internal, the rule
        # above says "internal scores 0.8-1.0" and the verifier waves it through
        # unless it can see for itself whose documents these are.
        owner_block = f"{owner}\n\n" if owner else ""
        prompt = (
            f"{owner_block}"
            f"Intent description:\n{intent_description}\n\n"
            f"Attribution:\n"
            f"  value is about: {subject_entity or '(the document owner)'} "
            f"(scope: {subject_scope})\n"
            f"  speaker's exact words: {spoken_context or '(none)'}\n"
            f"  section being edited: {target or '(unknown)'}\n\n"
            f"Before:\n{before_content}\n\n"
            f"After:\n{after_content}"
        )
        try:
            result = await guarded_run(self._agent, prompt)
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
                attribution_fit=raw.attribution_fit,
                quality_score=quality_score,
                verifier_note=raw.verifier_note,
            )
        except Exception:
            logger.error("VerifierAgent.verify() failed", exc_info=True)
            return VerifierResult(
                factual_consistency=0.0,
                formatting_integrity=0.0,
                intent_fulfillment=0.0,
                attribution_fit=0.0,
                quality_score=0.0,
                verifier_note="Verification failed",
            )
