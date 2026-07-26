"""EvaluationAgent — scores document section relevance for a given change intent.

Uses the OpenAI Agents SDK (agents.Agent + agents.Runner) with structured output
to return a float relevance score in [0.0, 1.0] for a (intent, chunk) pair.
Supports concurrent batch scoring via asyncio.gather.
"""

from __future__ import annotations

import asyncio
import logging
import os

from agents import Agent, ModelSettings, Runner
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

BOILERPLATE GUARD — name-dropping the topic is NOT enough. The right section must actually
CONTAIN the concrete thing the change modifies or attaches to: the specific value, number,
price, duration, response time, parameter, named method, rule, or the table row / list /
enumerated field of that kind. A section that only describes the topic AREA in generic,
templated, or boilerplate prose — mentioning the subject by name but stating NO concrete
value, figure, parameter, rule, or list of the relevant kind — is the WRONG target: score
it 0.2-0.4 even though the topic word appears. Examples of boilerplate that must score LOW
for a value/spec change: "Training on this topic is delivered on a rolling basis and tracked
to completion", "Responsibilities and limits for response SLA are set out in this section",
"Owners must keep the document store aligned with the thresholds described here". Contrast
with a RIGHT target, which states the actual fact: "Training duration: 8 hours", a response
table with "P1 — 4 hours", "Output dimension: 364 features". Reserve 0.7+ for a section that
holds the real fact/field/row being changed (or, for an ADDITION, the actual list/table/count
the new item joins) — not merely the subject's name. This still obeys the rule above: when the
concrete fact IS present, score on topic match and never lower it for a differing value.

DOCUMENT MATCH IS DECISIVE for disambiguation. If the speaker named a specific document,
page, company, or organization (see "Spoken context"/topic) AND the candidate's "Document"
CLEARLY belongs to a DIFFERENT company/organization, score 0.0-0.3 even if the topic,
heading, and values match perfectly — editing the right kind of section in the WRONG
document is exactly the failure to avoid (many documents share the same headings and
similar tables). BUT do NOT penalize when the document matches the named one — there,
score on topic relevance as usual.

SUBJECT ATTRIBUTION IS DECISIVE. Every value belongs to some party, and a section is only
the right target when the intent's subject and the SECTION's subject are the SAME party.
The intent arrives with its attribution:
- subject_scope "internal" — the value is about the organization whose documents these
  are. Score on topic exactly as described above.
- subject_scope "third_party" — the value was stated about a DIFFERENT party (a competitor
  or peer firm, another product, a customer, a vendor, a partner, an investee/portfolio
  company, an industry benchmark, or a figure quoted from elsewhere). Such a value may
  ONLY land on a section that is itself ABOUT that party — a case file, profile,
  directory row, competitive-landscape entry, or comparison table for that party. A
  section stating the DOCUMENT OWNER'S OWN equivalent figure is the WRONG target no matter
  how perfectly the topic, heading, table shape, units, and magnitude line up: another
  party's fee, rate, ratio, limit, headcount, or target must NEVER overwrite this
  organization's own. That is the most damaging error you can make — score it 0.0-0.15.
- subject_scope "unspecified" — no attribution was recorded. Read the spoken context
  yourself: if it clearly states the value about a named party that is NOT this section's
  subject, treat it as third_party and apply the rule above. Otherwise score on topic as
  usual.
Attribution is about WHOSE value it is, not about topic. A section can be a flawless topic
match and still be the wrong subject; when the two conflict, attribution wins.

Return a JSON object with:
  relevance_score: float (0.0-1.0)
  section_subject: str (whose facts this section states — the party the section's values
    belong to: the document owner's own organization, or a specific named third party)
  subject_match: bool — decided as follows:
    * subject_scope "internal", or no third party in play: return true.
    * subject_scope "third_party": return true ONLY if you can point to positive
      evidence — in the Document title, the Section heading, or the Content — that this
      section is ABOUT that party: its case file, profile, directory row, comparison
      table, or competitive-landscape entry. If the section holds the document owner's
      own equivalent figure, OR you cannot tell whose figure it is, return false.
      Absence of evidence is FALSE here, not true. Most sections of an organization's
      own handbook never restate who they are about — an unlabelled row in the owner's
      own document is the OWNER's row, not a rival's, so "the section doesn't say whose
      this is" is a reason to reject, never a reason to allow.
  reasoning: str (one sentence explaining the score)
"""


class _EvalResult(BaseModel):
    relevance_score: float = Field(ge=0.0, le=1.0)
    section_subject: str = ""
    subject_match: bool = True
    reasoning: str


class EvalVerdict(BaseModel):
    """A scored (intent, section) pair, including the subject-attribution check."""

    relevance_score: float = 0.0
    section_subject: str = ""
    subject_match: bool = True
    reasoning: str = ""


class EvaluationAgent:
    """Scores how relevant a document section is for a given change intent."""

    def __init__(self, model: str = None, *, temperature: float | None = None):
        self.model = model or os.getenv("LDOC_EVAL_MODEL", "gpt-5.4-mini")
        self._agent = Agent(
            name="ConfluenceEvaluator",
            model=self.model,
            instructions=EVAL_INSTRUCTIONS,
            output_type=_EvalResult,
            model_settings=ModelSettings(temperature=temperature),
        )

    async def score(self, intent: ConfluenceIntent, chunk: ChunkRecord) -> float:
        """Score how relevant a document section is for a change intent.

        Returns a float in [0.0, 1.0]. Returns 0.0 on any exception.
        """
        return (await self.score_detail(intent, chunk)).relevance_score

    async def score_detail(
        self, intent: ConfluenceIntent, chunk: ChunkRecord
    ) -> EvalVerdict:
        """Score a section AND report whose subject it states (attribution check).

        Returns a zero-score verdict on any exception; ``subject_match`` is left
        True there so the failure mode is unchanged — the 0.0 score already fails
        the relevance gate, and a transient LLM error must not be reported as an
        attribution mismatch.
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
            f"  subject_scope: {intent.subject_scope}\n"
            f"  subject_entity: {intent.subject_entity or '(the document owner)'}\n"
            f"  spoken context: {spoken or '(none)'}\n\n"
            f"Candidate section:\n"
            f"  Document (page): {doc_label}\n"
            f"  Section heading: {chunk.section_heading}\n"
            f"  Content:\n{content}"
        )
        try:
            result = await guarded_run(self._agent, prompt)
            r: _EvalResult = result.final_output
            return EvalVerdict(
                relevance_score=r.relevance_score,
                section_subject=r.section_subject,
                subject_match=r.subject_match,
                reasoning=r.reasoning,
            )
        except Exception:
            logger.warning(
                "EvaluationAgent.score() failed for topic=%s heading=%s",
                intent.affected_topic,
                chunk.section_heading,
                exc_info=True,
            )
            return EvalVerdict(relevance_score=0.0, reasoning="evaluation failed")

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
