"""IntentExtractionAgent — extracts actionable document-change intents from meeting transcripts.

Uses the OpenAI Agents SDK (agents.Agent + agents.Runner) with structured output
to return a list of LocalDocIntent objects. Filters out low-confidence intents
(confidence < 0.5) before returning.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re

from agents import Agent, AgentOutputSchema, Runner
from pydantic import BaseModel

from agents_local.llm_runtime import guarded_run
from models import LocalDocIntent

logger = logging.getLogger(__name__)

# Long meeting transcripts that contain many changes overwhelm a single
# extraction call — the model summarizes and drops most of them. Splitting the
# transcript into focused segments and extracting from each keeps recall high.
_SEGMENT_WORD_TARGET = 180


def _segment_transcript(text: str, max_words: int = _SEGMENT_WORD_TARGET,
                        overlap_sentences: int = 1) -> list[str]:
    """Split a transcript into overlapping sentence-grouped segments.

    Short transcripts are returned as a single segment. Overlap ensures a change
    that straddles a boundary is fully present in at least one segment.
    """
    words = text.split()
    if len(words) <= int(max_words * 1.4):
        return [text]
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    segments: list[str] = []
    cur: list[str] = []
    wc = 0
    for s in sentences:
        cur.append(s)
        wc += len(s.split())
        if wc >= max_words:
            segments.append(" ".join(cur))
            cur = cur[-overlap_sentences:] if overlap_sentences else []
            wc = sum(len(x.split()) for x in cur)
    if cur:
        segments.append(" ".join(cur))
    return segments or [text]


def _norm(s: str | None) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def _value_key(value: str | None) -> str:
    """Normalize a value for dedup: its numeric core if present, else the text.

    Overlapping segments may emit the same change as "208 days" and "208"; both
    reduce to the number 208 so they collapse to one intent.
    """
    v = _norm(value)
    nums = re.findall(r"\d[\d,\.]*", v)
    return nums[0].replace(",", "") if nums else v


def _dedupe_intents(intents: list[LocalDocIntent]) -> list[LocalDocIntent]:
    """Collapse intents that overlapping segments extracted twice.

    Two intents are the same change when their affected_topic and the numeric
    core of their new_value match (so "208 days" and "208" collapse); keep the
    higher-confidence one.
    """
    best: dict[tuple[str, str], LocalDocIntent] = {}
    for i in intents:
        key = (_norm(i.affected_topic), _value_key(i.new_value))
        if key not in best or i.confidence > best[key].confidence:
            best[key] = i
    return list(best.values())

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
- Reported facts, figures, costs, spend, budgets, durations, dates, counts, and incidents count as document updates when they state a concrete value about something an organization records in a document — even when phrased as an observation or status report rather than a command ("I discovered...", "wanted to note...", "it took..."). The speaker need not say "update the doc"; the stated value IS the proposed new content (old_value null unless a prior value is given). For example:
    * "our EC2 and other instance spend is close to 1.4 million dollars" -> affected_topic "EC2 and instance spend", new_value "approximately 1.4 million dollars"
    * "it took 9 days to fix the AWS NAT gateway issue" -> affected_topic "AWS NAT gateway issue resolution time", new_value "9 days"
    * "that outage cost us 51 thousand USD" -> affected_topic "AWS NAT gateway outage cost", new_value "51 thousand USD"
  Extract each such fact as its own intent. The downstream retrieval step decides whether a matching document section exists, so do NOT withhold a concrete operational, financial, or incident fact just because there was no explicit edit verb.
- Product / strategy pivots count ("we're switching the product to X", "web app first, then mobile").
- Ignore hesitations and fillers ("ahmmm", "sorry", "you know") and purely social, logistical, or ephemeral chatter that no organization records in a document: greetings, lunch/coffee, weather, parking, kudos, scheduling, "can you hear me", and vague aspirations with no concrete value ("we should improve security someday"). A self-correction keeps the corrected value ("8 days, sorry, 9 days" -> 9 days).
- CRITICAL — discussing a topic is NOT changing it. Do NOT extract an intent when the speaker names a policy, section, or value but the statement concludes that it is NOT changing, is undecided, deferred, or hypothetical — even though it names a real topic and sometimes a number. These are the most common false positives; reject all of them:
    * affirmed unchanged: "no changes", "leave it as is", "keep it", "fine as written", "fine the way it's written", "already covers it", "it's current", "in good shape", "no edits needed", "nothing to change", "nothing to action", "everyone's happy with it".
    * deferred / no decision: "parked it", "tabled it", "floated ... but", "we'll revisit", "circle back", "next quarter", "no decision today", "for now".
    * hypothetical / unknown: "if we ever", "hypothetical for now", "someday", "at some point", "nobody's sure", "we'll check offline", "we'll confirm later", "no one knew" — when no new value is actually being put in place.
  A document change requires an AFFIRMATIVE new state the speaker is enacting now (or a concrete reported fact/value being recorded). Examples that yield NOTHING: "We looked at the travel and reimbursement policy and everyone's happy with it, leave it as is." / "Does anyone remember our current data retention period? Nobody's sure, we'll check offline." / "There was an idea to tweak the on-call rotation, but we parked it for next quarter." / "If we ever expand to the EU we'd revisit compliance, but that's hypothetical for now."
- A QUESTION or expression of not-knowing is never a change. If a topic appears only as a question ("does anyone remember/know what our X is?", "what's our current X?") or with uncertainty ("not sure", "nobody knows", "we'll find out") and NO new value is stated, return no intent for it. You must be able to name the exact new value from the transcript; if you cannot, do not extract the intent.
- Transient operational / status metrics are NOT document changes, even though they are numbers. A reported value is document-worthy ONLY when it is a GOVERNING value an organization writes into a policy, handbook, or spec — a policy limit or threshold, a retention period, an SLA target, a budget or cost figure, an incident's cost/duration/impact, or an official recorded target. Reject transient performance/status figures that live on a dashboard or in a standup, not in a governing document: current ticket/backlog counts, deals closed this week, signups, monthly active users/sessions, uptime percentages, crash rates, app-store ratings, candidates in the pipeline, a metric "ticking up/down" this week. Examples that yield NOTHING: "support backlog dropped from ninety to forty", "sales closed twelve deals", "we crossed a million monthly sessions", "uptime was 99.9%", "our rating crept to 4.6", "churn ticked down half a point this month".
- A metric going DOWN is not a section removal. Words like "dropped", "fell", "down", "decreased" describing a number are decreases, never an instruction to delete a section. Only treat remove/delete/eliminate/strike/"get rid of"/"take out" as a removal, and only when the speaker is removing a section or rule.

Rules:
1. Extract an intent for any specific document change AND for any concrete reported fact, figure, cost, duration, count, date, or incident that an organization would record in a document. When a concrete value is tied to a named subject, extract it; the retrieval step will discard it if no document covers that subject.
2. verbatim_snippets MUST contain exact quoted text from the transcript. When the speaker addresses a change to a SPECIFIC named document, page, company, or organization (e.g. "for the Northwind customer support SOP, change ..."), ALWAYS include that naming phrase in verbatim_snippets so the change can be routed to the right document — even though the document name is never the affected_topic (see rule 6).
3. confidence: 0.9+ only when the transcript is unambiguous; 0.5-0.89 for probable; < 0.5 for speculative.
4. old_value: the current state BEFORE the change (null if unknown). Never guess a value the transcript does not give.
5. new_value: the intended new state AFTER the change (for a relative change like "increased by 3 days", describe the delta, e.g. "current sick days + 3").
6. affected_topic MUST be the SPECIFIC subject being changed — the policy item, limit, field, window, or named section (e.g. "artifact retention depth", "badge re-enrollment grace", "pager duty carbon-copy rule"). Do NOT use the document, policy, handbook, system, or company name as the affected_topic. If the speaker says "in the X policy, change the Y from A to B", the affected_topic is Y, never X. (The only exception is a positional removal that references position rather than a subject — see the removals note above — where you include the document/area.)
7. If no actionable document change intents are found, return {"intents": []}.
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

    async def _extract_segment(self, segment: str) -> list[LocalDocIntent]:
        try:
            result = await guarded_run(
                self._agent, f"Meeting transcript:\n\n{segment}"
            )
            intent_list: _IntentList = result.final_output
            return list(intent_list.intents)
        except Exception:
            logger.error("IntentExtractionAgent segment extract failed", exc_info=True)
            return []

    async def extract(self, transcript: str) -> list[LocalDocIntent]:
        """Extract document-change intents from a meeting transcript.

        Long transcripts are segmented and extracted in parallel, then merged —
        a single call on a long multi-change meeting drops most of the changes.
        Returns an empty list for empty transcripts; filters confidence < 0.5.
        """
        if not transcript.strip():
            return []
        segments = _segment_transcript(transcript)
        if len(segments) == 1:
            intents = await self._extract_segment(segments[0])
        else:
            results = await asyncio.gather(
                *[self._extract_segment(s) for s in segments]
            )
            intents = _dedupe_intents([i for r in results for i in r])
            logger.info(
                "IntentExtractionAgent: %d segments -> %d unique intents",
                len(segments), len(intents),
            )
        return [i for i in intents if i.confidence >= 0.5]
