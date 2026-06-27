"""IntentExtractionAgent — extracts actionable document-change intents from meeting transcripts.

Uses the OpenAI Agents SDK (agents.Agent + agents.Runner) with structured output
to return a list of ConfluenceIntent objects. Filters out low-confidence intents
(confidence < 0.5) before returning.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re

from agents import Agent, AgentOutputSchema, Runner
from pydantic import BaseModel

from .llm_runtime import guarded_run
from .models import ConfluenceIntent

logger = logging.getLogger(__name__)

# Long meeting transcripts that contain many changes overwhelm a single
# extraction call — the model summarizes and drops most of them. Splitting the
# transcript into focused segments and extracting from each keeps recall high.
# The target is deliberately modest: too large and the per-segment call starts
# summarizing again. 230 words (up from 180) yields ~20% fewer segments — i.e.
# ~20% fewer LLM calls — which offsets the single extra glossary call below,
# while the larger window also reduces how often a change straddles a boundary.
# Tunable per deployment via env (cost vs. recall lever).
_SEGMENT_WORD_TARGET = int(os.getenv("LDOC_INTENT_SEGMENT_WORDS", "230"))
_SEGMENT_OVERLAP_SENTENCES = int(os.getenv("LDOC_INTENT_OVERLAP_SENTENCES", "2"))


def _segment_transcript(
    text: str,
    max_words: int = _SEGMENT_WORD_TARGET,
    overlap_sentences: int = _SEGMENT_OVERLAP_SENTENCES,
) -> list[str]:
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


def _dedupe_intents(intents: list[ConfluenceIntent]) -> list[ConfluenceIntent]:
    """Collapse intents that overlapping segments extracted twice.

    Two intents are the same change when their affected_topic and the numeric
    core of their new_value match (so "208 days" and "208" collapse); keep the
    higher-confidence one.
    """
    best: dict[tuple[str, str], ConfluenceIntent] = {}
    for i in intents:
        key = (_norm(i.affected_topic), _value_key(i.new_value))
        if key not in best or i.confidence > best[key].confidence:
            best[key] = i
    return list(best.values())


# ── Participant glossary (cross-segment coreference) ──────────────────────────
#
# Because the transcript is extracted one segment at a time, a change phrased as
# "he should also own X" in a late segment cannot be resolved to the person named
# only in an early segment. A cheap pass over the transcript builds a small
# people-and-roles glossary (names + any ownership/role facts, including
# newly-assigned ones); that glossary is injected as a prefix into every segment
# so the per-segment extractor can bind the pronoun. It runs ONLY when the
# transcript actually splits — short meetings, which already fit in one segment,
# pay nothing extra.
#
# Scaling: a single glossary call over a HUGE transcript hits the same failure the
# per-segment extraction avoids — the model summarizes a 12-person, 40-change
# meeting and silently drops a person, breaking every pronoun that points at them.
# So the glossary is itself map-reduced: the transcript is split into large blocks
# (~1200 words, much coarser than the 230-word extraction segments), people are
# extracted per block in parallel, then merged by name (notes appended in
# appearance order, so a later reassignment is reflected last). A normal meeting is
# one block — still a single call, cost-neutral — and only genuinely large meetings
# fan out.
_GLOSSARY_BLOCK_WORDS = int(os.getenv("LDOC_GLOSSARY_BLOCK_WORDS", "1200"))


class _GlossaryPerson(BaseModel):
    name: str
    notes: str = ""  # roles / ownership / responsibilities, incl. newly-assigned


class _Glossary(BaseModel):
    people: list[_GlossaryPerson]


GLOSSARY_INSTRUCTIONS = """\
You build a compact participant glossary from a meeting transcript. List every
PERSON who is mentioned or speaks, and for each one capture any role, title,
ownership, or responsibility stated about them — INCLUDING ones newly assigned
during this meeting (e.g. "Lakshman is now the owner of the security docs").

Rules:
- One entry per distinct person. Use the name as spoken (first name is fine).
- notes: a short factual phrase of their role(s)/ownership. If a responsibility
  was reassigned, say so and name who it was taken from if stated
  ("now owner of the security docs, previously Rahul"). Empty notes is fine when
  only the name is known.
- Do NOT invent roles. Only record what the transcript states or clearly implies.
- This is a reference list to help resolve pronouns later; it is not a list of
  changes. Capture people and roles, nothing else.
- If no people are identifiable, return {"people": []}.
"""


def _merge_people(people: list[_GlossaryPerson]) -> list[_GlossaryPerson]:
    """Merge per-block people into one list, keyed by normalized name.

    First-appearance order is preserved. Distinct note phrases for the same person
    are concatenated in the order they were seen, so a later reassignment
    ("security handed to Meera") lands AFTER the earlier fact and reflects the
    current state. Case/space-insensitive on the name so "Lakshman" and "lakshman"
    collapse.
    """
    merged: dict[str, _GlossaryPerson] = {}
    order: list[str] = []
    for p in people:
        name = (p.name or "").strip()
        if not name:
            continue
        key = _norm(name)
        note = (p.notes or "").strip()
        if key not in merged:
            merged[key] = _GlossaryPerson(name=name, notes=note)
            order.append(key)
        elif note and note.lower() not in merged[key].notes.lower():
            existing = merged[key].notes
            merged[key].notes = f"{existing}; {note}" if existing else note
    return [merged[k] for k in order]


def _format_glossary(glossary: _Glossary) -> str:
    """Render the glossary as a prompt prefix; empty string when there is none."""
    lines: list[str] = []
    for p in glossary.people:
        name = (p.name or "").strip()
        if not name:
            continue
        notes = (p.notes or "").strip()
        lines.append(f"- {name}: {notes}" if notes else f"- {name}")
    if not lines:
        return ""
    return (
        "Known people in this meeting, in order of appearance (use ONLY to resolve "
        'pronouns like "he/she/they" and bare first names to a specific person; a '
        "role or owner listed here may have been established in an earlier part of "
        "the meeting, and where one note follows another the LATER note is the more "
        "recent state. Do not treat this list as a source of changes):\n"
        + "\n".join(lines)
    )


class ParticipantGlossaryAgent:
    """Builds a people-and-roles glossary from the transcript (map-reduce).

    Small meetings are a single block (one call). Large ones are split into coarse
    blocks, extracted in parallel, and merged, so a single oversized summarization
    call never drops a person.
    """

    def __init__(self, model: str = None):
        # A narrow extraction — the cheap model is sufficient and keeps the added
        # cost negligible (offset by the ~20% fewer segment calls from the larger
        # window above).
        self.model = model or os.getenv("LDOC_GLOSSARY_MODEL", "gpt-4o-mini")
        self._agent = Agent(
            name="ParticipantGlossary",
            model=self.model,
            instructions=GLOSSARY_INSTRUCTIONS,
            output_type=_Glossary,
        )

    @staticmethod
    def _blocks(transcript: str, max_words: int = _GLOSSARY_BLOCK_WORDS) -> list[str]:
        """Split into coarse, sentence-aligned blocks for parallel people-extraction."""
        words = transcript.split()
        if len(words) <= int(max_words * 1.3):
            return [transcript]
        sentences = re.split(r"(?<=[.!?])\s+", transcript.strip())
        blocks: list[str] = []
        cur: list[str] = []
        wc = 0
        for s in sentences:
            cur.append(s)
            wc += len(s.split())
            if wc >= max_words:
                blocks.append(" ".join(cur))
                cur = []
                wc = 0
        if cur:
            blocks.append(" ".join(cur))
        return blocks or [transcript]

    async def _extract_block(self, block: str) -> list[_GlossaryPerson]:
        try:
            result = await guarded_run(self._agent, f"Meeting transcript:\n\n{block}")
            return list(result.final_output.people)
        except Exception:
            logger.error("ParticipantGlossaryAgent block extract failed", exc_info=True)
            return []

    async def build(self, transcript: str) -> str:
        """Return a formatted glossary prefix, or "" on any failure/empty result."""
        blocks = self._blocks(transcript)
        if len(blocks) == 1:
            people = await self._extract_block(blocks[0])
        else:
            results = await asyncio.gather(*[self._extract_block(b) for b in blocks])
            people = _merge_people([p for r in results for p in r])
            logger.info(
                "ParticipantGlossaryAgent: %d blocks -> %d people",
                len(blocks),
                len(people),
            )
        return _format_glossary(_Glossary(people=people))


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
- Pronoun / first-name resolution: a meeting is processed one SEGMENT at a time, so the person a change is about may have been named in an EARLIER segment you cannot see. When a "Known people in this meeting" list is provided above the transcript, use it to resolve a pronoun (he/she/they) or a bare first name to the specific person, and write that resolved name — never a bare pronoun — into the field that records the person (e.g. the new owner in an ownership_change, the new responsible party in a process_change). If a change reassigns or extends a responsibility to someone referred to only as "he/she/they" and the list identifies exactly one plausible person (e.g. the newly-named owner), use that name. The list is a REFERENCE for resolution only — never invent a change that the transcript segment itself does not state.
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
    intents: list[ConfluenceIntent]


class IntentExtractionAgent:
    """Extracts document-change intents from a meeting transcript using GPT."""

    def __init__(self, model: str = None):
        # Intent extraction is the recall funnel: one call per transcript (not
        # fanned out like eval), so the stronger model is essentially free here
        # but recovers easy changes that gpt-4o-mini intermittently drops.
        self.model = model or os.getenv("LDOC_INTENT_MODEL", "gpt-5.4-mini")
        # Use strict_json_schema=False because ConfluenceIntent.metadata is an
        # untyped dict, which generates additionalProperties=True in JSON schema —
        # incompatible with the Agents SDK strict schema mode. (Rule 1 auto-fix)
        self._agent = Agent(
            name="ConfluenceIntentExtractor",
            model=self.model,
            instructions=INSTRUCTIONS,
            output_type=AgentOutputSchema(_IntentList, strict_json_schema=False),
        )

    async def _extract_segment(
        self, segment: str, glossary: str = ""
    ) -> list[ConfluenceIntent]:
        prompt = f"Meeting transcript:\n\n{segment}"
        if glossary:
            prompt = f"{glossary}\n\n{prompt}"
        try:
            result = await guarded_run(self._agent, prompt)
            intent_list: _IntentList = result.final_output
            return list(intent_list.intents)
        except Exception:
            logger.error("IntentExtractionAgent segment extract failed", exc_info=True)
            return []

    async def extract(self, transcript: str) -> list[ConfluenceIntent]:
        """Extract document-change intents from a meeting transcript.

        Long transcripts are segmented and extracted in parallel, then merged —
        a single call on a long multi-change meeting drops most of the changes.
        When the transcript splits, one cheap glossary pass over the whole text is
        injected into every segment so a pronoun/first-name in a later segment
        (e.g. "he should also own X") resolves to a person named in an earlier one.
        Returns an empty list for empty transcripts; filters confidence < 0.5.
        """
        if not transcript.strip():
            return []
        segments = _segment_transcript(transcript)
        if len(segments) == 1:
            # Short meeting: everything is already in one call, so the glossary
            # would add cost without buying any cross-segment resolution.
            intents = await self._extract_segment(segments[0])
        else:
            glossary = await ParticipantGlossaryAgent().build(transcript)
            results = await asyncio.gather(
                *[self._extract_segment(s, glossary) for s in segments]
            )
            intents = _dedupe_intents([i for r in results for i in r])
            logger.info(
                "IntentExtractionAgent: %d segments (glossary=%s) -> %d unique intents",
                len(segments),
                "yes" if glossary else "no",
                len(intents),
            )
        return [i for i in intents if i.confidence >= 0.5]
