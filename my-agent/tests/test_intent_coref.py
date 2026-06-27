"""Cross-segment coreference for ownership chains in intent extraction.

Scenario the user asked for: docs say Rahul owns the security docs; the meeting
reassigns ownership to Lakshman early on, then 200+ words later says "he should
also take care of the payments docs". With 180-230 word segmentation, the "he"
sentence lands in a LATER segment that never names Lakshman. Without a shared
participant glossary, the segment's extractor cannot resolve "he" and emits a
useless new_value ("he"); with the glossary it resolves to "Lakshman".

These tests drive the glossary plumbing deterministically (no live LLM): a fake
``guarded_run`` plays an LLM that only resolves the pronoun when the glossary is
actually injected into the segment prompt.
"""

import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from review_pipeline.confluence_pipeline import intent_extraction as ie
from review_pipeline.confluence_pipeline.intent_extraction import (
    IntentExtractionAgent,
    ParticipantGlossaryAgent,
    _Glossary,
    _GlossaryPerson,
    _IntentList,
    _merge_people,
    _segment_transcript,
)
from review_pipeline.confluence_pipeline.models import ConfluenceIntent


class _Result:
    def __init__(self, final_output):
        self.final_output = final_output


def _ownership_intent(new_value: str) -> ConfluenceIntent:
    return ConfluenceIntent(
        intent_type="ownership_change",
        affected_topic="payments docs ownership",
        old_value=None,
        new_value=new_value,
        verbatim_snippets=["he should also take care of the payments docs"],
        confidence=0.9,
        rationale="ownership extended to the payments docs",
    )


# The sentence that only a later segment contains; the new owner ("Lakshman") is
# never named in this segment — only the pronoun "he".
_PRONOUN_SENTENCE = "he should also take care of the payments docs"


def _make_fake_run(record):
    """A fake guarded_run; records every prompt and resolves the pronoun only
    when the glossary (containing 'Lakshman') is present in the segment prompt."""

    async def fake_guarded_run(agent, prompt):
        record.append((getattr(agent, "name", ""), prompt))
        if getattr(agent, "name", "") == "ParticipantGlossary":
            return _Result(
                _Glossary(
                    people=[
                        _GlossaryPerson(
                            name="Lakshman",
                            notes="now owner of the security docs (previously Rahul)",
                        )
                    ]
                )
            )
        # Intent extractor.
        if _PRONOUN_SENTENCE in prompt:
            resolved = "Lakshman" if "Lakshman" in prompt else "he"
            return _Result(_IntentList(intents=[_ownership_intent(resolved)]))
        return _Result(_IntentList(intents=[]))

    return fake_guarded_run


def _long_transcript() -> str:
    """A transcript long enough to split, with the new owner named early and the
    pronoun reference far enough away to fall in a later segment."""
    head = (
        "Priya: Let's start with documentation ownership. "
        "Lakshman is now the owner of the security docs, taking over from Rahul. "
    )
    # ~300 words of unrelated filler to push the pronoun sentence past a boundary.
    filler = (
        "Priya: On the infra side the nightly batch window moved from two hours "
        "to ninety minutes and the staging refresh now runs every six hours. "
        "We also bumped the cache TTL and tightened the retry budget on the "
        "ingestion workers. The dashboards looked healthy through the week and "
        "nobody flagged regressions during the canary. We talked through the "
        "backlog grooming cadence and agreed to keep it weekly for now. "
    ) * 6
    tail = f"Lakshman: Sounds good, and {_PRONOUN_SENTENCE} going forward."
    return head + filler + tail


def test_pronoun_resolves_to_named_owner_across_segments(monkeypatch):
    transcript = _long_transcript()
    # Sanity: the transcript really does split, and the pronoun sentence is NOT
    # in the same segment as the one that names Lakshman as the new owner.
    segments = _segment_transcript(transcript)
    assert len(segments) >= 2, "transcript must split for this test to be meaningful"
    pronoun_seg = next(s for s in segments if _PRONOUN_SENTENCE in s)
    assert "Lakshman is now the owner" not in pronoun_seg

    record = []
    monkeypatch.setattr(ie, "guarded_run", _make_fake_run(record))

    intents = asyncio.run(IntentExtractionAgent().extract(transcript))

    ownership = [i for i in intents if i.intent_type == "ownership_change"]
    assert ownership, "expected an ownership_change intent"
    assert ownership[0].new_value == "Lakshman", (
        f"pronoun should resolve to the named owner, got {ownership[0].new_value!r}"
    )
    # The glossary builder must have run exactly once over the whole transcript.
    assert sum(1 for name, _ in record if name == "ParticipantGlossary") == 1


def test_no_glossary_call_for_short_single_segment_transcript(monkeypatch):
    # Cost guard: a short transcript stays one segment and must NOT pay for the
    # extra glossary call.
    record = []
    monkeypatch.setattr(ie, "guarded_run", _make_fake_run(record))

    short = "Priya: Lakshman now owns the security docs."
    asyncio.run(IntentExtractionAgent().extract(short))

    assert all(name != "ParticipantGlossary" for name, _ in record)


def test_segment_size_increased_but_modest():
    # "Increase the chunk size slightly" — bigger than the old 180, but not so big
    # that the extractor summarizes and drops changes.
    assert 200 <= ie._SEGMENT_WORD_TARGET <= 320


def test_merge_people_dedupes_and_preserves_reassignment_order():
    people = [
        _GlossaryPerson(name="Lakshman", notes="owner of security docs"),
        _GlossaryPerson(name="Rahul", notes="moving to platform team"),
        # Same person, different casing + a later reassignment fact.
        _GlossaryPerson(name="lakshman", notes="security handed to Meera"),
        _GlossaryPerson(name="Lakshman", notes="owner of security docs"),  # dup note
        _GlossaryPerson(name="Meera", notes="now owns security docs"),
    ]
    merged = _merge_people(people)

    names = [p.name for p in merged]
    assert names == ["Lakshman", "Rahul", "Meera"], (
        "dedupe by name, keep first-seen order"
    )
    lakshman = merged[0]
    # Later note appended after the earlier one (reflects current state last);
    # the duplicate note is not repeated.
    assert lakshman.notes == "owner of security docs; security handed to Meera"


def test_glossary_map_reduces_huge_transcript(monkeypatch):
    # A transcript past the block threshold must fan out into multiple parallel
    # block calls and merge — not one oversized summarization call.
    block_words = ie._GLOSSARY_BLOCK_WORDS
    huge = "Priya: " + "the team discussed many operational details today. " * 400
    assert len(huge.split()) > block_words * 2

    calls = []

    async def fake_guarded_run(agent, prompt):
        calls.append(getattr(agent, "name", ""))
        return _Result(_Glossary(people=[_GlossaryPerson(name="Priya", notes="lead")]))

    monkeypatch.setattr(ie, "guarded_run", fake_guarded_run)
    out = asyncio.run(ParticipantGlossaryAgent().build(huge))

    glossary_calls = [c for c in calls if c == "ParticipantGlossary"]
    assert len(glossary_calls) >= 2, "huge transcript should fan out across blocks"
    assert "Priya" in out  # merged result still renders
