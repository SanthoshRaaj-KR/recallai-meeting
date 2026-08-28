"""Grounding the proposal pipeline in the corpus it is allowed to edit.

Two failures, one cause: the pipeline had no way to say "this meeting is not about
these documents".

  * Retrieval returned top_k by RANK, with no absolute floor, so every intent got a
    full slate of candidate sections no matter how unrelated the meeting was. The
    reranker computed the missing signal and it was thrown away.
  * ``subject_scope="internal"`` was read off the SPEAKER's framing, so an outsider
    saying "our headcount is 46" was recorded as an internal change to the document
    owner's headcount — a flawless topic match, a clean minimal diff, and completely
    wrong.

Together those produced twenty-odd immaculate, contextually absurd proposals from a
meeting that had nothing to do with the corpus. These tests drive both halves with
fakes; the live end-to-end pair at the bottom checks the real thing.
"""

import asyncio
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from review_pipeline.confluence_pipeline import corpus_profile as cp
from review_pipeline.confluence_pipeline import pipeline as pl
from review_pipeline.confluence_pipeline.corpus_profile import (
    CorpusProfile,
    MeetingScope,
    get_corpus_profile,
    resolve_meeting_scope,
)
from review_pipeline.confluence_pipeline.editor import EditorDraft
from review_pipeline.confluence_pipeline.evaluation import EvalVerdict
from review_pipeline.confluence_pipeline.models import (
    ChunkRecord,
    ConfluenceIntent,
)
from review_pipeline.confluence_pipeline.pipeline import (
    PipelineConfig,
    _entity_named_in_chunk,
    propose,
)
from review_pipeline.confluence_pipeline.structural import is_information_request
from review_pipeline.confluence_pipeline.retrieval import (
    PineconeHybridIndex,
    RetrievalUnavailable,
)
from review_pipeline.confluence_pipeline.verifier import VerifierResult


def _chunk(cid="p1:3", heading="3.4 Operating team", doc="Firm Overview", content="Headcount: 17 full-time"):
    return ChunkRecord(
        chunk_id=cid,
        source_path=cid.split(":")[0],
        source_format="confluence",
        section_heading=heading,
        section_index=3,
        content=content,
        doc_title=doc,
    )


# ── The retrieval floor ───────────────────────────────────────────────────────


class _Reranked:
    """Stands in for a pinecone inference.rerank response."""

    def __init__(self, scored):
        self.data = [type("Hit", (), {"index": i, "score": s})() for i, s in scored]


def _index_with(scored, *, rerank_raises=False, n_chunks=2, dense_fails=False):
    """A PineconeHybridIndex whose fusion and rerank are both faked.

    `n_chunks` candidates go in; `scored` decides what the reranker says about them.
    `dense_fails` simulates a Pinecone error (None), as opposed to an empty result.
    """
    index = PineconeHybridIndex(create=False)
    chunks = [_chunk("p1:0", "Fee table"), _chunk("p2:0", "Reserve policy")][:n_chunks]

    def fake_search(which, query, top_k):
        if dense_fails:
            return None
        if which != "dense":
            return []
        return [
            (rank, {"_id": c.chunk_id, "content": c.content, "source_path": c.source_path,
                    "section_heading": c.section_heading, "section_index": c.section_index,
                    "doc_title": c.doc_title}, 0.5)
            for rank, c in enumerate(chunks)
        ]

    class _Inference:
        @staticmethod
        def rerank(**kwargs):
            if rerank_raises:
                raise RuntimeError("rerank unavailable")
            return _Reranked(scored)

    index._search_one = fake_search
    index._client = lambda: type("PC", (), {"inference": _Inference()})()
    return index


def test_below_floor_candidates_are_dropped():
    """An off-corpus intent scores ~0.001 and must come back with nothing.

    Returning "no section of this corpus is about that" is the whole point: every
    gate downstream scores on topic match, and an outsider's headcount is a perfect
    topic match for the document owner's headcount row.
    """
    index = _index_with([(0, 0.001), (1, 0.0004)])
    assert index.query("our headcount is now 46 people", top_k=5) == []


def test_above_floor_candidates_survive():
    index = _index_with([(0, 0.95), (1, 0.87)])
    hits = index.query("management fee rate on committed capital", top_k=5)
    assert [h.chunk_id for h in hits] == ["p1:0", "p2:0"]


def test_floor_is_applied_per_candidate_not_per_query():
    """A strong top hit must not carry its weak neighbours through with it."""
    index = _index_with([(0, 0.95), (1, 0.002)])
    hits = index.query("management fee rate on committed capital", top_k=5)
    assert [h.chunk_id for h in hits] == ["p1:0"]


def test_rerank_failure_fails_open():
    """RRF scores are reciprocal ranks — no absolute meaning, so no floor on them.

    Applying one anyway would turn a transient Pinecone outage into silently
    dropped changes, which is the failure this pipeline can least afford.
    """
    index = _index_with([], rerank_raises=True)
    hits = index.query("management fee rate on committed capital", top_k=5)
    assert [h.chunk_id for h in hits] == ["p1:0", "p2:0"]


def test_a_lone_candidate_is_still_gated():
    """The floor must not be skipped just because fusion produced one candidate.

    That is precisely the shape an off-corpus query takes when the sparse index is
    disabled — and an ungated single chunk reads downstream as "grounded".
    """
    index = _index_with([(0, 0.004)], n_chunks=1)
    assert index.query("our headcount is now 46 people", top_k=5) == []

    index = _index_with([(0, 0.93)], n_chunks=1)
    assert [h.chunk_id for h in index.query("management fee rate", top_k=5)] == ["p1:0"]


def test_total_search_failure_is_not_reported_as_no_match():
    """An outage must not impersonate "this corpus documents nothing about that".

    Empty now carries a meaning the pipeline acts on, so a swallowed Pinecone error
    returning [] would drop every change and report a clean run with 0 proposals.
    """
    index = _index_with([], dense_fails=True)
    with pytest.raises(RetrievalUnavailable):
        index.query("management fee rate on committed capital", top_k=5)


def test_entity_name_floor_requires_a_whole_word():
    """"Ace Capital" must not be found inside "replace"."""
    intent = ConfluenceIntent(
        intent_type="policy_update", affected_topic="fee", new_value="2.5%",
        verbatim_snippets=["Ace Capital charges 2.5%"], confidence=0.9, rationale="r",
        subject_entity="Ace Capital", subject_scope="third_party",
    )
    substring_only = _chunk(content="replace the committed-capital rate in this space")
    assert not _entity_named_in_chunk(intent, substring_only)

    real_mention = _chunk(content="Ace Capital charges 2.5% on committed capital")
    assert _entity_named_in_chunk(intent, real_mention)


# ── The corpus owner ──────────────────────────────────────────────────────────


def test_env_owner_short_circuits_every_call(monkeypatch):
    """A pinned owner must cost nothing — no listing, no fetch, no LLM call."""
    cp.reset_cache()
    monkeypatch.setenv("LDOC_CORPUS_OWNER", "ValleyNXT Ventures")
    monkeypatch.setenv("LDOC_CORPUS_OWNER_ALIASES", "ValleyNXT, Bharat Breakthrough Fund-I")

    def explode(_index):
        raise AssertionError("sampled Pinecone despite a pinned owner")

    monkeypatch.setattr(cp, "_sample_corpus", explode)

    profile = asyncio.run(get_corpus_profile(object()))
    assert profile.owner_name == "ValleyNXT Ventures"
    assert "ValleyNXT" in profile.aliases


def test_unreachable_corpus_returns_none(monkeypatch):
    """Fail open: an unresolvable owner leaves the prompts exactly as they were."""
    cp.reset_cache()
    monkeypatch.delenv("LDOC_CORPUS_OWNER", raising=False)

    def explode(_index):
        raise RuntimeError("pinecone down")

    monkeypatch.setattr(cp, "_sample_corpus", explode)
    assert asyncio.run(get_corpus_profile(object())) is None


def test_empty_corpus_returns_none(monkeypatch):
    cp.reset_cache()
    monkeypatch.delenv("LDOC_CORPUS_OWNER", raising=False)
    monkeypatch.setattr(cp, "_sample_corpus", lambda _index: [])
    assert asyncio.run(get_corpus_profile(object())) is None


def test_prompt_block_names_the_owner_and_its_aliases():
    block = CorpusProfile(
        owner_name="ValleyNXT Ventures",
        aliases=["ValleyNXT"],
        description="An operator-led venture firm; these pages are its operating docs.",
    ).prompt_block()
    assert "ValleyNXT Ventures" in block
    assert "also referred to as ValleyNXT" in block


def test_prompt_block_is_empty_when_the_owner_is_unknown():
    """An empty block must leave the prompt byte-identical to the pre-owner one."""
    assert CorpusProfile(owner_name="").prompt_block() == ""
    assert CorpusProfile(owner_name="   ").prompt_block() == ""


# ── A question is not a change ────────────────────────────────────────────────


def test_a_number_spelled_out_is_the_same_value():
    """"nine-month" -> "9-month" changes nothing and must not become a card.

    This is what survived every other gate on a real meeting: someone ASKED about
    the 9-month programme, and the editor rewrote a section that already said
    "nine-month". Token-identical to a human, a clean diff to every checker.
    """
    before = "Every company enters the same nine-month acceleration programme."
    after = "Every company enters the same 9-month acceleration programme."
    assert not pl._meaningful_change(before, after)

    # A real change to the same field must still register.
    real = "Every company enters the same 12-month acceleration programme."
    assert pl._meaningful_change(before, real)


def test_information_requests_are_not_changes():
    def _q(snippet, old=None):
        return ConfluenceIntent(
            intent_type="other", affected_topic="acceleration programme duration",
            old_value=old, new_value="9 months", verbatim_snippets=[snippet],
            confidence=0.9, rationale="r",
        )

    assert is_information_request(_q("give me a brief about the 9-month acceleration program"))
    assert is_information_request(_q("can you explain the reserve policy"))
    assert is_information_request(_q("I wanted to know what the fee structure is"))

    # A statement of change is untouched...
    assert not is_information_request(_q("we are moving the programme to twelve months"))
    # ...and so is a change merely PHRASED as a question, which states what it moves from.
    assert not is_information_request(
        _q("what if we move it to three days?", old="five days")
    )


# ── Whose meeting is this? ────────────────────────────────────────────────────


def _scope_fakes(monkeypatch, scope):
    """propose() with everything faked except the meeting-scope decision."""
    captured = {}

    async def fake_extract(self, transcript, owner=""):
        return [captured["intent"]]

    async def fake_profile(_index):
        return CorpusProfile(owner_name="ValleyNXT Ventures")

    async def fake_scope(_transcript, _profile):
        return scope

    async def fake_score_detail(self, i, c):
        # The section is the owner's own row and says so — the honest verdict for
        # a visitor's value landing here.
        return EvalVerdict(relevance_score=0.95, section_subject="ValleyNXT Ventures",
                           subject_match=False, reasoning="fake")

    async def fake_score(self, i, c):
        return 0.95

    async def fake_draft(self, i, section_content):
        return EditorDraft(before_content=section_content,
                           after_content=section_content.replace("17", "46"),
                           edit_type="replace")

    async def fake_verify(self, before, after, desc, **kwargs):
        return VerifierResult(factual_consistency=1.0, formatting_integrity=1.0,
                              intent_fulfillment=1.0, attribution_fit=1.0,
                              quality_score=1.0, verifier_note="clean")

    monkeypatch.setattr(pl, "get_corpus_profile", fake_profile)
    monkeypatch.setattr(pl, "resolve_meeting_scope", fake_scope)
    monkeypatch.setattr(pl.IntentExtractionAgent, "extract", fake_extract)
    monkeypatch.setattr(pl.EvaluationAgent, "score_detail", fake_score_detail)
    monkeypatch.setattr(pl.EvaluationAgent, "score", fake_score)
    monkeypatch.setattr(pl.ConfluenceEditorAgent, "draft", fake_draft)
    monkeypatch.setattr(pl.VerifierAgent, "verify", fake_verify)
    return captured


def _visitor_intent():
    """A founder pitching an investor: "we need SOC 2". No other party is named in
    the sentence, so the extractor can only call it internal."""
    return ConfluenceIntent(
        intent_type="compliance_update", affected_topic="SOC 2 compliance",
        new_value="SOC 2 Type II required before enterprise rollout",
        verbatim_snippets=["we need SOC 2 before we can sell to enterprises"],
        confidence=0.9, rationale="stated in the meeting",
        subject_scope="internal", subject_entity=None,
    )


def test_a_visitors_our_never_reaches_the_owners_documents(monkeypatch):
    """The whole bug, in one test. Topic matches, diff is clean, every other gate
    passes — and the owner is not even in the room."""
    captured = _scope_fakes(monkeypatch, MeetingScope(
        owner_is_party=False,
        speaker_organizations=["Genreal Software", "Next Value Ventures"],
        reasoning="a startup pitching an investor; neither is the document owner",
    ))
    captured["intent"] = _visitor_intent()
    chunk = _chunk(heading="6. Internal Controls", content="SOC 2 Type II: certified")

    intents, proposals = asyncio.run(
        propose("...", retriever=_Retriever([chunk]), config=PipelineConfig(session_id="t"))
    )
    assert intents[0].subject_scope == "third_party", "an outsider's 'our' stayed internal"
    assert intents[0].subject_entity == "Genreal Software"
    assert proposals == []


def test_an_ordinary_internal_meeting_is_untouched(monkeypatch):
    """The recall control: when the owner IS in the room, nothing is re-attributed."""
    captured = _scope_fakes(monkeypatch, MeetingScope(
        owner_is_party=True, speaker_organizations=["ValleyNXT Ventures"],
    ))
    captured["intent"] = _visitor_intent()
    chunk = _chunk(heading="6. Internal Controls", content="SOC 2 Type II: certified")

    intents, _ = asyncio.run(
        propose("...", retriever=_Retriever([chunk]), config=PipelineConfig(session_id="t"))
    )
    assert intents[0].subject_scope == "internal", "re-attributed an owner's own change"


def test_scope_resolution_failure_assumes_the_owner_is_present():
    """Fail open — a failed scope call must not silently gate a real meeting."""
    scope = asyncio.run(resolve_meeting_scope("some transcript", None))
    assert scope.owner_is_party is True


def test_no_identifiable_speaker_org_leaves_intents_internal(monkeypatch):
    """An unmarked "our" in a meeting naming no company is the owner's own."""
    captured = _scope_fakes(monkeypatch, MeetingScope(
        owner_is_party=True, speaker_organizations=[],
    ))
    captured["intent"] = _visitor_intent()
    chunk = _chunk(heading="6. Internal Controls", content="SOC 2 Type II: certified")
    intents, _ = asyncio.run(
        propose("...", retriever=_Retriever([chunk]), config=PipelineConfig(session_id="t"))
    )
    assert intents[0].subject_scope == "internal"


# ── The two halves together, through propose() ────────────────────────────────


def _install_fakes(monkeypatch, intent, chunk, *, subject_match=False, attribution_fit=1.0):
    async def fake_extract(self, transcript, owner=""):
        fake_extract.owner_seen = owner
        return [intent]

    fake_extract.owner_seen = None

    async def fake_score_detail(self, i, c):
        return EvalVerdict(
            relevance_score=0.95,
            section_subject="the document owner's own operating team",
            subject_match=subject_match,
            reasoning="fake",
        )

    async def fake_score(self, i, c):
        return 0.95

    async def fake_draft(self, i, section_content):
        return EditorDraft(
            before_content=section_content,
            after_content=section_content.replace("17", "46"),
            edit_type="replace",
        )

    async def fake_verify(self, before, after, desc, **kwargs):
        return VerifierResult(
            factual_consistency=1.0,
            formatting_integrity=1.0,
            intent_fulfillment=1.0,
            attribution_fit=attribution_fit,
            quality_score=1.0,
            verifier_note="clean minimal edit",
        )

    async def fake_profile(_index):
        return CorpusProfile(owner_name="ValleyNXT Ventures", aliases=["ValleyNXT"])

    monkeypatch.setattr(pl, "get_corpus_profile", fake_profile)
    monkeypatch.setattr(pl.IntentExtractionAgent, "extract", fake_extract)
    monkeypatch.setattr(pl.EvaluationAgent, "score_detail", fake_score_detail)
    monkeypatch.setattr(pl.EvaluationAgent, "score", fake_score)
    monkeypatch.setattr(pl.ConfluenceEditorAgent, "draft", fake_draft)
    monkeypatch.setattr(pl.VerifierAgent, "verify", fake_verify)
    return fake_extract


class _Retriever:
    def __init__(self, chunks):
        self._chunks = chunks

    def query(self, query_text, top_k=12, **kwargs):
        return list(self._chunks)


def _outsider_intent():
    """What an outsider's "our headcount is 46" becomes once the owner is known.

    The speaker said "our". Anchored to the document owner, that "our" is a
    different organization, so the value is third_party — and third_party is a
    shape the attribution guard already knows how to refuse.
    """
    return ConfluenceIntent(
        intent_type="decision",
        affected_topic="headcount",
        old_value=None,
        new_value="46 full-time",
        verbatim_snippets=["our headcount at Nimbus Retail Systems is now 46 people"],
        confidence=0.9,
        rationale="headcount stated in the meeting",
        subject_entity="Nimbus Retail Systems",
        subject_scope="third_party",
    )


def test_an_outsiders_own_value_never_lands_on_the_owners_row(monkeypatch):
    chunk = _chunk()
    _install_fakes(monkeypatch, _outsider_intent(), chunk, subject_match=False)
    retriever = _Retriever([chunk])

    intents, proposals = asyncio.run(
        propose("...", retriever=retriever, config=PipelineConfig(session_id="t"))
    )
    assert len(intents) == 1
    assert proposals == [], "an outsider's headcount was written into the owner's own row"


def test_the_owner_block_reaches_the_extractor(monkeypatch):
    """The extractor cannot anchor "our" without being told whose documents these are."""
    chunk = _chunk()
    fake_extract = _install_fakes(monkeypatch, _outsider_intent(), chunk)
    asyncio.run(propose("...", retriever=_Retriever([chunk]), config=PipelineConfig(session_id="t")))
    assert "ValleyNXT Ventures" in fake_extract.owner_seen


def test_ungrounded_intents_skip_evaluation_entirely(monkeypatch):
    """Retrieval returning nothing is a verdict, not a reason to score an empty pool.

    On a meeting with nothing to do with the corpus this is the entire eval bill —
    a dozen cheap calls per intent, for every intent.
    """
    scored = []

    async def counting_score(self, i, c):
        scored.append(c)
        return 0.95

    _install_fakes(monkeypatch, _outsider_intent(), _chunk())
    monkeypatch.setattr(pl.EvaluationAgent, "score", counting_score)

    intents, proposals = asyncio.run(
        propose("...", retriever=_Retriever([]), config=PipelineConfig(session_id="t"))
    )
    assert len(intents) == 1
    assert proposals == []
    assert scored == [], "evaluated a section pool that retrieval had already rejected"


def test_owner_is_passed_to_both_evaluation_stages(monkeypatch):
    """Stage 1 and stage 2 must judge attribution against the same owner."""
    seen = []
    real_init = pl.EvaluationAgent.__init__

    def spy_init(self, model=None, *, temperature=None, owner=""):
        seen.append(owner)
        real_init(self, model, temperature=temperature, owner=owner)

    chunk = _chunk()
    _install_fakes(monkeypatch, _outsider_intent(), chunk)
    monkeypatch.setattr(pl.EvaluationAgent, "__init__", spy_init)

    asyncio.run(propose("...", retriever=_Retriever([chunk]), config=PipelineConfig(session_id="t")))
    assert len(seen) == 2
    assert all("ValleyNXT Ventures" in o for o in seen)


# ── Live end-to-end ───────────────────────────────────────────────────────────


def _load_env():
    """Live tests only: the review_pipeline package never loads .env.local itself."""
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env.local")

OFF_CORPUS_TRANSCRIPT = """\
Ravi: Thanks for making time. Quick recap on where Nimbus Retail Systems is.
Ravi: Our monthly recurring revenue crossed four hundred and eighty thousand dollars last month.
Ravi: Headcount is now forty six people across engineering and support.
Priya: And the warehouse?
Ravi: Our Coimbatore warehouse ships twelve thousand orders a day now, up from eight thousand.
Ravi: We moved the checkout service from Django to Go last sprint, latency dropped to ninety milliseconds.
Priya: What are you raising?
Ravi: The Series A is eight million dollars at a forty million pre-money valuation.
Ravi: Our gross margin is sixty two percent and we burn two hundred thousand a month.
Priya: Understood. We will come back to you after the partner meeting.
"""

ON_CORPUS_TRANSCRIPT = """\
Anand: Two things to change in the fund documents before the next LP update.
Anand: Our management fee on committed capital moves from two percent to one point seven five percent.
Anand: And the reserve policy — we are taking our follow-on reserve ratio to fifty percent of the fund.
Suresh: Noted. Also headcount.
Suresh: Our headcount as at thirty June is now nineteen full time, not seventeen.
"""


@pytest.mark.live
def test_live_off_corpus_meeting_produces_no_proposals():
    """A meeting about somebody else's company must not edit this corpus at all."""
    from review_pipeline.confluence_pipeline import propose as live_propose

    _load_env()
    cp.reset_cache()
    _intents, proposals = asyncio.run(
        live_propose(
            OFF_CORPUS_TRANSCRIPT,
            retriever=PineconeHybridIndex(create=False),
            config=PipelineConfig(session_id="live-off-corpus"),
        )
    )
    assert proposals == [], (
        f"{len(proposals)} proposal(s) from a meeting about a different company: "
        + "; ".join(f"{p.source_chunk.doc_title} :: {p.source_chunk.section_heading}" for p in proposals)
    )


@pytest.mark.live
def test_live_on_corpus_meeting_still_produces_proposals():
    """The recall control. The grounding fix is only correct if this side holds."""
    from review_pipeline.confluence_pipeline import propose as live_propose

    _load_env()
    cp.reset_cache()
    _intents, proposals = asyncio.run(
        live_propose(
            ON_CORPUS_TRANSCRIPT,
            retriever=PineconeHybridIndex(create=False),
            config=PipelineConfig(session_id="live-on-corpus"),
        )
    )
    assert proposals, "the grounding gate silenced a meeting that IS about these documents"


# The realistic shape, and the one that can regress quietly: outsiders in the room
# saying "we" about themselves, alongside genuine changes to the corpus. Anchoring
# "internal" to the document owner must separate the two rather than suppressing both.
MIXED_TRANSCRIPT = """\
Anand: Before we get to your round, one housekeeping item on our side.
Anand: Our management fee on committed capital is moving from two percent to one point seven five percent.
Ravi: Understood. On Nimbus Retail Systems, our monthly recurring revenue crossed four hundred
and eighty thousand dollars and our headcount is now forty six people.
Ravi: Our gross margin is sixty two percent.
Suresh: Good. And on our side, our headcount as at thirty June is nineteen full time, not seventeen.
"""


@pytest.mark.live
def test_live_mixed_meeting_keeps_only_the_owners_own_changes():
    """Both halves in one room: the owner's changes land, the outsider's do not.

    Note both parties state a headcount. They are the same kind of value in the same
    units about the same topic — only attribution tells them apart.
    """
    from review_pipeline.confluence_pipeline import propose as live_propose

    _load_env()
    cp.reset_cache()
    _intents, proposals = asyncio.run(
        live_propose(
            MIXED_TRANSCRIPT,
            retriever=PineconeHybridIndex(create=False),
            config=PipelineConfig(session_id="live-mixed"),
        )
    )
    assert proposals, "suppressed the document owner's own changes along with the outsider's"
    blob = " ".join(f"{p.after_content} {p.intent.new_value}" for p in proposals).lower()
    for leaked in ("480", "four hundred and eighty", "forty six", "sixty two percent", "62%"):
        assert leaked not in blob, f"an outsider's figure ({leaked!r}) reached a proposal"
