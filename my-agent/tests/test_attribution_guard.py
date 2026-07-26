"""Subject-attribution guardrail in the Confluence proposal pipeline.

The failure this guards: a meeting quotes ANOTHER party's number — a competitor's
fee, a peer firm's terms, an industry benchmark — and it retrieves the document
owner's own equivalent row, because the topic, table shape, units and magnitude all
line up. Every pre-existing gate waves it through: relevance is genuinely high (same
topic), the deterministic recall floors fire (it IS the same kind of field), the
editor produces a clean minimal diff, and the verifier scores it perfect on factual
consistency, formatting and fulfillment. The edit is flawless — and writes a rival's
figure into this organization's document.

So attribution is enforced as its own hard gate, twice: at evaluation (before any
drafting cost) and again at verification (which catches values whose attribution the
extractor never recorded). These tests drive both deterministically with fakes.
"""

import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from review_pipeline.confluence_pipeline import pipeline as pl
from review_pipeline.confluence_pipeline.editor import EditorDraft
from review_pipeline.confluence_pipeline.evaluation import EvalVerdict
from review_pipeline.confluence_pipeline.models import ChunkRecord, ConfluenceIntent
from review_pipeline.confluence_pipeline.pipeline import (
    PipelineConfig,
    _entity_named_in_chunk,
    _entity_tokens,
    propose,
)
from review_pipeline.confluence_pipeline.verifier import VerifierResult

# A fee table: the document owner's OWN management fee. A competitor's fee is the
# same kind of value in the same shape — which is exactly why it lands here.
FEE_SECTION = (
    "| Year | Fee basis | Rate |\n"
    "| --- | --- | --- |\n"
    "| 1-4 (investment period) | Committed capital | 2.00% |\n"
    "| 5-8 | Net invested capital | 1.75% |"
)


def _chunk(heading="7. Fee Model Illustration", content=FEE_SECTION, doc="Fund Terms"):
    return ChunkRecord(
        chunk_id="p1:7",
        source_path="p1",
        source_format="confluence",
        section_heading=heading,
        section_index=7,
        content=content,
        doc_title=doc,
    )


def _intent(scope="third_party", entity="Meridian Capital"):
    return ConfluenceIntent(
        intent_type="policy_update",
        affected_topic="management fee rate",
        old_value="2.00%",
        new_value="2.50%",
        verbatim_snippets=["Meridian Capital charges a 2.5% management fee"],
        confidence=0.9,
        rationale="fee rate mentioned in the meeting",
        subject_entity=entity,
        subject_scope=scope,
    )


def _install_fakes(
    monkeypatch,
    intent,
    chunk,
    *,
    relevance=0.95,
    subject_match=False,
    attribution_fit=1.0,
):
    """Wire the pipeline's four LLM agents to deterministic fakes."""

    async def fake_extract(self, transcript):
        return [intent]

    async def fake_score_detail(self, i, c):
        return EvalVerdict(
            relevance_score=relevance,
            section_subject="the document owner's own fund",
            subject_match=subject_match,
            reasoning="fake",
        )

    async def fake_draft(self, i, section_content):
        return EditorDraft(
            before_content=section_content,
            after_content=section_content.replace("2.00%", "2.50%"),
            edit_type="replace",
        )

    async def fake_verify(self, before, after, desc, **kwargs):
        fake_verify.calls.append(kwargs)
        return VerifierResult(
            factual_consistency=1.0,
            formatting_integrity=1.0,
            intent_fulfillment=1.0,
            attribution_fit=attribution_fit,
            quality_score=1.0,
            verifier_note="clean minimal edit",
        )

    fake_verify.calls = []

    monkeypatch.setattr(pl.IntentExtractionAgent, "extract", fake_extract)
    monkeypatch.setattr(pl.EvaluationAgent, "score_detail", fake_score_detail)
    monkeypatch.setattr(pl.EvaluationAgent, "score", lambda self, i, c: _coro(relevance))
    monkeypatch.setattr(pl.ConfluenceEditorAgent, "draft", fake_draft)
    monkeypatch.setattr(pl.VerifierAgent, "verify", fake_verify)
    return fake_verify


def _coro(value):
    async def _inner():
        return value

    return _inner()


class _Retriever:
    def __init__(self, chunk):
        self.chunk = chunk

    def query(self, text, top_k):
        return [self.chunk]


def _run(intent, chunk, cfg=None):
    return asyncio.run(
        propose(
            "transcript text",
            retriever=_Retriever(chunk),
            config=cfg or PipelineConfig(),
        )
    )


# ── Evaluation-stage gate ─────────────────────────────────────────────────────


def test_third_party_value_is_rejected_from_the_owners_own_section(monkeypatch):
    """The reported bug: a rival's fee must not overwrite our fee, despite a
    near-perfect relevance score and a spotless verifier report."""
    intent, chunk = _intent(), _chunk()
    _install_fakes(monkeypatch, intent, chunk, relevance=0.95, subject_match=False)
    intents, proposals = _run(intent, chunk)
    assert len(intents) == 1  # the intent is still extracted and reported
    assert proposals == []  # but nothing is proposed


def test_third_party_value_survives_on_a_section_about_that_party(monkeypatch):
    """The guardrail must not be a blunt "reject anything about another company":
    a fact about a party IS editable on a section that is about that party
    (a case file, directory row, or competitive-landscape entry)."""
    intent = _intent()
    chunk = _chunk(heading="4. Case File - Meridian Capital", doc="Peer Fund Directory")
    _install_fakes(monkeypatch, intent, chunk, relevance=0.95, subject_match=True)
    _intents, proposals = _run(intent, chunk)
    assert len(proposals) == 1
    assert "2.50%" in proposals[0].after_content


def test_internal_value_is_not_gated(monkeypatch):
    """Recall guard: the ordinary case — our own change to our own doc — is
    untouched by the guardrail, even if the evaluator reports subject_match False."""
    intent = _intent(scope="internal", entity=None)
    chunk = _chunk()
    _install_fakes(monkeypatch, intent, chunk, relevance=0.95, subject_match=False)
    _intents, proposals = _run(intent, chunk)
    assert len(proposals) == 1


def test_unspecified_attribution_is_not_gated(monkeypatch):
    """Recall guard: an intent the extractor could not attribute stays ungated at
    the evaluation stage — the verifier backstop is what covers it."""
    intent = _intent(scope="unspecified", entity=None)
    chunk = _chunk()
    _install_fakes(monkeypatch, intent, chunk, relevance=0.95, subject_match=False)
    _intents, proposals = _run(intent, chunk)
    assert len(proposals) == 1


def test_attribution_rejection_survives_the_deterministic_rescue(monkeypatch):
    """A competitor's figure reliably trips the field-label / phrase-overlap recall
    floors precisely because it is the same kind of value. A gated candidate is
    zeroed, not merely lowered, so the rescue cannot tip it back over the line."""
    intent = _intent()
    # verbatim snippets that overlap the section body heavily -> phrase-overlap match
    intent.verbatim_snippets = [
        "management fee basis committed capital investment period rate"
    ]
    chunk = _chunk(
        content="Management fee basis committed capital investment period rate 2.00%"
    )
    # Score inside the rescue band (threshold 0.70, margin 0.10).
    _install_fakes(monkeypatch, intent, chunk, relevance=0.65, subject_match=False)
    _intents, proposals = _run(intent, chunk)
    assert proposals == []


def test_guard_can_be_disabled_for_ab_comparison(monkeypatch):
    """The pre-guardrail behaviour must remain reachable, so the precision gain and
    any recall cost can be measured against each other."""
    intent, chunk = _intent(), _chunk()
    _install_fakes(monkeypatch, intent, chunk, relevance=0.95, subject_match=False)
    cfg = PipelineConfig(attribution_guard=False)
    _intents, proposals = _run(intent, chunk, cfg=cfg)
    assert len(proposals) == 1  # the old, wrong behaviour


# ── Verification-stage backstop ───────────────────────────────────────────────


def test_verifier_backstop_drops_unrecorded_misattribution(monkeypatch):
    """When the extractor never recorded the attribution (scope "unspecified"), the
    evaluation gate never examines it — the verifier is what catches it."""
    intent = _intent(scope="unspecified", entity=None)
    chunk = _chunk()
    _install_fakes(
        monkeypatch, intent, chunk, relevance=0.95, subject_match=True,
        attribution_fit=0.1,
    )
    _intents, proposals = _run(intent, chunk)
    assert proposals == []


def test_verifier_receives_the_attribution_context(monkeypatch):
    """The verifier cannot judge attribution it cannot see: the speaker's words, the
    recorded subject, and the target document must all reach it."""
    intent = _intent()
    # A section that names the party, so the edit reaches the verification stage.
    chunk = _chunk(heading="4. Case File - Meridian Capital", doc="Peer Fund Directory")
    fake = _install_fakes(
        monkeypatch, intent, chunk, relevance=0.95, subject_match=True
    )
    _run(intent, chunk)
    assert fake.calls, "verifier was never called"
    kwargs = fake.calls[0]
    assert kwargs["subject_entity"] == "Meridian Capital"
    assert kwargs["subject_scope"] == "third_party"
    assert "Meridian Capital charges" in kwargs["spoken_context"]
    assert kwargs["doc_title"] == "Peer Fund Directory"
    assert kwargs["section_heading"] == "4. Case File - Meridian Capital"


def test_high_attribution_fit_passes(monkeypatch):
    """Recall guard: a well-attributed edit is unaffected by the backstop."""
    intent = _intent(scope="internal", entity=None)
    chunk = _chunk()
    _install_fakes(
        monkeypatch, intent, chunk, relevance=0.95, subject_match=True,
        attribution_fit=1.0,
    )
    _intents, proposals = _run(intent, chunk)
    assert len(proposals) == 1


# ── Deterministic name floor ──────────────────────────────────────────────────
#
# The model's attribution verdict is not stable run to run, and it is weakest exactly
# where the damage is done: a bare "| Reserve ratio | 1.4x |" row carries no visible
# owner, so the model reads it as subject-less and admits a rival's figure. A section
# can only be *about* a party it names, so that necessary condition is checked in code.


def test_generic_corporate_words_are_not_identity():
    """"Fund"/"Capital"/"Ventures" are shared by every party in a corpus and must
    never be the token that makes a name match."""
    assert _entity_tokens("Meridian Capital") == ["meridian"]
    assert _entity_tokens("Northwind Ventures Ltd") == ["northwind"]
    # All-generic name: fall back to the raw tokens rather than matching everything.
    assert _entity_tokens("The Fund") == ["fund"]


def test_shared_word_does_not_make_a_rival_match_the_owner():
    """The live failure this encodes: the rival "Meridian Bharat Fund" must not match
    the owner's own "Bharat Breakthrough Fund-I" on the shared word "bharat"."""
    intent = _intent(entity="Meridian Bharat Fund")
    owner_chunk = _chunk(
        heading="2.1 Corpus structure",
        content="| Base corpus | 200 crore | Target commitments at final close |",
        doc="Bharat Breakthrough Fund-I - Structure, Terms & Deployment Plan",
    )
    assert not _entity_named_in_chunk(intent, owner_chunk)


def test_section_naming_the_party_passes_the_floor():
    intent = _intent(entity="Astrophel Aerospace")
    chunk = _chunk(
        heading="2. Case File - Astrophel Aerospace",
        content="| What they do | Pune-based spacetech company |",
        doc="Portfolio Directory & Company Case Files",
    )
    assert _entity_named_in_chunk(intent, chunk)


def test_unnameable_third_party_matches_nothing():
    """A third-party value with no identifiable subject ("the market average") can
    never be shown to belong to a section, so it is rejected."""
    intent = _intent(entity=None)
    assert not _entity_named_in_chunk(intent, _chunk())


def test_name_floor_rejects_even_when_the_model_says_match(monkeypatch):
    """Both checks must agree. Here the model wrongly reports subject_match — the
    deterministic floor still rejects, because the section never names the party."""
    intent, chunk = _intent(), _chunk()  # "Meridian Capital" absent from the fee table
    _install_fakes(monkeypatch, intent, chunk, relevance=0.95, subject_match=True)
    _intents, proposals = _run(intent, chunk)
    assert proposals == []


# ── Model plumbing ────────────────────────────────────────────────────────────


def test_intent_attribution_defaults_are_backward_compatible():
    """Intents constructed without attribution (older callers, stored shapes) stay
    valid and ungated."""
    i = ConfluenceIntent(
        intent_type="policy_update",
        affected_topic="fee",
        new_value="2.5%",
        verbatim_snippets=[],
        confidence=0.9,
        rationale="r",
    )
    assert i.subject_scope == "unspecified"
    assert i.subject_entity is None
