"""Reaffirmation guard: "deciding not to change it" is not a change.

The extraction instructions have always said discussing a topic is not changing it, but
that rule fails on the CONTRASTIVE form that dominates real meetings:

    "Meridian charges two point five percent. Our fee stays at two percent."

The "ours" clause reads as an affirmative statement of a value, so the extractor emits
an intent whose new_value is what the document ALREADY says. The no-op filter ought to
catch that — except the editor, told to make the section satisfy the intent, instead
manufactures a difference: on the live corpus it rewrote "walk every company's rating"
into "walk every amber and red company's rating", and on the pre-guardrail code it put
the investment-period fee (2%) onto the post-investment-period row (1.75%).

So reaffirmations are dropped before retrieval, on either of two independent signals:
the extractor's own ``change_polarity`` label, and a high-precision regex over the
speaker's quoted words (the label slips on contrastive phrasing; the markers are absent
when the speaker reaffirms implicitly).
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
from review_pipeline.confluence_pipeline.pipeline import PipelineConfig, propose
from review_pipeline.confluence_pipeline.structural import is_reaffirmation_phrasing
from review_pipeline.confluence_pipeline.verifier import VerifierResult

FEE_SECTION = (
    "| Term | Detail |\n"
    "| --- | --- |\n"
    "| Management fee - investment period | 2.00% per annum on committed capital |\n"
    "| Management fee - post investment period | 1.75% per annum on net asset value |"
)


def _chunk():
    return ChunkRecord(
        chunk_id="p1:3",
        source_path="p1",
        source_format="confluence",
        section_heading="3.1 Headline terms",
        section_index=3,
        content=FEE_SECTION,
        doc_title="Fund Terms",
    )


def _intent(snippets, polarity="change", topic="management fee", new_value="2 percent"):
    return ConfluenceIntent(
        intent_type="policy_update",
        affected_topic=topic,
        old_value=None,
        new_value=new_value,
        verbatim_snippets=snippets,
        confidence=0.9,
        rationale="stated in the meeting",
        subject_scope="internal",
        change_polarity=polarity,
    )


def _install_fakes(monkeypatch, intent):
    async def fake_extract(self, transcript, owner=""):
        return [intent]

    async def fake_score_detail(self, i, c):
        return EvalVerdict(relevance_score=0.95, subject_match=True, reasoning="fake")

    async def fake_draft(self, i, section_content):
        # The editor manufactures a difference rather than returning a no-op — the
        # behaviour that makes reaffirmations damaging instead of merely useless.
        return EditorDraft(
            before_content=section_content,
            after_content=section_content.replace("1.75%", "2%"),
            edit_type="replace",
        )

    async def fake_verify(self, before, after, desc, **kwargs):
        return VerifierResult(
            factual_consistency=1.0, formatting_integrity=1.0, intent_fulfillment=1.0,
            attribution_fit=1.0, quality_score=1.0, verifier_note="clean",
        )

    monkeypatch.setattr(pl.IntentExtractionAgent, "extract", fake_extract)
    monkeypatch.setattr(pl.EvaluationAgent, "score_detail", fake_score_detail)
    monkeypatch.setattr(pl.ConfluenceEditorAgent, "draft", fake_draft)
    monkeypatch.setattr(pl.VerifierAgent, "verify", fake_verify)


class _Retriever:
    def __init__(self, chunk):
        self.chunk = chunk

    def query(self, text, top_k):
        return [self.chunk]


def _run(chunk, cfg=None):
    return asyncio.run(
        propose("transcript", retriever=_Retriever(chunk), config=cfg or PipelineConfig())
    )


# ── Quoted-words detection ────────────────────────────────────────────────────


def test_contrastive_reaffirmations_are_detected():
    """The three live phrasings that produced wrong cards on the real corpus."""
    assert is_reaffirmation_phrasing(_intent(["Our management fee stays at two percent."]))
    assert is_reaffirmation_phrasing(
        _intent(["Ours stays at nine months, and we are not changing it."])
    )
    assert is_reaffirmation_phrasing(
        _intent(["Ours is monthly for amber and red only, and we are keeping it that way."])
    )


def test_real_changes_are_not_flagged():
    """Recall guard — the case B phrasings must all survive untouched."""
    for words in (
        "We are moving it from one point four x to one point six x.",
        "We are taking our cohort size up from ten companies to twelve companies.",
        "Our scout stipend goes from fifty thousand rupees to seventy five thousand.",
        "Our capital call notice period moves from fifteen business days to twenty.",
        "The IC memo page cap goes from twelve pages to fifteen pages.",
        "Set the retention period to ninety days.",
        "We are increasing the reserve ratio.",
    ):
        assert not is_reaffirmation_phrasing(_intent([words])), words


def test_detection_reads_only_the_speakers_words():
    """Like has_explicit_removal_verb, the marker must be in the QUOTED words — an
    extractor paraphrase that happens to say "stays" must not drop a real change."""
    intent = _intent(["We are moving the fee to two point five percent."])
    intent.rationale = "the fee stays at the new level thereafter"
    assert not is_reaffirmation_phrasing(intent)


# ── Pipeline enforcement ──────────────────────────────────────────────────────


def test_reaffirmation_label_drops_the_intent(monkeypatch):
    intent = _intent(["Our management fee is two percent."], polarity="reaffirmation")
    _install_fakes(monkeypatch, intent)
    intents, proposals = _run(_chunk())
    assert len(intents) == 1  # still reported as an intent
    assert proposals == []  # but never proposed


def test_quoted_words_drop_the_intent_even_when_mislabelled(monkeypatch):
    """The live failure: the model labels a contrastive reaffirmation "change"."""
    intent = _intent(["Our management fee stays at two percent."], polarity="change")
    _install_fakes(monkeypatch, intent)
    _intents, proposals = _run(_chunk())
    assert proposals == []


def test_a_real_change_still_produces_a_card(monkeypatch):
    intent = _intent(
        ["Our post investment period fee moves from one point seven five to two percent."],
        polarity="change",
    )
    _install_fakes(monkeypatch, intent)
    _intents, proposals = _run(_chunk())
    assert len(proposals) == 1


def test_guard_can_be_disabled_for_ab_comparison(monkeypatch):
    """The pre-guard behaviour stays reachable — and shows the damage it allowed:
    the reaffirmed 2% lands on the 1.75% post-investment-period row."""
    intent = _intent(["Our management fee stays at two percent."], polarity="reaffirmation")
    _install_fakes(monkeypatch, intent)
    _intents, proposals = _run(_chunk(), cfg=PipelineConfig(reaffirmation_guard=False))
    assert len(proposals) == 1
    assert "1.75%" in proposals[0].before_content


def test_polarity_defaults_to_change():
    """Backward compatibility: an intent with no polarity behaves as it always did."""
    i = ConfluenceIntent(
        intent_type="policy_update", affected_topic="fee", new_value="2%",
        verbatim_snippets=[], confidence=0.9, rationale="r",
    )
    assert i.change_polarity == "change"
