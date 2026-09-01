"""Vendored Confluence proposal pipeline (Confluence, Pinecone-hybrid retrieval).

This is the confluence-branch ``pipeline/run.py`` EDIT path, ported into my-agent
with one change: document retrieval goes through the Pinecone-native hybrid index
(dense ``llama-text-embed-v2`` + sparse ``pinecone-sparse-english-v0`` + rerank)
instead of FAISS/OpenAI. Every proposal-shaping step — intent extraction,
relevance evaluation (with the deterministic field-label / phrase-overlap recall
floors), drafting, verification, no-op/fulfillment filtering, and the thresholds
— is preserved exactly so the proposal quality matches the proven pipeline.

Renames and corpus-wide removals (which need the whole-corpus chunk list) are out
of scope for this Confluence v1; edits and in-section additions are covered.
"""

from __future__ import annotations

import asyncio
import difflib
import logging
import os
import re
from collections.abc import Awaitable, Callable
from typing import Any, Optional

from pydantic import BaseModel

from .corpus_profile import get_corpus_profile, resolve_meeting_scope
from .editor import ConfluenceEditorAgent
from .evaluation import EvaluationAgent
from .intent_extraction import IntentExtractionAgent
from .models import ChunkRecord, ConfluenceIntent, ConfluenceProposal
from .retrieval import PineconeHybridIndex
from .structural import (
    classify_kind,
    is_cross_cutting,
    is_information_request,
    is_reaffirmation_phrasing,
)
from .verifier import VerifierAgent

logger = logging.getLogger(__name__)

EmitFn = Callable[[str], Awaitable[None]]


class PipelineConfig(BaseModel):
    session_id: str = "confluence"
    # Keep a HIGH precision bar on the eval score — wrong sections score <=0.5 and
    # the "plausibly related but wrong" band (0.5-0.69) is exactly what produced the
    # wrong-row landings (response-time on the P4 row). Lowering this hurt precision
    # without fixing recall, so it stays at 0.70.
    relevance_threshold: float = 0.70
    # Two-stage eval (cost lever). Eval is the OpenAI-cost driver, so we split it:
    #  Stage 1 — cheap gpt-4o-mini scores a WIDE pool (retrieval_top_k) for recall.
    #  Stage 2 — the strong gpt-5.4-mini RE-scores only the top `eval_stage2_top_k`
    #            survivors (+ any deterministic field-label match) for precision.
    # So the expensive model runs ~5x/intent instead of 12-20x, while the wide cheap
    # pool keeps recall (ambiguous-term intents like "training" don't get crowded out
    # of a tiny pool). Models via LDOC_EVAL_STAGE1_MODEL / LDOC_EVAL_MODEL.
    retrieval_top_k: int = 12
    eval_stage2_top_k: int = 5
    # One target per intent. Allowing 2 was tested and reverted: every intent sprayed
    # a second card onto a plausible-but-wrong chunk (response-times on the wrong SLA
    # row, +5ms on unrelated API endpoints), and the verifier can't catch a locally-
    # plausible-but-misplaced edit. Recall is driven by the intent funnel + retrieval,
    # not by widening targets.
    max_targets_per_intent: int = 1
    cross_cutting_top_k: int = 40
    cross_cutting_max_targets: int = 30
    # A deterministic match (exact field-label / verbatim phrase) only CONFIRMS a
    # near-miss: if the strong model scored a section within this margin BELOW the
    # threshold, the match tips it over the line. It does NOT rescue a score the model
    # rated clearly-wrong (below threshold - margin) — there the lexical match is
    # almost always coincidental (e.g. "training" appearing in an unrelated section).
    floor_rescue_margin: float = 0.10
    # Precision-first: drop drafts the verifier isn't confident actually applied
    # the change. Correct edits score ~0.85-1.0; 0.6 cuts the weak/uncertain ones
    # (the user prefers a missed change over a wrong card).
    min_fulfillment: float = 0.6
    # Drop cards whose draft fails verification on factual consistency.
    min_factual: float = 0.6
    # Subject-attribution guardrail. A meeting quotes other parties' numbers all the
    # time (a competitor's fee, a peer firm's terms, an industry benchmark); those
    # retrieve the document owner's OWN equivalent row — same topic, same table, same
    # units — and every other gate waves them through, because the resulting edit is
    # genuinely well-formed. So a value attributed to a third party may only land on a
    # section that is itself about that party. Enforced twice: at evaluation (cheap,
    # before drafting) and again at verification (catches an attribution the extractor
    # never recorded). Set LDOC_ATTRIBUTION_GUARD=0 to disable for A/B comparison.
    attribution_guard: bool = os.getenv("LDOC_ATTRIBUTION_GUARD", "1").strip().lower() in {
        "1", "true", "yes", "on",
    }
    min_attribution: float = 0.5
    # Reaffirmation guard. "Deciding not to change something" is not a change, but in
    # the contrastive form that dominates real meetings — "they charge X; ours stays at
    # Y" — the "ours" clause reads as an affirmative statement of a value, and the
    # extractor emits it as an intent. Its new_value is what the document ALREADY says,
    # so the no-op filter should catch it — except the editor, told to make the section
    # satisfy the intent, instead edits a neighbouring row or rewrites a sentence to
    # manufacture a difference. Cheapest to drop these before retrieval: they cost
    # nothing to detect and every stage after this point can only do damage with them.
    # Set LDOC_REAFFIRMATION_GUARD=0 to disable for A/B comparison.
    reaffirmation_guard: bool = os.getenv("LDOC_REAFFIRMATION_GUARD", "1").strip().lower() in {
        "1", "true", "yes", "on",
    }


# ── Deterministic recall helpers (ported verbatim) ────────────────────────────

_LABEL_STOPWORDS = {
    "the", "a", "an", "of", "for", "to", "and", "or", "per", "by", "in", "on",
    "is", "are", "be", "value", "parameter", "parameters", "standard", "target",
    "current", "new", "max", "min",
}

_PLACEHOLDER_VALUES = {
    "", "tbd", "tba", "unknown", "n/a", "na", "to be determined",
    "to be decided", "?", "none", "null",
}


def _sig_tokens(text: str) -> list[str]:
    return [
        t for t in re.findall(r"[a-z0-9]+", (text or "").lower())
        if len(t) >= 3 and t not in _LABEL_STOPWORDS
    ]


def _field_labels(content: str) -> list[str]:
    labels: list[str] = []
    for line in (content or "").splitlines():
        s = line.strip()
        if not s:
            continue
        if "|" in s:
            cells = [c.strip() for c in s.strip("|").split("|")]
            if len(cells) >= 2 and cells[0] and not set(cells[0]) <= set("-: "):
                labels.append(cells[0])
        else:
            m = re.match(r"^([A-Za-z][\w '/&\-]{2,45})\s*[:\-–]\s+\S", s)
            if m:
                labels.append(m.group(1))
    return labels


def _field_label_match(intent: ConfluenceIntent, chunk: ChunkRecord) -> bool:
    topic_toks = _sig_tokens(getattr(intent, "affected_topic", ""))
    if len(topic_toks) < 2:
        return False
    topic_set = set(topic_toks)
    for label in _field_labels(chunk.content):
        ltoks = set(_sig_tokens(label))
        if not ltoks:
            continue
        hits = sum(1 for t in topic_set if any(t in lt or lt in t for lt in ltoks))
        if hits >= 2 and hits >= len(topic_set) - 1:
            return True
    return False


def _phrase_overlap_match(intent: ConfluenceIntent, chunk: ChunkRecord) -> bool:
    verbatim = " ".join(getattr(intent, "verbatim_snippets", None) or [])
    vtoks = set(_sig_tokens(verbatim))
    if len(vtoks) < 4:
        return False
    body = (chunk.content or "").lower()
    matched = sum(1 for t in vtoks if t in body)
    return matched >= 3 and matched >= 0.6 * len(vtoks)


def _has_concrete_value(i: ConfluenceIntent) -> bool:
    return (i.new_value or "").strip().lower() not in _PLACEHOLDER_VALUES


# Words that appear in essentially every section: requiring one as a name token would
# match the whole corpus and silently disable the floor. Always dropped.
_ENTITY_STOPWORDS = {"the", "and", "for"}
# Generic corporate words carry no identity — "Fund", "Ventures", "Ltd" are shared by
# every party in the corpus, so they must never be what makes a name "match". Dropped
# only when something distinctive survives, so an all-generic name still has a token.
_ENTITY_GENERIC_TOKENS = {
    "fund", "funds", "capital", "ventures", "venture", "partners", "partner",
    "company", "companies", "corp", "corporation", "inc", "incorporated",
    "ltd", "limited", "llp", "llc", "plc", "pvt", "private", "group", "holdings",
    "technologies", "technology", "labs", "systems", "solutions", "services",
    "firm", "team", "their", "them", "our", "market", "industry", "average",
}


def _entity_tokens(entity: str) -> list[str]:
    """The distinctive tokens of a party's name, generic corporate words removed."""
    toks = [
        t for t in re.findall(r"[a-z0-9]+", (entity or "").lower())
        if len(t) >= 3 and t not in _ENTITY_STOPWORDS
    ]
    sig = [t for t in toks if t not in _ENTITY_GENERIC_TOKENS]
    return sig or toks


def _entity_named_in_chunk(intent: ConfluenceIntent, chunk: ChunkRecord) -> bool:
    """True when the intent's third-party subject is actually NAMED in this section.

    A necessary condition for another party's value to belong here: a section can only
    be *about* a party that it names somewhere. This is the deterministic floor under
    the model's attribution judgement, and it exists because that judgement is weakest
    exactly where the damage is done — a bare table row like "| Reserve ratio | 1.4x |"
    carries no visible owner, so the model reads it as subject-less and lets a rival's
    figure in. It is also stable run to run, which the model's verdict is not.

    EVERY distinctive token must appear, so "Meridian Bharat Fund" cannot match the
    owner's own "Bharat Breakthrough Fund-I" on the shared word "bharat". A third-party
    value with no nameable subject at all matches nothing, which is the safe answer.
    """
    tokens = _entity_tokens(intent.subject_entity or "")
    if not tokens:
        return False
    haystack = f"{chunk.doc_title}\n{chunk.section_heading}\n{chunk.content}".lower()
    # Word boundaries, not substring containment. "Ace Capital" reduces to the single
    # distinctive token "ace", which a bare `in` finds inside "replace", "space" and
    # "trace" — so a section reading "replace the committed-capital rate" would satisfy
    # the floor and let Ace's figure onto the document owner's own row.
    return all(re.search(rf"\b{re.escape(t)}\b", haystack) for t in tokens)


# Spelled-out numbers and their digits are the SAME value. Without this, an editor
# asked to record "the 9-month programme" rewrites a section that already says
# "nine-month" and the diff looks real to every token-level check — a card that
# proposes no change at all, on a document that was already correct.
_NUMBER_WORDS = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
    "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
    "eleven": "11", "twelve": "12", "thirteen": "13", "fourteen": "14",
    "fifteen": "15", "sixteen": "16", "seventeen": "17", "eighteen": "18",
    "nineteen": "19", "twenty": "20", "thirty": "30", "forty": "40",
    "fifty": "50", "sixty": "60", "seventy": "70", "eighty": "80", "ninety": "90",
    "hundred": "100", "thousand": "1000", "million": "1000000",
}


def _value_tokens(text: str) -> "Counter[str]":
    """Alphanumeric tokens, with number words folded onto their digits.

    Hyphenated compounds are split so "nine-month" and "9-month" compare equal.
    """
    from collections import Counter

    raw = re.findall(r"[a-z0-9][a-z0-9$%.]*", (text or "").lower().replace("-", " "))
    return Counter(_NUMBER_WORDS.get(t, t) for t in raw)


def _meaningful_change(before: str, after: str) -> bool:
    """True only if a real value actually changed.

    Drops cards where before/after differ only in whitespace, punctuation, or the
    way a number is spelled — those are the weak, review-noise cards that propose
    nothing. Compares the multiset of value tokens.
    """
    return _value_tokens(before) != _value_tokens(after)


def _query_text(intent: ConfluenceIntent) -> str:
    return " ".join(
        p for p in (
            intent.affected_topic, intent.old_value, intent.new_value,
            " ".join(intent.verbatim_snippets or []),
        ) if p
    ).strip()


def _narrow_to_changed_lines(before: str, after: str) -> tuple[str, str]:
    """Reduce a whole-section before/after to just the line(s) that changed.

    The editor reproduces the ENTIRE section as before_content, so two edits to
    different rows of one table/section share an identical before_content and
    collapse into a single card in `_dedupe_same_row` — silently dropping all but
    one. Narrowing each edit to only its changed line(s) gives different-row
    sibling edits distinct anchors, and the review card shows just the changed
    row. The narrowed text stays a valid (tighter) anchor for the storage
    write-back — `apply_section_edit` re-derives the phrase diff and is scoped by
    the proposal's section heading regardless.
    """
    b_lines = (before or "").splitlines()
    a_lines = (after or "").splitlines()
    if not b_lines or not a_lines:
        return before, after
    sm = difflib.SequenceMatcher(a=b_lines, b=a_lines, autojunk=False)
    b_keep: list[str] = []
    a_keep: list[str] = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        b_keep.extend(b_lines[i1:i2])
        a_keep.extend(a_lines[j1:j2])
    nb = "\n".join(b_keep).strip()
    na = "\n".join(a_keep).strip()
    # Pure insertion/deletion (one side empty) or no line-level delta: keep the
    # full text so the write-back still has both sides to diff against.
    if not nb or not na:
        return before, after
    return nb, na


def _changed_anchor(before: str, after: str) -> str:
    """The normalized OLD text of the minimal word-diff between before and after.

    Used as the de-dup key so two edits to DIFFERENT cells of the SAME row (e.g.
    the Standard and Professional price in one pricing header row) — which share a
    before line but change different words — are recognised as distinct and both
    survive, while a true duplicate (same words changed) or a same-cell conflict
    collapses to the best. Empty when nothing changed.
    """
    b = (before or "").split()
    a = (after or "").split()
    sm = difflib.SequenceMatcher(a=b, b=a, autojunk=False)
    olds = [
        " ".join(b[i1:i2])
        for tag, i1, i2, _j1, _j2 in sm.get_opcodes()
        if tag != "equal"
    ]
    return " ".join(olds).strip().lower()


# ── Orchestrator ──────────────────────────────────────────────────────────────


async def propose(
    transcript_text: str,
    *,
    retriever: PineconeHybridIndex,
    config: Optional[PipelineConfig] = None,
    emit: Optional[EmitFn] = None,
) -> tuple[list[ConfluenceIntent], list[ConfluenceProposal]]:
    """Run the proven edit pipeline over Pinecone-hybrid retrieval.

    Returns ``(intents, proposals)``. ``intents`` is every extracted intent (for
    diagnostics / the meeting summary); ``proposals`` are the surviving cards.
    """
    cfg = config or PipelineConfig()

    async def _emit(stage: str) -> None:
        if emit:
            await emit(stage)

    # ── Stage 1-2: intent extraction ─────────────────────────────────────────
    await _emit("intent_extraction")
    if not (transcript_text or "").strip():
        return [], []
    # Whose documents these are. Resolved once per run (memoized for an hour), and
    # None whenever it cannot be told — in which case every prompt below is exactly
    # the prompt this pipeline used before the corpus owner was known.
    profile = await get_corpus_profile(retriever)
    owner_block = profile.prompt_block() if profile else ""
    intents = await IntentExtractionAgent().extract(transcript_text, owner=owner_block)
    if not intents:
        return [], []

    # Is the document owner even in this meeting? Extraction cannot answer that —
    # it sees one 230-word segment at a time, so it can only attribute a value when
    # the sentence itself names a party. In a meeting held entirely between OTHER
    # organizations, every unmarked "we need SOC 2" / "our competitors are X" reads
    # as the owner's own, and lands on the owner's own compliance and competitive
    # sections. One look at the whole transcript decides it for every intent at once.
    scope = await resolve_meeting_scope(transcript_text, profile)
    if not scope.owner_is_party:
        outsider = scope.outsider_entity()
        reattributed = 0
        for i in intents:
            if i.subject_scope == "internal" or (
                i.subject_scope == "unspecified" and not i.subject_entity
            ):
                # Not a new gate — just the correct attribution. "Our" said by a
                # visitor is a third party's value, and the attribution guard already
                # knows what to do with one of those: it may only land on a section
                # that is itself about that party.
                i.subject_scope = "third_party"
                i.subject_entity = i.subject_entity or outsider or None
                reattributed += 1
        logger.info(
            "confluence_pipeline.propose: %r is not a party to this meeting (speakers "
            "represent %s) — re-attributed %d/%d intent(s) away from the document owner",
            profile.owner_name if profile else None,
            scope.speaker_organizations or "[unknown]", reattributed, len(intents),
        )

    edit_intents = [
        i for i in intents
        if classify_kind(i) == "edit" and _has_concrete_value(i)
    ]
    if cfg.reaffirmation_guard:
        kept: list[ConfluenceIntent] = []
        for i in edit_intents:
            # Either signal is enough to reject: the model's own label, or the
            # speaker's quoted words. They fail independently — the label slips on the
            # contrastive form, the lexical markers are absent when the speaker
            # reaffirms implicitly — so neither alone closes the gap.
            by_label = i.change_polarity == "reaffirmation"
            by_words = is_reaffirmation_phrasing(i)
            # Asking about a document is not changing it — same family of failure as
            # a reaffirmation (a value stated with no intent to move it), so it is
            # dropped at the same point, before retrieval can find it a home.
            if is_information_request(i):
                logger.info(
                    "confluence_pipeline: question guard — %r (%s) came from a request "
                    "for information, not a change; dropped",
                    i.affected_topic, i.new_value,
                )
                continue
            if by_label or by_words:
                logger.info(
                    "confluence_pipeline: reaffirmation guard — %r (%s) restates an "
                    "existing value, not a change (label=%s, quoted_words=%s); dropped",
                    i.affected_topic, i.new_value, by_label, by_words,
                )
                continue
            kept.append(i)
        edit_intents = kept
    logger.info("confluence_pipeline.propose: %d intents (%d editable)", len(intents), len(edit_intents))
    if not edit_intents:
        return intents, []

    # ── Stage 3-4: per-intent hybrid retrieval ───────────────────────────────
    await _emit("rag_retrieval")

    async def _retrieve(intent: ConfluenceIntent) -> tuple[ConfluenceIntent, list[ChunkRecord], int]:
        cross = is_cross_cutting(intent)
        top_k = cfg.cross_cutting_top_k if cross else cfg.retrieval_top_k
        max_targets = cfg.cross_cutting_max_targets if cross else cfg.max_targets_per_intent
        chunks = await asyncio.to_thread(retriever.query, _query_text(intent), top_k)
        return intent, chunks, max_targets

    retrieved = await asyncio.gather(*[_retrieve(i) for i in edit_intents])

    # Retrieval now applies an absolute relevance floor, so an empty candidate list is
    # a real verdict: this corpus documents nothing about that intent. Drop those here
    # rather than paying a stage-1 fan-out per intent to rediscover it — on a meeting
    # that has nothing to do with these documents that is the entire eval bill.
    intent_candidates = [t for t in retrieved if t[1]]
    ungrounded = len(retrieved) - len(intent_candidates)
    logger.info(
        "confluence_pipeline.propose: %d editable intent(s) — %d grounded in the corpus, "
        "%d with no section about them (owner=%r)",
        len(retrieved), len(intent_candidates), ungrounded,
        profile.owner_name if profile else None,
    )
    if not intent_candidates:
        return intents, []

    # ── Stage 5: two-stage evaluation (cheap wide filter → strong precise gate) ─
    await _emit("evaluation")
    eval_cheap = EvaluationAgent(
        model=os.getenv("LDOC_EVAL_STAGE1_MODEL", "gpt-4o-mini"), temperature=0.0, owner=owner_block,
    )
    eval_fine = EvaluationAgent(owner=owner_block)  # LDOC_EVAL_MODEL, default gpt-5.4-mini

    async def _score_pool(intent: ConfluenceIntent, chunks: list[ChunkRecord], max_targets: int):
        if not chunks:
            return []
        # Stage 1: cheap model scores the WHOLE pool, only to rank/shortlist.
        cheap = await asyncio.gather(*[eval_cheap.score(intent, c) for c in chunks])
        order = sorted(range(len(chunks)), key=lambda i: cheap[i], reverse=True)
        # Hand the strong model the top survivors (>= max_targets so cross-cutting
        # intents aren't starved), plus any deterministic field-label match the cheap
        # model must not be allowed to drop.
        n_survivors = max(cfg.eval_stage2_top_k, max_targets)
        survivors_idx = set(order[:n_survivors])
        for i, c in enumerate(chunks):
            if _field_label_match(intent, c):
                survivors_idx.add(i)
        survivors = [chunks[i] for i in sorted(survivors_idx)]
        # Stage 2: the strong model re-scores only the shortlist (the precision gate).
        # Only the STRONG model's attribution verdict is enforced — stage 1 stays a
        # pure cheap ranker, as designed.
        fine = await asyncio.gather(*[eval_fine.score_detail(intent, c) for c in survivors])
        scores: list[float] = []
        rescue_floor = cfg.relevance_threshold - cfg.floor_rescue_margin
        gate_attribution = cfg.attribution_guard and intent.subject_scope == "third_party"
        for c, verdict in zip(survivors, fine):
            s = verdict.relevance_score
            # Subject-attribution guardrail: this value was stated about someone else,
            # so it may only land on a section that is about that party. Both checks
            # must agree, because either alone leaks — the deterministic name check
            # holds the line on subject-less table rows (where the model's judgement is
            # weakest), while the model's verdict distinguishes "names the party" from
            # "is about the party" (a passing mention is not a case file). Zeroing,
            # rather than merely lowering, also denies a rejected candidate the
            # deterministic rescue below: a competitor's figure reliably trips the
            # field-label and phrase-overlap floors precisely because it IS the same
            # kind of value.
            # Only computed when the gate will actually consult it: the name floor
            # lowercases the whole section body (up to 12k chars) per candidate, and
            # for an internal intent — the normal case — the answer is discarded.
            named_here = gate_attribution and _entity_named_in_chunk(intent, c)
            if gate_attribution and not (verdict.subject_match and named_here):
                logger.info(
                    "confluence_pipeline: attribution gate — %r (%s) is about %r, "
                    "section %r is about %r (named_here=%s, model_match=%s); rejected",
                    intent.affected_topic, intent.new_value, intent.subject_entity,
                    c.section_heading, verdict.section_subject,
                    named_here, verdict.subject_match,
                )
                scores.append(0.0)
                continue
            # Confirm a NEAR-MISS only: the model scored it just under the gate AND a
            # deterministic match backs it up → tip it over. A clearly-low score
            # (< rescue_floor) is a confident rejection and is left to fail, even with
            # a lexical match (almost always coincidental there).
            if rescue_floor <= s < cfg.relevance_threshold and (
                _field_label_match(intent, c) or _phrase_overlap_match(intent, c)
            ):
                s = cfg.relevance_threshold
            scores.append(s)
        ranked = sorted(zip(survivors, scores), key=lambda cs: cs[1], reverse=True)
        kept: list[tuple[ConfluenceIntent, ChunkRecord]] = []
        seen: set[tuple[str, str]] = set()
        for chunk, score in ranked:
            if score < cfg.relevance_threshold:
                continue
            key = (chunk.source_path, chunk.section_heading)
            if key in seen:
                continue
            seen.add(key)
            kept.append((intent, chunk))
            if len(kept) >= max_targets:
                break
        return kept

    eval_results = await asyncio.gather(
        *[_score_pool(i, chunks, mt) for i, chunks, mt in intent_candidates]
    )
    qualified = [pair for rl in eval_results for pair in rl]
    if not qualified:
        return intents, []

    # ── Stage 6: drafting ────────────────────────────────────────────────────
    await _emit("drafting")
    editor = ConfluenceEditorAgent()
    drafts = await asyncio.gather(*[editor.draft(i, c.content) for i, c in qualified])

    # ── Stage 7: verification ────────────────────────────────────────────────
    await _emit("verification")
    verifier = VerifierAgent()
    verifications = await asyncio.gather(*[
        verifier.verify(
            d.before_content,
            d.after_content,
            f"{i.intent_type}: {i.new_value}",
            spoken_context=" ".join(i.verbatim_snippets or []),
            subject_entity=i.subject_entity,
            subject_scope=i.subject_scope,
            doc_title=c.doc_title,
            section_heading=c.section_heading,
            owner=owner_block,
        )
        for (i, c), d in zip(qualified, drafts)
    ])

    # ── Stage 8: assemble, suppress no-ops + unfulfilled ─────────────────────
    await _emit("ready_for_review")
    proposals: list[ConfluenceProposal] = []
    for (intent, chunk), draft, ver in zip(qualified, drafts, verifications):
        # Narrow the editor's whole-section draft to just the changed line(s) so
        # sibling edits to the same section no longer collide in _dedupe_same_row.
        before, after = _narrow_to_changed_lines(draft.before_content, draft.after_content)
        if after.strip() == before.strip() or not _meaningful_change(before, after):
            logger.info("confluence_pipeline.propose: dropping no-op/trivial edit on %r", chunk.section_heading)
            continue
        if ver.intent_fulfillment < cfg.min_fulfillment:
            logger.info("confluence_pipeline.propose: dropping unfulfilled edit on %r (%.2f)",
                        chunk.section_heading, ver.intent_fulfillment)
            continue
        if ver.factual_consistency < cfg.min_factual:
            logger.info("confluence_pipeline.propose: dropping factually-weak edit on %r (%.2f)",
                        chunk.section_heading, ver.factual_consistency)
            continue
        # Attribution backstop. Runs on EVERY draft, not just third_party intents:
        # its job is to catch the value whose attribution the extractor never
        # recorded, which the evaluation-stage gate therefore never examined.
        if cfg.attribution_guard and ver.attribution_fit < cfg.min_attribution:
            logger.info(
                "confluence_pipeline.propose: dropping mis-attributed edit on %r "
                "(attribution %.2f) — %r is not said about this section's subject",
                chunk.section_heading, ver.attribution_fit, intent.new_value,
            )
            continue
        edit_type = draft.edit_type if draft.edit_type in ("replace", "append", "delete_section") else "replace"
        prop = ConfluenceProposal.create(cfg.session_id, intent, chunk)
        prop.before_content = before
        prop.after_content = after
        prop.edit_type = edit_type
        prop.factual_consistency = ver.factual_consistency
        prop.formatting_integrity = ver.formatting_integrity
        prop.intent_fulfillment = ver.intent_fulfillment
        prop.quality_score = ver.quality_score
        prop.confidence = ver.quality_score
        prop.verifier_note = ver.verifier_note
        proposals.append(prop)

    # ── Stage 9: collapse same-row collisions ────────────────────────────────
    # Different intents can land on the SAME row (e.g. standard- and enterprise-
    # response-time both rewriting the one P1 row, or a method edit and a no-op
    # twin on the same classifier row). For `replace` edits sharing identical
    # before_content, keep only the highest-quality draft. Appends never collide
    # on a before-row, so they pass through untouched.
    proposals = _dedupe_same_row(proposals)

    logger.info("confluence_pipeline.propose: %d proposal(s) from %d qualified section(s)",
                len(proposals), len(qualified))
    return intents, proposals


def _dedupe_same_row(proposals: list[ConfluenceProposal]) -> list[ConfluenceProposal]:
    """Keep the best `replace` per (page, changed-phrase); pass appends through.

    Keying on the CHANGED PHRASE (not the whole before-row) is what lets two edits
    to different cells of the SAME row both survive — e.g. changing the Standard
    AND the Professional price in one pricing header row, which share a before line
    but alter different words. A true duplicate (same words changed) or a same-cell
    conflict still shares an anchor and collapses to the highest-quality draft.
    """
    best: dict[tuple[str, str], ConfluenceProposal] = {}
    passthrough: list[ConfluenceProposal] = []
    for p in proposals:
        if p.edit_type != "replace" or not (p.before_content or "").strip():
            passthrough.append(p)
            continue
        anchor = _changed_anchor(p.before_content or "", p.after_content or "")
        if not anchor:  # no detectable change — fall back to the whole before-row
            anchor = " ".join((p.before_content or "").split()).lower()
        key = (p.source_chunk.source_path, anchor)
        cur = best.get(key)
        if cur is None or (p.quality_score, p.confidence) > (cur.quality_score, cur.confidence):
            best[key] = p
    # Preserve original order: walk proposals, emit each kept item once.
    kept = set(id(p) for p in best.values()) | set(id(p) for p in passthrough)
    return [p for p in proposals if id(p) in kept]
