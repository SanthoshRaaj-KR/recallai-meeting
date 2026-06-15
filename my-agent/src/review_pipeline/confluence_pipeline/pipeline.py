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
import logging
import re
from collections.abc import Awaitable, Callable
from typing import Any, Optional

from pydantic import BaseModel

from .editor import ConfluenceEditorAgent
from .evaluation import EvaluationAgent
from .intent_extraction import IntentExtractionAgent
from .models import ChunkRecord, ConfluenceIntent, ConfluenceProposal
from .retrieval import PineconeHybridIndex
from .structural import classify_kind, is_cross_cutting
from .verifier import VerifierAgent

logger = logging.getLogger(__name__)

EmitFn = Callable[[str], Awaitable[None]]


class PipelineConfig(BaseModel):
    session_id: str = "confluence"
    relevance_threshold: float = 0.70
    retrieval_top_k: int = 20
    max_targets_per_intent: int = 1
    cross_cutting_top_k: int = 40
    cross_cutting_max_targets: int = 30
    # Precision-first: drop drafts the verifier isn't confident actually applied
    # the change. Correct edits score ~0.85-1.0; 0.6 cuts the weak/uncertain ones
    # (the user prefers a missed change over a wrong card).
    min_fulfillment: float = 0.6
    # Drop cards whose draft fails verification on factual consistency.
    min_factual: float = 0.6


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

_LABEL_FLOOR = 0.85
_PHRASE_FLOOR = 0.80
_PHRASE_LLM_GATE = 0.40


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


def _meaningful_change(before: str, after: str) -> bool:
    """True only if a real token actually changed.

    Drops cards where before/after differ only in whitespace or punctuation (a
    near-no-op that slipped past the strict strip() check) — those are the weak,
    review-noise cards. Compares the multiset of alphanumeric/value tokens.
    """
    from collections import Counter

    pat = r"[a-z0-9][a-z0-9$%./-]*"
    b = Counter(re.findall(pat, (before or "").lower()))
    a = Counter(re.findall(pat, (after or "").lower()))
    return b != a


def _query_text(intent: ConfluenceIntent) -> str:
    return " ".join(
        p for p in (
            intent.affected_topic, intent.old_value, intent.new_value,
            " ".join(intent.verbatim_snippets or []),
        ) if p
    ).strip()


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
    intents = await IntentExtractionAgent().extract(transcript_text)
    if not intents:
        return [], []

    edit_intents = [
        i for i in intents
        if classify_kind(i) == "edit" and _has_concrete_value(i)
    ]
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

    intent_candidates = await asyncio.gather(*[_retrieve(i) for i in edit_intents])

    # ── Stage 5: evaluation (LLM score + deterministic recall floors) ────────
    await _emit("evaluation")
    eval_agent = EvaluationAgent()

    async def _score_pool(intent: ConfluenceIntent, chunks: list[ChunkRecord], max_targets: int):
        if not chunks:
            return []
        raw = await asyncio.gather(*[eval_agent.score(intent, c) for c in chunks])
        scores: list[float] = []
        for c, s in zip(chunks, raw):
            if _field_label_match(intent, c):
                s = max(s, _LABEL_FLOOR)
            elif s >= _PHRASE_LLM_GATE and _phrase_overlap_match(intent, c):
                s = max(s, _PHRASE_FLOOR)
            scores.append(s)
        ranked = sorted(zip(chunks, scores), key=lambda cs: cs[1], reverse=True)
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
        verifier.verify(d.before_content, d.after_content, f"{i.intent_type}: {i.new_value}")
        for (i, _c), d in zip(qualified, drafts)
    ])

    # ── Stage 8: assemble, suppress no-ops + unfulfilled ─────────────────────
    await _emit("ready_for_review")
    proposals: list[ConfluenceProposal] = []
    for (intent, chunk), draft, ver in zip(qualified, drafts, verifications):
        before, after = draft.before_content, draft.after_content
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

    logger.info("confluence_pipeline.propose: %d proposal(s) from %d qualified section(s)",
                len(proposals), len(qualified))
    return intents, proposals
