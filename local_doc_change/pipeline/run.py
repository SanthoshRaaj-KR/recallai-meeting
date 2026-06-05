"""Pipeline orchestrator for the local document change pipeline.

Orchestrates all 8 stages:
  1. transcript_source  — validate transcript
  2. intent_extraction  — extract document change intents via IntentExtractionAgent
  3. rag_indexing       — build or load the BM25/FAISS document index
  4. rag_retrieval      — retrieve candidate sections per intent via HybridRetriever
  5. evaluation         — score sections with EvaluationAgent; filter by threshold
  6. drafting           — draft before/after edits via LocalDocEditorAgent
  7. verification       — verify edit quality via VerifierAgent
  8. ready_for_review   — assemble and return LocalDocProposal list

Each extracted intent is routed by kind (see agents_local.structural):
  - "edit"    → retrieve → evaluate → draft → verify (localized change). An edit
                intent may now produce MULTIPLE proposals when several sections
                qualify (e.g. one change that affects two documents).
  - "rename"  → literal corpus-wide value replacement: one proposal per section
                that mentions the old value.
  - "removal" → whole-section deletion: resolve which ordered sections the
                instruction targets, one delete proposal per section.

Progress is reported via an asyncio.Queue (one stage name per message).
None sentinel closes the stream when the pipeline finishes or errors.
"""

from __future__ import annotations

import asyncio
import datetime
import logging
import os
import re
import uuid
from typing import Optional

import openai
from pydantic import BaseModel

from models import LocalDocProposal
from models.rag import ChunkRecord

logger = logging.getLogger(__name__)

# ── Stage names (must match test_pipeline.py exactly) ────────────────────────

PIPELINE_STAGES: list[str] = [
    "transcript_source",
    "intent_extraction",
    "rag_indexing",
    "rag_retrieval",
    "evaluation",
    "drafting",
    "verification",
    "ready_for_review",
]


# ── PipelineConfig ────────────────────────────────────────────────────────────


class PipelineConfig(BaseModel):
    """Configuration for a single run_pipeline() invocation."""

    session_id: str
    doc_folder: str
    use_embeddings: bool = True
    rerank: bool = True
    contextual_retrieval: bool = True
    top_k: int = 3
    relevance_threshold: float = 0.7
    # How many candidate sections to pull per edit intent before evaluation.
    # Wider than top_k so cross-section / cross-document changes can surface.
    retrieval_top_k: int = 8
    # Cap on how many sections a single edit intent may change (scalability).
    max_targets_per_intent: int = 4
    # Drafts below this intent-fulfilment score are dropped as failed (no-op /
    # unfulfilled) rather than shown as a misleading accept-able card.
    min_fulfillment: float = 0.4
    skip_contextual: bool = False   # test flag: skip GPT-4o-mini context enrichment
    openai_api_key: Optional[str] = None  # falls back to OPENAI_API_KEY env var


# ── Internal helpers ──────────────────────────────────────────────────────────


async def _emit(q: Optional[asyncio.Queue], stage: str) -> None:
    """Emit a stage name to the progress queue (no-op if queue is None)."""
    if q is not None:
        await q.put(stage)


def _utcnow() -> str:
    return datetime.datetime.utcnow().isoformat() + "Z"


def _get_openai_client(config: PipelineConfig) -> openai.AsyncOpenAI:
    """Return an AsyncOpenAI client using config.openai_api_key or env var."""
    return openai.AsyncOpenAI(
        api_key=config.openai_api_key or os.getenv("OPENAI_API_KEY")
    )


def _ordered_sections(chunks: list[ChunkRecord], source_path: str) -> list[ChunkRecord]:
    """All chunks for one document, in document order, one per heading."""
    by_heading: dict[str, ChunkRecord] = {}
    for c in chunks:
        if c.source_path != source_path:
            continue
        # Keep the first chunk per heading (oversized sections split into .0/.1).
        if c.section_heading not in by_heading:
            by_heading[c.section_heading] = c
    return sorted(by_heading.values(), key=lambda c: c.section_index)


def _make_proposal(
    config: PipelineConfig,
    intent,
    chunk: ChunkRecord,
    before_content: str,
    after_content: str,
    edit_type: str,
    *,
    factual: float,
    formatting: float,
    fulfillment: float,
    quality: float,
    note: str,
) -> LocalDocProposal:
    return LocalDocProposal(
        proposal_id=str(uuid.uuid4()),
        session_id=config.session_id,
        intent=intent,
        source_chunk=chunk,
        before_content=before_content,
        after_content=after_content,
        edit_type=edit_type,
        confidence=quality,
        factual_consistency=factual,
        formatting_integrity=formatting,
        intent_fulfillment=fulfillment,
        quality_score=quality,
        verifier_note=note,
        status="pending",
        created_at=_utcnow(),
    )


# ── Pipeline orchestrator ─────────────────────────────────────────────────────


async def run_pipeline(
    transcript: str,
    config: PipelineConfig,
    progress_queue: Optional[asyncio.Queue] = None,
) -> list[LocalDocProposal]:
    """Run the full 8-stage local document change pipeline.

    Returns proposals ready for human review. Empty list if no qualifying edits
    are found.
    """
    q = progress_queue

    # ── Stage 1: transcript_source ────────────────────────────────────────────
    await _emit(q, "transcript_source")
    if not transcript or not transcript.strip():
        logger.info("run_pipeline: empty transcript — no proposals to generate")
        await _emit(q, "ready_for_review")
        return []

    # ── Stage 2: intent_extraction ────────────────────────────────────────────
    await _emit(q, "intent_extraction")
    from agents_local.intent_extraction import IntentExtractionAgent  # noqa: PLC0415
    from agents_local.structural import classify_kind  # noqa: PLC0415

    intent_agent = IntentExtractionAgent()
    intents = await intent_agent.extract(transcript)
    if not intents:
        logger.info("run_pipeline: no intents extracted from transcript")
        await _emit(q, "ready_for_review")
        return []

    # Route each intent by kind.
    edit_intents = [i for i in intents if classify_kind(i) == "edit"]
    rename_intents = [i for i in intents if classify_kind(i) == "rename"]
    removal_intents = [i for i in intents if classify_kind(i) == "removal"]
    logger.info(
        "run_pipeline: %d intents (edit=%d rename=%d removal=%d)",
        len(intents), len(edit_intents), len(rename_intents), len(removal_intents),
    )

    # ── Stage 3: rag_indexing ─────────────────────────────────────────────────
    await _emit(q, "rag_indexing")
    from rag.indexer import build_index  # noqa: PLC0415

    openai_client = _get_openai_client(config) if config.use_embeddings else None
    index = build_index(
        folder_path=config.doc_folder,
        use_embeddings=config.use_embeddings,
        contextual_retrieval=config.contextual_retrieval and not config.skip_contextual,
        openai_client=openai_client,
    )
    all_chunks: list[ChunkRecord] = index.chunks

    # ── Stage 4: rag_retrieval ────────────────────────────────────────────────
    await _emit(q, "rag_retrieval")
    from rag.retriever import HybridRetriever  # noqa: PLC0415

    retriever = HybridRetriever(index, rerank=config.rerank)

    # Edit intents: collect candidate sections (wide net for recall).
    intent_candidates: list[tuple] = []
    for intent in edit_intents:
        query_text = " ".join(
            p for p in (intent.affected_topic, intent.old_value, intent.new_value) if p
        ).strip()
        results = retriever.query(query_text, top_k=config.retrieval_top_k)
        intent_candidates.append((intent, [r.chunk for r in results]))

    # Removal intents: pick the target document, resolve which sections to drop.
    removal_targets: list[tuple] = []  # (intent, chunk) per section to delete
    if removal_intents:
        from agents_local.structural import RemovalResolverAgent  # noqa: PLC0415

        resolver = RemovalResolverAgent()

        # Positional removals ("the last 4 points", "sections 5-8") need the
        # ordered section list and an LLM to resolve the position. Named/topical
        # removals ("no benefits", "drop the travel section") map directly to the
        # strongest-retrieved section, which is far more reliable than asking an
        # LLM to connect loose wording to a specific heading.
        _POSITIONAL_RE = re.compile(
            r"\b(last|first|final|bottom|top|next|preceding|following)\b"
            r"|\d+\s*(to|through|thru|[-–])\s*\d+|sections?\s+\d",
            re.IGNORECASE,
        )

        async def _plan_removal(intent) -> list[tuple]:
            snippet = " ".join(intent.verbatim_snippets or []) or intent.new_value or ""
            instruction = (
                f'Instruction: "{snippet}". '
                f"Topic: {intent.affected_topic}. "
                f"Intended change: {intent.new_value}."
            )
            results = retriever.query(
                f"{intent.affected_topic} {intent.new_value}".strip(),
                top_k=config.retrieval_top_k,
            )
            if not results:
                return []
            target_path = results[0].chunk.source_path
            sections = _ordered_sections(all_chunks, target_path)
            headings = [c.section_heading for c in sections]
            by_heading = {c.section_heading: c for c in sections}

            text = f"{instruction} {snippet}"
            is_positional = bool(_POSITIONAL_RE.search(text))

            to_delete = await resolver.resolve(instruction, headings)
            if to_delete:
                return [(intent, by_heading[h]) for h in to_delete if h in by_heading]

            # Named removal with no resolver hit: delete the strongest matching
            # content section in the target document (skip a leading title/intro).
            if not is_positional:
                for r in results:
                    if r.chunk.source_path != target_path:
                        continue
                    if r.chunk.section_index == 0 and len(sections) > 1:
                        continue  # avoid deleting the document title/intro
                    return [(intent, r.chunk)]
            return []

        removal_plans = await asyncio.gather(
            *[_plan_removal(i) for i in removal_intents]
        )
        removal_targets = [pair for plan in removal_plans for pair in plan]

    # ── Stage 5: evaluation ───────────────────────────────────────────────────
    await _emit(q, "evaluation")
    from agents_local.evaluation import EvaluationAgent  # noqa: PLC0415

    eval_agent = EvaluationAgent()

    async def _score_candidates(intent, chunks: list[ChunkRecord]) -> list[tuple]:
        """Keep EVERY section that scores above threshold (deduped, capped)."""
        if not chunks:
            return []
        scores = await asyncio.gather(*[eval_agent.score(intent, c) for c in chunks])
        ranked = sorted(zip(chunks, scores), key=lambda cs: cs[1], reverse=True)
        kept: list[tuple] = []
        seen: set[tuple[str, str]] = set()
        for chunk, score in ranked:
            if score < config.relevance_threshold:
                continue
            key = (chunk.source_path, chunk.section_heading)
            if key in seen:
                continue
            seen.add(key)
            kept.append((intent, chunk))
            if len(kept) >= config.max_targets_per_intent:
                break
        return kept

    eval_results = await asyncio.gather(
        *[_score_candidates(intent, chunks) for intent, chunks in intent_candidates]
    )
    qualified: list[tuple] = [pair for rl in eval_results for pair in rl]

    # ── Stage 6: drafting ─────────────────────────────────────────────────────
    await _emit(q, "drafting")
    from agents_local.editor import LocalDocEditorAgent  # noqa: PLC0415

    editor = LocalDocEditorAgent()
    drafts = await asyncio.gather(
        *[editor.draft(intent, chunk.content) for intent, chunk in qualified]
    ) if qualified else []

    # ── Stage 7: verification ─────────────────────────────────────────────────
    await _emit(q, "verification")
    from agents_local.verifier import VerifierAgent  # noqa: PLC0415

    verifier = VerifierAgent()
    verifications = await asyncio.gather(
        *[
            verifier.verify(
                draft.before_content,
                draft.after_content,
                f"{intent.intent_type}: {intent.new_value}",
            )
            for (intent, chunk), draft in zip(qualified, drafts)
        ]
    ) if qualified else []

    # ── Stage 8: ready_for_review ─────────────────────────────────────────────
    await _emit(q, "ready_for_review")
    proposals: list[LocalDocProposal] = []

    # Edit proposals — suppress no-ops and clearly-unfulfilled drafts.
    for (intent, chunk), draft, verification in zip(qualified, drafts, verifications):
        before = draft.before_content
        after = draft.after_content
        if after.strip() == before.strip():
            logger.info("run_pipeline: dropping no-op edit on %r", chunk.section_heading)
            continue
        if verification.intent_fulfillment < config.min_fulfillment:
            logger.info(
                "run_pipeline: dropping unfulfilled edit on %r (fulfillment=%.2f)",
                chunk.section_heading, verification.intent_fulfillment,
            )
            continue
        proposals.append(
            _make_proposal(
                config, intent, chunk, before, after,
                edit_type=draft.edit_type if draft.edit_type in
                ("replace", "append", "delete_section") else "replace",
                factual=verification.factual_consistency,
                formatting=verification.formatting_integrity,
                fulfillment=verification.intent_fulfillment,
                quality=verification.quality_score,
                note=verification.verifier_note,
            )
        )

    # Rename proposals — deterministic literal replacement across the corpus.
    if rename_intents:
        from agents_local.structural import (  # noqa: PLC0415
            find_rename_targets,
            infer_replacement_target,
        )

        inferred: Optional[str] = None
        for intent in rename_intents:
            new_value = (intent.new_value or "").strip()
            if not new_value:
                continue
            old_value = (intent.old_value or "").strip()
            # If the transcript gave no old value (or one that isn't in the
            # docs), infer the dominant cross-document brand name.
            if not old_value or not find_rename_targets(old_value, all_chunks):
                if inferred is None:
                    inferred = infer_replacement_target(all_chunks) or ""
                old_value = inferred
            if not old_value:
                continue
            pattern = re.compile(re.escape(old_value), re.IGNORECASE)
            for chunk, original in find_rename_targets(
                old_value, all_chunks, max_targets=config.max_targets_per_intent * 3
            ):
                new_body = pattern.sub(new_value, original)
                if new_body == original:
                    continue
                proposals.append(
                    _make_proposal(
                        config, intent, chunk, original, new_body,
                        edit_type="replace",
                        factual=1.0, formatting=1.0, fulfillment=1.0, quality=0.9,
                        note=(
                            f"Replaces '{old_value}' with '{new_value}' in this "
                            f"section (deterministic rename)."
                        ),
                    )
                )

    # Removal proposals — whole-section deletions.
    seen_removals: set[tuple[str, str]] = set()
    for intent, chunk in removal_targets:
        key = (chunk.source_path, chunk.section_heading)
        if key in seen_removals:
            continue
        seen_removals.add(key)
        proposals.append(
            _make_proposal(
                config, intent, chunk, chunk.content, "",
                edit_type="delete_section",
                factual=1.0, formatting=1.0, fulfillment=1.0, quality=0.9,
                note="Proposes removing this entire section.",
            )
        )

    logger.info(
        "run_pipeline: completed — session=%s proposals=%d",
        config.session_id,
        len(proposals),
    )
    return proposals
