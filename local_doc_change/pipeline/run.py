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

Progress is reported via an asyncio.Queue (one stage name per message).
None sentinel closes the stream when the pipeline finishes or errors.
"""

from __future__ import annotations

import asyncio
import datetime
import logging
import os
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


# ── Pipeline orchestrator ─────────────────────────────────────────────────────


async def run_pipeline(
    transcript: str,
    config: PipelineConfig,
    progress_queue: Optional[asyncio.Queue] = None,
) -> list[LocalDocProposal]:
    """Run the full 8-stage local document change pipeline.

    Parameters
    ----------
    transcript:
        Meeting transcript text to extract intents from.
    config:
        Pipeline configuration (doc_folder, model flags, etc.).
    progress_queue:
        Optional asyncio.Queue; each stage name is put() before that stage runs.
        A None sentinel is put() when the pipeline finishes (or errors).

    Returns
    -------
    list[LocalDocProposal]
        Proposals ready for human review. Empty list if no qualifying edits found.
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

    intent_agent = IntentExtractionAgent()
    intents = await intent_agent.extract(transcript)
    if not intents:
        logger.info("run_pipeline: no intents extracted from transcript")
        await _emit(q, "ready_for_review")
        return []

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

    # ── Stage 4: rag_retrieval ────────────────────────────────────────────────
    await _emit(q, "rag_retrieval")
    from rag.retriever import HybridRetriever  # noqa: PLC0415

    retriever = HybridRetriever(index, rerank=config.rerank)

    # Collect (intent, candidates) pairs
    intent_candidates: list[tuple] = []
    for intent in intents:
        query_text = f"{intent.affected_topic} {intent.new_value or ''}".strip()
        results = retriever.query(query_text, top_k=config.top_k)
        chunks = [r.chunk for r in results]
        intent_candidates.append((intent, chunks))

    # ── Stage 5: evaluation ───────────────────────────────────────────────────
    await _emit(q, "evaluation")
    from agents_local.evaluation import EvaluationAgent  # noqa: PLC0415

    eval_agent = EvaluationAgent()

    async def _score_candidates(intent, chunks: list[ChunkRecord]) -> list[tuple]:
        """Score all chunks for an intent concurrently; return qualifying pairs."""
        if not chunks:
            return []
        scores = await asyncio.gather(*[eval_agent.score(intent, chunk) for chunk in chunks])
        qualified = [
            (intent, chunk)
            for chunk, score in zip(chunks, scores)
            if score >= config.relevance_threshold
        ]
        if qualified:
            # Take the best-scoring chunk only (highest score)
            best_idx = max(range(len(chunks)), key=lambda i: scores[i])
            if scores[best_idx] >= config.relevance_threshold:
                return [(intent, chunks[best_idx])]
        return []

    # Run evaluation concurrently for all intents
    eval_results = await asyncio.gather(
        *[_score_candidates(intent, chunks) for intent, chunks in intent_candidates]
    )
    qualified: list[tuple] = [pair for result_list in eval_results for pair in result_list]

    if not qualified:
        logger.info("run_pipeline: no sections qualified above threshold %.2f", config.relevance_threshold)
        await _emit(q, "ready_for_review")
        return []

    # ── Stage 6: drafting ─────────────────────────────────────────────────────
    await _emit(q, "drafting")
    from agents_local.editor import LocalDocEditorAgent  # noqa: PLC0415

    editor = LocalDocEditorAgent()
    drafts = await asyncio.gather(
        *[editor.draft(intent, chunk.content) for intent, chunk in qualified]
    )

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
    )

    # ── Stage 8: ready_for_review ─────────────────────────────────────────────
    await _emit(q, "ready_for_review")
    proposals: list[LocalDocProposal] = []
    for (intent, chunk), draft, verification in zip(qualified, drafts, verifications):
        proposal = LocalDocProposal(
            proposal_id=str(uuid.uuid4()),
            session_id=config.session_id,
            intent=intent,
            source_chunk=chunk,
            before_content=draft.before_content,
            after_content=draft.after_content,
            confidence=verification.quality_score,
            factual_consistency=verification.factual_consistency,
            formatting_integrity=verification.formatting_integrity,
            intent_fulfillment=verification.intent_fulfillment,
            quality_score=verification.quality_score,
            verifier_note=verification.verifier_note,
            status="pending",
            created_at=_utcnow(),
        )
        proposals.append(proposal)

    logger.info(
        "run_pipeline: completed — session=%s proposals=%d",
        config.session_id,
        len(proposals),
    )
    return proposals
