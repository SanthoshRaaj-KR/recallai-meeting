"""Deterministic pipeline orchestrator (ARCH-V3-01).

Wires the seven stages in order with bounded per-intent fan-out and per-stage
StageTrace emission (OBS-V3-01):

  Stage 0: transcript_source.load_transcript   — sets ctx.transcript_text
  Stage 1: extract.extract_intents             — ChangeIntentV3[]
  Stage 2: retrieve.retrieve_candidates        — per-intent (fan-out)
  Stage 3: rerank.rerank_candidates            — per-intent (fan-out)
  Stage 4: iterate.iterative_retrieve          — per-intent (fan-out, low-confidence only)
  Stage 5: contradict.detect_contradictions    — fact_update intents only
  Stage 6: plan_ops.plan_operation             — per (intent, candidate) pair
  Stage 7: gate.apply_grounding_gate_v3        — per PlannedOperation

Control flow is plain Python (Pattern 4 — no LLM supervisor).

LLM agents live INSIDE individual stages; ``run`` only coordinates.

SECURITY (T-11-19, T-11-20, T-11-21):
  * graph_user_id is read from ctx — NEVER from a ContextVar inside this module.
  * Broad top-level try/except: a stage exception is caught, logged, and
    recorded in the context; it NEVER propagates to the event loop.
  * Incremental persist of proposals by (session_id, dedup_key) makes re-runs
    idempotent.

STAGE IMPORTS AT MODULE LEVEL (test seam):
  All stage callables are imported at module level so tests can patch them
  via ``monkeypatch.setattr(run, "<symbol>", stub)`` OR by patching the
  symbol in both its home module AND this module (mirror the e2e_eval pattern
  from e2e_proposal_quality_v2_eval._swap).

See RESEARCH §Orchestration Design and §Pattern 4 for the full rationale.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
import uuid
from typing import Any, Dict, List, Optional

from confluence_logic.pipeline.contracts import (
    ChangeIntentV3,
    ContradictionGroup,
    PlannedOperation,
    ProposalCardV3,
    SectionCandidate,
    StageTrace,
)
from confluence_logic.pipeline.context import PipelineContext

# ---------------------------------------------------------------------------
# Stage imports at module level (TEST SEAM — patch these symbols in tests)
# ---------------------------------------------------------------------------

# Stage 0 — transcript source
from confluence_logic.pipeline.stages.transcript_source import (
    load_transcript,
)

# Stage 1 — extraction
from confluence_logic.pipeline.stages.extract import (
    extract_intents,
)

# Stage 2 — retrieval
from confluence_logic.pipeline.stages.retrieve import (
    retrieve_candidates,
)

# Stage 3 — rerank
from confluence_logic.pipeline.stages.rerank import (
    rerank_candidates,
)

# Stage 4 — bounded iterative retrieval
from confluence_logic.pipeline.stages.iterate import (
    iterative_retrieve,
)

# Stage 5 — contradiction sweep
from confluence_logic.pipeline.stages.contradict import (
    detect_contradictions,
)

# Stage 6 — operation planning
from confluence_logic.pipeline.stages.plan_ops import (
    plan_operation,
)

# Stage 7 — grounding + confidence gate
from confluence_logic.pipeline.stages.gate import (
    apply_grounding_gate_v3,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Concurrency cap (lifted from review/api.py _run_pipeline)
# ---------------------------------------------------------------------------

_DRAFTER_CONCURRENCY: int = int(os.getenv("JARVIS_DRAFTER_CONCURRENCY", "6"))


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _emit_trace(ctx: PipelineContext, trace: StageTrace) -> None:
    """Emit a StageTrace via ctx.trace_bus; no-op if trace_bus is None."""
    try:
        bus = getattr(ctx, "trace_bus", None)
        if bus is not None:
            job_id = getattr(ctx, "session_id", None)
            bus.emit(trace, job_id=job_id)
    except Exception as exc:  # noqa: BLE001
        logger.warning("run: trace emit error (non-fatal): %s", exc)


def _stage_start(ctx: PipelineContext, stage: str, candidates_in: Optional[int] = None) -> float:
    """Emit stage_start trace and return the monotonic start time."""
    t = time.monotonic()
    _emit_trace(
        ctx,
        StageTrace(
            stage=stage,
            phase="start",
            candidates_in=candidates_in,
        ),
    )
    return t


def _stage_end(
    ctx: PipelineContext,
    stage: str,
    t_start: float,
    *,
    candidates_in: Optional[int] = None,
    candidates_out: Optional[int] = None,
    dropped: int = 0,
    drop_reason: Optional[str] = None,
    gate: Optional[str] = None,
) -> None:
    """Emit stage_end trace with latency."""
    latency_ms = round((time.monotonic() - t_start) * 1000.0, 1)
    _emit_trace(
        ctx,
        StageTrace(
            stage=stage,
            phase="end",
            latency_ms=latency_ms,
            candidates_in=candidates_in,
            candidates_out=candidates_out,
            dropped=dropped,
            drop_reason=drop_reason,
            gate=gate,
        ),
    )


def _op_to_proposal_card(
    op: PlannedOperation,
    intent: ChangeIntentV3,
    gate_result: Any,
) -> ProposalCardV3:
    """Convert a PlannedOperation + GateResult into a ProposalCardV3."""
    # Map PlannedOperationKind to legacy ChangeType for backward compat
    change_type_map: Dict[str, str] = {
        "edit_section": "edit",
        "append": "edit",
        "create_page": "create",
        "archive_deprecate": "delete",
    }
    change_type = change_type_map.get(op.operation, "edit")

    # Build a short summary (≤120 chars)
    summary_raw = (
        f"{op.operation.replace('_', ' ').title()}: {op.page_title}"
        + (f" / {op.section_heading}" if op.section_heading else "")
    )
    change_summary = summary_raw[:120] if len(summary_raw) > 120 else summary_raw

    return ProposalCardV3(
        change_type=change_type,
        page_id=op.page_id,
        page_title=op.page_title,
        section_heading=op.section_heading,
        before_content=op.before_content,
        after_content=op.after_content,
        status="pending",
        rationale=op.rationale,
        change_summary=change_summary,
        operation_action=op.operation,
        group_id=op.group_id,
        # v3 confidence fields from gate result
        confidence_score=getattr(gate_result, "confidence_score", None),
        confidence_bin=getattr(gate_result, "confidence_bin", None),
        # v3 evidence spans from intent
        evidence=list(intent.evidence) if intent.evidence else None,
        transcript_evidence=[span.text for span in (intent.evidence or [])],
    )


# ---------------------------------------------------------------------------
# Per-intent fan-out coroutine
# ---------------------------------------------------------------------------


async def _per_intent(
    intent: ChangeIntentV3,
    ctx: PipelineContext,
    sem: asyncio.Semaphore,
) -> tuple:
    """Run stages 2-4 for one intent under the shared semaphore.

    Returns a list of PlannedOperation candidates (pre-gate) for this intent.
    A stage exception isolates this intent without crashing the run.
    """
    async with sem:
        try:
            corpus = getattr(ctx, "section_corpus", None)

            # Stage 2: retrieve
            retrieval = await retrieve_candidates(intent, corpus=corpus)

            # Stage 3: rerank (if enabled)
            if ctx.rerank_enabled and retrieval.candidates:
                reranked = await rerank_candidates(intent, retrieval)
            else:
                reranked = retrieval.candidates

            # Stage 4: iterative retrieval — only when reranked top score is low
            top_score = 0.0
            if reranked:
                top_score = max(
                    (c.rerank_score or c.rrf_score or 0.0) for c in reranked
                )

            _RETRIEVAL_FLOOR = float(os.getenv("JARVIS_RETRIEVAL_FLOOR", "0.4"))
            if top_score < _RETRIEVAL_FLOOR:
                iterate_result = await iterative_retrieve(
                    intent,
                    corpus=corpus if corpus is not None else [],
                    max_iterations=ctx.retrieval_max_iters,
                )
                if iterate_result.candidates:
                    reranked = iterate_result.candidates
                elif iterate_result.no_existing_target:
                    # Signal no existing target — plan_ops will route to create_page
                    no_target_candidate = SectionCandidate(
                        no_existing_target=True,
                        page_title="",
                        source="iterate",
                    )
                    reranked = [no_target_candidate]

            return reranked, intent

        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "run: per-intent stage 2-4 failed for intent subject=%r: %s — isolating",
                getattr(intent, "subject", "?"),
                exc,
            )
            return [], intent


# ---------------------------------------------------------------------------
# run — public orchestrator entry point
# ---------------------------------------------------------------------------


async def run(ctx: PipelineContext) -> List[ProposalCardV3]:
    """Run the full v3 pipeline for the given context.

    Stages execute in deterministic order with bounded per-intent fan-out.
    A single intent's stage exception is isolated and never propagates to
    the event loop (T-11-20).

    Args:
        ctx: PipelineContext carrying session_id, user_id, graph_user_id,
             bot_id, transcript_log, trace_bus, and killswitch snapshots.
             ctx.transcript_text starts as "" and is populated by Stage 0.

    Returns:
        List of surviving ProposalCardV3 (all passed the grounding gate).
        Empty list on total failure — never raises.
    """
    session_id = ctx.session_id
    graph_user_id = ctx.graph_user_id

    proposals: List[ProposalCardV3] = []

    try:
        # ----------------------------------------------------------------
        # Stage 0: Transcript source
        # ----------------------------------------------------------------
        t0 = _stage_start(ctx, "transcript_source")
        try:
            entries = await load_transcript(session_id, ctx)
            # ctx.transcript_text is now populated by load_transcript
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "run: transcript_source failed for session=%s: %s — using empty transcript",
                session_id, exc,
            )
            entries = []
        _stage_end(
            ctx, "transcript_source", t0,
            candidates_out=len(entries),
        )

        transcript_text = ctx.transcript_text or ""

        if not transcript_text.strip():
            logger.warning(
                "run: session=%s — empty transcript after Stage 0; no intents will be extracted",
                session_id,
            )

        # ----------------------------------------------------------------
        # Stage 1: Extraction (once per run)
        # ----------------------------------------------------------------
        t1 = _stage_start(ctx, "extract", candidates_in=1)
        intents: List[ChangeIntentV3] = []
        try:
            intents = await extract_intents(transcript_text, ctx=ctx)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "run: extract_intents failed for session=%s: %s — no intents",
                session_id, exc,
            )
            intents = []
        _stage_end(
            ctx, "extract", t1,
            candidates_in=1,
            candidates_out=len(intents),
        )

        if not intents:
            logger.info(
                "run: session=%s — no intents extracted; pipeline complete with 0 proposals",
                session_id,
            )
            return proposals

        # ----------------------------------------------------------------
        # Stages 2-4: Per-intent retrieve → rerank → iterate (fan-out)
        # ----------------------------------------------------------------
        sem = asyncio.Semaphore(_DRAFTER_CONCURRENCY)

        t24 = _stage_start(ctx, "retrieve_rerank_iterate", candidates_in=len(intents))

        fan_out_tasks = [_per_intent(intent, ctx, sem) for intent in intents]
        fan_out_results = await asyncio.gather(*fan_out_tasks, return_exceptions=True)

        # Build (intent, candidates) pairs, filtering exceptions
        intent_candidate_pairs: List[tuple] = []
        for i, result in enumerate(fan_out_results):
            intent = intents[i]
            if isinstance(result, Exception):
                logger.warning(
                    "run: fan-out failed for intent=%r: %s — skipping",
                    getattr(intent, "subject", "?"), result,
                )
                continue
            candidates, _ = result
            intent_candidate_pairs.append((intent, candidates))

        total_candidates = sum(len(c) for _, c in intent_candidate_pairs)
        _stage_end(
            ctx, "retrieve_rerank_iterate", t24,
            candidates_in=len(intents),
            candidates_out=total_candidates,
        )

        # ----------------------------------------------------------------
        # Stage 5: Contradiction sweep (fact_update intents)
        # ----------------------------------------------------------------
        contradiction_ops: List[PlannedOperation] = []

        if ctx.contradiction_enabled:
            fact_update_intents = [
                (intent, cands)
                for intent, cands in intent_candidate_pairs
                if intent.kind in ("fact_update", "deprecation") and intent.old_value
            ]

            if fact_update_intents:
                t5 = _stage_start(ctx, "contradict", candidates_in=len(fact_update_intents))
                all_groups: List[ContradictionGroup] = []

                # Build workspace_pages list from the corpus for contradiction
                corpus = getattr(ctx, "section_corpus", None)
                workspace_pages: List[Dict[str, Any]] = []
                if corpus is not None:
                    try:
                        if hasattr(corpus, "pages"):
                            workspace_pages = list(corpus.pages)
                        elif isinstance(corpus, list):
                            workspace_pages = corpus
                    except Exception:
                        workspace_pages = []

                contradict_tasks = [
                    detect_contradictions(
                        intent,
                        workspace_pages=workspace_pages,
                        graph_user_id=graph_user_id,
                        trace=ctx.trace_bus,
                    )
                    for intent, _cands in fact_update_intents
                ]
                contradict_results = await asyncio.gather(*contradict_tasks, return_exceptions=True)

                for i, cr in enumerate(contradict_results):
                    if isinstance(cr, Exception):
                        intent = fact_update_intents[i][0]
                        logger.warning(
                            "run: contradict failed for intent=%r: %s — skipping",
                            getattr(intent, "subject", "?"), cr,
                        )
                        continue
                    if cr:
                        all_groups.extend(cr)

                # Flatten ContradictionGroup.operations into the contradiction_ops list
                for group in all_groups:
                    for op in group.operations:
                        # Ensure group_id is set for UI grouping
                        if not op.group_id:
                            object.__setattr__(
                                op,
                                "group_id",
                                str(uuid.uuid4()),
                            ) if hasattr(op, "__dataclass_fields__") else None
                        contradiction_ops.append(op)

                _stage_end(
                    ctx, "contradict", t5,
                    candidates_in=len(fact_update_intents),
                    candidates_out=len(contradiction_ops),
                )

        # ----------------------------------------------------------------
        # Stage 6 + 7: plan_ops + gate (per (intent, candidate) pair)
        # ----------------------------------------------------------------
        t67 = _stage_start(ctx, "plan_ops_gate", candidates_in=total_candidates)

        planned_count = 0
        dropped_count = 0

        for intent, candidates in intent_candidate_pairs:
            if not candidates:
                logger.info(
                    "run: no candidates for intent=%r — skipping plan_ops",
                    getattr(intent, "subject", "?"),
                )
                continue

            # Use the top candidate (after rerank)
            top_candidate = candidates[0]

            try:
                op = await plan_operation(intent, top_candidate)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "run: plan_operation failed for intent=%r: %s — skipping",
                    getattr(intent, "subject", "?"), exc,
                )
                op = None

            if op is None:
                logger.info(
                    "run: plan_operation returned None for intent=%r — no card emitted",
                    getattr(intent, "subject", "?"),
                )
                continue

            planned_count += 1

            # Stage 7: grounding gate
            try:
                gate_result = await apply_grounding_gate_v3(
                    op,
                    transcript_text=transcript_text,
                    current_page_content="",
                    user_id=ctx.user_id,
                    retrieval_score=top_candidate.rerank_score or top_candidate.rrf_score or 0.0,
                    is_fact_update=(intent.kind in ("fact_update", "deprecation")),
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "run: apply_grounding_gate_v3 failed for intent=%r: %s — treating as dropped",
                    getattr(intent, "subject", "?"), exc,
                )
                dropped_count += 1
                continue

            if gate_result.is_dropped and not gate_result.is_flagged:
                dropped_count += 1
                logger.info(
                    "run: gate dropped card for intent=%r gate=%s reasons=%s",
                    getattr(intent, "subject", "?"),
                    gate_result.gate,
                    gate_result.drop_reasons,
                )
                continue

            # Build and collect ProposalCardV3
            card = _op_to_proposal_card(op, intent, gate_result)
            proposals.append(card)

        # Also convert contradiction_ops to cards (gate applied separately)
        for op in contradiction_ops:
            planned_count += 1
            try:
                gate_result = await apply_grounding_gate_v3(
                    op,
                    transcript_text=transcript_text,
                    current_page_content="",
                    user_id=ctx.user_id,
                    retrieval_score=0.5,
                    is_fact_update=True,  # contradiction ops are always fact_update-grade
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "run: contradiction gate failed for op page=%r: %s — treating as dropped",
                    op.page_id, exc,
                )
                dropped_count += 1
                continue

            if gate_result.is_dropped and not gate_result.is_flagged:
                dropped_count += 1
                continue

            # Find the closest intent for evidence spans (use first fact_update)
            best_intent = next(
                (
                    intent
                    for intent, _ in intent_candidate_pairs
                    if intent.kind in ("fact_update", "deprecation")
                ),
                None,
            )
            if best_intent is None:
                # Construct a minimal proxy intent for the card builder
                from confluence_logic.pipeline.contracts import EvidenceSpan
                best_intent = ChangeIntentV3(
                    kind="fact_update",
                    subject=op.page_title,
                    evidence=[EvidenceSpan(text=op.rationale or "")],
                )

            card = _op_to_proposal_card(op, best_intent, gate_result)
            proposals.append(card)

        _stage_end(
            ctx, "plan_ops_gate", t67,
            candidates_in=planned_count,
            candidates_out=len(proposals),
            dropped=dropped_count,
        )

        logger.info(
            "run: session=%s complete — %d proposals, %d dropped",
            session_id, len(proposals), dropped_count,
        )

    except Exception as exc:  # noqa: BLE001
        # Broad top-level swallow — never propagate to event loop (T-11-20)
        logger.error(
            "run: unhandled exception for session=%s: %s — returning partial results",
            session_id, exc,
        )

    return proposals
