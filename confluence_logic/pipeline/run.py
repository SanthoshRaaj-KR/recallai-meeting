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
import html
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
from confluence_logic.pipeline.retrieval.corpus import fetch_section_corpus

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Concurrency cap (lifted from review/api.py _run_pipeline)
# ---------------------------------------------------------------------------

_DRAFTER_CONCURRENCY: int = int(os.getenv("JARVIS_DRAFTER_CONCURRENCY", "6"))
_PLAN_OPS_MAX_CANDIDATES_PER_INTENT: int = int(
    os.getenv("JARVIS_PLAN_OPS_MAX_CANDIDATES_PER_INTENT", "5")
)
_PLAN_OPS_EXPAND_MIN_RERANK: float = float(
    os.getenv("JARVIS_PLAN_OPS_EXPAND_MIN_RERANK", "0.75")
)


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
        verifier_note=(
            "Low confidence: " + "; ".join(getattr(gate_result, "drop_reasons", []) or [])
            if getattr(gate_result, "is_flagged", False)
            else None
        ),
        # v3 evidence spans from intent
        evidence=list(intent.evidence) if intent.evidence else None,
        transcript_evidence=[span.text for span in (intent.evidence or [])],
    )


def _row_get(row: Any, key: str, default: Any = None) -> Any:
    """Read a dict key or object attribute from corpus rows/pages."""
    if isinstance(row, dict):
        return row.get(key, default)
    return getattr(row, key, default)


def _corpus_to_workspace_pages(corpus: Any) -> List[Dict[str, Any]]:
    """Return page-shaped dicts with synthesized HTML from the section corpus.

    The contradiction stage and plan/gate steps need page-level content, while
    the production corpus is section-shaped. This groups sections by page_id and
    emits a small HTML document per page so existing section extraction helpers
    can operate without a live Confluence fetch.
    """
    if not corpus:
        return []

    if hasattr(corpus, "pages"):
        try:
            return list(corpus.pages)
        except Exception:
            return []

    pages: Dict[str, Dict[str, Any]] = {}
    for row in (corpus if isinstance(corpus, list) else []):
        page_id = (_row_get(row, "page_id", "") or "").strip()
        if not page_id:
            continue
        title = (
            _row_get(row, "title", None)
            or _row_get(row, "page_title", None)
            or ""
        )
        space_key = _row_get(row, "space_key", "") or ""
        heading = (
            _row_get(row, "heading", None)
            or _row_get(row, "section_heading", None)
            or ""
        )
        text = (
            _row_get(row, "content_html", None)
            or _row_get(row, "text", None)
            or _row_get(row, "section_text", None)
            or ""
        )
        page = pages.setdefault(
            page_id,
            {
                "page_id": page_id,
                "title": title,
                "page_title": title,
                "space_key": space_key,
                "_sections": [],
            },
        )
        if title and not page.get("title"):
            page["title"] = title
            page["page_title"] = title
        if isinstance(row, dict) and row.get("content_html") and not heading:
            page["_sections"].append(str(row.get("content_html") or ""))
        else:
            section_html = (
                f"<h2>{html.escape(str(heading))}</h2>\n"
                f"<p>{html.escape(str(text))}</p>"
            )
            page["_sections"].append(section_html)

    out: List[Dict[str, Any]] = []
    for page in pages.values():
        sections = page.pop("_sections", [])
        page["content_html"] = "\n".join(sections)
        out.append(page)
    return out


def _page_html_for_id(page_id: Optional[str], workspace_pages: List[Dict[str, Any]]) -> str:
    if not page_id:
        return ""
    for page in workspace_pages:
        if str(page.get("page_id") or "") == str(page_id):
            return page.get("content_html") or page.get("text") or ""
    return ""


def _page_html_for_candidate(
    candidate: SectionCandidate,
    workspace_pages: List[Dict[str, Any]],
) -> str:
    return _page_html_for_id(candidate.page_id, workspace_pages)


def _dedupe_candidates(candidates: List[SectionCandidate]) -> List[SectionCandidate]:
    """Preserve rank order while removing repeated page/section candidates."""
    seen: set = set()
    out: List[SectionCandidate] = []
    for cand in candidates:
        key = (cand.page_id, cand.section_heading)
        if key in seen:
            continue
        seen.add(key)
        out.append(cand)
    return out


def _op_key(intent: ChangeIntentV3, op: PlannedOperation) -> tuple:
    return (
        intent.dedup_key or intent.subject,
        op.page_id or "",
        op.page_title or "",
        op.section_heading or "",
        op.operation,
        (op.after_content or "")[:160],
    )


def _candidate_retrieval_score(candidate: SectionCandidate) -> float:
    return candidate.rerank_score or candidate.rrf_score or candidate.score or 0.0


def _topic_tokens(text: str) -> set:
    stop = {
        "this", "that", "with", "from", "into", "page", "section",
        "update", "change", "create", "add", "move", "make",
    }
    return {
        token
        for token in "".join(ch.lower() if ch.isalnum() else " " for ch in text or "").split()
        if len(token) >= 4 and token not in stop
    }


def _candidate_has_topic_overlap(
    intent: ChangeIntentV3,
    candidate: SectionCandidate,
    page_html: str,
) -> bool:
    needles = _topic_tokens(" ".join([intent.subject, intent.target_hint, intent.instruction]))
    if not needles:
        return True
    haystack = " ".join([
        candidate.page_title or "",
        candidate.section_heading or "",
        candidate.section_text or "",
        page_html or "",
    ]).lower()
    return any(token in haystack for token in needles)


def _subject_tokens_match_page_title(intent: ChangeIntentV3, candidate: SectionCandidate) -> bool:
    """Return True iff at least one content-bearing subject token appears in the page title.

    Prevents routing intents about Person A to Person B's page just because Person A is
    mentioned somewhere in Person B's page content (e.g., 'Virat Kohli' intent → Tilak
    Varma page that references Virat Kohli in a comparison section).
    """
    subject_tokens = _topic_tokens(intent.subject)
    if not subject_tokens:
        return True  # no specific subject tokens — no restriction
    title_tokens = _topic_tokens(candidate.page_title or "")
    return bool(subject_tokens & title_tokens)


def _candidate_qualifies_for_planning(
    intent: ChangeIntentV3,
    candidate: SectionCandidate,
    page_html: str,
    rank: int,
) -> bool:
    """Limit candidate expansion so recall improves without spraying wrong pages."""
    if intent.kind in ("decision", "action_item") and not _candidate_has_topic_overlap(
        intent, candidate, page_html
    ):
        return False

    if rank == 0:
        return True

    # Non-rank-0 expansion: require subject tokens to appear in the page TITLE,
    # not just anywhere in the page content. Without this, a page about Player B
    # can match an intent about Player A because Player A is mentioned in Player B's
    # page body — leading to changes being proposed on the wrong page.
    if not _subject_tokens_match_page_title(intent, candidate):
        return False

    old_value = (intent.old_value or "").strip().lower()
    if old_value and old_value in (page_html or "").lower():
        return True

    # Only expand non-verbatim candidates when the reranker gave a strong score.
    # RRF-only scores are intentionally not used here because their scale is tiny
    # and not calibrated for wrong-page risk.
    if candidate.rerank_score is not None and candidate.rerank_score >= _PLAN_OPS_EXPAND_MIN_RERANK:
        return True

    return False


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

        # Load the user's full Confluence section corpus once per run. Retrieval
        # uses it for BM25, contradiction uses it for workspace-wide scans, and
        # the gate uses synthesized page content for token grounding.
        try:
            await fetch_section_corpus(graph_user_id, ctx)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "run: fetch_section_corpus failed for user=%s: %s — continuing with dense-only retrieval",
                graph_user_id, exc,
            )
            ctx.section_corpus = []

        workspace_pages = _corpus_to_workspace_pages(getattr(ctx, "section_corpus", None))

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
        # Stage 5: Contradiction sweep (fact_update / deprecation intents)
        # ----------------------------------------------------------------
        contradiction_ops: List[tuple[PlannedOperation, ChangeIntentV3]] = []
        # Intents whose cards are produced via a contradiction group are skipped
        # in the per-candidate plan_ops loop below, so the same change is not
        # emitted twice (once grouped, once ungrouped). De-dup is intent-scoped.
        intents_handled_by_contradiction: set = set()

        if ctx.contradiction_enabled:
            fact_update_intents = [
                (intent, cands)
                for intent, cands in intent_candidate_pairs
                if intent.kind in ("fact_update", "deprecation") and intent.old_value
            ]

            if fact_update_intents:
                t5 = _stage_start(ctx, "contradict", candidates_in=len(fact_update_intents))

                page_lookup: Dict[str, Dict[str, Any]] = {
                    p.get("page_id"): p for p in workspace_pages if p.get("page_id")
                }

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
                    intent = fact_update_intents[i][0]
                    if isinstance(cr, Exception):
                        logger.warning(
                            "run: contradict failed for intent=%r: %s — skipping",
                            getattr(intent, "subject", "?"), cr,
                        )
                        continue
                    if not cr:
                        continue

                    produced_any = False
                    for group in cr:
                        # Prefer pre-planned operations when an upstream stage (or
                        # a test stub) already filled them. The REAL
                        # detect_contradictions leaves operations=[] and only
                        # fills affected_pages, so the else-branch is the
                        # production path that actually turns a contradiction
                        # into proposal cards (without it, contradiction groups
                        # never become cards — they would be silently dropped).
                        group_ops: List[PlannedOperation] = list(group.operations)
                        if not group_ops:
                            for affected in group.affected_pages:
                                page = page_lookup.get(affected.page_id) or {}
                                candidate = SectionCandidate(
                                    page_id=affected.page_id,
                                    page_title=affected.page_title,
                                    section_heading=affected.section_heading,
                                    source="contradict",
                                )
                                page_html = page.get("content_html") or ""
                                try:
                                    planned = await plan_operation(intent, candidate, page_html)
                                except Exception as exc:  # noqa: BLE001
                                    logger.warning(
                                        "run: contradiction plan_operation failed page=%s: %s",
                                        affected.page_id, exc,
                                    )
                                    planned = None
                                if planned is None:
                                    continue
                                planned.group_id = affected.group_id or planned.group_id
                                group_ops.append(planned)

                        # Every op in the group shares ONE group_id so the UI
                        # renders them as a single contradiction decision (each
                        # still individually accept/reject-able — UI-V3-01).
                        shared_group_id = (
                            group.affected_pages[0].group_id
                            if group.affected_pages else None
                        )
                        for op in group_ops:
                            if not op.group_id:
                                op.group_id = shared_group_id or str(uuid.uuid4())
                            contradiction_ops.append((op, intent))
                            produced_any = True

                    if produced_any:
                        intents_handled_by_contradiction.add(id(intent))

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
        emitted_op_keys: set = set()

        # Multi-candidate planner. This replaces the legacy top-candidate-only
        # loop below, but leaves that code in place as an inert fallback path so
        # this patch stays small around the existing worktree edits.
        for intent, candidates in list(intent_candidate_pairs):
            if id(intent) in intents_handled_by_contradiction:
                continue
            if not candidates:
                continue

            qualified_candidates: List[SectionCandidate] = []
            for rank, candidate in enumerate(_dedupe_candidates(candidates)):
                if len(qualified_candidates) >= _PLAN_OPS_MAX_CANDIDATES_PER_INTENT:
                    break
                page_html = _page_html_for_candidate(candidate, workspace_pages)
                if _candidate_qualifies_for_planning(intent, candidate, page_html, rank):
                    qualified_candidates.append(candidate)

            for candidate in qualified_candidates:
                page_html = _page_html_for_candidate(candidate, workspace_pages)
                try:
                    op = await plan_operation(intent, candidate, page_html)
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "run: plan_operation failed for intent=%r page=%s: %s — skipping",
                        getattr(intent, "subject", "?"), candidate.page_id, exc,
                    )
                    continue
                if op is None:
                    continue

                op_key = _op_key(intent, op)
                if op_key in emitted_op_keys:
                    continue
                emitted_op_keys.add(op_key)
                planned_count += 1

                try:
                    gate_result = await apply_grounding_gate_v3(
                        op,
                        transcript_text=transcript_text,
                        current_page_content=page_html,
                        user_id=ctx.graph_user_id,
                        retrieval_score=_candidate_retrieval_score(candidate),
                        is_fact_update=(intent.kind in ("fact_update", "deprecation")),
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "run: apply_grounding_gate_v3 failed for intent=%r page=%s: %s — treating as dropped",
                        getattr(intent, "subject", "?"), candidate.page_id, exc,
                    )
                    dropped_count += 1
                    continue

                if gate_result.is_dropped and not gate_result.is_flagged:
                    dropped_count += 1
                    logger.info(
                        "run: gate dropped card for intent=%r page=%s gate=%s reasons=%s",
                        getattr(intent, "subject", "?"),
                        candidate.page_id,
                        gate_result.gate,
                        gate_result.drop_reasons,
                    )
                    continue

                proposals.append(_op_to_proposal_card(op, intent, gate_result))

        # Also convert contradiction_ops to cards (gate applied separately)
        for op, c_intent in contradiction_ops:
            op_key = _op_key(c_intent, op)
            if op_key in emitted_op_keys:
                dropped_count += 1
                continue
            emitted_op_keys.add(op_key)

            planned_count += 1
            page_html = _page_html_for_id(op.page_id, workspace_pages)
            try:
                gate_result = await apply_grounding_gate_v3(
                    op,
                    transcript_text=transcript_text,
                    current_page_content=page_html,
                    user_id=ctx.graph_user_id,
                    # Contradiction detection is a verbatim old_value match — stronger
                    # signal than semantic retrieval, so score above the 0.5 midpoint.
                    retrieval_score=0.75,
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

            card = _op_to_proposal_card(op, c_intent, gate_result)
            proposals.append(card)

        _stage_end(
            ctx, "plan_ops_gate", t67,
            candidates_in=total_candidates,
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
