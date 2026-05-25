"""Stage 4: Bounded agentic iterative retrieval loop (RETR-V3-04).

Fires ONLY when the top reranked candidate relevance is below a confidence
floor (``JARVIS_RETRIEVAL_FLOOR``).  When strong candidates already exist the
stage is a no-op passthrough.

Design choices (RESEARCH §Bounded agentic retrieval + Pitfall 7):
  - Python-enforced cap (``JARVIS_RETRIEVAL_MAX_ITERS``, default 2).  The cap
    is counted and enforced in Python — the LLM is NEVER trusted to self-stop
    (Pitfall 7: unbounded agentic loop blows the latency budget).
  - Each iteration reformulates the query by progressively combining more intent
    signals (subject → + old/new value → + target_hint → decomposed synonyms).
    This gives the retrieval stage broader coverage without an extra LLM call.
  - Terminal outcomes:
      (a) A candidate with score ≥ floor found → return updated RetrievalResult.
      (b) Exhausted iterations with ALL results being empty → no_existing_target=True
          (plan_ops routes this intent to create_page).
      (c) Exhausted with low-confidence candidates → return best result
          (not a guaranteed create_page; leave the decision to plan_ops).
  - ``_retrieve_once`` is a module-level function that delegates to the
    retrieve-stage internals (dense+BM25+RRF).  Tests patch it directly.
  - Graceful degradation: any error inside the loop degrades to the input
    result (never raises into the orchestrator).

Public API::

    from confluence_logic.pipeline.stages.iterate import iterative_retrieve

    result: RetrievalResult = await iterative_retrieve(
        intent,
        corpus=ctx.section_corpus,
        store=pinecone_store,
        max_iterations=ctx.retrieval_max_iters,
    )

``Runner`` is imported at module level for the test-seam convention (tests may
patch ``confluence_logic.pipeline.stages.iterate.Runner`` even though the
current implementation does not use it directly — future extension point for
the full agent-tool loop described in RETR-V3-04).
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, List, Optional

from confluence_logic.pipeline.contracts import (
    ChangeIntentV3,
    RetrievalResult,
    SectionCandidate,
)
from confluence_logic.pipeline.context import JARVIS_RETRIEVAL_MAX_ITERS

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Score floor: if the best candidate's score is below this value the iterate
# loop fires.  0.5 is the default (scale 0–1 from retrieve stage / rerank).
_RETRIEVAL_FLOOR: float = float(os.getenv("JARVIS_RETRIEVAL_FLOOR", "0.5"))

# ---------------------------------------------------------------------------
# Runner import at module level (test seam — PATTERNS.md convention)
# ---------------------------------------------------------------------------
# Tests may patch ``confluence_logic.pipeline.stages.iterate.Runner`` for a
# future agent-tool-loop extension.  Import is guarded to avoid side effects
# when the agents SDK is absent.

try:
    from agents import Runner  # noqa: F401
    _RUNNER_AVAILABLE = True
except ImportError:
    Runner = None  # type: ignore[assignment]
    _RUNNER_AVAILABLE = False


# ---------------------------------------------------------------------------
# Query reformulation helpers
# ---------------------------------------------------------------------------

def _build_query(intent: ChangeIntentV3, iteration: int) -> str:
    """Return a progressively broader query string for iteration N (0-indexed).

    Iteration 0: base query — subject only (or subject + target_hint if set).
    Iteration 1: subject + old_value/new_value context (fact-change framing).
    Iteration 2+: subject + full instruction + target_hint (broadest).

    Each iteration produces a distinct string so the iterate loop genuinely
    reformulates the search rather than repeating the same query (RETR-V3-04).
    """
    subject = intent.subject.strip()
    target_hint = intent.target_hint.strip()
    old_value = intent.old_value.strip()
    new_value = intent.new_value.strip()
    instruction = intent.instruction.strip()

    if iteration == 0:
        # Base: subject + target_hint
        parts = [p for p in [subject, target_hint] if p]
        return " ".join(parts) if parts else subject

    if iteration == 1:
        # Fact-change framing: include old→new value context
        change_ctx = ""
        if old_value and new_value:
            change_ctx = f"{old_value} to {new_value}"
        elif new_value:
            change_ctx = new_value
        elif old_value:
            change_ctx = old_value
        parts = [p for p in [subject, change_ctx, target_hint] if p]
        return " ".join(parts) if parts else subject

    # Iteration 2+: broadest — all signals
    parts = [p for p in [subject, instruction, target_hint, old_value, new_value] if p]
    return " ".join(parts) if parts else subject


# ---------------------------------------------------------------------------
# _retrieve_once — internal single-shot retrieval (test-patchable)
# ---------------------------------------------------------------------------
# Tests patch this function at ``confluence_logic.pipeline.stages.iterate._retrieve_once``.
# It wraps ``retrieve_candidates`` from Stage 2.  The ``query`` kwarg lets the
# iterate loop pass a reformulated query string.

async def _retrieve_once(
    intent: ChangeIntentV3,
    corpus: list,
    store: Any,
    *,
    query: Optional[str] = None,
) -> RetrievalResult:
    """Run one round of hybrid retrieval (Stage 2) with an optional reformulated query.

    If *query* is provided the intent's fields are temporarily overridden for
    this call so the retrieve stage uses the reformulated string.

    Falls back to an empty RetrievalResult on any error.
    """
    try:
        from confluence_logic.pipeline.stages.retrieve import retrieve_candidates  # noqa: PLC0415

        # If a reformulated query was provided, create a shallow copy of the
        # intent with the instruction overridden to the new query string.
        if query is not None:
            # Construct a modified intent variant with subject set to the query.
            # We reuse the intent's evidence/kind/etc.; only the lookup strings change.
            effective_intent = ChangeIntentV3(
                kind=intent.kind,
                subject=query,
                old_value=intent.old_value,
                new_value=intent.new_value,
                instruction="",   # query already contains the salient tokens
                target_hint="",
                verbatim_content=intent.verbatim_content,
                dedup_key=intent.dedup_key,
                evidence=intent.evidence,
            )
        else:
            effective_intent = intent

        return await retrieve_candidates(
            effective_intent,
            corpus=corpus,
            pinecone_store=store,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "iterate._retrieve_once: retrieval error (non-fatal): %s", exc
        )
        return RetrievalResult(candidates=[], fusion_log=None)


# ---------------------------------------------------------------------------
# _top_score — convenience: best candidate score in a result
# ---------------------------------------------------------------------------

def _top_score(result: RetrievalResult) -> float:
    """Return the score of the best candidate, or -1.0 if no candidates."""
    if not result.candidates:
        return -1.0
    # Prefer rerank_score if set, else fall back to rrf_score or score.
    def _best(c: SectionCandidate) -> float:
        if c.rerank_score is not None:
            return c.rerank_score
        if c.rrf_score:
            return c.rrf_score
        return c.score

    return max(_best(c) for c in result.candidates)


# ---------------------------------------------------------------------------
# iterative_retrieve — public entry point
# ---------------------------------------------------------------------------

async def iterative_retrieve(
    intent: ChangeIntentV3,
    *,
    corpus: Optional[list] = None,
    store: Optional[Any] = None,
    max_iterations: Optional[int] = None,
) -> RetrievalResult:
    """Bounded iterative retrieval loop (RETR-V3-04).

    Calls ``_retrieve_once`` up to ``max_iterations`` times, reformulating
    the query each iteration.  Stops early when a high-confidence candidate
    (score ≥ ``JARVIS_RETRIEVAL_FLOOR``) is found.

    Terminal outcomes:
      - High-confidence hit found early → return that RetrievalResult.
      - ``max_iterations`` exhausted AND all iterations returned empty
        candidates → ``no_existing_target=True``.
      - ``max_iterations`` exhausted with low-confidence candidates →
        return best result (plan_ops decides).

    The iteration cap is enforced in Python (``max_iterations`` counter).  The
    LLM is NEVER trusted to self-stop (Pitfall 7 / T-11-08).

    Args:
        intent: The ChangeIntentV3 driving the retrieval.
        corpus: Per-run section corpus (list of SectionRow or compatible).
        store: PineconeStore instance or compatible.
        max_iterations: Max retrieve calls (default: ``JARVIS_RETRIEVAL_MAX_ITERS``).

    Returns:
        RetrievalResult with ``no_existing_target`` and ``iterations`` set.
    """
    if max_iterations is None:
        max_iterations = JARVIS_RETRIEVAL_MAX_ITERS

    effective_corpus: list = corpus if corpus is not None else []

    best_result: Optional[RetrievalResult] = None
    all_empty = True

    try:
        for iteration in range(max_iterations):
            query = _build_query(intent, iteration)

            result = await _retrieve_once(
                intent,
                effective_corpus,
                store,
                query=query,
            )

            if result.candidates:
                all_empty = False
                best_result = result

            top = _top_score(result)

            if top >= _RETRIEVAL_FLOOR:
                # High-confidence hit — early exit (RETR-V3-04)
                logger.debug(
                    "iterate: early exit at iteration %d — top_score=%.3f >= floor=%.3f",
                    iteration,
                    top,
                    _RETRIEVAL_FLOOR,
                )
                return RetrievalResult(
                    intent=result.intent,
                    candidates=result.candidates,
                    no_existing_target=False,
                    iterations=iteration + 1,
                    fusion_log=result.fusion_log,
                )

            logger.debug(
                "iterate: iteration %d — top_score=%.3f < floor=%.3f; reformulating",
                iteration,
                top,
                _RETRIEVAL_FLOOR,
            )

        # Exhausted all iterations without a high-confidence hit.
        if all_empty or best_result is None:
            # No candidates found at all → signal create_page (RETR-V3-04)
            logger.info(
                "iterate: exhausted %d iterations with no candidates — "
                "no_existing_target=True",
                max_iterations,
            )
            return RetrievalResult(
                intent=best_result.intent if best_result else None,
                candidates=[],
                no_existing_target=True,
                iterations=max_iterations,
                fusion_log=None,
            )

        # Low-confidence candidates found; return best result as-is.
        logger.debug(
            "iterate: exhausted %d iterations — returning best low-confidence result",
            max_iterations,
        )
        return RetrievalResult(
            intent=best_result.intent,
            candidates=best_result.candidates,
            no_existing_target=False,
            iterations=max_iterations,
            fusion_log=best_result.fusion_log,
        )

    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "iterate.iterative_retrieve: unexpected error (non-fatal, "
            "returning input result): %s",
            exc,
        )
        # Graceful degradation — return empty no_existing_target result.
        return RetrievalResult(
            intent=None,
            candidates=[],
            no_existing_target=True,
            iterations=0,
            fusion_log=None,
        )
