"""Stage 3: LLM pointwise-parallel reranking (RETR-V3-03).

Reranks the RRF-fused candidates by true relevance using a ``gpt-5.4-nano``
agent that scores each candidate independently (pointwise) under
``asyncio.gather`` + ``asyncio.Semaphore`` concurrency cap.

Design choices (RESEARCH §Reranker decision + §Anti-Patterns):
  - Pointwise NOT listwise: each candidate gets its own nano call, which is
    more calibratable and lower-variance than one big listwise prompt.
  - ``_llm_listwise_rerank`` assembles the final ranked doc-id list from the
    per-candidate scores — its name reflects the *output shape* (a ranked list)
    while the internal implementation uses pointwise scoring.  Tests patch
    ``_llm_listwise_rerank`` directly to inject a pre-computed order.
  - RRF fallback: if ALL nano calls fail OR ``JARVIS_V3_RERANK_ENABLED`` is
    False, candidates are returned in their original RRF order (no crash, no
    silent drop — T-11-09 mitigation).
  - Per-candidate exception isolation: a single nano failure does NOT fail the
    whole batch; that candidate keeps its RRF rank position.
  - ``top_k`` truncation: only the top-k candidates proceed to downstream
    stages (RETR-V3-03 — caller defaults to ``JARVIS_RERANK_TOP_K``).

Public API::

    from confluence_logic.pipeline.stages.rerank import rerank_candidates

    ranked: list[SectionCandidate] = await rerank_candidates(
        intent, retrieval_result, top_k=5
    )

Module-level ``Agent`` singleton + ``Runner`` import at module level (test
seam — tests patch ``confluence_logic.pipeline.stages.rerank.Runner``).
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import List, Optional

from pydantic import BaseModel

from confluence_logic.pipeline.contracts import (
    ChangeIntentV3,
    RetrievalResult,
    SectionCandidate,
)
from confluence_logic.pipeline.model_config import MODEL_NANO

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Maximum number of concurrent nano rerank calls (rate-limit guard).
_RERANK_CONCURRENCY: int = int(os.getenv("JARVIS_RERANK_CONCURRENCY", "6"))

# Default top-k — how many candidates survive the rerank stage.
JARVIS_RERANK_TOP_K: int = int(os.getenv("JARVIS_RERANK_TOP_K", "5"))


# ---------------------------------------------------------------------------
# Pydantic schema for per-candidate score output
# ---------------------------------------------------------------------------

class _RerankScore(BaseModel):
    """Score emitted by the nano agent for one candidate (0–10)."""
    score: float = 0.0
    rationale: str = ""


# ---------------------------------------------------------------------------
# Module-level agent singleton + Runner import (test seam)
# ---------------------------------------------------------------------------
# Runner is imported at module level so tests can patch
# ``confluence_logic.pipeline.stages.rerank.Runner`` directly.

try:
    from agents import Agent, AgentOutputSchema, Runner  # noqa: F401
    _AGENT_SDK_AVAILABLE = True
except ImportError:
    Agent = None  # type: ignore[assignment]
    AgentOutputSchema = None  # type: ignore[assignment]
    Runner = None  # type: ignore[assignment]
    _AGENT_SDK_AVAILABLE = False

_RERANK_SYSTEM_PROMPT = """\
You are a relevance-scoring assistant. Given a meeting intent (subject,
old_value, new_value, instruction) and a single Confluence section candidate,
score how relevant the section is to the intent on a scale from 0 to 10
(10 = perfect match, 0 = completely unrelated).

Return a JSON object with two fields:
  "score": <float between 0 and 10>
  "rationale": <one sentence justifying the score>
"""

_rerank_agent = None  # lazy init on first use


def _get_rerank_agent():
    """Lazily build the module-level singleton rerank agent."""
    global _rerank_agent
    if _rerank_agent is None and _AGENT_SDK_AVAILABLE and Agent is not None:
        _rerank_agent = Agent(
            name="RerankAgent",
            model=MODEL_NANO,
            instructions=_RERANK_SYSTEM_PROMPT,
            output_type=AgentOutputSchema(_RerankScore, strict_json_schema=False),
        )
    return _rerank_agent


# ---------------------------------------------------------------------------
# _doc_id_for_candidate — canonical doc-id matching retrieve.py convention
# ---------------------------------------------------------------------------

def _doc_id(candidate: SectionCandidate) -> str:
    """Return ``"{page_id}::{section_heading or ''}"`` as the composite doc id."""
    heading = candidate.section_heading or ""
    return f"{candidate.page_id}::{heading}"


# ---------------------------------------------------------------------------
# _score_one — score a single candidate (pointwise nano call)
# ---------------------------------------------------------------------------

async def _score_one(
    intent: ChangeIntentV3,
    candidate: SectionCandidate,
    sem: asyncio.Semaphore,
) -> tuple[str, float]:
    """Score one candidate against the intent with a nano Runner.run call.

    Returns:
        (doc_id, score_0_to_10)

    On any error, returns (doc_id, -1.0) so the caller can detect failures
    and fall back gracefully (per-candidate isolation — RETR-V3-03).
    """
    doc_id = _doc_id(candidate)
    if not _AGENT_SDK_AVAILABLE or Runner is None:
        return doc_id, -1.0

    agent = _get_rerank_agent()
    if agent is None:
        return doc_id, -1.0

    prompt = (
        f"Intent subject: {intent.subject}\n"
        f"Old value: {intent.old_value}\n"
        f"New value: {intent.new_value}\n"
        f"Instruction: {intent.instruction}\n\n"
        f"Section heading: {candidate.section_heading or '(none)'}\n"
        f"Section text (excerpt): {candidate.section_text[:400]}"
    )

    try:
        async with sem:
            result = await Runner.run(agent, prompt)
        if isinstance(result.final_output, _RerankScore):
            score_obj = result.final_output
        else:
            score_obj = _RerankScore.model_validate(result.final_output)
        return doc_id, float(score_obj.score)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "rerank._score_one: nano call failed for %r (non-fatal, "
            "candidate keeps RRF rank): %s",
            doc_id,
            exc,
        )
        return doc_id, -1.0


# ---------------------------------------------------------------------------
# _llm_listwise_rerank — assembles a ranked doc-id list from pointwise scores
# ---------------------------------------------------------------------------
# Tests patch this function directly to inject a pre-computed ranking.
# In production it runs N pointwise nano calls and sorts by score descending.

async def _llm_listwise_rerank(
    intent: ChangeIntentV3,
    candidates: List[SectionCandidate],
) -> List[str]:
    """Return a ranked list of doc-ids for *candidates* against *intent*.

    Internally runs one ``gpt-5.4-nano`` ``Runner.run`` call per candidate
    under ``asyncio.gather`` + ``Semaphore`` (pointwise parallel scoring —
    RETR-V3-03, RESEARCH §Anti-Patterns).

    Returns:
        List of ``"{page_id}::{section_heading}"`` strings, best → worst.

    Raises:
        Exception — propagated to ``rerank_candidates`` which catches it
        and falls back to RRF order (T-11-09).
    """
    if not candidates:
        return []

    sem = asyncio.Semaphore(_RERANK_CONCURRENCY)
    score_tasks = [_score_one(intent, c, sem) for c in candidates]
    results = await asyncio.gather(*score_tasks, return_exceptions=True)

    scored: list[tuple[str, float]] = []
    all_failed = True
    for i, r in enumerate(results):
        if isinstance(r, Exception):
            logger.warning(
                "rerank._llm_listwise_rerank: task %d raised (non-fatal): %s", i, r
            )
            # Fallback score: -1.0 (preserves relative RRF order for failures)
            scored.append((_doc_id(candidates[i]), -1.0))
        else:
            doc_id, score = r
            scored.append((doc_id, score))
            if score >= 0.0:
                all_failed = False

    if all_failed:
        # Raise so rerank_candidates sees total failure and returns RRF order.
        raise RuntimeError(
            "rerank._llm_listwise_rerank: all nano calls failed — falling back to RRF order"
        )

    # Sort by score descending; ties preserve original RRF order (stable sort).
    scored.sort(key=lambda kv: -kv[1])
    return [doc_id for doc_id, _ in scored]


# ---------------------------------------------------------------------------
# rerank_candidates — public entry point
# ---------------------------------------------------------------------------

async def rerank_candidates(
    intent: ChangeIntentV3,
    retrieval: RetrievalResult,
    *,
    top_k: Optional[int] = None,
) -> List[SectionCandidate]:
    """Rerank RRF-fused candidates by true relevance and return the top-k.

    Uses ``_llm_listwise_rerank`` to obtain a relevance-ordered list of doc-ids,
    then maps back to the original ``SectionCandidate`` objects (hallucinated
    doc-ids from the LLM are silently dropped — RETR-V3-03).

    Fallback (T-11-09): if ``_llm_listwise_rerank`` raises, candidates are
    returned in their original RRF order.  No candidate is dropped due to
    rerank failure.

    Args:
        intent: The driving ChangeIntentV3.
        retrieval: RetrievalResult from Stage 2 (contains RRF-ordered candidates).
        top_k: How many candidates to keep (default: ``JARVIS_RERANK_TOP_K``).

    Returns:
        Re-ordered SectionCandidate list, length ≤ top_k.
    """
    if top_k is None:
        top_k = JARVIS_RERANK_TOP_K

    candidates = list(retrieval.candidates)

    if not candidates:
        return []

    # Build a lookup from doc_id → candidate to handle hallucinated ids and
    # to support the case where _llm_listwise_rerank returns a subset.
    candidate_by_id: dict[str, SectionCandidate] = {}
    for c in candidates:
        candidate_by_id[_doc_id(c)] = c

    try:
        ranked_ids = await _llm_listwise_rerank(intent, candidates)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "rerank_candidates: _llm_listwise_rerank failed (non-fatal, "
            "returning RRF order): %s",
            exc,
        )
        # Fallback: original RRF order, truncated to top_k.
        return candidates[:top_k]

    # Map ranked doc-ids back to candidates; drop hallucinated ids (RETR-V3-03).
    reranked: List[SectionCandidate] = []
    seen_ids: set[str] = set()
    for doc_id in ranked_ids:
        if doc_id in candidate_by_id and doc_id not in seen_ids:
            reranked.append(candidate_by_id[doc_id])
            seen_ids.add(doc_id)

    # Append any remaining candidates not returned by the reranker (preserves
    # completeness — we never lose candidates due to the LLM returning a partial
    # list, though we don't promote them past the top_k).
    for c in candidates:
        doc_id = _doc_id(c)
        if doc_id not in seen_ids:
            reranked.append(c)
            seen_ids.add(doc_id)

    # Truncate to top_k.
    return reranked[:top_k]
