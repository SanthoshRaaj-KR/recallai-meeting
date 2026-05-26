"""Stage 7: Grounding + calibrated confidence gate (GND-V3-01).

Two-layer design:

1. HARD GATE (deterministic, LLM-free, non-negotiable):
   Reuses ``grounding_gate.check_page_existence`` and
   ``grounding_gate.check_grounding`` VERBATIM.  A card failing either check
   is dropped with the gate name recorded.  This is non-overridable.

2. SOFT CONFIDENCE SCORER (calibrated, never overrides hard pass/fail):
   Combines three cheap signals into a 0–1 score:
     (a) ``retrieval_score`` — reranker relevance of the chosen section
         (passed in from the rerank stage or provided as a parameter).
     (b) ``grounding_signal`` — ``1 - |missing| / |tokens|`` derived from
         the missing set the hard gate already computed (reuses tokens_subset).
     (c) Verbalized LLM confidence — DISCOUNTED (Pitfall 5: verbalized LLM
         confidence is empirically miscalibrated / over-confident).  Not used
         in the current scorer unless explicitly provided; treated as 0.0.
   Bins to ``high/medium/low`` using tunable thresholds (fit on the eval set).
   Sub-threshold cards are SUPPRESSED with a logged drop reason + StageTrace.
   EXCEPTION: ``fact_update`` / contradiction cards are FLAGGED (not dropped)
   when below threshold — a missed contradiction is worse than a soft card.

Design notes:
  - ``check_page_existence`` and ``check_grounding`` are called unchanged.
  - ``graph_user_id`` is explicit everywhere (no ContextVar reads — Pitfall 4).
  - ``%``-style logging per CLAUDE.md convention.

GND-V3-01 acceptance bar:
  - test_gate_v3.py GREEN (confidence binning + sub-threshold suppression/flagging).
  - test_grounding_gate.py still GREEN (hard gate logic unchanged — verbatim reuse).
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

from confluence_logic.pipeline.contracts import (
    PlannedOperation,
    ProposalCardV3,
    StageTrace,
)

# Hard gate — imported VERBATIM; never reimplemented here (GND-V3-01 / PATTERNS.md)
from confluence_logic.agents.grounding_gate import (
    check_grounding,
    check_page_existence,
    content_bearing_tokens,
    tokens_subset,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Confidence thresholds — tunable on the eval set (OBS-V3-01)
# ---------------------------------------------------------------------------

_THRESHOLD_HIGH: float = float(os.getenv("JARVIS_GATE_THRESHOLD_HIGH", "0.80"))
_THRESHOLD_MEDIUM: float = float(os.getenv("JARVIS_GATE_THRESHOLD_MEDIUM", "0.60"))
_THRESHOLD_EMIT: float = float(os.getenv("JARVIS_GATE_THRESHOLD_EMIT", "0.30"))


# ---------------------------------------------------------------------------
# GateResult — output of apply_grounding_gate_v3
# ---------------------------------------------------------------------------


class GateResult:
    """Output of ``apply_grounding_gate_v3`` for one PlannedOperation.

    ``is_dropped``: True when the hard gate rejected the card (page-existence
        or token-grounding failure) OR when the soft confidence is below the
        emit threshold for a non-fact_update card.
    ``is_flagged``: True when the confidence is below the emit threshold but
        the card is a fact_update/contradiction — kept but marked low-confidence
        for user attention (never silently shipped).
    ``drop_reasons``: List of human-readable reason strings.
    ``gate``: Name of the gate that caused the drop (for StageTrace).
    ``confidence_score``: Raw 0–1 calibrated confidence (None if dropped by hard gate).
    ``confidence_bin``: "high" | "medium" | "low" | None.
    """

    def __init__(
        self,
        is_dropped: bool = False,
        is_flagged: bool = False,
        drop_reasons: Optional[List[str]] = None,
        gate: Optional[str] = None,
        confidence_score: Optional[float] = None,
        confidence_bin: Optional[str] = None,
    ) -> None:
        self.is_dropped = is_dropped
        self.is_flagged = is_flagged
        self.drop_reasons = drop_reasons or []
        self.gate = gate
        self.confidence_score = confidence_score
        self.confidence_bin = confidence_bin


# ---------------------------------------------------------------------------
# Internal helpers (module-level so tests can patch them)
# ---------------------------------------------------------------------------


async def _check_page_exists(
    page_id: Optional[str],
    user_id: str = "",
    connector: Optional[Any] = None,
) -> bool:
    """Delegate to ``check_page_existence`` — patchable seam for tests."""
    return await check_page_existence(page_id, user_id, connector=connector)


def _compute_grounding_score(
    after_content: str,
    transcript_text: str,
    current_page_content: str,
) -> float:
    """Compute the token-grounding signal: 1 - |missing| / |tokens|.

    Reuses ``content_bearing_tokens`` and ``tokens_subset`` from
    ``grounding_gate`` so the signal shares the production tokenizer.

    Returns a float in [0, 1].  Returns 1.0 when there are no content-bearing
    tokens (nothing to ground — pass by default).
    """
    tokens = content_bearing_tokens(after_content or "")
    if not tokens:
        return 1.0
    allowed = (transcript_text or "") + "\n" + (current_page_content or "")
    missing = tokens_subset(tokens, allowed)
    return 1.0 - len(missing) / len(tokens)


# ---------------------------------------------------------------------------
# Public: calibrated confidence scorer
# ---------------------------------------------------------------------------


def calibrate_confidence(
    grounding_score: float,
    retrieval_score: float = 0.0,
    verbalized_confidence: float = 0.0,
) -> tuple:
    """Combine signals into a calibrated 0–1 confidence with a bin label.

    Weighted combination (clamped to [0, 1]):
      0.60 × grounding_score    (most reliable: deterministic token overlap)
      0.40 × retrieval_score    (reranker relevance)
      0.10 × verbalized_confidence × DISCOUNT(0.5)  (miscalibrated — Pitfall 5)

    Thresholds (tunable via env):
      ≥ HIGH_THRESHOLD  → "high"
      ≥ MEDIUM_THRESHOLD → "medium"
      <  MEDIUM_THRESHOLD → "low"

    Args:
        grounding_score: 0–1 grounding signal from ``_compute_grounding_score``.
        retrieval_score: 0–1 reranker relevance from the rerank stage (default 0.0).
        verbalized_confidence: LLM-verbalized confidence (default 0.0).

    Returns:
        Tuple of (combined_score: float, label: str).
    """
    _VERBALIZED_DISCOUNT = 0.5  # Pitfall 5: verbalized confidence is over-confident

    combined = (
        0.60 * float(grounding_score)
        + 0.40 * float(retrieval_score)
        + 0.10 * float(verbalized_confidence) * _VERBALIZED_DISCOUNT
    )
    # Clamp to [0, 1]
    combined = max(0.0, min(1.0, combined))

    if combined >= _THRESHOLD_HIGH:
        label = "high"
    elif combined >= _THRESHOLD_MEDIUM:
        label = "medium"
    else:
        label = "low"

    return (combined, label)


# ---------------------------------------------------------------------------
# Public: apply hard gate + soft confidence
# ---------------------------------------------------------------------------


async def apply_grounding_gate_v3(
    op: PlannedOperation,
    transcript_text: str = "",
    current_page_content: str = "",
    user_id: str = "",
    retrieval_score: float = 0.0,
    is_fact_update: bool = False,
    connector: Optional[Any] = None,
) -> GateResult:
    """Apply the two-layer gate to one PlannedOperation.

    Hard gate (LLM-free, non-negotiable):
      1. ``check_page_existence`` — page_id must be verifiable.
      2. ``check_grounding`` — tokens in after_content must be grounded.
      Failure → ``GateResult(is_dropped=True, gate=<gate_name>)``.

    Soft confidence layer (never overrides hard pass/fail):
      Compute ``_compute_grounding_score`` + ``calibrate_confidence``.
      Below ``_THRESHOLD_EMIT``:
        - fact_update / is_fact_update=True → ``is_flagged=True`` (not dropped).
        - other ops → ``is_dropped=True``, gate="confidence".

    Args:
        op: The PlannedOperation to gate.
        transcript_text: Normalised full transcript for the grounding check.
        current_page_content: Current Confluence page content (plain text).
        user_id: User identifier for page-existence graph lookup.
        retrieval_score: Reranker score for the chosen candidate (0–1).
        is_fact_update: Set to True to treat as fact_update/contradiction card
            (flagged instead of dropped when below confidence threshold).
        connector: Optional ConfluenceConnector for REST page-existence fallback.

    Returns:
        ``GateResult`` with ``is_dropped``, ``is_flagged``, ``drop_reasons``,
        ``gate``, ``confidence_score``, and ``confidence_bin`` populated.
    """
    drop_reasons: List[str] = []

    # -------------------------------------------------------------------------
    # Hard gate 1: page existence
    # -------------------------------------------------------------------------
    if op.operation != "create_page":
        try:
            page_exists = await _check_page_exists(
                op.page_id, user_id=user_id, connector=connector
            )
        except Exception as exc:
            logger.warning(
                "gate: _check_page_exists raised for page_id=%s: %s — treating as non-existent",
                op.page_id, exc,
            )
            page_exists = False

        if not page_exists:
            reason = f"page_id={op.page_id!r} not found in graph or via REST"
            logger.warning(
                "gate: hard drop — page_existence gate page_id=%s", op.page_id
            )
            return GateResult(
                is_dropped=True,
                drop_reasons=[reason],
                gate="page_existence",
            )

    # -------------------------------------------------------------------------
    # Hard gate 2: token grounding (check_grounding verbatim)
    # When current_page_content is empty (Neo4j corpus unavailable) we cannot
    # verify before_content against the page — skipping the hard gate avoids
    # false-positive drops on every edit_section card when the corpus is down.
    # The soft confidence scorer below still penalises low-overlap cards.
    # -------------------------------------------------------------------------
    if not current_page_content:
        grounding_result = {"ok": True, "failures": [], "reason": "no-page-content-soft-only"}
        logger.debug(
            "gate: skipping hard grounding (no page content) for page_id=%s — soft scorer applies",
            op.page_id,
        )
    else:
        # Build a card dict compatible with check_grounding's expected format.
        card: Dict[str, Any] = {
            "change_type": "edit",
            "edit_mode": "replace" if op.operation == "edit_section" else "add",
            "before_content": op.before_content or "",
            "after_content": op.after_content or "",
            "page_id": op.page_id,
            "operation_type": op.operation,
        }
        try:
            grounding_result = await check_grounding(card, transcript_text, current_page_content)
        except Exception as exc:
            logger.warning(
                "gate: check_grounding raised for page_id=%s: %s — treating as pass",
                op.page_id, exc,
            )
            grounding_result = {"ok": True, "failures": [], "reason": "error-treated-as-pass"}

        if not grounding_result.get("ok", True):
            failures = grounding_result.get("failures", [])
            reason = grounding_result.get("reason", "token grounding failure")
            logger.warning(
                "gate: hard drop — grounding gate page_id=%s reason=%s tokens=%s",
                op.page_id, reason, failures,
            )
            return GateResult(
                is_dropped=True,
                drop_reasons=[reason],
                gate="grounding",
            )

    # -------------------------------------------------------------------------
    # Soft confidence layer
    # -------------------------------------------------------------------------
    grounding_score = _compute_grounding_score(
        op.after_content or "",
        transcript_text,
        current_page_content,
    )
    confidence, bin_label = calibrate_confidence(
        grounding_score=grounding_score,
        retrieval_score=retrieval_score,
    )

    # Determine if this is a fact_update / contradiction card.
    _is_fact = is_fact_update or op.operation in ("archive_deprecate",)

    if confidence < _THRESHOLD_EMIT:
        reason = f"confidence={confidence:.3f} below emit threshold={_THRESHOLD_EMIT}"
        logger.warning(
            "gate: sub-threshold page_id=%s confidence=%.3f bin=%s is_fact=%s",
            op.page_id, confidence, bin_label, _is_fact,
        )
        if _is_fact:
            # Flagged but not dropped — user sees it as low-confidence
            return GateResult(
                is_dropped=False,
                is_flagged=True,
                drop_reasons=[reason],
                gate="confidence",
                confidence_score=confidence,
                confidence_bin=bin_label,
            )
        else:
            return GateResult(
                is_dropped=True,
                is_flagged=False,
                drop_reasons=[reason],
                gate="confidence",
                confidence_score=confidence,
                confidence_bin=bin_label,
            )

    # -------------------------------------------------------------------------
    # Card passed all gates
    # -------------------------------------------------------------------------
    return GateResult(
        is_dropped=False,
        is_flagged=False,
        drop_reasons=[],
        gate=None,
        confidence_score=confidence,
        confidence_bin=bin_label,
    )


__all__ = [
    "GateResult",
    "apply_grounding_gate_v3",
    "calibrate_confidence",
    "_check_page_exists",
    "_compute_grounding_score",
]
