"""Stage 5: Workspace contradiction sweep (CON-V3-01).

For each ``fact_update`` (or ``deprecation``) intent, finds EVERY section in
the pre-indexed section corpus whose text contains the ``old_value`` verbatim
AND whose terms overlap the ``subject`` tokens.  Each candidate is validated
by a ``MODEL_NANO`` entailment pass; only sections that *contradict* the new
intent survive (precision).  All surviving sections are grouped under one
``ContradictionGroupResult`` sharing a single ``group_id`` so the UI can
render them as a single logical decision (UI-V3-01).

Design notes:
  - Recall pass is deterministic (over the pre-indexed corpus — RAG-first).
  - Precision pass uses ``MODEL_NANO`` entailment classification
    (``{contradicts, neutral, entailed}``).
  - Entailment errors KEEP the candidate flagged low-confidence (a missed
    contradiction is the worst failure — degrade toward recall).
  - ``graph_user_id`` is an explicit parameter; ContextVar never read.
  - ``%``-style logging per CLAUDE.md convention.

CON-V3-01 acceptance bar: contradiction recall = 100% on the SOC2 fixture
(both pg-soc2 and pg-security-overview found and grouped under one group_id).
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import uuid
from typing import Any, Dict, List, Optional

from confluence_logic.pipeline.contracts import (
    AffectedPage,
    ChangeIntentV3,
    ContradictionGroup,
    StageTrace,
)
from confluence_logic.pipeline.model_config import MODEL_NANO

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Concurrency cap for the nano entailment fan-out
# ---------------------------------------------------------------------------

_ENTAIL_SEM_SIZE: int = int(os.getenv("JARVIS_CONTRADICT_CONCURRENCY", "6"))




# ---------------------------------------------------------------------------
# Internal helpers (module-level so tests can patch them)
# ---------------------------------------------------------------------------


async def _find_pages_stating_old_value(
    old_value: str,
    subject: str,
    workspace_pages: List[Dict[str, Any]],
) -> List[str]:
    """Return page_ids whose content contains old_value verbatim (recall pass).

    Deterministic — scans the pre-indexed workspace_pages corpus.  RAG-first:
    no live Confluence API scan.

    Matching rule:
      (a) Normalized page content contains old_value (case-insensitive substring).
      (b) Normalized page content OR title contains at least one subject token.

    Subject tokenisation reuses ``content_bearing_tokens`` from grounding_gate
    so the lexical signal shares the production tokenizer.
    """
    if not old_value or not workspace_pages:
        return []

    # Lazy import to avoid circular imports at module load time.
    try:
        from confluence_logic.agents.grounding_gate import content_bearing_tokens
    except Exception:
        # Graceful degradation: plain whitespace tokeniser if gate unavailable.
        def content_bearing_tokens(text: str) -> List[str]:  # type: ignore[misc]
            return [t.lower() for t in text.split() if len(t) >= 2]

    subject_tokens = set(content_bearing_tokens(subject)) if subject else set()
    old_norm = old_value.strip().lower()

    matched: List[str] = []
    for page in workspace_pages:
        page_id = page.get("page_id")
        if not page_id:
            continue

        # Build text blob: title + content_html stripped of HTML tags
        title = page.get("title") or ""
        html = page.get("content_html") or ""
        text_blob = _strip_html(html)
        combined_norm = (title + " " + text_blob).lower()

        # (a) old_value verbatim presence
        if old_norm not in combined_norm:
            continue

        # (b) subject token overlap
        if subject_tokens:
            blob_tokens = set(content_bearing_tokens(title + " " + text_blob))
            if not subject_tokens & blob_tokens:
                continue

        matched.append(page_id)

    return matched


def _strip_html(html: str) -> str:
    """Strip HTML tags returning plain text (no new deps — stdlib re)."""
    return re.sub(r"<[^>]+>", " ", html or "")


async def _entailment_classify(
    page_id: str,
    page_content: str,
    subject: str,
    old_value: str,
    new_value: Optional[str],
) -> str:
    """Classify one page section as {contradicts, neutral, entailed}.

    Uses MODEL_NANO via the OpenAI chat completions API directly (not the
    Agents SDK — lower overhead for a single classification call).

    Returns 'contradicts', 'neutral', or 'entailed'.
    Raises on API error so the caller can degrade to keeping the candidate.
    """
    try:
        import openai  # lazy import — no side effects at module load
        client = openai.AsyncOpenAI()
        prompt = (
            f"Subject: {subject}\n"
            f"Old value: {old_value}\n"
            f"New value: {new_value or '(deprecated)'}\n"
            f"Page excerpt: {page_content[:800]}\n\n"
            f"Does the page excerpt assert that '{subject}' is '{old_value}' "
            f"in a way that contradicts '{new_value}'?\n"
            f"Reply with exactly one word: contradicts, neutral, or entailed."
        )
        response = await client.chat.completions.create(
            model=MODEL_NANO,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=10,
            temperature=0,
        )
        label = (response.choices[0].message.content or "").strip().lower()
        if label in {"contradicts", "neutral", "entailed"}:
            return label
        # Unexpected label — treat as neutral (keep precision)
        logger.warning(
            "contradict: unexpected entailment label %r for page %s — treating neutral",
            label, page_id,
        )
        return "neutral"
    except Exception as exc:
        # Re-raise so the caller keeps the candidate (recall-biased degradation)
        raise exc


async def _is_page_fully_superseded(
    page_id: str,
    page_content: str,
    subject: str,
    old_value: str,
) -> bool:
    """Heuristic: return True if the page appears to be entirely about old_value.

    Simple deterministic heuristic for deprecation intents — if the page's
    content is dominated by the old_value references and there is no newer
    context, flag it for archive_deprecate.

    Designed to be patchable in tests.
    """
    if not page_content or not old_value:
        return False
    norm = page_content.lower()
    old_norm = old_value.lower()
    # Rough density heuristic: old_value appears 3+ times relative to length
    count = norm.count(old_norm)
    density = count / (len(norm.split()) + 1)
    return density >= 0.05


# ---------------------------------------------------------------------------
# Public entrypoint
# ---------------------------------------------------------------------------


async def detect_contradictions(
    intent: ChangeIntentV3,
    workspace_pages: Optional[List[Dict[str, Any]]] = None,
    graph_user_id: str = "",
    trace: Optional[Any] = None,
) -> List[ContradictionGroup]:
    """Workspace contradiction sweep for one intent (CON-V3-01).

    Algorithm:
      1. Recall pass (deterministic): find every workspace page whose content
         contains ``intent.old_value`` verbatim AND whose terms overlap
         ``intent.subject`` tokens.
      2. Precision pass (LLM ``MODEL_NANO``): classify each candidate as
         {contradicts, neutral, entailed} in parallel (bounded ``Semaphore``).
         Only ``contradicts`` survivors proceed.
         On entailment error → KEEP the candidate flagged low-confidence
         (recall-biased degradation per RESEARCH §Retry/Fallback).
      3. Group all survivors under ONE ``ContradictionGroupResult`` with a
         shared ``group_id``.  Deprecation candidates flagged for
         ``archive_deprecate`` if ``_is_page_fully_superseded`` returns True.

    Args:
        intent: The ChangeIntentV3 driving this sweep.  Only processed when
            ``kind`` is ``fact_update`` or ``deprecation`` and
            ``old_value`` is non-empty.
        workspace_pages: Pre-fetched list of page dicts (``page_id``,
            ``title``, ``content_html``).  When None the function returns []
            (corpus not available — graceful degradation).
        graph_user_id: Explicit user scoping key (never read via ContextVar).
        trace: Optional TraceBus for StageTrace events.

    Returns:
        List of ContradictionGroupResult (at most one per intent in normal
        cases, more if the corpus is partitioned).
    """
    # Guard: only fact_update + deprecation with old_value produce contradictions.
    if intent.kind not in ("fact_update", "deprecation"):
        return []
    old_value = (intent.old_value or "").strip()
    if not old_value:
        return []
    if not workspace_pages:
        return []

    subject = intent.subject or ""
    new_value = intent.new_value or None

    # --- Recall pass ---------------------------------------------------------
    try:
        candidate_page_ids = await _find_pages_stating_old_value(
            old_value, subject, workspace_pages
        )
    except Exception as exc:
        logger.warning(
            "contradict: _find_pages_stating_old_value failed for subject=%r: %s",
            subject, exc,
        )
        candidate_page_ids = []

    if not candidate_page_ids:
        return []

    # Build a lookup: page_id → page dict
    page_lookup: Dict[str, Dict[str, Any]] = {
        p["page_id"]: p for p in workspace_pages if p.get("page_id")
    }

    # --- Precision pass (parallel nano entailment) ---------------------------
    sem = asyncio.Semaphore(_ENTAIL_SEM_SIZE)

    async def _classify_one(pid: str) -> tuple[str, str]:
        """Returns (pid, label) or raises."""
        page = page_lookup.get(pid, {})
        html = page.get("content_html") or ""
        content = _strip_html(html)
        async with sem:
            return (pid, await _entailment_classify(pid, content, subject, old_value, new_value))

    tasks = [_classify_one(pid) for pid in candidate_page_ids]
    raw_results = await asyncio.gather(*tasks, return_exceptions=True)

    # --- Resolve survivors ----------------------------------------------------
    survivors: List[tuple[str, bool]] = []  # (page_id, low_confidence)
    for i, result in enumerate(raw_results):
        pid = candidate_page_ids[i]
        if isinstance(result, Exception):
            # Entailment error → KEEP flagged low-confidence (recall-biased)
            logger.warning(
                "contradict: entailment classify failed for page %s: %s — keeping (low-confidence)",
                pid, result,
            )
            survivors.append((pid, True))
        else:
            _, label = result
            if label == "contradicts":
                survivors.append((pid, False))
            elif label == "neutral":
                # Precision drop: mentions old_value for an unrelated reason
                logger.warning(
                    "contradict: page %s label=neutral for subject=%r old_value=%r — dropping",
                    pid, subject, old_value,
                )
            else:
                # entailed: the page already says the new value (no contradiction)
                logger.warning(
                    "contradict: page %s label=%r for subject=%r — dropping",
                    pid, label, subject,
                )

    if not survivors:
        return []

    # --- Group under one group_id --------------------------------------------
    group_id = intent.dedup_key or str(uuid.uuid4())

    affected: List[AffectedPage] = []
    for pid, low_conf in survivors:
        page = page_lookup.get(pid, {})
        page_title = page.get("title", "")
        html = page.get("content_html") or ""
        content = _strip_html(html)

        # Determine recommended operation
        recommended_op = "edit_section"
        if intent.kind == "deprecation":
            try:
                is_superseded = await _is_page_fully_superseded(
                    pid, content, subject, old_value
                )
                if is_superseded:
                    recommended_op = "archive_deprecate"
            except Exception as exc:
                logger.warning(
                    "contradict: _is_page_fully_superseded failed for page %s: %s",
                    pid, exc,
                )

        affected.append(AffectedPage(
            page_id=pid,
            page_title=page_title,
            section_heading=None,  # section-level resolution is in plan_ops
            group_id=group_id,
            recommended_op=recommended_op,
            low_confidence=low_conf,
        ))

    group = ContradictionGroup(
        subject=subject,
        old_value=old_value,
        new_value=new_value,
        affected_pages=affected,
        operations=[],  # populated by plan_ops stage
    )

    logger.warning(
        "contradict: intent subject=%r old_value=%r → %d affected pages group_id=%s",
        subject, old_value, len(affected), group_id,
    )

    return [group]


__all__ = [
    "detect_contradictions",
    "_find_pages_stating_old_value",
    "_is_page_fully_superseded",
]
