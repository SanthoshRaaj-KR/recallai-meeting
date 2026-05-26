"""Stage 1: Transcript → ChangeIntentV3[] structured final-state extraction (EXT-V3-01).

Adapts the Phase 10 FactExtractionAgent to produce typed ChangeIntentV3 with:
  * ``kind`` — Literal discriminator (decision/fact_update/action_item/new_workstream/deprecation)
  * ``evidence`` — ≥1 EvidenceSpan whose char offsets are computed by Python str.find
    against the normalised transcript (NEVER trusted from the LLM — Pitfall 3).
  * final-state collapse + dedup by (normalised subject, kind) — LAST-wins

Security: evidence char offsets are computed in Python via str.find.  Intents whose
verbatim quote cannot be located in the transcript are rejected (logged, never emitted
with a fabricated span) — this is EXT-V3-01 grounding-at-extraction and a mitigation
for T-11-06 (Tampering via LLM-emitted offsets).

Chunking/overlap and the FINAL-STATE collapse rules are reused verbatim from
confluence_logic/agents/fact_extraction_agent.py (FACT_EXTRACTION_PROMPT rules A/B/C).
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
from typing import Any, Dict, List, Optional

from agents import Agent, AgentOutputSchema, Runner
from pydantic import BaseModel

from confluence_logic.pipeline.contracts import ChangeIntentV3, EvidenceSpan
from confluence_logic.pipeline.model_config import MODEL_WORKER

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level chunking constants (reused from fact_extraction_agent)
# ---------------------------------------------------------------------------

JARVIS_FACT_CHUNK_CHARS: int = int(os.getenv("JARVIS_FACT_CHUNK_CHARS", "55000"))
JARVIS_FACT_CHUNK_OVERLAP: int = int(os.getenv("JARVIS_FACT_CHUNK_OVERLAP", "2000"))

# ---------------------------------------------------------------------------
# LLM output schema — what the agent emits per intent
# ---------------------------------------------------------------------------


class _RawIntent(BaseModel):
    """Raw per-intent JSON emitted by the extraction agent.

    ``kind`` maps to ChangeIntentV3.kind.
    ``verbatim_quote`` is an *optional* verbatim string the LLM believes appears
    in the transcript.  Char offsets are NEVER requested from the LLM (Pitfall 3)
    — Python str.find computes them afterwards.
    """

    kind: str = "fact_update"
    subject: str = ""
    old_value: str = ""
    new_value: str = ""
    instruction: str = ""
    target_hint: str = ""
    verbatim_content: str = ""
    dedup_key: str = ""
    verbatim_quote: str = ""   # LLM-supplied quote; offsets computed in Python


class _RawExtractionOutput(BaseModel):
    """Wrapper schema returned by the extraction agent."""

    intents: List[_RawIntent] = []


# ---------------------------------------------------------------------------
# Extraction agent prompt — extends FACT_EXTRACTION_PROMPT (reuses rules A/B/C)
# ---------------------------------------------------------------------------

_EXTRACT_V3_PROMPT = (
    "You are a meeting intelligence assistant. Read the meeting transcript carefully and "
    "extract every documentation-worthy change as a structured intent.\n\n"
    "For EACH distinct change discussed (no skipping), output one object in 'intents' with:\n"
    "  kind: one of 'decision', 'fact_update', 'action_item', 'new_workstream', 'deprecation'\n"
    "  subject: WHAT is being changed (e.g. 'SOC2 audit schedule', 'payments on-call owner')\n"
    "  old_value: the current value being replaced (empty if additive/create)\n"
    "  new_value: the COMPLETE new value to apply — include ALL changed facts in one field. "
    "If the quarter changed AND a specific date was mentioned, put both: e.g. 'Q4 (December 26)'. "
    "Do not split related facts across new_value and instruction; capture the full replacement here. "
    "(empty for removal/deprecation)\n"
    "  instruction: human-readable description of the change\n"
    "  target_hint: short hint for which Confluence page/section this affects\n"
    "  verbatim_content: for add/create intents, exact quoted content from transcript\n"
    "  dedup_key: short normalized identifier for this change (e.g. 'soc2-audit-quarter')\n"
    "  verbatim_quote: the EXACT words spoken in the meeting that most directly STATE this "
    "change (not background context — the actual moment it was decided or announced). "
    "Must be 5–30 words. Copy character-for-character including capitalization and punctuation "
    "exactly as they appear in the transcript — do NOT paraphrase or summarise. "
    "For a date/value change, quote where the new date or value was spoken. "
    "For an assignment, quote where the person was named for the task. "
    "This string must be findable by Python str.find() — any paraphrase will cause the intent to be dropped.\n\n"
    "Kind selection:\n"
    "  decision      — a definitive choice made by the group\n"
    "  fact_update   — a specific factual value is changing (version, name, date, metric)\n"
    "  action_item   — a task assigned to someone\n"
    "  new_workstream — a new project/workstream/service was decided\n"
    "  deprecation   — something is being REMOVED, retired, or archived. Use this when a PAGE or "
    "SECTION is being deleted (e.g. 'remove the X section', 'delete the Y section', 'take out "
    "the Z section from the page'). Set new_value='' for deprecation intents.\n\n"
    "    CRITICAL RULE A — FINAL STATE ONLY:\n"
    "    Extract the NET FINAL agreed state, not intermediate positions. "
    "If the group first proposes X and then reverts or revises to Y, produce ONE intent for Y. "
    "If the final decision is 'keep as-is / no change', produce ZERO intents for that topic. "
    "NEVER produce two intents for the same topic with different new_values — that means you "
    "captured an intermediate step that was overruled.\n"
    "    CRITICAL RULE B — NO DUPLICATES:\n"
    "    Each distinct change must appear exactly once. If the same update was mentioned at "
    "multiple points in the meeting, extract it only once with the most complete information.\n"
    "    CRITICAL RULE C — verbatim_content for add/create:\n"
    "    For kind='new_workstream' or create intents where specific items were named, copy "
    "the EXACT items into verbatim_content. Do NOT paraphrase.\n\n"
    "Return JSON only — no markdown, no explanation. "
    "Return {\"intents\": []} if no documentation changes were discussed."
)

# ---------------------------------------------------------------------------
# Module-level singleton agent
# ---------------------------------------------------------------------------

_extract_agent = Agent(
    name="ExtractV3Agent",
    model=MODEL_WORKER,
    instructions=_EXTRACT_V3_PROMPT,
    output_type=AgentOutputSchema(_RawExtractionOutput, strict_json_schema=False),
)

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _norm_key(text: str) -> str:
    """Normalize text for dedup keying (lowercase, strip punct, collapse whitespace, cap at 60)."""
    t = re.sub(r"[^\w\s]", " ", (text or "").lower())
    return re.sub(r"\s+", " ", t).strip()[:60]


def _find_evidence(transcript: str, raw: _RawIntent) -> Optional[EvidenceSpan]:
    """Compute a single EvidenceSpan by locating a quote verbatim in the transcript.

    Search order:
    1. ``raw.verbatim_quote`` — LLM-supplied quote (most specific)
    2. ``raw.new_value``      — fallback: find the new value in the transcript
    3. ``raw.old_value``      — fallback: find the old value in the transcript
    4. ``raw.subject``        — last resort: find the subject

    Char offsets are computed ONLY via Python str.find — never from the LLM (Pitfall 3).
    Returns None if nothing can be located verbatim.
    """
    candidates = [
        raw.verbatim_quote,
        raw.new_value,
        raw.old_value,
        raw.subject,
    ]
    for quote in candidates:
        if not quote or not quote.strip():
            continue
        start = transcript.find(quote)
        if start != -1:
            return EvidenceSpan(
                text=quote,
                start=start,
                end=start + len(quote),
            )
    return None


def _coerce_kind(raw_kind: str) -> Optional[str]:
    """Validate and normalise a raw kind string to one of the five ChangeIntentKind values."""
    valid = {"decision", "fact_update", "action_item", "new_workstream", "deprecation"}
    k = (raw_kind or "").strip().lower()
    return k if k in valid else None


def _merge_intents(
    chunks: List[List[_RawIntent]],
) -> List[_RawIntent]:
    """Merge per-chunk raw intent lists with LAST-wins dedup by (norm_subject, kind).

    Preserves the FINAL-STATE collapse: the last intent seen for a given
    (normalised subject, kind) key is the agreed final state.
    """
    key_order: List[tuple] = []
    intent_last: Dict[tuple, _RawIntent] = {}

    for chunk in chunks:
        for intent in chunk:
            k = (_norm_key(intent.subject), (intent.kind or "").strip().lower())
            if not all(k):
                continue
            if k not in intent_last:
                key_order.append(k)
            intent_last[k] = intent  # LAST wins — final state of the discussion

    return [intent_last[k] for k in key_order]


# ---------------------------------------------------------------------------
# Internal LLM extraction — patchable by tests
# ---------------------------------------------------------------------------


async def _run_llm_extraction(text: str) -> List[Dict[str, Any]]:
    """Run the extraction agent on a single text chunk.

    Returns a list of raw intent dicts.  This function is a separate coroutine
    so that tests can patch ``confluence_logic.pipeline.stages.extract._run_llm_extraction``
    without touching the agent or the Runner.
    """
    result = await Runner.run(_extract_agent, text)
    if isinstance(result.final_output, _RawExtractionOutput):
        output = result.final_output
    else:
        output = _RawExtractionOutput.model_validate(result.final_output)
    return [i.model_dump() for i in output.intents]


async def _extract_chunk(text: str) -> List[_RawIntent]:
    """Run LLM extraction on a single chunk, returning validated _RawIntent objects."""
    raw_dicts = await _run_llm_extraction(text)
    intents: List[_RawIntent] = []
    for d in raw_dicts:
        try:
            intents.append(_RawIntent.model_validate(d))
        except Exception as exc:
            logger.warning("_extract_chunk: invalid intent dict %s: %s", d, exc)
    return intents


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def extract_intents(
    transcript: str,
    ctx: Any = None,
) -> List[ChangeIntentV3]:
    """Extract typed ChangeIntentV3 from a meeting transcript (EXT-V3-01).

    Args:
        transcript: Normalised full transcript text.  Used both as LLM input
            and as the reference string for Python str.find offset computation.
        ctx: Optional PipelineContext.  When provided, ctx.transcript_text is
            updated to *transcript* so downstream stages reference the same
            normalised string.  Ignored when running in standalone/test mode.

    Returns:
        A list of ChangeIntentV3 objects, each carrying ≥1 verbatim EvidenceSpan
        whose char offsets were computed via str.find (never from the LLM).
        Intents whose evidence quote cannot be located verbatim in the transcript
        are silently dropped (logged at WARNING level).
        On any LLM error the stage returns [] (graceful degradation, never raises).
    """
    try:
        text = transcript or ""
        if not text.strip():
            return []

        # Update ctx.transcript_text if a PipelineContext was supplied
        if ctx is not None and hasattr(ctx, "transcript_text"):
            ctx.transcript_text = text

        # ---------------------
        # Chunk the transcript
        # ---------------------
        if len(text) <= JARVIS_FACT_CHUNK_CHARS:
            chunk_texts = [text]
        else:
            chunk_texts = []
            start = 0
            while start < len(text):
                end = start + JARVIS_FACT_CHUNK_CHARS
                chunk_texts.append(text[start:end])
                if end >= len(text):
                    break
                start = end - JARVIS_FACT_CHUNK_OVERLAP
            logger.info(
                "extract_intents: transcript %d chars → %d chunks of ~%d chars each",
                len(text), len(chunk_texts), JARVIS_FACT_CHUNK_CHARS,
            )

        # ----------------------------------
        # Run LLM extraction (parallel chunks)
        # ----------------------------------
        if len(chunk_texts) == 1:
            # Single chunk — avoid asyncio.gather overhead
            raw_chunks = [await _extract_chunk(chunk_texts[0])]
        else:
            results = await asyncio.gather(
                *[_extract_chunk(c) for c in chunk_texts],
                return_exceptions=True,
            )
            raw_chunks = []
            for i, r in enumerate(results):
                if isinstance(r, Exception):
                    logger.warning(
                        "extract_intents: chunk %d failed (non-fatal): %s", i, r
                    )
                else:
                    raw_chunks.append(r)

        # ---------------------------
        # Final-state collapse + dedup
        # ---------------------------
        merged = _merge_intents(raw_chunks)

        # ------------------------------------
        # Coerce to ChangeIntentV3 + compute Python offsets
        # ------------------------------------
        intents: List[ChangeIntentV3] = []
        for raw in merged:
            kind = _coerce_kind(raw.kind)
            if kind is None:
                logger.warning(
                    "extract_intents: dropping intent with invalid kind %r (subject=%r)",
                    raw.kind, raw.subject,
                )
                continue

            # Compute evidence span via Python str.find — NEVER trust LLM offsets
            span = _find_evidence(text, raw)
            if span is None:
                logger.warning(
                    "extract_intents: dropping intent %r — no verbatim quote found in transcript "
                    "(EXT-V3-01 grounding-at-extraction)",
                    raw.subject,
                )
                continue

            # Compute dedup_key in Python if LLM didn't supply one
            dedup_key = (
                raw.dedup_key.strip()
                if raw.dedup_key.strip()
                else f"{_norm_key(raw.subject)}-{kind}"
            )

            try:
                intent = ChangeIntentV3(
                    kind=kind,  # type: ignore[arg-type]
                    subject=raw.subject,
                    old_value=raw.old_value,
                    new_value=raw.new_value,
                    instruction=raw.instruction,
                    target_hint=raw.target_hint,
                    verbatim_content=raw.verbatim_content,
                    dedup_key=dedup_key,
                    evidence=[span],
                )
            except Exception as exc:
                logger.warning(
                    "extract_intents: ChangeIntentV3 construction failed for %r: %s",
                    raw.subject, exc,
                )
                continue

            intents.append(intent)

        return intents

    except Exception as exc:
        logger.warning("extract_intents: stage failed (non-fatal), returning []: %s", exc)
        return []
