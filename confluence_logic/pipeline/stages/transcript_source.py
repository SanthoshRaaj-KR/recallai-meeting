"""Stage 0: Transcript source selection with killswitch + fallback (SPK-V3-01).

Selects the best available transcript source for the post-meeting extraction
pipeline:

  1. When ``JARVIS_RECALL_TRANSCRIPT_ENABLED=1`` AND ``ctx.bot_id`` is set,
     attempts to fetch the Recall.ai diarized transcript via
     ``recall_transcript.fetch_recall_transcript``.  If the fetch returns a
     non-empty result the diarized transcript is used (real participant names).

  2. Otherwise (killswitch OFF, no bot_id, or Recall fetch returned None/[]),
     falls back cleanly to the existing LiveKit transcript_log supplied as
     ``ctx.transcript_log`` — behavior is byte-identical to the pre-plan path.

BOUNDARY (agent_worker.py intentionally untouched):
  The live LiveKit STT path (``agent_worker.py`` ``_post_transcript`` /
  ``jarvis_agentic.py`` ``receive_livekit_transcript``) defaults
  ``speaker="Meeting"`` because it captures MIXED audio with no per-speaker
  attribution.  This stage does NOT change the live path — it provides a
  post-meeting override when Recall diarization is available and enabled.

OBSERVABILITY (OBS-V3-01):
  Emits a ``StageTrace`` event via ``ctx.trace_bus.emit(...)`` recording which
  source was selected (``source="recall"`` | ``source="livekit_fallback"``)
  and the entry count.

OUTPUT:
  Populates ``ctx.transcript_text`` with the joined transcript (one
  ``"<participant>: <text>"`` line per entry).  Returns the raw entry list for
  downstream use (the extract stage may re-read the entries for offset
  computation).  Never raises — falls back on any error.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

from confluence_logic.pipeline.contracts import StageTrace
from confluence_logic.pipeline.recall_transcript import (
    RECALL_TRANSCRIPT_ENABLED,
    fetch_recall_transcript,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _entries_from_transcript_log(
    transcript_log: Optional[List[Any]],
) -> List[Dict[str, Any]]:
    """Normalize the raw LiveKit transcript_log into the shared entry shape.

    Each entry in transcript_log is a dict that ``_format_transcript`` in
    ``review/api.py`` reads via ``entry.get('participant','Unknown')``.
    We preserve that shape verbatim so downstream rendering is unchanged.
    """
    if not transcript_log:
        return []
    out = []
    for raw in transcript_log:
        if not isinstance(raw, dict):
            continue
        text = str(raw.get("text") or raw.get("transcript", "")).strip()
        if not text:
            continue
        out.append(
            {
                "participant": str(raw.get("participant") or "Meeting"),
                "text": text,
                "timestamp": float(raw.get("timestamp") or raw.get("created_at") or 0.0),
                "source": "livekit",
            }
        )
    return out


def _build_transcript_text(entries: List[Dict[str, Any]]) -> str:
    """Join entries into the normalized full-transcript string.

    Format: ``"<participant>: <text>\\n"`` per entry.
    This is the string used by the extract stage for evidence-span
    offset computation (EXT-V3-01).
    """
    return "\n".join(
        f"{e.get('participant', 'Meeting')}: {e.get('text', '')}" for e in entries
    )


# ---------------------------------------------------------------------------
# Public API — Stage 0
# ---------------------------------------------------------------------------

async def load_transcript(
    session_id: str,
    ctx: Any,
) -> List[Dict[str, Any]]:
    """Select the best available transcript source and populate ctx.transcript_text.

    Branch logic:
      - If RECALL_TRANSCRIPT_ENABLED AND ctx.bot_id is set, attempt to fetch
        the Recall.ai diarized transcript.  On a truthy non-empty result use
        it (real participant names, source="recall").
      - Otherwise fall back to ctx.transcript_log (source="livekit_fallback").

    Populates ``ctx.transcript_text`` (joined ``"<participant>: <text>"`` lines).

    Emits a ``StageTrace`` via ``ctx.trace_bus`` recording the chosen source
    and entry count (OBS-V3-01).

    Args:
        session_id: The post-meeting session id (for log correlation).
        ctx: A ``PipelineContext`` instance carrying bot_id, transcript_log,
             trace_bus, and killswitch snapshots.

    Returns:
        The selected list of normalized entry dicts.  Never raises.
    """
    start_ts = time.monotonic()
    chosen_source = "livekit_fallback"
    entries: List[Dict[str, Any]] = []

    try:
        # --- Attempt Recall diarized transcript when killswitch is ON -------
        bot_id: Optional[str] = getattr(ctx, "bot_id", None)

        if RECALL_TRANSCRIPT_ENABLED and bot_id:
            try:
                recall_entries = await fetch_recall_transcript(bot_id)
                if recall_entries:
                    entries = recall_entries
                    chosen_source = "recall"
                    logger.info(
                        "transcript_source: session=%s using Recall diarized transcript "
                        "(%d entries)",
                        session_id,
                        len(entries),
                    )
                else:
                    logger.info(
                        "transcript_source: session=%s Recall fetch returned empty/None — "
                        "falling back to livekit log",
                        session_id,
                    )
            except Exception as exc:
                logger.warning(
                    "transcript_source: session=%s Recall fetch error — falling back: %s",
                    session_id,
                    exc,
                )

        # --- Fallback: LiveKit transcript_log --------------------------------
        if chosen_source != "recall":
            transcript_log = getattr(ctx, "transcript_log", None)
            entries = _entries_from_transcript_log(transcript_log)
            logger.info(
                "transcript_source: session=%s using livekit_fallback (%d entries)",
                session_id,
                len(entries),
            )

        # --- Populate ctx.transcript_text ------------------------------------
        ctx.transcript_text = _build_transcript_text(entries)

    except Exception as exc:
        logger.warning(
            "transcript_source: session=%s unexpected error — returning empty: %s",
            session_id,
            exc,
        )
        entries = []
        chosen_source = "livekit_fallback"

    # --- Emit StageTrace (OBS-V3-01) ----------------------------------------
    latency_ms = (time.monotonic() - start_ts) * 1000.0
    trace = StageTrace(
        stage="transcript_source",
        phase="end",
        latency_ms=round(latency_ms, 1),
        candidates_in=None,
        candidates_out=len(entries),
        dropped=0,
        drop_reason=None,
        gate=None,
    )
    # Carry the chosen source in a non-standard field so callers can inspect it.
    # We attach it to the trace object dynamically (Pydantic extra='ignore' is
    # not set, so we store it as a plain attribute after construction).
    object.__setattr__(trace, "_source", chosen_source)  # type: ignore[arg-type]

    try:
        trace_bus = getattr(ctx, "trace_bus", None)
        job_id: Optional[str] = getattr(ctx, "session_id", None)
        if trace_bus is not None:
            trace_bus.emit(trace, job_id=job_id)
    except Exception:
        pass  # Tracing must never break the pipeline

    # Attach chosen source to ctx for downstream inspection / tests.
    ctx.__dict__["_transcript_source"] = chosen_source

    return entries
