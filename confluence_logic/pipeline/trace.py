"""TraceBus — per-stage observability emitter (OBS-V3-01).

Wraps an ``asyncio.Queue`` registry with the no-op-if-no-consumer semantics
mirrored from ``review/api.py`` ``_emit`` / ``_job_queues`` / ``_SENTINEL``.

Usage:
    bus = TraceBus()
    q = bus.register(job_id)   # called by the SSE endpoint
    bus.emit(trace)            # called by each pipeline stage
    bus.close(job_id)          # puts _SENTINEL to stop event_generator

Emit shape (stable SSE event strings — sync-sage-bot consumer depends on these):
  phase="start" -> {"type": "stage_start",    "stage": <name>}
  phase="end"   -> {"type": "stage_progress", "stage": <name>, "candidates": <out>}

Both events carry extra StageTrace fields (latency_ms, dropped, drop_reason,
gate) for structured logging — the SSE consumer ignores unknown keys gracefully.

If no queue is registered for a job_id, emit is a no-op (RESEARCH Pitfall 1:
the StageIndicator advances to the latest received stage, so early events
before the SSE consumer connects are acceptable losses).
"""

from __future__ import annotations

import asyncio
import logging
from typing import Dict, Optional

from confluence_logic.pipeline.contracts import StageTrace

logger = logging.getLogger(__name__)

_SENTINEL = object()  # Signals event_generator to break out of its loop.


class TraceBus:
    """Per-job SSE queue registry + structured logger for StageTrace events.

    One TraceBus instance is created per pipeline run and attached to
    PipelineContext.trace_bus.  The SSE endpoint registers a consumer queue
    via ``register(job_id)``; ``emit`` is a no-op when no consumer is present.
    """

    def __init__(self) -> None:
        self._queues: Dict[str, asyncio.Queue] = {}

    def register(self, job_id: str) -> asyncio.Queue:
        """Create and return a new asyncio.Queue for the given job_id.

        Called by the SSE endpoint when the consumer first connects.  If a
        queue already exists (e.g. reconnect), the existing queue is replaced
        with a fresh one to avoid delivering stale events.
        """
        q: asyncio.Queue = asyncio.Queue()
        self._queues[job_id] = q
        return q

    def emit(self, trace: StageTrace, job_id: Optional[str] = None) -> None:
        """Emit a StageTrace event to the SSE queue and to structured logs.

        Args:
            trace: The StageTrace to emit.
            job_id: The pipeline job id.  When None (e.g. in unit tests with
                no consumer registered), emit is still a no-op for the queue
                but always writes the structured log line.

        SSE event shapes (stable — sync-sage-bot SSE consumer depends on the
        ``type`` and ``stage`` keys):
          phase="start" -> {"type": "stage_start",    "stage": ...}
          phase="end"   -> {"type": "stage_progress", "stage": ..., "candidates": N}
        """
        # Build the SSE event dict.
        if trace.phase == "start":
            event: dict = {"type": "stage_start", "stage": trace.stage}
        else:
            event = {
                "type": "stage_progress",
                "stage": trace.stage,
                "candidates": trace.candidates_out,
            }

        # Carry extra fields for diagnostic consumers (ignored by the UI).
        if trace.latency_ms is not None:
            event["latency_ms"] = trace.latency_ms
        if trace.dropped:
            event["dropped"] = trace.dropped
        if trace.drop_reason:
            event["drop_reason"] = trace.drop_reason
        if trace.gate:
            event["gate"] = trace.gate

        # --- Queue emit (no-op if no consumer registered) ---
        if job_id is not None:
            q = self._queues.get(job_id)
            if q is None and self._queues:
                # job_id lookup miss (session_id vs pipeline job UUID mismatch
                # from _emit_trace in run.py) — broadcast to all registered
                # queues. Each TraceBus has exactly one queue per pipeline run.
                for q in self._queues.values():
                    q.put_nowait(event)
            elif q is not None:
                q.put_nowait(event)

        # --- Structured log (always) ---
        logger.info(
            "stage=%s phase=%s latency_ms=%s candidates_in=%s candidates_out=%s "
            "dropped=%s drop_reason=%s gate=%s",
            trace.stage,
            trace.phase,
            trace.latency_ms,
            trace.candidates_in,
            trace.candidates_out,
            trace.dropped,
            trace.drop_reason,
            trace.gate,
        )

    def close(self, job_id: str) -> None:
        """Put the sentinel object onto the job's queue to stop event_generator.

        Called after the pipeline run completes (success or failure) so the
        SSE endpoint's event_generator loop exits cleanly.
        """
        q = self._queues.get(job_id)
        if q is not None:
            q.put_nowait(_SENTINEL)
