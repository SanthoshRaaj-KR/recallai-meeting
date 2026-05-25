"""Deterministic pipeline orchestrator (ARCH-V3-01).

Implementation comes in Wave 4 (Plan 11-10).  This stub defines the
``run`` coroutine signature so test collection succeeds and imports are
stable (ARCH-V3-01 isolation test).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from confluence_logic.pipeline.context import PipelineContext


async def run(ctx: "PipelineContext") -> None:
    """Run the full v3 pipeline for the given context.

    Stages: extract → retrieve → rerank → iterate → contradict → plan_ops → gate.

    This stub raises NotImplementedError until Wave 4 implements the full
    orchestrator.  Wave 0 tests only import this symbol — they do not call it.
    """
    raise NotImplementedError(
        "pipeline.run is a Wave 4 deliverable (Plan 11-10). "
        "This stub exists for import-time collection only."
    )
