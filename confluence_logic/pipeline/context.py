"""PipelineContext dataclass + centralized v3 killswitch flags (ARCH-V3-01).

All four v3 killswitches are defined here so every stage reads the same
parsed flag without duplicating the WR-07 robust off-parse idiom.

WR-07 rule: an empty or garbled env var MUST NOT flip the default to False.
Only explicit off-values {0, false, no, off} disable a flag.  This closes
the threat T-11-03 (Tampering via killswitch parsing).

Killswitches defined here:
  JARVIS_PIPELINE_V3_ENABLED       — whole v3 pipeline (default ON)
  JARVIS_V3_RERANK_ENABLED         — reranking stage  (default ON)
  JARVIS_V3_CONTRADICTION_ENABLED  — contradiction sweep (default ON)
  JARVIS_V3_EDITOR_LLM             — LLM EditorAgent vs deterministic
                                     apply_structured (default ON = LLM)

Integer config:
  JARVIS_RETRIEVAL_MAX_ITERS       — bounded agentic retrieval loop cap (default 2)

CRITICAL (T-11-02 / Pitfall 1/4):
  ``graph_user_id`` is an EXPLICIT field on PipelineContext.  Stages MUST
  NEVER read the ``confluence_graph_user_id`` ContextVar — it does not survive
  ``asyncio.to_thread`` boundaries and causes cross-user contamination in
  multi-tenant deployments.  The ContextVar is set ONLY around the locked
  EditorAgent call in ``pipeline/apply.py`` (its own tools require it).
"""

from __future__ import annotations

import dataclasses
import logging
import os
from typing import TYPE_CHECKING, Any, List, Optional

if TYPE_CHECKING:
    from confluence_logic.pipeline.trace import TraceBus

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Robust off-parse helper (WR-07) — reused for every killswitch below.
# ---------------------------------------------------------------------------

def _parse_flag(env_var: str, default: str = "1") -> bool:
    """Parse a boolean env flag using the WR-07 robust idiom.

    An unset or empty env var returns the default.  Only the explicit
    off-values {0, false, no, off} (case-insensitive) return False.

    Args:
        env_var: Name of the environment variable to read.
        default: Default value string to use when the env var is unset or
                 empty (``"1"`` = True, ``"0"`` = False).

    Returns:
        True unless the resolved raw value is in {0, false, no, off}.
    """
    raw = (os.getenv(env_var) or default).strip().lower()
    return raw not in {"0", "false", "no", "off"}


# ---------------------------------------------------------------------------
# V3 killswitch flags (module-level constants — reload-friendly for tests)
# ---------------------------------------------------------------------------

# Master switch: set to 0/false/no/off to fall back to the Phase 10 pipeline.
JARVIS_PIPELINE_V3_ENABLED: bool = _parse_flag("JARVIS_PIPELINE_V3_ENABLED")

# Stage-level killswitches (fine-grained rollout / A-B testing):
JARVIS_V3_RERANK_ENABLED: bool = _parse_flag("JARVIS_V3_RERANK_ENABLED")
JARVIS_V3_CONTRADICTION_ENABLED: bool = _parse_flag("JARVIS_V3_CONTRADICTION_ENABLED")

# Editor choice: True = LLM EditorAgent (default, user-stated preference);
# False = deterministic apply_structured (for reorder/archive where
# non-fabrication is provably safer — Phase 10 Pitfall 5).
JARVIS_V3_EDITOR_LLM: bool = _parse_flag("JARVIS_V3_EDITOR_LLM")

# Bounded agentic retrieval loop iteration cap (RETR-V3-04).
JARVIS_RETRIEVAL_MAX_ITERS: int = int(os.getenv("JARVIS_RETRIEVAL_MAX_ITERS", "2"))


# ---------------------------------------------------------------------------
# PipelineContext — explicit per-run state threaded through every stage
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class PipelineContext:
    """Explicit per-run state container threaded through every pipeline stage.

    All fields are explicit parameters — no ContextVar reads inside stages
    (T-11-02 / Pitfall 1/4).  The only place the ContextVar is touched is
    ``pipeline/apply.py`` around the EditorAgent call boundary.

    Fields:
        session_id: Unique identifier for the post-meeting review session.
        user_id: Supabase user id (for auth + Supabase store writes).
        graph_user_id: Neo4j graph scoping key.  NEVER read the
            ``confluence_graph_user_id`` ContextVar inside a stage — use
            this field instead (T-11-02, Pitfall 1).
        bot_id: Recall.ai bot id for the Stage 0 transcript fetch (Plan 11).
            None when running in local/test mode without a real bot.
        transcript_text: Normalised full transcript string used for char-
            offset computation (EXT-V3-01 evidence spans).  Starts as ``""``
            and is populated by the Stage 0 transcript source, NOT pre-filled
            by api.py.
        transcript_log: Raw LiveKit/Supabase fallback transcript entries.
            Stage 0 uses this when the Recall diarised transcript is
            unavailable or when the v3 pipeline is running in local mode.
            None when the diarised Recall transcript was fetched successfully.
        trace_bus: TraceBus instance for emitting StageTrace events to SSE
            and structured logs (OBS-V3-01).
        pipeline_v3_enabled: Per-run snapshot of JARVIS_PIPELINE_V3_ENABLED.
        rerank_enabled: Per-run snapshot of JARVIS_V3_RERANK_ENABLED.
        contradiction_enabled: Per-run snapshot of JARVIS_V3_CONTRADICTION_ENABLED.
        editor_llm: Per-run snapshot of JARVIS_V3_EDITOR_LLM.
        retrieval_max_iters: Per-run snapshot of JARVIS_RETRIEVAL_MAX_ITERS.
        section_corpus: Cache slot filled once per run by pipeline/retrieval/
            corpus.py.  Shared by BM25 index build and contradiction fan-out
            so only one Neo4j fetch is needed.  Type is Any to avoid importing
            the corpus module here (would create a cycle).
    """

    # --- Identity / session ---
    session_id: str
    user_id: str
    graph_user_id: str  # EXPLICIT — never read ContextVar in stages (T-11-02)

    # --- Bot / transcript ---
    bot_id: Optional[str] = None          # Recall.ai bot id for Stage 0
    transcript_text: str = ""             # populated by Stage 0, NOT by api.py
    transcript_log: Optional[List[Any]] = None  # raw fallback entries

    # --- Observability ---
    trace_bus: Any = None  # TraceBus — Any to avoid circular import at dataclass def

    # --- Per-run killswitch snapshots (default to module-level flags) ------
    # Storing snapshots lets tests override flags per-context without
    # reloading the module.
    pipeline_v3_enabled: bool = dataclasses.field(
        default_factory=lambda: JARVIS_PIPELINE_V3_ENABLED
    )
    rerank_enabled: bool = dataclasses.field(
        default_factory=lambda: JARVIS_V3_RERANK_ENABLED
    )
    contradiction_enabled: bool = dataclasses.field(
        default_factory=lambda: JARVIS_V3_CONTRADICTION_ENABLED
    )
    editor_llm: bool = dataclasses.field(
        default_factory=lambda: JARVIS_V3_EDITOR_LLM
    )
    retrieval_max_iters: int = dataclasses.field(
        default_factory=lambda: JARVIS_RETRIEVAL_MAX_ITERS
    )

    # --- Per-run corpus cache (filled by retrieval/corpus.py) --------------
    section_corpus: Any = None  # SectionCorpus | None; shared across stages
