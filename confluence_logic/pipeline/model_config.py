"""Centralized model ID constants for the Phase 11 v3 pipeline (ARCH-V3-01).

GPT-5-mini is the HARD CEILING for every stage in Phase 11 (per CLAUDE.md model
ceiling and RESEARCH locked constraint D1). No stage may use a model exceeding
gpt-5-mini capability.

Suggested split (matches RESEARCH §Model ceiling):
  MODEL_WORKER  — extraction / drafting / plan_ops / verification / orchestration
                  → gpt-5-mini  (default, env JARVIS_AGENT_MODEL)
  MODEL_NANO    — routing / rerank / entailment / classification
                  → gpt-5.4-nano  (default, env JARVIS_NANO_MODEL)

Both constants are env-overridable.  An override that resolves to an above-ceiling
model ID is rejected at import via the ALLOWED_MODELS guard.

NOTE: The actual OpenAI API acceptance of these specific model IDs is a manual
verification item (RESEARCH VALIDATION § Manual-Only) — the constants are kept
overridable so callers can supply accepted IDs via env vars while keeping the
defaults as the intended targets.
"""

from __future__ import annotations

import os

# ---------------------------------------------------------------------------
# Allowed model IDs — ceiling enforcement
# ---------------------------------------------------------------------------
# Any model ID not in this set is rejected at import.  The set covers the
# concrete IDs planned for Phase 11 plus safe aliases the OpenAI API accepts.
# Expanding this set is a deliberate action (add the model + update this list).

ALLOWED_MODELS: frozenset[str] = frozenset(
    {
        # GPT-5-mini capability ceiling (worker models)
        "gpt-5-mini",
        # Nano-tier models (≤ gpt-5-mini, preferred for routing/rerank/entailment)
        "gpt-5.4-nano",
        # Accepted sub-ceiling aliases already in use across the codebase
        "gpt-4o-mini",
        "gpt-4o",
        # Legacy alias used in jarvis_agentic / older phases — allowed but prefer
        # replacing with gpt-5-mini for Phase 11 agents.
        "gpt-5.4-mini",
    }
)

# ---------------------------------------------------------------------------
# Model constants — read from env, default to concrete ceiling-respecting IDs
# ---------------------------------------------------------------------------

MODEL_WORKER: str = (
    os.getenv("JARVIS_AGENT_MODEL", "gpt-5-mini").strip() or "gpt-5-mini"
)
"""Worker model for extraction / drafting / plan_ops / verification / orchestration.

Default: ``gpt-5-mini`` (env override: ``JARVIS_AGENT_MODEL``).
"""

MODEL_NANO: str = (
    os.getenv("JARVIS_NANO_MODEL", "gpt-5.4-nano").strip() or "gpt-5.4-nano"
)
"""Nano-tier model for routing / rerank / entailment (lowest cost/latency).

Default: ``gpt-5.4-nano`` (env override: ``JARVIS_NANO_MODEL``).
"""

# ---------------------------------------------------------------------------
# Ceiling guard — reject above-ceiling or unknown model IDs at import time
# ---------------------------------------------------------------------------

def _assert_within_ceiling(name: str, model_id: str) -> None:
    """Raise ValueError if *model_id* is not in ALLOWED_MODELS."""
    if model_id not in ALLOWED_MODELS:
        raise ValueError(
            f"[model_config] {name}={model_id!r} is not in ALLOWED_MODELS. "
            f"GPT-5-mini is the hard ceiling for Phase 11. "
            f"Allowed: {sorted(ALLOWED_MODELS)}"
        )


_assert_within_ceiling("MODEL_WORKER", MODEL_WORKER)
_assert_within_ceiling("MODEL_NANO", MODEL_NANO)
