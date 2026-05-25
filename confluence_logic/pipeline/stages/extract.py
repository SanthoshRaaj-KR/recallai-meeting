"""Stage 1: Transcript → ChangeIntentV3[] extraction (EXT-V3-01).

Implementation comes in Wave 3 (Plan 11-03).  This stub exists so the
package is importable in isolation (ARCH-V3-01 isolation test).
"""

from __future__ import annotations

# No live-service imports at module level — preserves isolation guarantee.
# The real implementation will import openai/agents at function scope or
# lazily to keep the import-time side effects clean.
