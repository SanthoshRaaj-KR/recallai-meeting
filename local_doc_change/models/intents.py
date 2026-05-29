"""Intent models for extracted document-change intents from meeting transcripts.

Defines the IntentType enumeration and LocalDocIntent Pydantic model.
These are the authoritative contracts for Waves 1-4 of Phase 12.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

IntentType = Literal[
    "policy_update",
    "process_change",
    "ownership_change",
    "compliance_update",
    "decision",
    "technology_migration",
    "onboarding_update",
    "other",
]


class LocalDocIntent(BaseModel):
    """A single actionable document-change intent extracted from a meeting transcript."""

    intent_type: IntentType
    affected_topic: str
    old_value: str | None = None
    new_value: str
    verbatim_snippets: list[str]
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str
    metadata: dict = {}
