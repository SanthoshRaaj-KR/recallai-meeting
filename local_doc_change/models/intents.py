"""Intent models for extracted document-change intents from meeting transcripts.

Defines the IntentType enumeration and LocalDocIntent Pydantic model.
These are the authoritative contracts for Waves 1-4 of Phase 12.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

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

    # The LLM frequently returns numeric old/new values as JSON numbers (e.g.
    # 2360 instead of "2360"); without coercion the whole extraction batch fails
    # validation and every change in it is silently dropped. Coerce to str.
    model_config = ConfigDict(coerce_numbers_to_str=True)

    intent_type: IntentType
    affected_topic: str
    old_value: str | None = None
    new_value: str
    verbatim_snippets: list[str]
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str
    metadata: dict = {}
