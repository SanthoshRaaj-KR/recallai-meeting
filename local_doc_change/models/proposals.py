"""Proposal models for proposed local document changes awaiting human review.

Defines ProposalStatus and LocalDocProposal Pydantic model.
These are the authoritative contracts for Waves 1-4 of Phase 12.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel

from .intents import LocalDocIntent
from .rag import ChunkRecord

ProposalStatus = Literal["pending", "accepted", "rejected"]


class LocalDocProposal(BaseModel):
    """A proposed edit to a local document, pending human review."""

    proposal_id: str  # uuid4
    session_id: str
    intent: LocalDocIntent
    source_chunk: ChunkRecord
    before_content: str
    after_content: str
    edit_type: str = "replace"  # "replace" | "append" | "delete_section"
    confidence: float  # 0.0–1.0 from VerifierAgent
    factual_consistency: float  # 0.0–1.0
    formatting_integrity: float
    intent_fulfillment: float
    quality_score: float  # composite
    verifier_note: str
    status: ProposalStatus = "pending"
    created_at: str  # ISO timestamp

    @classmethod
    def create(
        cls,
        session_id: str,
        intent: LocalDocIntent,
        chunk: ChunkRecord,
    ) -> LocalDocProposal:
        """Factory method that initialises all required fields with sensible defaults.

        Quality scores all start at 0.0 and are populated by VerifierAgent in Wave 3.
        """
        return cls(
            proposal_id=str(uuid.uuid4()),
            session_id=session_id,
            intent=intent,
            source_chunk=chunk,
            before_content="",
            after_content="",
            confidence=0.0,
            factual_consistency=0.0,
            formatting_integrity=0.0,
            intent_fulfillment=0.0,
            quality_score=0.0,
            verifier_note="",
            status="pending",
            created_at=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        )
