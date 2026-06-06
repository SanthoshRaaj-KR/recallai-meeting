"""Pydantic data model contracts for the local document change pipeline.

All models are import-only schema definitions — no business logic.
"""

from __future__ import annotations

from .intents import IntentType, LocalDocIntent
from .proposals import LocalDocProposal, ProposalStatus
from .rag import ChunkRecord, RetrievalResult

__all__ = [
    "IntentType",
    "LocalDocIntent",
    "ChunkRecord",
    "RetrievalResult",
    "LocalDocProposal",
    "ProposalStatus",
]
