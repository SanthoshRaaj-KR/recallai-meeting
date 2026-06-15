"""Data models for the vendored Confluence proposal pipeline.

These are copied verbatim (semantics-preserving) from the confluence-branch
``local_doc_change/models/`` so the proposal logic is byte-for-byte the same.
They are consolidated into one module here so the vendored package is
self-contained inside my-agent (no dependency on the local_doc_change tree).
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

# ── Intent ───────────────────────────────────────────────────────────────────

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


class ConfluenceIntent(BaseModel):
    """A single actionable document-change intent extracted from a transcript."""

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


# ── RAG chunk ────────────────────────────────────────────────────────────────


class ChunkRecord(BaseModel):
    """A single indexed section of a document."""

    model_config = ConfigDict(frozen=False)

    chunk_id: str  # f"{file_hash}:{section_index}"
    source_path: str  # path / identifier of the source document
    source_format: str  # "docx" | "odt" | "pdf" | "txt" | "md" | "rtf"
    section_heading: str  # extracted heading or synthetic label
    section_index: int  # position in document
    content: str  # raw text of section
    doc_title: str = ""  # the document/page title (first heading); used to route
    context_prefix: str = ""  # contextual description prepended at embed time
    token_count: int = 0


class RetrievalResult(BaseModel):
    """A ranked retrieval result combining dense, sparse, and RRF/rerank scores."""

    chunk: ChunkRecord
    bm25_rank: Optional[int] = None
    dense_rank: Optional[int] = None
    rrf_score: float = 0.0
    rerank_score: Optional[float] = None
    final_rank: int = 0


# ── Proposal ─────────────────────────────────────────────────────────────────

ProposalStatus = Literal["pending", "accepted", "rejected"]


class ConfluenceProposal(BaseModel):
    """A proposed edit to a document, pending human review."""

    proposal_id: str  # uuid4
    session_id: str
    intent: ConfluenceIntent
    source_chunk: ChunkRecord
    before_content: str
    after_content: str
    edit_type: str = "replace"  # "replace" | "append" | "delete_section"
    confidence: float  # 0.0-1.0 from VerifierAgent
    factual_consistency: float  # 0.0-1.0
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
        intent: ConfluenceIntent,
        chunk: ChunkRecord,
    ) -> "ConfluenceProposal":
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
