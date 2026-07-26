"""Data models for the Confluence proposal pipeline."""

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

# Who a stated value belongs to. Every figure in a meeting is *about* someone, and
# an edit is only correct when the intent's subject and the target section's subject
# are the same party. Without this, a competitor's fee/limit/headcount mentioned in
# passing retrieves the document owner's own equivalent row — same topic, same table
# shape, same units — and gets written straight into it.
#   internal     — the value belongs to the organization whose documents these are.
#   third_party  — the value was stated about a different named party (competitor,
#                  customer, vendor, portfolio company, industry benchmark, a figure
#                  quoted from elsewhere). Gated: may only land on a section that is
#                  itself about that party.
#   unspecified  — no attribution signal; treated as ungated so recall is unaffected,
#                  with the evaluator free to catch an attribution the extractor missed.
SubjectScope = Literal["internal", "third_party", "unspecified"]

# Whether the speaker is CHANGING a value or merely restating one that already holds.
# "Discussing a topic is not changing it" is already in the extraction instructions, but
# it fails on the contrastive form that dominates real meetings — "Competitor does X;
# ours stays at Y" — where the "ours" clause reads like an affirmative statement of a
# value. The extractor then emits an intent whose new_value is the value the document
# ALREADY has, and the editor, told to make the section satisfy it, invents a change to
# a neighbouring row or rewrites a sentence. Labelling the intent is a far easier task
# for the model than suppressing it, and gives a deterministic place to drop it.
ChangePolarity = Literal["change", "reaffirmation"]


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
    # Attribution. Defaults keep every existing construction site (and the stored
    # shape of older intents) valid — an intent with no attribution is "unspecified"
    # and behaves exactly as it did before the guardrail existed.
    subject_entity: str | None = None  # the party the value is about, as named
    subject_scope: SubjectScope = "unspecified"
    # Defaults to "change" so an intent carrying no polarity behaves exactly as before.
    change_polarity: ChangePolarity = "change"
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
    version: int | None = None  # Confluence page version at index time
    content_hash: str = ""     # SHA-256 of page content for freshness checks


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
