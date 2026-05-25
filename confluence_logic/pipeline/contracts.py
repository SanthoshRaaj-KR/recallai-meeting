"""Typed stage I/O contracts for the v3 pipeline (ARCH-V3-01).

Every cross-stage data shape is defined once here.  Each model is importable
standalone — ``from confluence_logic.pipeline.contracts import X`` — with no
live-service side effects at import time (no OpenAI client, no Pinecone,
no Neo4j driver).

Pydantic conventions mirror ``confluence_logic/core/schemas.py``:
  * ``Literal`` discriminators for action/operation/kind fields.
  * ``Optional[...] = None`` defaults on fields that may be absent for some op
    shapes (keeps partial LLM outputs valid — see per-model comments below).
  * ``Field(min_length=1)`` only where a non-empty collection is a hard
    correctness requirement (EXT-V3-01: no intent without evidence).
  * ``Field(default=None, max_length=120)`` for change_summary (D-07).
  * ``if TYPE_CHECKING:`` guard for any cross-module type refs that would
    otherwise create import cycles.
  * ``model_config = {"arbitrary_types_allowed": True}`` if needed for AST
    refs (none required in this file — all fields are stdlib/Pydantic types).

Per-model required-field convention (enforced in prose / comments, NOT by
Pydantic — so partial LLM outputs still validate and we can normalise
downstream):
  * edit_section  -> page_id, section_heading, before_content, after_content
  * append        -> page_id, after_content
  * create_page   -> page_title, after_content (new page body)
  * archive_deprecate -> page_id, page_title

EXCEPTION: ChangeIntentV3.evidence uses Field(min_length=1) — a hard
correctness requirement (EXT-V3-01: no intent without evidence span).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, List, Literal, Optional

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    # Guard — avoids a runtime circular import when page_parser is imported
    # elsewhere.  No field below actually references ASTRoot at runtime.
    from confluence_logic.agents.page_parser import ASTRoot  # noqa: F401


# ---------------------------------------------------------------------------
# EvidenceSpan — verbatim transcript evidence (EXT-V3-01)
# ---------------------------------------------------------------------------
# Fields use short names (text/start/end) so they align with the test suite
# and the sync-sage-bot TS ChangeItem.transcript_evidence convention.
# Offsets are filled in Python via str.find(text) against the normalised
# transcript — NEVER trust LLM-emitted integer offsets (anti-pattern noted in
# RESEARCH §Architecture Patterns Pattern 1 and §Structured-Output Schemas).

class EvidenceSpan(BaseModel):
    """A verbatim substring of the meeting transcript supporting an intent.

    ``text`` is the LLM-emitted quote; ``start``/``end`` char offsets are
    computed deterministically in Python (str.find) by the extract stage —
    -1 means not-yet-resolved.  An EvidenceSpan with text="" is considered
    invalid and the extract stage rejects the parent intent.
    """

    text: str  # verbatim transcript substring (LLM-emitted)
    start: int = -1  # filled by Python str.find against normalised transcript
    end: int = -1


# ---------------------------------------------------------------------------
# ChangeIntentV3 — one extracted meeting decision/fact/action (EXT-V3-01)
# ---------------------------------------------------------------------------

ChangeIntentKind = Literal[
    "decision",
    "fact_update",
    "action_item",
    "new_workstream",
    "deprecation",
]


class ChangeIntentV3(BaseModel):
    """A typed, evidence-backed change extracted from the meeting transcript.

    ``kind`` is the Literal discriminator; ``evidence`` requires at least one
    EvidenceSpan (Field(min_length=1) — EXT-V3-01).  All other fields are
    Optional so partial LLM outputs validate and can be normalised downstream.

    ``dedup_key`` is a normalised ``(subject, kind)`` composite computed in
    Python (not by the LLM) so re-running a job upserts idempotently.
    """

    kind: ChangeIntentKind
    subject: str
    old_value: str = ""
    new_value: Optional[str] = ""
    instruction: str = ""
    target_hint: str = ""
    verbatim_content: str = ""  # for add/create (Phase 4/8 rule preserved)
    dedup_key: str = ""  # normalised (subject, kind) — computed in Python
    evidence: List[EvidenceSpan] = Field(
        min_length=1,
        description="At least one verbatim evidence span required (EXT-V3-01).",
    )


# ---------------------------------------------------------------------------
# SectionCandidate — one retrieval hit resolved to page→section (RETR-V3-02)
# ---------------------------------------------------------------------------

class SectionCandidate(BaseModel):
    """A candidate Confluence section returned by the hybrid retrieval stage.

    Resolved to ``(page_id, section_heading)`` before drafting — the drafter
    never picks the section (RESEARCH anti-pattern §Architecture Patterns).

    ``dense_rank`` / ``lexical_rank`` / ``rrf_score`` support the RRF fusion
    (RETR-V3-01); ``rerank_score`` is filled by the rerank stage (RETR-V3-03).

    ``score`` is the raw retrieval score from the originating signal (dense
    cosine similarity or BM25 score); ``source`` identifies which signal
    produced this candidate (``"dense"``, ``"lexical"``, or ``"fused"``).
    These additive fields allow tests to inject pre-scored candidates and let
    the retrieve stage attribute each candidate to its signal source.
    """

    page_id: str
    page_title: str = ""
    space_key: str = ""
    section_heading: Optional[str] = None
    section_text: str = ""
    dense_rank: Optional[int] = None
    lexical_rank: Optional[int] = None
    rrf_score: float = 0.0
    rerank_score: Optional[float] = None
    # Additive fields for signal attribution and raw score tracking
    score: float = 0.0       # raw retrieval score from the originating signal
    source: str = ""         # "dense" | "lexical" | "fused"


# ---------------------------------------------------------------------------
# RetrievalResult — output of the retrieve + rerank + iterate stages
# ---------------------------------------------------------------------------

class RetrievalResult(BaseModel):
    """Ranked candidates for one ChangeIntentV3 after full retrieval (RETR-V3-01..04).

    ``no_existing_target=True`` signals that bounded agentic retrieval (Stage
    4) exhausted its iteration budget without finding a viable candidate — the
    orchestrator routes this intent to ``create_page`` in plan_ops.
    ``iterations`` counts how many agentic-loop retries were needed (0 = no
    loop triggered).

    ``fusion_log`` is a structured record of the RRF fusion step emitted for
    observability (RETR-V3-01: fusion must be logged/traced).  It carries the
    fused order (list of ``(doc_id, rrf_score)`` tuples) and signal metadata.
    None only when no candidates were found (empty corpus + no dense hits).

    ``intent`` is Optional so test fixtures can construct ``RetrievalResult``
    without a full intent (test isolation — unit tests for rerank/iterate only
    need candidate data, not the driving intent).
    """

    intent: Optional[ChangeIntentV3] = None
    candidates: List[SectionCandidate]
    no_existing_target: bool = False
    iterations: int = 0
    # fusion_log accepts dict (production) or a descriptive string (test fixtures)
    fusion_log: Optional[Any] = None  # RETR-V3-01: fused order + signal counts


# ---------------------------------------------------------------------------
# AffectedPage — one affected page/section in a ContradictionGroup
# ---------------------------------------------------------------------------

class AffectedPage(BaseModel):
    """One workspace page/section flagged by the contradiction sweep (CON-V3-01).

    ``page_id`` and ``group_id`` are always set.  ``recommended_op`` is the
    op shape the contradiction sweep suggests (``edit_section`` by default,
    ``archive_deprecate`` when the page is fully superseded).
    ``low_confidence`` is True when the entailment classifier errored and the
    page was kept with degraded certainty (recall-biased fallback).
    """

    page_id: str
    page_title: str = ""
    section_heading: Optional[str] = None
    group_id: str = ""
    recommended_op: str = "edit_section"  # "edit_section" | "archive_deprecate"
    low_confidence: bool = False


# ---------------------------------------------------------------------------
# ContradictionGroup — cross-page contradiction set (CON-V3-01)
# ---------------------------------------------------------------------------

class ContradictionGroup(BaseModel):
    """A group of operations sharing one logical decision (CON-V3-01).

    For a ``fact_update`` intent whose ``old_value`` appears in multiple
    Confluence pages, the contradiction sweep emits ONE ContradictionGroup
    carrying one ``PlannedOperation`` per affected page/section, all sharing
    the same ``group_id`` so the UI renders them as one decision (UI-V3-01).

    ``affected_pages`` is the stage-5 output list (AffectedPage items with
    page_id, group_id, recommended_op) — populated by contradict.py.
    ``operations`` is the stage-6 output list (PlannedOperation items with
    full content) — populated by plan_ops.py.
    """

    subject: str
    old_value: str
    new_value: Optional[str] = None
    operations: List["PlannedOperation"] = Field(default_factory=list)  # forward ref
    affected_pages: List[AffectedPage] = Field(default_factory=list)  # stage-5 output


# ---------------------------------------------------------------------------
# PlannedOperation — one atomic planned edit (OPS-V3-01)
# ---------------------------------------------------------------------------

PlannedOperationKind = Literal[
    "edit_section",
    "append",
    "create_page",
    "archive_deprecate",
]


class PlannedOperation(BaseModel):
    """One atomic planned operation produced by plan_ops (OPS-V3-01).

    ``operation`` is exactly one of the four enumerated values — ambiguous ops
    are not emitted (the stage degrades to skip with a logged reason instead).

    Per-operation required-field convention (enforced by prompt/code, NOT by
    Pydantic so partial LLM outputs still validate):
      * edit_section      -> page_id, section_heading, before_content, after_content
      * append            -> page_id, after_content
      * create_page       -> page_title, after_content
      * archive_deprecate -> page_id, page_title

    ``group_id`` links operations belonging to the same ContradictionGroup.
    """

    operation: PlannedOperationKind
    page_id: Optional[str] = None
    page_title: str
    space_key: Optional[str] = None
    section_heading: Optional[str] = None
    before_content: Optional[str] = None   # for edit_section (old text)
    after_content: Optional[str] = None    # for edit_section / append / create_page
    rationale: str = ""
    group_id: Optional[str] = None


# Resolve forward reference in ContradictionGroup.operations
ContradictionGroup.model_rebuild()


# ---------------------------------------------------------------------------
# ProposalCardV3 — superset of the Phase 10 ChangeItem (UI-V3-01)
# ---------------------------------------------------------------------------
# ChangeItem fields (sync-sage-bot/src/types.ts lines 17-68) are mirrored
# here as snake_case so the same JSON payload renders in ProposalCardV2.tsx.
# All existing ChangeItem fields are Optional so a Phase 10 payload validates
# unchanged (superset proved by test_pipeline_contracts_v3).
# New v3 fields: confidence_score, confidence_bin, group_id, evidence.

ChangeType = Literal["create", "edit", "delete", "title"]
ChangeStatus = Literal["pending", "approved", "rejected", "executed", "executing", "failed"]
ConfidenceBin = Literal["high", "medium", "low"]


class ProposalCardV3(BaseModel):
    """A review-ready proposal card carrying full v3 metadata (UI-V3-01).

    This model is a strict superset of the Phase 10 ``ChangeItem`` (Pydantic
    + TS): all ChangeItem fields are preserved under the same names so a Phase
    10 payload validates unchanged (backward compatibility).  New v3 additive
    fields: ``confidence_score``, ``confidence_bin``, ``group_id``,
    ``evidence``.

    ``change_summary`` is capped at 120 chars (D-07) — a Pydantic
    ValidationError on overflow causes the plan_ops stage to skip the card
    rather than emit a malformed one.
    """

    # --- Phase 10 ChangeItem fields (backward compat) ----------------------
    change_type: Optional[ChangeType] = None
    page_id: Optional[str] = None
    page_title: Optional[str] = None
    section_heading: Optional[str] = None
    before_content: Optional[str] = None
    after_content: Optional[str] = None
    status: Optional[ChangeStatus] = None
    rationale: Optional[str] = None
    change_summary: Optional[str] = Field(default=None, max_length=120)

    # Phase 10 additive fields (from Plan 08-03/08-04/10-08)
    breadcrumb: Optional[str] = None
    page_url: Optional[str] = None
    operation_action: Optional[str] = None  # StructuredOperationAction string
    reorder_payload: Optional[dict] = None
    regenerate_available: Optional[bool] = None
    last_error: Optional[str] = None
    verifier_note: Optional[str] = None
    transcript_evidence: Optional[List[str]] = None

    # --- Phase 11 v3 new fields --------------------------------------------
    confidence_score: Optional[float] = None  # 0.0–1.0 calibrated confidence
    confidence_bin: Optional[ConfidenceBin] = None
    group_id: Optional[str] = None  # links contradiction-group operations
    evidence: Optional[List[EvidenceSpan]] = None  # typed evidence spans


# ---------------------------------------------------------------------------
# StageTrace — per-stage observability event (OBS-V3-01)
# ---------------------------------------------------------------------------

StagePhase = Literal["start", "end"]


class StageTrace(BaseModel):
    """A trace event emitted by TraceBus before/after each pipeline stage.

    Carries latency, candidate counts, and drop metadata for structured logs
    and SSE emission.  ``gate`` identifies which deterministic gate dropped
    candidates (e.g. "grounding_gate", "page_existence").
    """

    stage: str
    phase: StagePhase
    latency_ms: Optional[float] = None
    candidates_in: Optional[int] = None
    candidates_out: Optional[int] = None
    dropped: int = 0
    drop_reason: Optional[str] = None
    gate: Optional[str] = None


# ---------------------------------------------------------------------------
# EditorInstruction — typed envelope for EditorAgent handoff (EDIT-V3-01)
# ---------------------------------------------------------------------------
# The real deliverable of D2: a machine-checkable instruction envelope whose
# fields are validated BEFORE being rendered into a prompt string for
# EditorAgent.handle_prepared_query().  This makes the instruction
# machine-inspectable: page_id present? section resolved? old_text grounded?

EditorInstructionOp = Literal[
    "edit_section",
    "append",
    "create_page",
    "archive_deprecate",
]


class EditorInstruction(BaseModel):
    """A fully-resolved, machine-checkable instruction envelope (EDIT-V3-01).

    Passed to ``pipeline/editor/instruction.py:render_editor_prompt`` which
    renders it into the exact natural-language string consumed by
    ``editor_agent.handle_prepared_query``.

    Pre-send validator (in render_editor_prompt) rejects any instruction
    missing its required fields or whose old_text is absent on the live page
    (section-anchor preflight, SAFE-V3-01).

    Per-operation required-field convention:
      * edit_section      -> page_id, section_heading, old_text, new_text
      * append            -> page_id, new_text
      * create_page       -> page_title, new_page_body
      * archive_deprecate -> page_id, page_title; confirm_hard_delete=True
                            only for permanent deletion (default = label/archive)
    """

    operation: EditorInstructionOp
    page_id: Optional[str] = None       # required for all but create_page
    page_title: str
    space_key: Optional[str] = None
    section_heading: Optional[str] = None
    old_text: Optional[str] = None      # required for edit_section; grounded on live page
    new_text: Optional[str] = None      # required for edit_section / append
    new_page_body: Optional[str] = None  # required for create_page
    rationale: str = ""
    confirm_hard_delete: bool = False    # archive_deprecate defaults to label/archive
