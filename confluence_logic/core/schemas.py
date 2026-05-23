from pydantic import BaseModel, Field
from typing import Any, Dict, List, Literal, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    # Imported only for type checking — avoids a runtime circular import between
    # core.schemas, agents.page_parser, and agents.fact_extraction_agent.
    from confluence_logic.agents.fact_extraction_agent import ChangeIntent
    from confluence_logic.agents.page_parser import ASTRoot

class CandidatePage(BaseModel):
    page_id: str
    title: str
    heading: Optional[str] = None
    is_root_section: bool = False
    space_key: str
    snippet: str

class SearchResponse(BaseModel):
    candidates: List[CandidatePage]
    message: str

class LivePageResponse(BaseModel):
    page_id: str
    expected_version: int
    available_headings: List[str]
    section_html: Optional[str] = None
    message: str

class PreviewResponse(BaseModel):
    success: bool
    diff: str
    message: str

class PageSectionInput(BaseModel):
    heading: Optional[str] = None
    content: str

class CreatePageResponse(BaseModel):
    success: bool
    page_id: Optional[str]
    title: Optional[str]
    space_key: Optional[str]
    version: Optional[int]
    message: str

class CommitResponse(BaseModel):
    success: bool
    version: Optional[int]
    message: str

class MasterVoiceDecision(BaseModel):
    immediate_reply: str
    needs_clarification: bool
    clarification_question: Optional[str] = None
    proceed_reply: Optional[str] = None
    execution_request: Optional[str] = None
    intent: str = "edit"
    rationale: Optional[str] = None

class ResolverDecision(BaseModel):
    action: str
    page_title: Optional[str] = None
    page_id: Optional[str] = None
    heading: Optional[str] = None
    reframed_request: str
    rationale: str


# ---------------------------------------------------------------------------
# Phase 10 — Structure-aware drafter contract (PROP-V2-02, PROP-V2-06, D-03/D-06)
# ---------------------------------------------------------------------------
# These models are the JSON schema the StructureAwareDrafter LLM is constrained
# to emit. The ``action`` field is a Pydantic Literal: any LLM output that
# tries to return a non-enumerated value (e.g. "freeform_prose",
# "rewrite_section") fails validation and is rejected at the agent boundary.
# This is the structural fix for Failure Mode 2 (flow destruction on reorder):
# the LLM literally cannot return a regenerated <ol> for a reorder intent
# because there is no schema-level field for it to ride on.
#
# Per-action required-field convention (enforced in prose by the drafter
# prompt, NOT by Pydantic — we keep every action's field Optional so partial
# outputs validate and we can normalise / re-stamp downstream):
#   * replace        -> page_id, section_heading, old_text, new_text
#   * insert_after   -> page_id, section_heading, anchor_text, new_text
#   * reorder        -> page_id, section_heading, from_index, to_index
#   * delete_section -> page_id, section_heading
#   * create_section -> page_id, parent_heading, new_heading, new_content
#   * create_page    -> title, content, parent_page_id (optional), space_key (optional)
#   * skip           -> reason
# ---------------------------------------------------------------------------


StructuredOperationAction = Literal[
    "replace",
    "insert_after",
    "reorder",
    "delete_section",
    "create_section",
    "create_page",
    "skip",
]


class StructuredOperation(BaseModel):
    """One of the six D-02 EditorAgent instruction shapes (plus ``skip``).

    Emitted by the StructureAwareDrafter LLM via a constrained JSON output
    schema. The ``action`` Literal is the structural guarantee that ordered
    procedures are reordered (not rewritten) — there is no top-level prose
    payload the LLM can use to smuggle a regenerated ``<ol>`` in for a
    reorder intent. Defense-in-depth: the drafter's post-validate also
    strips ``new_text`` / ``new_content`` for ``action="reorder"`` ops
    (Pitfall 5 in 10-RESEARCH.md).
    """

    action: StructuredOperationAction

    # Universal identifiers (re-stamped from page_meta by the drafter)
    page_id: Optional[str] = None
    section_heading: Optional[str] = None
    ast_path: Optional[str] = None

    # replace fields
    old_text: Optional[str] = None
    new_text: Optional[str] = None

    # insert_after fields (anchor_text is the existing block to insert AFTER)
    anchor_text: Optional[str] = None

    # reorder fields (0-based indices into the OrderedList.items)
    from_index: Optional[int] = None
    to_index: Optional[int] = None

    # create_section fields
    parent_heading: Optional[str] = None
    new_heading: Optional[str] = None
    new_content: Optional[str] = None

    # create_page fields
    title: Optional[str] = None
    content: Optional[str] = None
    parent_page_id: Optional[str] = None
    space_key: Optional[str] = None

    # skip fields
    reason: Optional[str] = None

    # UI metadata (carried through the verifier and into the ProposalCard).
    # ≤120 chars per D-07; if the LLM exceeds, Pydantic raises ValidationError
    # and the drafter falls back to action="skip" rather than emitting a
    # malformed card.
    change_summary: Optional[str] = Field(default=None, max_length=120)


class StructureAwareDrafterInput(BaseModel):
    """Input contract for ``draft_operation``.

    ``page_ast`` is the parsed AST from ``confluence_logic.agents.page_parser``
    (Plan 10-02). ``page_meta`` is the small dict the upstream pipeline
    already builds (page_id, page_title, space_key, page_url, ancestors).
    ``transcript_window`` is the relevant slice of the meeting transcript
    — typically the ``_find_relevant_transcript_window`` excerpt from
    ``drafter_agent.py`` centered on the intent's subject.
    """

    # Typed via ``Any`` rather than the real ChangeIntent / ASTRoot to avoid
    # an import-time circular dependency (core.schemas is imported by
    # downstream modules that themselves import from agents.*). Validation
    # is exercised end-to-end by the drafter test suite.
    intent: Any = Field(description="A ChangeIntent (fact_extraction_agent.py)")
    page_ast: Any = Field(description="An ASTRoot from page_parser.py")
    page_meta: Dict[str, Any] = Field(default_factory=dict)
    transcript_window: str = ""

    model_config = {"arbitrary_types_allowed": True}
