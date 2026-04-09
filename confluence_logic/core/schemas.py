from pydantic import BaseModel
from typing import List, Optional

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
