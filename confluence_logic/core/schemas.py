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
