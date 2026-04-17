from typing import List, Optional

from pydantic import BaseModel


class CandidateArtifact(BaseModel):
    artifact_id: str
    title: str
    relative_path: str
    artifact_family: str
    file_format: str
    section_label: Optional[str] = None
    snippet: str


class ArtifactSearchResponse(BaseModel):
    candidates: List[CandidateArtifact]
    message: str


class LiveArtifactResponse(BaseModel):
    artifact_id: str
    expected_version_token: str
    available_targets: List[str]
    section_html: Optional[str] = None
    message: str


class ArtifactPreviewResponse(BaseModel):
    success: bool
    diff: str
    message: str


class ArtifactSectionInput(BaseModel):
    label: Optional[str] = None
    content: str


class CreateArtifactResponse(BaseModel):
    success: bool
    artifact_id: Optional[str]
    title: Optional[str]
    relative_path: Optional[str]
    file_format: Optional[str]
    version_token: Optional[str]
    message: str


class ArtifactCommitResponse(BaseModel):
    success: bool
    version_token: Optional[str]
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
    artifact_title: Optional[str] = None
    artifact_id: Optional[str] = None
    section_label: Optional[str] = None
    reframed_request: str
    rationale: str
