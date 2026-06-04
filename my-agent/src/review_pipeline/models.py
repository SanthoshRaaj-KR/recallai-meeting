from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


ChangeType = Literal["create", "edit", "delete", "title"]
EditMode = Literal["replace", "append", "create_section", "task_status"]


@dataclass
class ChangeIntent:
    """One atomic documentation change extracted from the transcript."""

    instruction: str = ""
    subject: str = ""
    target_hint: str = ""
    old_value: str = ""
    new_value: str = ""
    action: str = "replace"
    rationale: str = ""
    evidence: list[str] = field(default_factory=list)
    source: str = "intent_extractor"
    page_id: str | None = None
    page_title: str | None = None


@dataclass
class ExtractedMeeting:
    """Structured meeting understanding used by the proposal pipeline."""

    title: str = "Meeting Review"
    summary: str = ""
    key_topics: list[str] = field(default_factory=list)
    decisions: list[str] = field(default_factory=list)
    action_items: list[dict[str, Any]] = field(default_factory=list)
    participants: list[str] = field(default_factory=list)
    moments: list[dict[str, str]] = field(default_factory=list)
    change_intents: list[ChangeIntent] = field(default_factory=list)


@dataclass
class PageCandidate:
    """A live Confluence page considered for a specific intent."""

    page_id: str | None
    title: str
    space_key: str = ""
    url: str | None = None
    html: str = ""
    text: str = ""
    version: int | None = None
    source: str = "search"
    score: float = 0.0
    sections: list[dict[str, str]] = field(default_factory=list)


@dataclass
class Proposal:
    """Reviewable Confluence change card returned to sync-sage-bot."""

    id: str
    change_type: ChangeType
    page_id: str | None
    page_title: str
    section_heading: str | None
    before_content: str | None
    after_content: str | None
    timestamp: str
    session_id: str
    status: str = "pending"
    source: str = "my-agent-pipeline"
    rationale: str | None = None
    generation_query: str | None = None
    transcript_evidence: list[str] = field(default_factory=list)
    confidence: Literal["high", "medium", "low"] = "medium"
    risk: Literal["safe", "review", "risky"] = "review"
    verifier_note: str | None = None
    edit_mode: EditMode | None = None
    change_summary: str | None = None
    page_url: str | None = None
    breadcrumb: list[str] | None = None
    confidence_score: float | None = None
    confidence_bin: Literal["high", "medium", "low"] | None = None

    def to_dict(self) -> dict[str, Any]:
        data = {
            "id": self.id,
            "change_type": self.change_type,
            "page_id": self.page_id,
            "page_title": self.page_title,
            "section_heading": self.section_heading,
            "before_content": self.before_content,
            "after_content": self.after_content,
            "timestamp": self.timestamp,
            "session_id": self.session_id,
            "status": self.status,
            "source": self.source,
            "rationale": self.rationale,
            "generation_query": self.generation_query,
            "transcript_evidence": self.transcript_evidence,
            "confidence": self.confidence,
            "risk": self.risk,
            "verifier_note": self.verifier_note,
            "edit_mode": self.edit_mode,
            "change_summary": self.change_summary,
            "page_url": self.page_url,
            "breadcrumb": self.breadcrumb,
            "confidence_score": self.confidence_score,
            "confidence_bin": self.confidence_bin,
        }
        return {k: v for k, v in data.items() if v is not None}