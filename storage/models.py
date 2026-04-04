"""
Canonical Pydantic models for meeting records.

MeetingRecord is the single source of truth for both JSON persistence on disk
and Pinecone vector metadata. ActionItem represents a structured action item
extracted from the meeting transcript.

Design constraints:
- All timestamps (start_ts, end_ts, summarized_at) are Unix epoch integers.
  Never use datetime objects — Pinecone $gte/$lte filters require integers.
- All fields are flat scalars or lists of scalars (no nested JSON blobs).
  This prevents schema divergence between the JSON file and Pinecone metadata.
- status defaults to "complete"; set to "partial" for mid-meeting summaries
  to prevent partial data from being mistaken for complete meeting records.
"""

import logging
from typing import Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ActionItem(BaseModel):
    """A structured action item extracted from a meeting."""

    owner: str
    task: str
    due: Optional[str] = None


class MeetingRecord(BaseModel):
    """
    Canonical schema for a single meeting record.

    This model drives both JSON persistence (via MetadataStore) and
    Pinecone upsert (via PineconeClient). Any schema change here propagates
    to both storage systems automatically.
    """

    meeting_id: str
    channel_id: str
    channel_name: str
    start_ts: int                                          # Unix epoch integer — NEVER datetime
    end_ts: Optional[int] = None
    duration_seconds: Optional[int] = None
    participants: list[str] = Field(default_factory=list)
    summary_text: str
    topics_covered: list[str] = Field(default_factory=list)
    action_items: list[ActionItem] = Field(default_factory=list)
    decisions: list[str] = Field(default_factory=list)
    series_name: Optional[str] = None
    recurrence_pattern: Optional[str] = None
    status: str = "complete"                               # "complete" or "partial"
    raw_transcript_chars: Optional[int] = None
    summarized_at: Optional[int] = None                   # Unix epoch integer
