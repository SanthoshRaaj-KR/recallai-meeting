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


class BatchSummaryOutput(BaseModel):
    """Structured output schema for a single rolling-summarizer batch.

    Returned by RollingSummarizerAgent for each flushed sentence buffer.
    All fields are flat scalars or lists of strings — consistent with the
    project-wide constraint of no nested JSON objects.
    """

    summary_text: str                       # 2-4 sentence prose summary of this batch
    key_points: list[str] = Field(default_factory=list)   # short bullet-point phrases
    speakers: list[str] = Field(default_factory=list)     # speaker names active in this batch


class MeetingIndexEntry(BaseModel):
    """Index entry for the per-meeting JSON index maintained by MeetingWriterAgent.

    This is the fast-lookup path used by the History Manager to select a meeting
    before loading the full .md file content. One entry per meeting, upserted
    after each batch flush so the index reflects the meeting's current state.

    Design constraints (same as MeetingRecord):
    - start_ts is a Unix epoch integer — never a datetime object.
    - All fields are flat scalars or lists of strings — no nested objects.
    """

    meeting_id: str
    title: str                              # derived from channel name or meeting metadata
    date: str                               # ISO date string (YYYY-MM-DD)
    channel_id: str
    channel_name: str
    overview: str                           # 1-2 sentence description, updated each batch
    md_path: str                            # relative path to the .md file
    participants: list[str] = Field(default_factory=list)  # accumulated unique names
    start_ts: int                           # Unix epoch integer — NEVER datetime
