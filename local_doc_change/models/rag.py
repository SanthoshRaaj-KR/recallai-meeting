"""RAG data models for chunked document records and retrieval results.

Defines ChunkRecord (indexed document section) and RetrievalResult
(ranked retrieval output combining BM25 + dense + RRF scores).
These are the authoritative contracts for Waves 1-4 of Phase 12.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict


class ChunkRecord(BaseModel):
    """A single indexed section of a local document."""

    model_config = ConfigDict(frozen=False)

    chunk_id: str  # f"{file_hash}:{section_index}"
    source_path: str  # absolute path to original file
    source_format: str  # "docx" | "odt" | "pdf" | "txt" | "md" | "rtf"
    section_heading: str  # extracted heading or synthetic label
    section_index: int  # position in document
    content: str  # raw text of section
    context_prefix: str = ""  # contextual description prepended at embed time
    token_count: int = 0


class RetrievalResult(BaseModel):
    """A ranked retrieval result combining BM25, dense, and RRF scores."""

    chunk: ChunkRecord
    bm25_rank: Optional[int] = None
    dense_rank: Optional[int] = None
    rrf_score: float = 0.0
    rerank_score: Optional[float] = None
    final_rank: int = 0
