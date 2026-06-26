"""Confluence adapter for the vendored Confluence proposal pipeline.

Drives the proven Confluence proposal *logic* (vendored byte-for-byte into
``review_pipeline.confluence_pipeline``) against Confluence pages, using Pinecone-native
hybrid retrieval (dense + sparse integrated inference + rerank) — nothing runs on
a local vector store. The proposal agents (intent extraction → evaluation →
editing → verification) are unchanged; only retrieval is Pinecone-backed.

Flow:
  transcript → confluence_pipeline.propose(retriever=PineconeHybridIndex)
            → ConfluenceProposal[]  (source_chunk.source_path = Confluence page_id)
            → my-agent Proposal[]  (only chunks with a non-empty page_id survive)

Proposals always go through PineconeHybridIndex — no fallback path.
"""
from __future__ import annotations

import datetime as dt
import logging
import os
import uuid
from collections.abc import Awaitable, Callable
from typing import Any

from .confluence_pipeline import PineconeHybridIndex, PipelineConfig, propose
from .confluence_pipeline.models import ConfluenceProposal
from .models import ExtractedMeeting, Proposal
from .text_utils import format_transcript, html_to_text, looks_like_storage_html, normalize_ws

logger = logging.getLogger(__name__)

EmitFn = Callable[[dict[str, Any]], Awaitable[None]]


def _utcnow() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def _confluence_page_url(page_id: str) -> str | None:
    """Construct a Confluence page URL from env when available."""
    domain = os.getenv("ATLASSIAN_DOMAIN", "").strip()
    if domain and page_id:
        return f"https://{domain}/wiki/spaces/~/pages/{page_id}"
    return None


def _confidence_bin(score: float) -> str:
    if score >= 0.8:
        return "high"
    if score >= 0.6:
        return "medium"
    return "low"


def _proposal_from_native(
    p: ConfluenceProposal,
    session_id: str,
) -> dict[str, Any] | None:
    """Map a vendored ConfluenceProposal to a my-agent Proposal dict.

    source_chunk.source_path is the Confluence page_id (set by reindex_live).
    Returns None when the chunk has no page_id — those are dropped.
    """
    page_id = str(p.source_chunk.source_path or "").strip()
    if not page_id:
        return None

    edit_type = p.edit_type or "replace"
    if edit_type == "delete_section":
        change_type, edit_mode, after = "delete", "replace", ""
    else:
        change_type = "edit"
        edit_mode = "append" if edit_type == "append" else "replace"
        after = p.after_content

    # Defensive: the editor operates on clean markdown, but if any residual
    # Confluence XHTML slipped through it must not reach the write-back path —
    # replace_text_in_storage html.escape()s `after`, which would surface literal
    # &lt;ac:…&gt; on the page. Strip tags so the anchor/replacement stay clean.
    before = p.before_content or ""
    if looks_like_storage_html(before):
        before = html_to_text(before)
    if after and looks_like_storage_html(after):
        after = html_to_text(after)

    quality = float(p.quality_score or p.confidence or 0.0)
    bin_ = _confidence_bin(quality)
    intent = p.intent
    topic = normalize_ws(intent.affected_topic or "")
    rationale = normalize_ws(intent.rationale or "") or (
        f"{topic}: {intent.new_value}".strip(": ") or "Change proposed from the meeting transcript."
    )
    heading = normalize_ws(p.source_chunk.section_heading or "") or None
    page_title = str(p.source_chunk.doc_title or topic or "Confluence Page")

    proposal = Proposal(
        id=str(uuid.uuid4()),
        change_type=change_type,  # type: ignore[arg-type]
        page_id=page_id,
        page_title=page_title,
        section_heading=heading,
        before_content=before,
        after_content=after,
        timestamp=_utcnow(),
        session_id=session_id,
        rationale=rationale,
        transcript_evidence=[normalize_ws(str(s)) for s in (intent.verbatim_snippets or [])][:3],
        confidence=bin_,  # type: ignore[arg-type]
        risk="safe" if bin_ == "high" else "review",
        verifier_note=normalize_ws(p.verifier_note or "") or None,
        edit_mode=edit_mode,  # type: ignore[arg-type]
        change_summary=(f"Update '{heading}' on {page_title}" if heading else f"Update {topic}".strip()) or None,
        page_url=_confluence_page_url(page_id),
        source="confluence-pipeline",
        confidence_score=quality,
        confidence_bin=bin_,  # type: ignore[arg-type]
    )
    return proposal.to_dict()


async def run_confluence_pipeline(
    *,
    session_id: str,
    transcript: list[dict[str, Any]],
    memory_context: str | None = None,
    emit: EmitFn | None = None,
) -> tuple[ExtractedMeeting, list[dict[str, Any]]]:
    """Propose Confluence changes via PineconeHybridIndex pipeline."""
    from memory_compaction import add_memory_context

    async def _emit(event: dict[str, Any]) -> None:
        if emit:
            await emit(event)

    await _emit({"type": "stage_start", "stage": "transcript_source"})
    transcript_text = add_memory_context(format_transcript(transcript), memory_context)
    if not transcript_text:
        return ExtractedMeeting(title="Meeting Review", summary="No transcript captured."), []

    async def _stage(stage: str) -> None:
        await _emit({"type": "stage_start", "stage": stage})

    retriever = PineconeHybridIndex()
    cfg = PipelineConfig(session_id=session_id)
    intents, ld_proposals = await propose(
        transcript_text, retriever=retriever, config=cfg, emit=_stage,
    )

    proposals: list[dict[str, Any]] = []
    dropped = 0
    for p in ld_proposals:
        mapped = _proposal_from_native(p, session_id)
        if mapped:
            proposals.append(mapped)
            await _emit({"type": "proposal_ready", **mapped})
        else:
            dropped += 1

    meeting = ExtractedMeeting(
        title="Meeting Review",
        summary=(
            f"{len(intents)} change intent(s) detected; "
            f"{len(proposals)} Confluence proposal(s) ready for review."
        ),
    )
    logger.info(
        "confluence_pipeline adapter: %d intents -> %d proposals (%d dropped: no Confluence page)",
        len(intents), len(proposals), dropped,
    )
    return meeting, proposals
