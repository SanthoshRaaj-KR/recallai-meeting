"""Confluence adapter for the vendored Confluence proposal pipeline.

Drives the proven Confluence proposal *logic* (vendored byte-for-byte into
``review_pipeline.confluence_pipeline``) against Confluence pages, using Pinecone-native
hybrid retrieval (dense + sparse integrated inference + rerank) — nothing runs on
a local vector store. The proposal agents (intent extraction → evaluation →
editing → verification) are unchanged; only retrieval is Pinecone-backed.

Flow:
  transcript → confluence_pipeline.propose(retriever=PineconeHybridIndex)
            → ConfluenceProposal[]  (source_chunk.source_path = corpus filename)
            → map filename → Confluence page_id via corpus_page_map.json
            → my-agent Proposal[]  (only pages with a real page_id survive)

Enabled via ``MY_AGENT_USE_LOCAL_DOC_PIPELINE`` (default on).
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import os
import uuid
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from .confluence_pipeline import PineconeHybridIndex, PipelineConfig, propose
from .confluence_pipeline.models import ConfluenceProposal
from .models import ExtractedMeeting, Proposal
from .pipeline import ProposalPipeline
from .text_utils import format_transcript, normalize_ws

logger = logging.getLogger(__name__)

EmitFn = Callable[[dict[str, Any]], Awaitable[None]]


def enabled() -> bool:
    return os.getenv("MY_AGENT_USE_LOCAL_DOC_PIPELINE", "1").strip().lower() in {
        "1", "true", "yes", "on",
    }


def _utcnow() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def _default_page_map() -> str:
    # repo_root/local_doc_change/corpus_page_map.json  (my-agent/src/review_pipeline/..)
    return str(Path(__file__).resolve().parents[3] / "local_doc_change" / "corpus_page_map.json")


_page_map_cache: dict[str, dict] | None = None


def _page_map() -> dict[str, dict]:
    global _page_map_cache
    if _page_map_cache is None:
        path = os.getenv("MY_AGENT_LDOC_PAGE_MAP") or _default_page_map()
        try:
            _page_map_cache = json.loads(Path(path).read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("could not load page map %s: %s", path, exc)
            _page_map_cache = {}
    return _page_map_cache


def _confidence_bin(score: float) -> str:
    if score >= 0.8:
        return "high"
    if score >= 0.6:
        return "medium"
    return "low"


def _proposal_from_native(
    p: ConfluenceProposal,
    page_map: dict[str, dict],
    session_id: str,
) -> dict[str, Any] | None:
    """Map a vendored ConfluenceProposal to a my-agent Proposal dict.

    Returns None when the source document does not map to a Confluence page —
    those are dropped so only real Confluence pages become review cards (this
    also filters cross-company noise from the combined test corpus).
    """
    base = os.path.basename(str(p.source_chunk.source_path))
    meta = page_map.get(base)
    if not meta:
        return None
    page_id = str(meta.get("page_id") or "")
    if not page_id:
        return None

    edit_type = p.edit_type or "replace"
    if edit_type == "delete_section":
        change_type, edit_mode, after = "delete", "replace", ""
    else:
        change_type = "edit"
        edit_mode = "append" if edit_type == "append" else "replace"
        after = p.after_content

    quality = float(p.quality_score or p.confidence or 0.0)
    bin_ = _confidence_bin(quality)
    intent = p.intent
    topic = normalize_ws(intent.affected_topic or "")
    rationale = normalize_ws(intent.rationale or "") or (
        f"{topic}: {intent.new_value}".strip(": ") or "Change proposed from the meeting transcript."
    )
    heading = normalize_ws(p.source_chunk.section_heading or "") or None
    page_title = str(meta.get("title") or p.source_chunk.doc_title or "Confluence Page")

    proposal = Proposal(
        id=str(uuid.uuid4()),
        change_type=change_type,  # type: ignore[arg-type]
        page_id=page_id,
        page_title=page_title,
        section_heading=heading,
        before_content=p.before_content,
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
        page_url=str(meta.get("url") or "") or None,
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
    pipeline: ProposalPipeline | None = None,  # accepted for signature compat; unused
) -> tuple[ExtractedMeeting, list[dict[str, Any]]]:
    """Drop-in replacement for ProposalPipeline.run that proposes via Confluence logic."""
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

    page_map = _page_map()
    proposals: list[dict[str, Any]] = []
    dropped = 0
    for p in ld_proposals:
        mapped = _proposal_from_native(p, page_map, session_id)
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
