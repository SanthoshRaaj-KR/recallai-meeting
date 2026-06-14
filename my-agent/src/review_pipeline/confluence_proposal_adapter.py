"""Confluence adapter for the local_doc_change pipeline.

Runs the PROVEN local-doc-change proposal pipeline against Confluence pages
without modifying any of its logic. The local-doc pipeline reads a folder of
local files and proposes edits; this adapter:

  1. reuses the existing meeting extraction + Confluence vector RAG to pick the
     pages relevant to the meeting (the prev builder's "knowledge confluence"),
  2. materializes those pages as local markdown files in a temp folder,
  3. invokes ``local_doc_change/run_json.py`` as a SUBPROCESS in the local-doc
     virtualenv (which already has docling/faiss) — so my-agent stays free of
     those heavy deps and the pipeline logic is byte-for-byte unchanged,
  4. maps each returned ``LocalDocProposal`` back to a Confluence page and emits
     the same ``Proposal`` dicts the rest of my-agent already consumes.

Enabled via ``MY_AGENT_USE_LOCAL_DOC_PIPELINE`` (default on). The python
interpreter and pipeline location are configurable via ``LDOC_PYTHON`` and
``LDOC_DIR`` so deployment can point them elsewhere.
"""
from __future__ import annotations

import asyncio
import datetime as dt
import json
import logging
import os
import re
import tempfile
import uuid
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from .models import ExtractedMeeting, PageCandidate, Proposal
from .pipeline import ProposalPipeline
from .text_utils import extract_sections, html_to_text, normalize_ws

logger = logging.getLogger(__name__)

EmitFn = Callable[[dict[str, Any]], Awaitable[None]]


def enabled() -> bool:
    return os.getenv("MY_AGENT_USE_LOCAL_DOC_PIPELINE", "1").strip().lower() in {
        "1", "true", "yes", "on",
    }


def _utcnow() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def _ldoc_dir() -> Path:
    override = os.getenv("LDOC_DIR")
    if override:
        return Path(override)
    # default: sibling local_doc_change/ next to the repo root (my-agent/src/...)
    return Path(__file__).resolve().parents[3] / "local_doc_change"


def _ldoc_python() -> str:
    override = os.getenv("LDOC_PYTHON")
    if override:
        return override
    base = _ldoc_dir()
    win = base / ".venv" / "Scripts" / "python.exe"
    nix = base / ".venv" / "bin" / "python"
    if win.exists():
        return str(win)
    if nix.exists():
        return str(nix)
    return "python"


_MAX_CANDIDATE_PAGES = int(os.getenv("MY_AGENT_LDOC_MAX_PAGES", "16"))
_SUBPROCESS_TIMEOUT = int(os.getenv("MY_AGENT_LDOC_TIMEOUT", "900"))


def _cell_md(cell_html: str) -> str:
    """One table cell -> single-line markdown text (pipes escaped)."""
    return normalize_ws(html_to_text(cell_html)).replace("|", r"\|")


def _table_to_md(table_html: str) -> str:
    """Convert a Confluence storage <table> to a markdown table (one row per line).

    Preserving rows is what lets the pipeline target a single table row (e.g. the
    'Managed Threat Hunting' price) instead of editing one flattened blob.
    """
    rows = re.findall(r"<tr[^>]*>(.*?)</tr>", table_html, re.DOTALL | re.IGNORECASE)
    md: list[str] = []
    header_done = False
    for row in rows:
        cells = [_cell_md(c) for c in re.findall(r"<t[hd][^>]*>(.*?)</t[hd]>", row, re.DOTALL | re.IGNORECASE)]
        if not any(cells):
            continue
        md.append("| " + " | ".join(cells) + " |")
        if not header_done:
            md.append("| " + " | ".join("---" for _ in cells) + " |")
            header_done = True
    return "\n".join(md)


def _html_to_markdown(html_str: str) -> str:
    """Convert Confluence storage HTML to markdown, keeping tables as rows.

    Prose around tables is flattened (fine for paragraphs); tables become proper
    markdown so the local-doc pipeline can make surgical, per-row edits.
    """
    if not html_str:
        return ""
    parts: list[str] = []
    pos = 0
    for m in re.finditer(r"<table[^>]*>.*?</table>", html_str, re.DOTALL | re.IGNORECASE):
        pre = normalize_ws(html_to_text(html_str[pos:m.start()]))
        if pre:
            parts.append(pre)
        tbl = _table_to_md(m.group(0))
        if tbl:
            parts.append(tbl)
        pos = m.end()
    tail = normalize_ws(html_to_text(html_str[pos:]))
    if tail:
        parts.append(tail)
    return "\n\n".join(p for p in parts if p.strip()).strip()


def _materialize_page(page: PageCandidate, folder: Path) -> Path | None:
    """Write a Confluence page as a markdown file the local-doc chunker can read.

    Section HTML is converted to markdown (tables preserved, newlines kept) so the
    pipeline edits the right row; headings are preserved so the proposal's
    section_heading maps back to the same Confluence section for write-back.
    """
    if not page.page_id:
        return None
    sections = page.sections or extract_sections(page.html)
    lines: list[str] = [f"# {normalize_ws(page.title) or 'Untitled'}", ""]
    for section in sections:
        heading = normalize_ws(str(section.get("heading") or ""))
        section_html = str(section.get("html") or "")
        # Prefer table-preserving markdown from HTML; fall back to the (already
        # flattened) RAG text when no HTML is available.
        body = _html_to_markdown(section_html) if section_html else normalize_ws(str(section.get("text") or ""))
        if not body:
            continue
        if heading:
            lines.append(f"## {heading}")
        lines.append(body)
        lines.append("")
    if len(lines) <= 2 and page.html:
        body = _html_to_markdown(page.html)
        if body:
            lines.append(body)
    # filename encodes the page_id so the proposal maps back unambiguously
    path = folder / f"page_{page.page_id}.md"
    path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")
    return path


_PAGE_ID_RE = re.compile(r"^page_(?P<pid>.+)\.md$", re.IGNORECASE)


def _page_id_from_path(source_path: str) -> str | None:
    m = _PAGE_ID_RE.match(os.path.basename(str(source_path)))
    return m.group("pid") if m else None


def _confidence_bin(score: float) -> str:
    if score >= 0.8:
        return "high"
    if score >= 0.6:
        return "medium"
    return "low"


def _proposal_from_ldoc(
    raw: dict[str, Any],
    pages_by_id: dict[str, PageCandidate],
    session_id: str,
) -> dict[str, Any] | None:
    chunk = raw.get("source_chunk") or {}
    intent = raw.get("intent") or {}
    page_id = _page_id_from_path(chunk.get("source_path") or "")
    if not page_id:
        return None
    page = pages_by_id.get(page_id)
    edit_type = str(raw.get("edit_type") or "replace")
    before = raw.get("before_content")
    after = raw.get("after_content")
    if edit_type == "delete_section":
        change_type = "delete"
        edit_mode = "replace"
        after = ""
    else:
        change_type = "edit"
        edit_mode = "append" if edit_type == "append" else "replace"

    quality = float(raw.get("quality_score") or raw.get("confidence") or 0.0)
    bin_ = _confidence_bin(quality)
    topic = normalize_ws(str(intent.get("affected_topic") or ""))
    new_value = normalize_ws(str(intent.get("new_value") or ""))
    rationale = normalize_ws(str(intent.get("rationale") or "")) or (
        f"{topic}: {new_value}".strip(": ") or "Change proposed from the meeting transcript."
    )
    section_heading = normalize_ws(str(chunk.get("section_heading") or "")) or None

    proposal = Proposal(
        id=str(uuid.uuid4()),
        change_type=change_type,  # type: ignore[arg-type]
        page_id=page_id,
        page_title=(page.title if page else normalize_ws(str(chunk.get("doc_title") or "")) or "Confluence Page"),
        section_heading=section_heading,
        before_content=before,
        after_content=after,
        timestamp=_utcnow(),
        session_id=session_id,
        rationale=rationale,
        transcript_evidence=[normalize_ws(str(s)) for s in (intent.get("verbatim_snippets") or [])][:3],
        confidence=bin_,  # type: ignore[arg-type]
        risk="safe" if bin_ == "high" else "review",
        verifier_note=normalize_ws(str(raw.get("verifier_note") or "")) or None,
        edit_mode=edit_mode,  # type: ignore[arg-type]
        change_summary=(f"Update '{section_heading}' on {page.title}" if page and section_heading
                        else f"Update {topic}".strip()) or None,
        page_url=(page.url if page else None),
        source="local-doc-pipeline",
        confidence_score=quality,
        confidence_bin=bin_,  # type: ignore[arg-type]
    )
    return proposal.to_dict()


def _page_from_hits(page_id: str, hits: list[Any]) -> PageCandidate:
    """Build a PageCandidate from RAG section hits (no live Confluence needed).

    Uses the indexed Confluence content (the prev builder's knowledge) so the
    pipeline can propose changes even when the live REST API is unreachable.
    """
    title = ""
    sections: list[dict[str, str]] = []
    seen_headings: set[str] = set()
    version = None
    for h in hits:
        title = title or normalize_ws(getattr(h, "title", "") or "")
        version = version or getattr(h, "version", None)
        heading = normalize_ws(getattr(h, "heading", "") or "")
        text = normalize_ws(getattr(h, "text", "") or "")
        if not text:
            continue
        key = heading.lower()
        if key in seen_headings:
            continue
        seen_headings.add(key)
        sections.append({"heading": heading or title, "text": text, "html": ""})
    return PageCandidate(
        page_id=page_id,
        title=title or f"Page {page_id}",
        url=None,
        html="",
        version=version,
        source="rag_content",
        sections=sections,
    )


async def run_local_doc_pipeline(
    *,
    session_id: str,
    transcript: list[dict[str, Any]],
    memory_context: str | None = None,
    emit: EmitFn | None = None,
    pipeline: ProposalPipeline | None = None,
) -> tuple[ExtractedMeeting, list[dict[str, Any]]]:
    """Drop-in replacement for ProposalPipeline.run that proposes via local-doc logic.

    Reuses ``pipeline`` (the caller's ProposalPipeline) for extraction, RAG and
    Confluence I/O when provided, so the caller's ``summary_response`` stays valid.
    """
    from memory_compaction import add_memory_context

    from .text_utils import format_transcript

    async def _emit(event: dict[str, Any]) -> None:
        if emit:
            await emit(event)

    pp = pipeline or ProposalPipeline()
    await _emit({"type": "stage_start", "stage": "transcript_source"})
    transcript_text = add_memory_context(format_transcript(transcript), memory_context)
    if not transcript_text:
        return ExtractedMeeting(title="Meeting Review", summary="No transcript captured."), []

    # 1. Reuse the proven extraction + Confluence RAG to pick relevant pages.
    await _emit({"type": "stage_start", "stage": "fact_extraction"})
    meeting = await pp._extract_meeting(transcript, transcript_text)
    fallback = pp._intents_from_action_items(meeting)
    if fallback:
        meeting.change_intents = pp._merge_intents(meeting.change_intents, fallback)[:30]
    if not meeting.change_intents:
        return meeting, []

    await _emit({"type": "stage_start", "stage": "rag_retrieval"})
    intent_sections = await pp._retrieve_all_sections(meeting.change_intents)
    # Group retrieved sections by page so each candidate page can be materialized
    # from the RAG content itself — this is the prev builder's indexed Confluence
    # knowledge, and works even when the live Confluence REST API is unreachable.
    hits_by_page: dict[str, list[Any]] = {}
    order: list[str] = []
    for _intent, hits in intent_sections:
        for hit in hits:
            if not hit.page_id:
                continue
            if hit.page_id not in hits_by_page:
                hits_by_page[hit.page_id] = []
                order.append(hit.page_id)
            hits_by_page[hit.page_id].append(hit)
    page_ids = order[:_MAX_CANDIDATE_PAGES]
    if not page_ids:
        await _emit({"type": "stage_progress", "stage": "rag_retrieval", "pages_found": 0})
        return meeting, []

    # Prefer the full live page (gives version/url for write-back); fall back to a
    # page built from the RAG hits when Confluence is unreachable (e.g. 403).
    pages_by_id: dict[str, PageCandidate] = {}
    live = rag_only = 0
    for pid in page_ids:
        page: PageCandidate | None = None
        try:
            page = await pp._fetch_page_for_retrieval(pid, "local_doc_adapter")
        except Exception as exc:
            logger.debug("live fetch failed for page %s (%s) — using RAG content", pid, exc)
            page = None
        if page is not None and (page.sections or page.html):
            live += 1
        else:
            page = _page_from_hits(pid, hits_by_page[pid])
            rag_only += 1
        if page.page_id:
            pages_by_id[page.page_id] = page
    await _emit({"type": "stage_progress", "stage": "rag_retrieval",
                 "pages_live": live, "pages_from_rag": rag_only})
    if not pages_by_id:
        return meeting, []

    # 2. Materialize pages -> temp markdown folder.
    await _emit({"type": "stage_start", "stage": "section_fetch"})
    tmp = Path(tempfile.mkdtemp(prefix="ldoc_conf_"))
    docs = tmp / "docs"
    docs.mkdir(parents=True, exist_ok=True)
    materialized = 0
    for page in pages_by_id.values():
        if _materialize_page(page, docs):
            materialized += 1
    await _emit({"type": "stage_progress", "stage": "section_fetch", "pages_materialized": materialized})

    # 3. Run the UNCHANGED local-doc pipeline as a subprocess in its own venv.
    await _emit({"type": "stage_start", "stage": "drafting"})
    request = {
        "transcript": transcript_text,
        "doc_folder": str(docs),
        "session_id": session_id,
        "config": {"use_embeddings": True, "rerank": False, "contextual_retrieval": False},
    }
    req_path = tmp / "request.json"
    out_path = tmp / "proposals.json"
    req_path.write_text(json.dumps(request, ensure_ascii=False), encoding="utf-8")

    env = dict(os.environ)
    env.setdefault("LDOC_VECTOR_DB", "faiss")  # deterministic local index for the candidate set
    runner = str(_ldoc_dir() / "run_json.py")
    proc = await asyncio.create_subprocess_exec(
        _ldoc_python(), runner, str(req_path), str(out_path),
        cwd=str(_ldoc_dir()),
        env=env,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        _stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=_SUBPROCESS_TIMEOUT)
    except asyncio.TimeoutError:
        proc.kill()
        logger.error("local-doc pipeline subprocess timed out after %ss", _SUBPROCESS_TIMEOUT)
        return meeting, []
    if proc.returncode != 0:
        logger.error("local-doc pipeline subprocess failed (rc=%s): %s",
                     proc.returncode, (stderr or b"").decode("utf-8", "replace")[-2000:])
        return meeting, []

    try:
        raw_proposals = json.loads(out_path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.error("could not read local-doc proposals: %s", exc)
        return meeting, []

    # 4. Map LocalDocProposal -> my-agent Proposal dicts.
    await _emit({"type": "stage_start", "stage": "verification"})
    proposals: list[dict[str, Any]] = []
    for raw in raw_proposals:
        mapped = _proposal_from_ldoc(raw, pages_by_id, session_id)
        if mapped:
            proposals.append(mapped)
            await _emit({"type": "proposal_ready", **mapped})
    logger.info("local-doc adapter produced %d proposal(s) from %d candidate page(s)",
                len(proposals), len(pages_by_id))
    return meeting, proposals
