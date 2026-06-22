"""Markdown section chunker for the vendored Confluence pipeline.

The Confluence pipeline operates on documents split into *sections* (one
ChunkRecord per heading-delimited section). Inputs are Confluence pages
materialized as markdown by ``reindex_live``, so a lightweight,
dependency-free markdown splitter is enough — and it keeps each whole section
(including tables) intact so the editor can make surgical, per-row edits.
"""

from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path  # used for _slug stem extraction

from .models import ChunkRecord

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")
# Sections larger than this many words are split into sub-chunks that share the
# heading, so a single huge section never blows past Pinecone metadata limits.
_MAX_SECTION_WORDS = int(os.getenv("MY_AGENT_LDOC_SECTION_WORDS", "1200"))


def _slug(path: str) -> str:
    stem = Path(path).stem
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", stem)[:80] or hashlib.md5(path.encode()).hexdigest()[:12]


def _split_long(text: str, max_words: int) -> list[str]:
    words = text.split()
    if len(words) <= max_words:
        return [text]
    # Split on blank-line paragraph boundaries, packing up to max_words each.
    paras = re.split(r"\n\s*\n", text)
    out: list[str] = []
    cur: list[str] = []
    cur_wc = 0
    for para in paras:
        pw = len(para.split())
        if cur and cur_wc + pw > max_words:
            out.append("\n\n".join(cur))
            cur, cur_wc = [], 0
        cur.append(para)
        cur_wc += pw
    if cur:
        out.append("\n\n".join(cur))
    return out or [text]


def chunk_markdown_text(text: str, source_path: str, source_format: str = "md") -> list[ChunkRecord]:
    """Split markdown text into section-level ChunkRecords."""
    lines = text.splitlines()
    doc_title = ""
    # Document title = first level-1 heading, else filename stem.
    for ln in lines:
        m = _HEADING_RE.match(ln)
        if m and len(m.group(1)) == 1:
            doc_title = m.group(2).strip()
            break
    if not doc_title:
        doc_title = Path(source_path).stem.replace("_", " ").strip()

    # Group lines into (heading, body) sections. Level-1 title line itself is not
    # a separate section; content before the first sub-heading is the intro.
    sections: list[tuple[str, list[str]]] = []
    cur_heading = doc_title
    cur_body: list[str] = []
    started = False
    for ln in lines:
        m = _HEADING_RE.match(ln)
        if m:
            level = len(m.group(1))
            heading_text = m.group(2).strip()
            if level == 1 and not started:
                # the document title line; skip, keep gathering intro under title
                cur_heading = heading_text or doc_title
                continue
            # flush current section
            if cur_body and any(s.strip() for s in cur_body):
                sections.append((cur_heading, cur_body))
            cur_heading = heading_text
            cur_body = []
            started = True
        else:
            cur_body.append(ln)
    if cur_body and any(s.strip() for s in cur_body):
        sections.append((cur_heading, cur_body))

    slug = _slug(source_path)
    chunks: list[ChunkRecord] = []
    idx = 0
    for heading, body_lines in sections:
        body = "\n".join(body_lines).strip()
        if not body:
            continue
        for piece in _split_long(body, _MAX_SECTION_WORDS):
            chunks.append(
                ChunkRecord(
                    chunk_id=f"{slug}:{idx}",
                    source_path=source_path,
                    source_format=source_format,
                    section_heading=heading or doc_title,
                    section_index=idx,
                    content=piece,
                    doc_title=doc_title,
                    token_count=len(piece.split()),
                )
            )
            idx += 1
    return chunks


