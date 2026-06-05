"""Docling-based document chunker for local file formats.

Converts any supported file (DOCX, ODT, PDF, TXT, MD, RTF) into a list of
ChunkRecord objects keyed by section heading. Uses a single module-level
DocumentConverter instance (warm-up at import time) to avoid repeated heavy
ML loads — identical to the pattern in confluence_logic/ingestion/doc_pipeline.py.

Security: only reads files with known-safe extensions (.docx, .odt, .pdf,
.txt, .md, .rtf). Unknown extensions raise ValueError without reading content
(T-12-02 path traversal mitigation).
"""

from __future__ import annotations

import hashlib
import logging
import re
import warnings
from pathlib import Path
from typing import TYPE_CHECKING

# Suppress docling verbose startup warnings
warnings.filterwarnings("ignore", category=UserWarning)

from docling.document_converter import DocumentConverter  # noqa: E402

from models.rag import ChunkRecord  # noqa: E402

logger = logging.getLogger(__name__)

# ── Module-level singleton (heavy ML load happens once at import) ─────────────
_CONVERTER = DocumentConverter()

# ── Supported extensions (T-12-02: allowlist, not blocklist) ─────────────────
_SAFE_EXTENSIONS = {"docx", "odt", "pdf", "txt", "md", "rtf"}


# ── Public API ────────────────────────────────────────────────────────────────


def chunk_document(file_path: str) -> list[ChunkRecord]:
    """Convert a local file into a list of ChunkRecord objects.

    Parameters
    ----------
    file_path:
        Absolute or relative path to the document.  Must have a supported
        extension (.docx, .odt, .pdf, .txt, .md, .rtf).

    Returns
    -------
    list[ChunkRecord]
        One record per section (heading-delimited).  Large sections are split
        further by :func:`_split_oversized_section`.  Returns ``[]`` for RTF
        when ``striprtf`` is not installed.
    """
    path = Path(file_path)
    ext = path.suffix.lower().lstrip(".")

    # T-12-02: allowlist enforcement
    if ext not in _SAFE_EXTENSIONS:
        raise ValueError(
            f"Unsupported file extension '.{ext}'. "
            f"Allowed: {', '.join(sorted(_SAFE_EXTENSIONS))}"
        )

    # RTF: use striprtf if available, otherwise graceful skip
    if ext == "rtf":
        return _handle_rtf(file_path, ext)

    # Plain text: read raw to preserve line structure. docling reflows .txt into
    # one paragraph, collapsing "SECTION N." headings into the body line and
    # destroying section granularity (and breaking write-back). Raw read keeps
    # headings on their own lines so _promote_plaintext_headings can detect them.
    if ext == "txt":
        try:
            raw = path.read_text(encoding="utf-8", errors="replace")
        except Exception as exc:
            logger.warning("Failed to read text file %s: %s", file_path, exc)
            return []
        return _markdown_to_chunks(raw, file_path, ext)

    # ODT: docling does NOT support OpenDocument Text (.odt is not in its allowed
    # input formats), so a docling convert raises and the document yields zero
    # chunks — i.e. .odt files would be silently invisible to the pipeline. Read
    # them structurally via odfpy instead (headings + paragraphs in document
    # order), which is already the write-back backend for .odt.
    if ext == "odt":
        try:
            markdown = _read_odt_markdown(file_path)
        except Exception as exc:
            logger.warning("odfpy ODT read failed for %s: %s", file_path, exc)
            return []
        return _markdown_to_chunks(markdown, file_path, ext)

    # All other formats: docling single-API conversion
    try:
        result = _CONVERTER.convert(str(path))
        markdown = result.document.export_to_markdown()
    except Exception as exc:
        logger.warning("docling conversion failed for %s: %s", file_path, exc)
        return []

    return _markdown_to_chunks(markdown, file_path, ext)


def _split_oversized_section(text: str, max_tokens: int = 800) -> list[str]:
    """Split a large section at paragraph boundaries.

    Greedily accumulates `\\n\\n`-separated paragraphs until the next
    paragraph would push the word count past *max_tokens*.  Always returns
    at least one part.

    Parameters
    ----------
    text:
        Section body text (no heading line).
    max_tokens:
        Approximate word-count ceiling per part.
    """
    paragraphs = re.split(r"\n\n+", text.strip())
    parts: list[str] = []
    current: list[str] = []
    current_count = 0

    for para in paragraphs:
        words = len(para.split())
        if current and current_count + words > max_tokens:
            parts.append("\n\n".join(current))
            current = [para]
            current_count = words
        else:
            current.append(para)
            current_count += words

    if current:
        parts.append("\n\n".join(current))

    return parts if parts else [text]


# ── Internal helpers ──────────────────────────────────────────────────────────


def _file_hash(path: str) -> str:
    """Return first 8 hex chars of the MD5 of the file's bytes."""
    return hashlib.md5(Path(path).read_bytes()).hexdigest()[:8]


def _read_odt_markdown(file_path: str) -> str:
    """Convert an .odt file to Markdown by walking its body in document order.

    Headings (text:h) become ``# heading`` lines and paragraphs (text:p) become
    body lines, preserving order so :func:`_markdown_to_chunks` can split the
    document into heading-delimited sections. Mirrors the node walk that
    SafeApply.apply_odt uses for write-back, so read and write stay symmetric.
    """
    from odf import teletype
    from odf.opendocument import load

    doc = load(file_path)
    lines: list[str] = []

    def _local(child) -> str:
        # Loaded odfpy elements are generic ``Element`` instances; the reliable
        # type signal is the qualified name's local part (text:h -> "h").
        qn = getattr(child, "qname", None)
        return qn[1] if qn else child.__class__.__name__.lower()

    def _walk(node) -> None:
        for child in getattr(node, "childNodes", []):
            local = _local(child)
            if local == "h":
                text = teletype.extractText(child).strip()
                if text:
                    lines.append(f"# {text}")
            elif local == "p":
                text = teletype.extractText(child).strip()
                if text:
                    lines.append(text)
            elif local in ("list", "section", "table", "table-cell", "list-item"):
                _walk(child)  # recurse into common containers

    _walk(doc.text)
    return "\n\n".join(lines)


def _handle_rtf(file_path: str, ext: str) -> list[ChunkRecord]:
    """Attempt RTF conversion via striprtf; return [] if unavailable."""
    try:
        from striprtf.striprtf import rtf_to_text  # type: ignore[import]
    except ImportError:
        logger.warning(
            "striprtf not installed; skipping RTF file %s. "
            "Install striprtf to enable RTF chunking.",
            file_path,
        )
        return []

    try:
        raw = Path(file_path).read_text(encoding="utf-8", errors="replace")
        plain_text = rtf_to_text(raw)
    except Exception as exc:
        logger.warning("RTF read/conversion failed for %s: %s", file_path, exc)
        return []

    return _markdown_to_chunks(plain_text, file_path, ext)


def _markdown_to_chunks(
    markdown: str,
    file_path: str,
    source_format: str,
) -> list[ChunkRecord]:
    """Convert Markdown (or plain text) into a list of ChunkRecord objects."""
    sections = re.split(r"\n(?=#+ )", markdown)
    if not sections:
        sections = [markdown]

    # Detect whether ANY section has a real heading
    has_headings = any(
        re.match(r"^#+ ", s.lstrip()) for s in sections
    )

    # Plain-text docs (esp. .txt) often carry headings that docling does NOT
    # promote to Markdown '#': e.g. "SECTION 1. ..." or ALL-CAPS title lines.
    # Promote those to Markdown headings before giving up to page-level windows,
    # so retrieval and write-back operate at section granularity.
    if not has_headings:
        promoted = _promote_plaintext_headings(markdown)
        if promoted is not None:
            markdown = promoted
            sections = re.split(r"\n(?=#+ )", markdown)
            has_headings = True

    if not has_headings:
        return _page_level_fallback(markdown, file_path, source_format)

    file_hash = _file_hash(file_path)
    chunks: list[ChunkRecord] = []
    global_index = 0

    for section in sections:
        section = section.strip()
        if not section:
            continue

        lines = section.split("\n")
        heading_match = re.match(r"^#+ (.*)", lines[0].strip())
        if heading_match:
            heading = heading_match.group(1).strip()
            body = "\n".join(lines[1:]).strip()
        else:
            heading = "Page intro"
            body = section

        if not body:
            body = section

        word_count = len(body.split())
        base_chunk_id = f"{file_hash}:{global_index}"

        if word_count > 800:
            sub_parts = _split_oversized_section(body, max_tokens=800)
            for sub_i, part in enumerate(sub_parts):
                chunks.append(
                    ChunkRecord(
                        chunk_id=f"{base_chunk_id}.{sub_i}",
                        source_path=str(Path(file_path).resolve()),
                        source_format=source_format,
                        section_heading=heading,
                        section_index=global_index,
                        content=part,
                        token_count=len(part.split()),
                    )
                )
        else:
            chunks.append(
                ChunkRecord(
                    chunk_id=base_chunk_id,
                    source_path=str(Path(file_path).resolve()),
                    source_format=source_format,
                    section_heading=heading,
                    section_index=global_index,
                    content=body,
                    token_count=word_count,
                )
            )

        global_index += 1

    return chunks


def _looks_like_plaintext_heading(line: str) -> bool:
    """Heuristic: is this line a section heading in a plain-text document?

    Domain-agnostic — detects two common conventions without hardcoding any
    specific document's wording:
      1. Numbered section markers: "SECTION 1. ...", "1. ...", "1.2 ..." etc.
      2. Short ALL-CAPS title lines (the alphabetic characters are all upper).
    """
    s = line.strip()
    if not s or len(s) > 70:
        return False
    # Numbered section conventions (SECTION N., N., N.N) followed by a word
    if re.match(r"^(SECTION\s+)?\d+(\.\d+)*\.?\s+\S", s, re.IGNORECASE) and s.upper() == s:
        return True
    # ALL-CAPS heading line: needs >=3 letters, and every letter is uppercase
    letters = [c for c in s if c.isalpha()]
    if len(letters) >= 3 and all(c.isupper() for c in letters):
        return True
    return False


def _promote_plaintext_headings(text: str) -> str | None:
    """Prefix detected plain-text heading lines with '# '.

    Returns the rewritten text if at least one heading was found, else None
    (signalling the caller to fall back to page-level windowing).
    """
    out_lines: list[str] = []
    found = 0
    for line in text.split("\n"):
        if _looks_like_plaintext_heading(line):
            out_lines.append(f"# {line.strip()}")
            found += 1
        else:
            out_lines.append(line)
    if found == 0:
        return None
    return "\n".join(out_lines)


def _page_level_fallback(
    content: str,
    file_path: str,
    source_format: str,
) -> list[ChunkRecord]:
    """Split headingless content into overlapping 400-token windows.

    Heading label format: ``"Page 1, Window {i+1}"`` to satisfy the
    test assertion ``"Window" in c.section_heading or "Page" in c.section_heading``.
    """
    file_hash = _file_hash(file_path)
    words = content.split()
    window_size = 400
    overlap = 80
    step = window_size - overlap

    chunks: list[ChunkRecord] = []
    i = 0
    window_index = 0

    if not words:
        return [
            ChunkRecord(
                chunk_id=f"{file_hash}:0",
                source_path=str(Path(file_path).resolve()),
                source_format=source_format,
                section_heading="Page 1, Window 1",
                section_index=0,
                content=content,
                token_count=0,
            )
        ]

    while i < len(words):
        window_words = words[i : i + window_size]
        window_text = " ".join(window_words)
        chunks.append(
            ChunkRecord(
                chunk_id=f"{file_hash}:{window_index}",
                source_path=str(Path(file_path).resolve()),
                source_format=source_format,
                section_heading=f"Page 1, Window {window_index + 1}",
                section_index=window_index,
                content=window_text,
                token_count=len(window_words),
            )
        )
        i += step
        window_index += 1
        if i >= len(words):
            break

    return chunks
