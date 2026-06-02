"""Tests for the local-document chunker — RAG-01, RAG-06, RAG-07.

All tests fail with ImportError until Wave 1 implements rag/chunker.py.
The import is deferred into each test body so pytest can collect without errors.
"""

from __future__ import annotations

import pytest

from tests.fixtures import FIXTURES_DIR


def test_docx_heading_extraction():
    """RAG-01: DOCX section-level chunking extracts 3 headed sections."""
    from rag.chunker import chunk_document  # ImportError until Wave 1

    chunks = chunk_document(str(FIXTURES_DIR / "policy.docx"))
    assert len(chunks) == 3
    headings = [c.section_heading for c in chunks]
    assert "Data Retention Policy" in headings
    assert "Access Control" in headings
    assert "Incident Response" in headings


def test_pdf_heading_extraction():
    """RAG-06: PDF chunking via docling detects headings."""
    from rag.chunker import chunk_document  # ImportError until Wave 1
    from pathlib import Path

    pdf_path = FIXTURES_DIR / "sample.pdf"
    if not pdf_path.exists():
        pytest.skip("PDF fixture not yet created")
    chunks = chunk_document(str(pdf_path))
    assert len(chunks) >= 1
    assert all(c.source_format == "pdf" for c in chunks)


def test_markdown_heading_extraction():
    """Section-level chunking of Markdown files."""
    from rag.chunker import chunk_document  # ImportError until Wave 1

    chunks = chunk_document(str(FIXTURES_DIR / "handbook.md"))
    assert len(chunks) == 3
    headings = [c.section_heading for c in chunks]
    assert "Onboarding Process" in headings


def test_no_heading_fallback():
    """RAG-07: Page-level fallback triggers when no headings detected."""
    from rag.chunker import chunk_document  # ImportError until Wave 1

    chunks = chunk_document(str(FIXTURES_DIR / "no_headings.txt"))
    assert len(chunks) >= 1
    for c in chunks:
        assert "Window" in c.section_heading or "Page" in c.section_heading


def test_chunk_metadata_complete():
    """Every chunk carries required metadata fields."""
    from rag.chunker import chunk_document  # ImportError until Wave 1

    chunks = chunk_document(str(FIXTURES_DIR / "handbook.md"))
    for c in chunks:
        assert c.chunk_id
        assert c.source_path
        assert c.source_format in ("docx", "odt", "pdf", "txt", "md", "rtf")
        assert c.section_heading
        assert isinstance(c.section_index, int)
        assert c.content


def test_oversized_section_split():
    """Sections >800 tokens are split at paragraph boundaries."""
    from rag.chunker import _split_oversized_section  # ImportError until Wave 1

    long_text = ("This is a long paragraph. " * 60).strip()
    parts = _split_oversized_section(long_text, max_tokens=800)
    assert len(parts) >= 1
    for p in parts:
        assert len(p.split()) <= 850  # generous upper bound
