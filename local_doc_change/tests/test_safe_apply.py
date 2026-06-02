"""Tests for SafeApply write-back layer — WRITE-01, WRITE-02.

All tests fail with ImportError until Wave 4 implements pipeline/safe_apply.py.
The import is deferred into each test body so pytest can collect without errors.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from tests.fixtures import FIXTURES_DIR


def test_backup_created(tmp_path):
    """WRITE-02: Backup file created before write-back."""
    from pipeline.safe_apply import SafeApply  # ImportError until Wave 4

    src = FIXTURES_DIR / "handbook.md"
    target = tmp_path / "handbook.md"
    shutil.copy(src, target)
    applier = SafeApply()
    applier.apply_markdown(
        file_path=str(target),
        section_heading="Onboarding Process",
        new_content="New hires must complete orientation within their first THREE days.",
    )
    backups = list(tmp_path.glob("handbook.backup.*.md"))
    assert len(backups) == 1, "Exactly one backup must be created"


def test_docx_writeback(tmp_path):
    """WRITE-01: DOCX write-back produces valid DOCX with expected change."""
    import docx
    from pipeline.safe_apply import SafeApply  # ImportError until Wave 4

    src = FIXTURES_DIR / "policy.docx"
    target = tmp_path / "policy.docx"
    shutil.copy(src, target)
    applier = SafeApply()
    applier.apply_docx(
        file_path=str(target),
        section_heading="Data Retention Policy",
        new_content="All records must be retained for 5 years per updated regulations.",
    )
    doc = docx.Document(str(target))
    full_text = " ".join(p.text for p in doc.paragraphs)
    assert "5 years" in full_text
    assert "7 years" not in full_text


def test_markdown_writeback(tmp_path):
    """Markdown write-back replaces section content correctly."""
    from pipeline.safe_apply import SafeApply  # ImportError until Wave 4

    src = FIXTURES_DIR / "handbook.md"
    target = tmp_path / "handbook.md"
    shutil.copy(src, target)
    applier = SafeApply()
    applier.apply_markdown(
        file_path=str(target),
        section_heading="Benefits Enrollment",
        new_content="Employees must enroll in benefits within 60 days of start date.",
    )
    content = target.read_text(encoding="utf-8")
    assert "60 days" in content
    assert "30 days" not in content


def test_no_write_without_approval(tmp_path):
    """File must not change if SafeApply is constructed but apply is not called."""
    from pipeline.safe_apply import SafeApply  # ImportError until Wave 4

    src = FIXTURES_DIR / "handbook.md"
    target = tmp_path / "handbook.md"
    shutil.copy(src, target)
    original_mtime = target.stat().st_mtime
    # Just constructing SafeApply must not touch the file
    SafeApply()
    assert target.stat().st_mtime == original_mtime


def test_audit_log_appended(tmp_path):
    """Audit JSON is appended after write-back."""
    import json
    from pipeline.safe_apply import SafeApply  # ImportError until Wave 4

    src = FIXTURES_DIR / "handbook.md"
    target = tmp_path / "handbook.md"
    shutil.copy(src, target)
    audit_dir = tmp_path / "audit"
    audit_dir.mkdir()
    applier = SafeApply(audit_dir=str(audit_dir))
    applier.apply_markdown(
        file_path=str(target),
        section_heading="Performance Review",
        new_content="Reviews are conducted bi-annually in June and December.",
        session_id="test-session-001",
    )
    audit_file = audit_dir / "test-session-001.json"
    assert audit_file.exists()
    entries = json.loads(audit_file.read_text(encoding="utf-8"))
    assert isinstance(entries, list)
    assert len(entries) == 1
    assert entries[0]["section_heading"] == "Performance Review"
