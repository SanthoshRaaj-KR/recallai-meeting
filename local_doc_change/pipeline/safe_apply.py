"""SafeApply — atomic file write-back with backup and audit trail.

Supports DOCX, Markdown (.md), plain text (.txt), and ODT files.
PDF files are read-only and raise ValueError on write attempts.

Safety guarantees:
- Backup is always created BEFORE any write (backup failure aborts the write).
- No file I/O occurs on SafeApply construction (beyond creating the audit dir).
- Each accepted write appends a JSON entry to audit/{session_id}.json.

T-12-10 mitigation: _backup() runs before every write; failure raises exception.
T-12-13 mitigation: session_id sanitised (only [a-zA-Z0-9_-] allowed) before
  use as an audit log filename.
"""

from __future__ import annotations

import datetime
import json
import logging
import re
import shutil
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def _sanitise_session_id(session_id: str) -> str:
    """Remove path-traversal / shell-injection characters from session_id.

    T-12-13: Only alphanumerics, underscore, and hyphen are allowed.
    Any other character is replaced with '_'.
    """
    return re.sub(r"[^a-zA-Z0-9_-]", "_", session_id)


class SafeApply:
    """Atomic write-back layer for local document sections.

    Supported formats: .docx, .md, .txt, .odt
    Read-only: .pdf (raises ValueError on any apply attempt)
    """

    def __init__(self, audit_dir: str = "local_doc_change/audit") -> None:
        self.audit_dir = Path(audit_dir)
        self.audit_dir.mkdir(parents=True, exist_ok=True)

    # ── Private helpers ─────────────────────────────────────────────────────

    def _backup(self, file_path: str) -> str:
        """Create a timestamped backup of file_path before writing.

        Returns the backup file path as a string.
        Raises if the copy fails — this prevents any write from proceeding.
        """
        p = Path(file_path)
        ts = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%S")
        backup_path = p.parent / f"{p.stem}.backup.{ts}{p.suffix}"
        shutil.copy2(str(p), str(backup_path))
        return str(backup_path)

    def _append_audit(self, session_id: str, entry: dict) -> None:
        """Append a JSON entry to audit/{session_id}.json.

        T-12-13: session_id is sanitised before constructing the filename.
        The file contains a JSON array; entries are appended in-place.
        """
        safe_sid = _sanitise_session_id(session_id)
        audit_file = self.audit_dir / f"{safe_sid}.json"
        if audit_file.exists():
            try:
                entries: list = json.loads(audit_file.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                entries = []
        else:
            entries = []
        entries.append(entry)
        audit_file.write_text(
            json.dumps(entries, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    def _build_audit_entry(
        self,
        proposal_id: str,
        session_id: str,
        file_path: str,
        section_heading: str,
        before_content: str,
        after_content: str,
        backup_path: str,
        intent_type: str = "",
        intent_topic: str = "",
    ) -> dict:
        return {
            "proposal_id": proposal_id,
            "session_id": session_id,
            "file_path": file_path,
            "section_heading": section_heading,
            "before_content": before_content,
            "after_content": after_content,
            "backup_path": backup_path,
            "applied_at": datetime.datetime.utcnow().isoformat() + "Z",
            "intent_type": intent_type,
            "intent_topic": intent_topic,
        }

    # ── Format-specific apply methods ───────────────────────────────────────

    def apply_docx(
        self,
        file_path: str,
        section_heading: str,
        new_content: str,
        session_id: str = "default",
        proposal_id: str = "",
    ) -> str:
        """Apply a section replacement to a DOCX file.

        Finds the paragraph whose text matches section_heading, then clears all
        subsequent non-heading paragraphs and replaces them with new_content.
        A backup is created before any write. Raises ValueError for PDF inputs.
        """
        if file_path.lower().endswith(".pdf"):
            raise ValueError("PDF is read-only; no write-back available")

        import docx as python_docx  # python-docx import alias

        # Capture before_content (all body text after the target heading)
        doc_read = python_docx.Document(file_path)
        before_content = ""
        found = False
        before_parts: list[str] = []
        for para in doc_read.paragraphs:
            if found:
                # Stop at the next heading
                if para.style.name.startswith("Heading"):
                    break
                before_parts.append(para.text)
            elif para.text.strip() == section_heading:
                found = True
        before_content = "\n".join(before_parts)

        # Create backup before any modification
        backup_path = self._backup(file_path)

        # Open document for modification
        doc = python_docx.Document(file_path)
        paragraphs = doc.paragraphs

        # Find the heading paragraph index
        heading_idx: Optional[int] = None
        for i, para in enumerate(paragraphs):
            if para.text.strip() == section_heading:
                heading_idx = i
                break

        if heading_idx is None:
            logger.warning(
                "apply_docx: heading %r not found in %s — appending new section",
                section_heading,
                file_path,
            )
            doc.add_heading(section_heading, level=1)
            doc.add_paragraph(new_content)
            doc.save(file_path)
        else:
            # Collect the body paragraph elements after the heading
            # (stopping before the next Heading paragraph or end of doc)
            body_paras_to_replace: list = []
            for i in range(heading_idx + 1, len(paragraphs)):
                para = paragraphs[i]
                if para.style.name.startswith("Heading"):
                    break
                body_paras_to_replace.append(para)

            if body_paras_to_replace:
                # Clear first body paragraph and set its text to new_content
                body_paras_to_replace[0].clear()
                body_paras_to_replace[0].add_run(new_content)
                # Remove remaining body paragraphs
                for para in body_paras_to_replace[1:]:
                    p_elem = para._element
                    p_elem.getparent().remove(p_elem)
            else:
                # No body paragraphs — insert a new one after the heading
                heading_elem = paragraphs[heading_idx]._element
                new_para = doc.add_paragraph(new_content)
                heading_elem.addnext(new_para._element)
                # Remove the paragraph that add_paragraph() appended at end
                new_para._element.getparent().remove(new_para._element)
                # Re-insert after heading
                heading_elem.addnext(new_para._element)

            doc.save(file_path)

        audit_entry = self._build_audit_entry(
            proposal_id=proposal_id,
            session_id=session_id,
            file_path=file_path,
            section_heading=section_heading,
            before_content=before_content,
            after_content=new_content,
            backup_path=backup_path,
        )
        self._append_audit(session_id, audit_entry)
        return backup_path

    def apply_markdown(
        self,
        file_path: str,
        section_heading: str,
        new_content: str,
        session_id: str = "default",
        proposal_id: str = "",
    ) -> str:
        """Apply a section replacement to a Markdown file.

        Finds the section by heading and replaces all content until the next
        heading (or end of file). A backup is created before any write.
        """
        content = Path(file_path).read_text(encoding="utf-8")

        # Capture before_content
        pattern_read = (
            r"(?m)^(#{1,6} " + re.escape(section_heading) + r"\n)(.*?)(?=^#{1,6} |\Z)"
        )
        match = re.search(pattern_read, content, flags=re.MULTILINE | re.DOTALL)
        before_content = match.group(2) if match else ""

        # Create backup before any modification
        backup_path = self._backup(file_path)

        # Replace section content
        def _replacement(m: re.Match) -> str:
            return m.group(1) + new_content + "\n\n"

        new_text = re.sub(
            r"(?m)(^#{1,6} " + re.escape(section_heading) + r"\n)(.*?)(?=^#{1,6} |\Z)",
            _replacement,
            content,
            flags=re.MULTILINE | re.DOTALL,
        )

        Path(file_path).write_text(new_text, encoding="utf-8")

        audit_entry = self._build_audit_entry(
            proposal_id=proposal_id,
            session_id=session_id,
            file_path=file_path,
            section_heading=section_heading,
            before_content=before_content,
            after_content=new_content,
            backup_path=backup_path,
        )
        self._append_audit(session_id, audit_entry)
        return backup_path

    def apply_txt(
        self,
        file_path: str,
        section_heading: str,
        new_content: str,
        session_id: str = "default",
        proposal_id: str = "",
    ) -> str:
        """Apply a section replacement to a plain text file.

        Uses synthetic heading patterns: "SECTION N." or "Page N, Window M".
        Falls back to appending a new section if the heading is not found.
        """
        content = Path(file_path).read_text(encoding="utf-8")

        # Pattern for synthetic headings (process.txt style)
        heading_pattern = (
            r"((?:SECTION \d+\..*?|Page \d+, Window \d+)[\r\n]+)"
            r"(.*?)(?=(?:SECTION \d+\.|Page \d+, Window)|\Z)"
        )
        match = re.search(
            r"(" + re.escape(section_heading) + r"[\r\n]+)(.*?)(?=(?:SECTION \d+\.|Page \d+, Window)|\Z)",
            content,
            flags=re.DOTALL,
        )
        before_content = match.group(2) if match else ""

        # Create backup
        backup_path = self._backup(file_path)

        if match:
            def _replacement_txt(m: re.Match) -> str:
                return m.group(1) + new_content + "\n\n"

            new_text = re.sub(
                r"(" + re.escape(section_heading) + r"[\r\n]+)(.*?)(?=(?:SECTION \d+\.|Page \d+, Window)|\Z)",
                _replacement_txt,
                content,
                flags=re.DOTALL,
            )
        else:
            # Fallback: append new section at end
            new_text = content.rstrip() + f"\n\n{section_heading}\n{new_content}"

        Path(file_path).write_text(new_text, encoding="utf-8")

        audit_entry = self._build_audit_entry(
            proposal_id=proposal_id,
            session_id=session_id,
            file_path=file_path,
            section_heading=section_heading,
            before_content=before_content,
            after_content=new_content,
            backup_path=backup_path,
        )
        self._append_audit(session_id, audit_entry)
        return backup_path

    def apply_odt(
        self,
        file_path: str,
        section_heading: str,
        new_content: str,
        session_id: str = "default",
        proposal_id: str = "",
    ) -> str:
        """Apply a section replacement to an ODT file using odfpy.

        Iterates child nodes of the text body to find and replace the section.
        Logs a warning for sections containing embedded tables or images.
        """
        from odf.opendocument import load
        from odf.text import H as OdfHeading
        from odf.text import P as OdfParagraph

        doc = load(file_path)
        before_content = ""
        backup_path = self._backup(file_path)

        # Walk child nodes to find heading and replace body paragraphs
        text_body = doc.text
        nodes = list(text_body.childNodes)
        heading_idx: Optional[int] = None

        for i, node in enumerate(nodes):
            node_text = node.getAttribute("text:outline-level") if hasattr(node, "getAttribute") else None
            # Get text content of node
            try:
                text_val = str(node) if node.nodeType == node.TEXT_NODE else ""
            except Exception:
                text_val = ""

            # Check if this is a heading with matching text
            if node.__class__.__name__ == "H":
                # odfpy heading element
                heading_text = "".join(
                    str(child) for child in node.childNodes
                    if hasattr(child, "data") or child.nodeType == child.TEXT_NODE
                )
                # Flatten to plain string
                heading_text_plain = _odf_node_text(node)
                if heading_text_plain.strip() == section_heading:
                    heading_idx = i
                    break

        if heading_idx is not None:
            # Collect body paragraphs between this heading and the next
            body_indices: list[int] = []
            complex_warning = False
            for i in range(heading_idx + 1, len(nodes)):
                node = nodes[i]
                cls_name = node.__class__.__name__
                if cls_name == "H":
                    break
                if cls_name in ("Table", "Frame"):
                    complex_warning = True
                body_indices.append(i)

            if complex_warning:
                logger.warning(
                    "COMPLEX_SECTION_WARNING: ODT section %r in %s contains "
                    "embedded tables or frames — text replacement only",
                    section_heading,
                    file_path,
                )

            # Capture before_content
            before_parts = [_odf_node_text(nodes[i]) for i in body_indices]
            before_content = "\n".join(before_parts)

            # Remove existing body paragraphs
            for i in reversed(body_indices):
                text_body.removeChild(nodes[i])

            # Insert new paragraph after heading
            new_para = OdfParagraph()
            from odf.element import Text as OdfText
            new_para.addText(new_content)

            # Re-query nodes after removal
            nodes_after = list(text_body.childNodes)
            new_heading_idx = next(
                (j for j, n in enumerate(nodes_after) if _odf_node_text(n).strip() == section_heading),
                None,
            )
            if new_heading_idx is not None and new_heading_idx + 1 <= len(nodes_after):
                if new_heading_idx + 1 < len(nodes_after):
                    text_body.insertBefore(new_para, nodes_after[new_heading_idx + 1])
                else:
                    text_body.appendChild(new_para)
            else:
                text_body.appendChild(new_para)
        else:
            # Heading not found — append
            logger.warning(
                "apply_odt: heading %r not found in %s — appending",
                section_heading,
                file_path,
            )
            new_para = OdfParagraph()
            new_para.addText(new_content)
            text_body.appendChild(new_para)

        doc.save(file_path)

        audit_entry = self._build_audit_entry(
            proposal_id=proposal_id,
            session_id=session_id,
            file_path=file_path,
            section_heading=section_heading,
            before_content=before_content,
            after_content=new_content,
            backup_path=backup_path,
        )
        self._append_audit(session_id, audit_entry)
        return backup_path

    def apply(
        self,
        file_path: str,
        section_heading: str,
        new_content: str,
        session_id: str = "default",
        proposal_id: str = "",
    ) -> str:
        """Route to the correct format-specific apply method.

        Supported: .docx, .md, .txt, .odt
        Read-only: .pdf (raises ValueError)
        Unknown: raises ValueError
        """
        suffix = Path(file_path).suffix.lower()
        if suffix == ".docx":
            return self.apply_docx(file_path, section_heading, new_content, session_id, proposal_id)
        elif suffix == ".md":
            return self.apply_markdown(file_path, section_heading, new_content, session_id, proposal_id)
        elif suffix == ".txt":
            return self.apply_txt(file_path, section_heading, new_content, session_id, proposal_id)
        elif suffix == ".odt":
            return self.apply_odt(file_path, section_heading, new_content, session_id, proposal_id)
        elif suffix == ".pdf":
            raise ValueError("PDF is read-only; no write-back supported")
        else:
            raise ValueError(f"Unsupported format: {suffix}")


def _odf_node_text(node) -> str:
    """Recursively extract plain text from an odfpy node."""
    parts: list[str] = []
    for child in node.childNodes:
        if hasattr(child, "data"):
            parts.append(child.data)
        elif hasattr(child, "childNodes"):
            parts.append(_odf_node_text(child))
    return "".join(parts)
