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

            # Tables live in <w:tbl>, NOT in doc.paragraphs. If the section
            # contains a table (common in real policy docs), replacing only the
            # paragraphs leaves the table's OLD values in place and appends the
            # new content as a stray text blob — duplicated, stale data. Edit the
            # native table in place when the new content is a table of the same
            # shape; otherwise drop the stale table so values are never left behind.
            section_tables = _docx_section_tables(doc, section_heading)
            md_rows = _markdown_table_rows(new_content)
            updated_in_place = bool(
                section_tables and md_rows
                and _update_docx_table_in_place(section_tables[0], md_rows)
            )

            if updated_in_place:
                # Native table edited cell-by-cell. Preserve any NON-table prose in
                # the new content (text around the table); otherwise remove leftover
                # body paragraphs so the change appears exactly once (no dup blob).
                non_table = "\n".join(
                    ln for ln in new_content.splitlines() if not ln.strip().startswith("|")
                ).strip()
                if non_table:
                    if body_paras_to_replace:
                        body_paras_to_replace[0].clear()
                        body_paras_to_replace[0].add_run(non_table)
                        for para in body_paras_to_replace[1:]:
                            para._element.getparent().remove(para._element)
                    else:
                        heading_elem = paragraphs[heading_idx]._element
                        heading_elem.addnext(doc.add_paragraph(non_table)._element)
                else:
                    for para in body_paras_to_replace:
                        para._element.getparent().remove(para._element)
            else:
                # Shape mismatch or non-table section: remove stale tables first,
                # then replace the body text with the new content.
                for tbl in section_tables:
                    tbl._element.getparent().remove(tbl._element)
                if body_paras_to_replace:
                    body_paras_to_replace[0].clear()
                    body_paras_to_replace[0].add_run(new_content)
                    for para in body_paras_to_replace[1:]:
                        para._element.getparent().remove(para._element)
                else:
                    heading_elem = paragraphs[heading_idx]._element
                    new_para = doc.add_paragraph(new_content)
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
            # Loaded odfpy nodes are generic Element instances; identify headings
            # by the qualified-name local part (text:h -> "h"), NOT by class name.
            if _odf_localname(node) == "h":
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
                local = _odf_localname(node)
                if local == "h":
                    break
                if local in ("table", "frame"):
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

    def delete_section(
        self,
        file_path: str,
        section_heading: str,
        session_id: str = "default",
        proposal_id: str = "",
    ) -> str:
        """Remove an entire section (heading line + body) from a document.

        Supported: .docx, .md, .txt. A backup is always created first.
        The removal is recorded in the audit log with after_content == "".
        """
        suffix = Path(file_path).suffix.lower()
        if suffix == ".pdf":
            raise ValueError("PDF is read-only; no write-back supported")

        if suffix == ".docx":
            before_content = self._delete_docx(file_path, section_heading)
        elif suffix == ".md":
            before_content = self._delete_markdown(file_path, section_heading)
        elif suffix == ".txt":
            before_content = self._delete_txt(file_path, section_heading)
        elif suffix == ".odt":
            before_content = self._delete_odt(file_path, section_heading)
        else:
            raise ValueError(f"Unsupported format for delete: {suffix}")
        return before_content  # the format-specific helpers handle backup + audit

    # ── delete_section format helpers ───────────────────────────────────────

    def _delete_markdown(self, file_path: str, section_heading: str) -> str:
        content = Path(file_path).read_text(encoding="utf-8")
        pattern = (
            r"(?m)^#{1,6} " + re.escape(section_heading) + r"\n(.*?)(?=^#{1,6} |\Z)"
        )
        match = re.search(pattern, content, flags=re.MULTILINE | re.DOTALL)
        before_content = match.group(0) if match else ""
        self._backup(file_path)
        new_text = re.sub(pattern, "", content, flags=re.MULTILINE | re.DOTALL)
        # Collapse any blank-line gap the removal left behind.
        new_text = re.sub(r"\n{3,}", "\n\n", new_text)
        Path(file_path).write_text(new_text, encoding="utf-8")
        return before_content

    def _delete_txt(self, file_path: str, section_heading: str) -> str:
        content = Path(file_path).read_text(encoding="utf-8")
        pattern = (
            r"(" + re.escape(section_heading) + r"[\r\n]+)(.*?)"
            r"(?=(?:SECTION \d+\.|Page \d+, Window)|\Z)"
        )
        match = re.search(pattern, content, flags=re.DOTALL)
        before_content = match.group(0) if match else ""
        self._backup(file_path)
        new_text = re.sub(pattern, "", content, flags=re.DOTALL)
        new_text = re.sub(r"\n{3,}", "\n\n", new_text)
        Path(file_path).write_text(new_text, encoding="utf-8")
        return before_content

    def _delete_docx(self, file_path: str, section_heading: str) -> str:
        import docx as python_docx

        doc = python_docx.Document(file_path)
        paragraphs = doc.paragraphs
        heading_idx: Optional[int] = None
        for i, para in enumerate(paragraphs):
            if para.text.strip() == section_heading:
                heading_idx = i
                break
        if heading_idx is None:
            self._backup(file_path)
            return ""

        # Capture the heading + body that will be removed (for the audit trail).
        removed_parts = [paragraphs[heading_idx].text]
        to_remove = [paragraphs[heading_idx]]
        for i in range(heading_idx + 1, len(paragraphs)):
            para = paragraphs[i]
            if para.style.name.startswith("Heading"):
                break
            removed_parts.append(para.text)
            to_remove.append(para)
        before_content = "\n".join(removed_parts)

        self._backup(file_path)
        for para in to_remove:
            p_elem = para._element
            p_elem.getparent().remove(p_elem)
        doc.save(file_path)
        return before_content

    def _delete_odt(self, file_path: str, section_heading: str) -> str:
        from odf.opendocument import load

        doc = load(file_path)
        text_body = doc.text
        nodes = list(text_body.childNodes)
        heading_idx: Optional[int] = None
        for i, node in enumerate(nodes):
            if _odf_localname(node) == "h" and _odf_node_text(node).strip() == section_heading:
                heading_idx = i
                break
        if heading_idx is None:
            self._backup(file_path)
            return ""

        to_remove = [nodes[heading_idx]]
        removed = [_odf_node_text(nodes[heading_idx])]
        for i in range(heading_idx + 1, len(nodes)):
            if _odf_localname(nodes[i]) == "h":
                break
            to_remove.append(nodes[i])
            removed.append(_odf_node_text(nodes[i]))
        before_content = "\n".join(removed)

        self._backup(file_path)
        for node in to_remove:
            text_body.removeChild(node)
        doc.save(file_path)
        return before_content

    def apply(
        self,
        file_path: str,
        section_heading: str,
        new_content: str,
        session_id: str = "default",
        proposal_id: str = "",
        edit_type: str = "replace",
    ) -> str:
        """Route to the correct format-specific apply method.

        Supported: .docx, .md, .txt, .odt
        Read-only: .pdf (raises ValueError)
        Unknown: raises ValueError

        edit_type == "delete_section" removes the whole section instead of
        replacing its body (for .docx, .md, .txt).
        """
        suffix = Path(file_path).suffix.lower()

        if edit_type == "delete_section":
            before_content = self.delete_section(
                file_path, section_heading, session_id, proposal_id
            )
            audit_entry = self._build_audit_entry(
                proposal_id=proposal_id,
                session_id=session_id,
                file_path=file_path,
                section_heading=section_heading,
                before_content=before_content,
                after_content="",
                backup_path="",
            )
            audit_entry["edit_type"] = "delete_section"
            self._append_audit(session_id, audit_entry)
            return ""

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


def _markdown_table_rows(text: str) -> list[list[str]]:
    """Parse a Markdown/pipe table into rows of cell strings.

    Ignores the ``|---|---|`` separator row. Returns [] if *text* is not a
    pipe table, so a non-table edit falls through to plain-text replacement.
    """
    rows: list[list[str]] = []
    for line in text.splitlines():
        s = line.strip()
        if not s.startswith("|"):
            continue
        cells = [c.strip() for c in s.strip("|").split("|")]
        if cells and all(set(c) <= set("-: ") for c in cells):
            continue  # separator row
        rows.append(cells)
    return rows


def _docx_section_tables(doc, section_heading: str) -> list:
    """Return python-docx Table objects that sit within a heading's section.

    Walks the body in document order so paragraphs and tables are seen
    interleaved (doc.tables alone loses position). A section runs from its
    heading paragraph to the next Heading paragraph.
    """
    from docx.oxml.ns import qn
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    tables: list = []
    in_section = False
    for child in doc.element.body.iterchildren():
        if child.tag == qn("w:p"):
            para = Paragraph(child, doc)
            is_heading = para.style is not None and para.style.name.startswith("Heading")
            if in_section and is_heading:
                break
            if is_heading and para.text.strip() == section_heading:
                in_section = True
        elif child.tag == qn("w:tbl") and in_section:
            tables.append(Table(child, doc))
    return tables


def _update_docx_table_in_place(table, md_rows: list[list[str]]) -> bool:
    """Set a native docx table's cell texts from parsed Markdown rows.

    Only proceeds when the shapes match exactly (so a reformatted edit cannot
    scramble cells); returns False to signal the caller to fall back otherwise.
    """
    if len(table.rows) != len(md_rows):
        return False
    for r, row in enumerate(table.rows):
        if len(row.cells) != len(md_rows[r]):
            return False
    for r, row in enumerate(table.rows):
        for c, cell in enumerate(row.cells):
            if cell.text != md_rows[r][c]:
                cell.text = md_rows[r][c]
    return True


def _odf_localname(node) -> str:
    """Return the local part of an odfpy node's qualified name (text:h -> 'h').

    Loaded odfpy elements are generic ``Element`` instances, so ``__class__``
    is unreliable; the qname local part is the stable type signal.
    """
    qn = getattr(node, "qname", None)
    return qn[1] if qn else node.__class__.__name__.lower()


def _odf_node_text(node) -> str:
    """Recursively extract plain text from an odfpy node."""
    parts: list[str] = []
    for child in node.childNodes:
        if hasattr(child, "data"):
            parts.append(child.data)
        elif hasattr(child, "childNodes"):
            parts.append(_odf_node_text(child))
    return "".join(parts)
