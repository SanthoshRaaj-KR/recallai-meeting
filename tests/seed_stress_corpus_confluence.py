"""Upload the first 20 docs from local_doc_change/stress_corpus/docs/ to Confluence.

Supports .md, .docx, and .odt files.  Content is uploaded as-is (no edits).

Run:
    conda run -n meetagents python tests/seed_stress_corpus_confluence.py
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import requests
from requests.auth import HTTPBasicAuth

# ── env ────────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
for line in (ROOT / ".env").read_text().splitlines():
    line = line.strip()
    if not line or line.startswith("#") or "=" not in line:
        continue
    k, _, v = line.partition("=")
    os.environ.setdefault(k.strip(), v.strip())

EMAIL  = os.environ["ATLASSIAN_USER_EMAIL"].strip()
TOKEN  = os.environ["ATLASSIAN_API_TOKEN"].strip()
DOMAIN = os.environ["ATLASSIAN_DOMAIN"].strip()
SPACE  = os.environ["ATLASSIAN_SPACE_KEY"].strip()

BASE_URL = f"https://{DOMAIN}/wiki/rest/api"
AUTH     = HTTPBasicAuth(EMAIL, TOKEN)
HEADERS  = {"Content-Type": "application/json", "Accept": "application/json"}

DOCS_DIR = ROOT / "local_doc_change" / "stress_corpus" / "docs"
FILES    = sorted(DOCS_DIR.iterdir())[:20]


# ── converters ─────────────────────────────────────────────────────────────────

def _md_to_html(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    # Convert headings
    def _heading(m: re.Match) -> str:
        level = len(m.group(1))
        return f"<h{level}>{m.group(2).strip()}</h{level}>"
    text = re.sub(r"^(#{1,6})\s+(.*)", _heading, text, flags=re.MULTILINE)
    # Bold / italic
    text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"\*(.+?)\*",     r"<em>\1</em>", text)
    # Inline code
    text = re.sub(r"`(.+?)`", r"<code>\1</code>", text)
    # Horizontal rules
    text = re.sub(r"^---+$", "<hr/>", text, flags=re.MULTILINE)
    # Tables — wrap whole table block
    lines = text.split("\n")
    out: list[str] = []
    in_table = False
    for line in lines:
        if re.match(r"\|.*\|", line):
            if not in_table:
                out.append("<table><tbody>")
                in_table = True
            if re.match(r"\|[\s\-:|]+\|", line):
                continue  # separator row
            cells = [c.strip() for c in line.strip("|").split("|")]
            out.append("<tr>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>")
        else:
            if in_table:
                out.append("</tbody></table>")
                in_table = False
            out.append(line)
    if in_table:
        out.append("</tbody></table>")
    text = "\n".join(out)
    # Paragraphs: blank-line-separated blocks not already in a tag
    blocks = re.split(r"\n{2,}", text)
    html_blocks: list[str] = []
    for block in blocks:
        block = block.strip()
        if not block:
            continue
        if block.startswith("<"):
            html_blocks.append(block)
        else:
            html_blocks.append(f"<p>{block}</p>")
    return "\n".join(html_blocks)


def _docx_to_html(path: Path) -> str:
    from docx import Document
    from docx.oxml.ns import qn

    doc = Document(str(path))
    parts: list[str] = []

    for block in doc.element.body:
        tag = block.tag.split("}")[-1] if "}" in block.tag else block.tag

        if tag == "p":
            from docx.text.paragraph import Paragraph
            para = Paragraph(block, doc)
            style = para.style.name if para.style else "Normal"
            text = para.text.strip()
            if not text:
                continue
            if style.startswith("Heading"):
                try:
                    level = int(style.split()[-1])
                except ValueError:
                    level = 2
                level = max(1, min(6, level))
                parts.append(f"<h{level}>{text}</h{level}>")
            else:
                parts.append(f"<p>{text}</p>")

        elif tag == "tbl":
            from docx.table import Table
            tbl = Table(block, doc)
            rows_html = ["<table><tbody>"]
            for row in tbl.rows:
                cells = "".join(f"<td>{c.text.strip()}</td>" for c in row.cells)
                rows_html.append(f"<tr>{cells}</tr>")
            rows_html.append("</tbody></table>")
            parts.append("\n".join(rows_html))

    return "\n".join(parts) or "<p>(empty document)</p>"


def _odt_to_html(path: Path) -> str:
    """Convert ODT to HTML via LibreOffice headless, fall back to text extraction."""
    try:
        with tempfile.TemporaryDirectory() as tmp:
            subprocess.run(
                ["soffice", "--headless", "--convert-to", "html", "--outdir", tmp, str(path)],
                capture_output=True, timeout=30,
            )
            html_files = list(Path(tmp).glob("*.html"))
            if html_files:
                raw = html_files[0].read_text(encoding="utf-8", errors="replace")
                # Extract body content only
                m = re.search(r"<body[^>]*>(.*?)</body>", raw, re.DOTALL | re.IGNORECASE)
                if m:
                    return m.group(1).strip() or "<p>(empty)</p>"
    except Exception:
        pass

    # Fallback: odfpy text extraction
    try:
        from odf.opendocument import load as odf_load
        from odf import text as odf_text
        from odf.element import Element

        def _iter_text(node: Element) -> str:
            parts: list[str] = []
            if hasattr(node, "data"):
                parts.append(str(node.data))
            for child in node.childNodes:
                parts.append(_iter_text(child))
            return "".join(parts)

        doc = odf_load(str(path))
        paragraphs: list[str] = []
        for elem in doc.text.childNodes:
            t = _iter_text(elem).strip()
            if t:
                paragraphs.append(f"<p>{t}</p>")
        return "\n".join(paragraphs) or "<p>(empty document)</p>"
    except Exception as exc:
        return f"<p>(Could not convert ODT: {exc})</p>"


def file_to_html(path: Path) -> tuple[str, str]:
    """Return (title, html_body). Title derived from filename."""
    stem = path.stem.replace("_", " ").replace("-", " ").title()
    suffix = path.suffix.lower()
    if suffix == ".md":
        return stem, _md_to_html(path)
    elif suffix == ".docx":
        return stem, _docx_to_html(path)
    elif suffix == ".odt":
        return stem, _odt_to_html(path)
    else:
        content = path.read_text(encoding="utf-8", errors="replace")
        return stem, f"<pre>{content}</pre>"


# ── Confluence create ──────────────────────────────────────────────────────────

def create_page(title: str, html: str) -> dict:
    payload = {
        "type": "page",
        "title": title,
        "space": {"key": SPACE},
        "body": {"storage": {"value": html, "representation": "storage"}},
    }
    resp = requests.post(f"{BASE_URL}/content", auth=AUTH, json=payload, headers=HEADERS, timeout=30)
    if not resp.ok:
        raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:200]}")
    data = resp.json()
    return {
        "id": data["id"],
        "url": f"https://{DOMAIN}/wiki" + data["_links"]["webui"],
    }


# ── main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    print(f"Uploading {len(FILES)} stress-corpus docs to Confluence space '{SPACE}'\n")
    created, failed = [], []

    for i, path in enumerate(FILES, 1):
        print(f"[{i:2d}/{len(FILES)}] {path.name} ... ", end="", flush=True)
        try:
            title, html = file_to_html(path)
            result = create_page(title, html)
            print(f"OK  id={result['id']}")
            created.append((path.name, result))
        except Exception as exc:
            print(f"FAILED — {exc}")
            failed.append((path.name, str(exc)))
        if i < len(FILES):
            time.sleep(0.4)

    print(f"\n{'─'*60}")
    print(f"Created {len(created)}/{len(FILES)}  Failed {len(failed)}/{len(FILES)}")
    if failed:
        print("\nFailed files:")
        for name, err in failed:
            print(f"  ✗ {name}: {err}")
    if created:
        print("\nCreated pages:")
        for name, r in created:
            print(f"  ✓ {name}  →  {r['url']}")


if __name__ == "__main__":
    main()
