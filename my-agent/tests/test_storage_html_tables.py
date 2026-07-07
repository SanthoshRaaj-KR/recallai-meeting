"""_storage_html must render markdown tables as Confluence storage XHTML.

Table additions (new rows/columns) arrive as markdown (``| a | b |``); the
write-back for append/create paths goes through _storage_html, which previously
wrapped each row in ``<p>…</p>`` — leaking literal pipes into the page. It must
emit ``<table>/<tr>/<td>`` instead.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from review_pipeline.pipeline import _storage_html  # noqa: E402


def test_markdown_table_becomes_storage_table_not_paragraphs():
    md = "| Name | Role |\n| --- | --- |\n| Asha | PM |\n| Ravi | Eng |"
    out = _storage_html(md)
    assert "<table>" in out and "</table>" in out
    assert "<th>Name</th>" in out and "<th>Role</th>" in out
    assert "<tr><td>Asha</td><td>PM</td></tr>" in out
    assert "<tr><td>Ravi</td><td>Eng</td></tr>" in out
    # No literal markdown pipes or separator dashes leaked into storage.
    assert "|" not in out
    assert "---" not in out


def test_added_row_only_renders_as_table_row():
    # append mode often sends just the new row(s).
    out = _storage_html("| Priya | Design |")
    assert "<tr><td>Priya</td><td>Design</td></tr>" in out
    assert "|" not in out


def test_added_column_widens_every_row():
    # A new column shows up as an extra cell in each markdown row.
    md = "| Name | Role | Location |\n| --- | --- | --- |\n| Asha | PM | NYC |"
    out = _storage_html(md)
    assert "<th>Name</th><th>Role</th><th>Location</th>" in out
    assert "<tr><td>Asha</td><td>PM</td><td>NYC</td></tr>" in out
    assert "|" not in out


def test_table_cell_emphasis_becomes_tags():
    out = _storage_html("| Plan | Price |\n| --- | --- |\n| Standard | *$8/device/year* |")
    assert "<td><em>$8/device/year</em></td>" in out
    assert "*" not in out


def test_table_mixed_with_heading_and_paragraph():
    md = "## Team\n\nIntro line.\n\n| Name | Role |\n| --- | --- |\n| Asha | PM |\n\nOutro line."
    out = _storage_html(md)
    assert "<h2>Team</h2>" in out
    assert "<p>Intro line.</p>" in out
    assert "<table>" in out and "<th>Name</th>" in out
    assert "<p>Outro line.</p>" in out
    assert "|" not in out


def test_non_table_content_unchanged():
    # Regression: existing heading/list/paragraph behaviour is preserved.
    md = "## Notes\n- first\n- second\nplain text"
    out = _storage_html(md)
    assert "<h2>Notes</h2>" in out
    assert "<ul>\n<li>first</li>\n<li>second</li>\n</ul>" in out
    assert "<p>plain text</p>" in out


def test_html_in_cell_is_escaped():
    # A cell value containing markup is escaped, never injected.
    out = _storage_html("| Field | Value |\n| --- | --- |\n| x | <script>alert(1)</script> |")
    assert "<script>" not in out
    assert "&lt;script&gt;" in out
