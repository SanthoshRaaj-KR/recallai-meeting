"""GREEN test suite for PageParser (PROP-V2-02).

Phase 10 — auto-propose-pipeline-quality-redesign-v2.

Wave 0 landed these as RED scaffolds with ``pytest.fail("Wave 0 RED")`` bodies.
Wave 1 (this commit) replaces the bodies with concrete assertions against the
``confluence_logic.agents.page_parser.PageParser`` Pydantic AST surface. Until
that module ships, pytest collection itself raises ``ModuleNotFoundError`` —
i.e. these assertions are the new RED gate Wave 1 implementation must turn
GREEN.
"""
import pytest

# The failing import below is intentional — it is the RED gate that
# confirms Wave 1 has not yet shipped the PageParser module. Once
# Wave 1 lands `confluence_logic/agents/page_parser.py` exporting
# PageParser + the AST node taxonomy, this import becomes a no-op and
# the test bodies become the real RED gate.
from confluence_logic.agents.page_parser import (  # noqa: F401
    PageParser,
    ASTRoot,
    OrderedList,
    ListItem,
    Heading,
    Paragraph,
    Macro,
)


def test_ordered_list_indices_match_source_order():
    """PROP-V2-02: PageParser produces stable ``<li>.index`` values matching Confluence source order."""
    html = "<h1>Steps</h1><ol><li>One</li><li>Two</li><li>Three</li></ol>"
    root = PageParser().parse(html)

    # First (and only) section is bounded by <h1>Steps</h1>.
    assert len(root.sections) == 1, f"expected exactly one section, got {len(root.sections)}"
    section = root.sections[0]
    assert section.heading is not None
    assert section.heading.text == "Steps"

    # The section's single block is the ordered list.
    assert len(section.blocks) == 1, f"expected one block in section, got {len(section.blocks)}"
    ol = section.blocks[0]
    assert isinstance(ol, OrderedList), f"expected OrderedList, got {type(ol).__name__}"
    assert ol.kind == "ordered_list"

    # Items carry 0-based indices matching the source <li> order.
    assert len(ol.items) == 3
    assert ol.items[0].index == 0
    assert ol.items[1].index == 1
    assert ol.items[2].index == 2

    # Text content also lines up with the source order.
    assert "One" in "".join(run.text for run in ol.items[0].runs)
    assert "Two" in "".join(run.text for run in ol.items[1].runs)
    assert "Three" in "".join(run.text for run in ol.items[2].runs)


def test_macro_preserved_as_opaque_body_html():
    """PROP-V2-02: ``ac:structured-macro`` round-trips byte-identical as a Macro node with opaque body_html."""
    html = (
        '<ac:structured-macro ac:name="info">'
        "<ac:rich-text-body>warning</ac:rich-text-body>"
        "</ac:structured-macro>"
    )
    root = PageParser().parse(html)

    # No headings → everything sits in a leading Section(heading=None).
    assert len(root.sections) >= 1
    pre_heading_section = root.sections[0]
    assert pre_heading_section.heading is None
    assert pre_heading_section.section_index == 0

    # The structured macro becomes an opaque Macro node — no descent into the body.
    macro_blocks = [b for b in pre_heading_section.blocks if isinstance(b, Macro)]
    assert len(macro_blocks) == 1, "expected exactly one Macro node"
    macro = macro_blocks[0]
    assert macro.kind == "macro"
    assert macro.name == "info"
    # body_html must preserve the inner XML verbatim (drafter never edits inside macros).
    assert "<ac:rich-text-body>warning</ac:rich-text-body>" in macro.body_html, (
        f"macro.body_html lost the inner XML, got: {macro.body_html!r}"
    )

    # Round-trip: ASTRoot.raw_html stores the input HTML verbatim for editor_dispatcher.
    assert root.raw_html == html


def test_round_trip_text_runs_preserves_bold_italic():
    """PROP-V2-02: text runs with ``<strong>``/``<em>`` survive parse → render round-trip."""
    html = "<p>This is <strong>bold</strong> and <em>italic</em></p>"
    root = PageParser().parse(html)

    assert len(root.sections) >= 1
    leading = root.sections[0]
    assert leading.heading is None

    # The single paragraph carries TextRun objects with bold/italic flags.
    paragraphs = [b for b in leading.blocks if isinstance(b, Paragraph)]
    assert len(paragraphs) == 1, f"expected one paragraph, got {len(paragraphs)}"
    para = paragraphs[0]
    assert para.kind == "paragraph"

    bolds = [r for r in para.runs if r.bold]
    italics = [r for r in para.runs if r.italic]
    assert any("bold" in r.text for r in bolds), (
        f"expected a bold TextRun containing 'bold', got runs={para.runs!r}"
    )
    assert any("italic" in r.text for r in italics), (
        f"expected an italic TextRun containing 'italic', got runs={para.runs!r}"
    )

    # The plain prose around the inline emphasis must survive too.
    full_text = "".join(r.text for r in para.runs)
    assert "This is" in full_text
    assert "and" in full_text


def test_section_boundary_split_by_heading_level():
    """PROP-V2-02: PageParser splits the document into sections bounded by Heading nodes by level."""
    html = "<h1>A</h1><p>x</p><h1>B</h1><p>y</p>"
    root = PageParser().parse(html)

    # Two H1 headings → two sections (no pre-heading content present).
    section_headings = [s.heading.text for s in root.sections if s.heading is not None]
    assert section_headings == ["A", "B"], (
        f"expected section headings ['A', 'B'], got {section_headings!r}"
    )

    section_a = next(s for s in root.sections if s.heading and s.heading.text == "A")
    section_b = next(s for s in root.sections if s.heading and s.heading.text == "B")

    # Each section owns exactly its one paragraph — never the other's.
    para_a_text = "".join(
        r.text for b in section_a.blocks if isinstance(b, Paragraph) for r in b.runs
    )
    para_b_text = "".join(
        r.text for b in section_b.blocks if isinstance(b, Paragraph) for r in b.runs
    )
    assert "x" in para_a_text and "y" not in para_a_text, (
        f"section A leaked content: para_a_text={para_a_text!r}"
    )
    assert "y" in para_b_text and "x" not in para_b_text, (
        f"section B leaked content: para_b_text={para_b_text!r}"
    )

    # ast_path uniqueness: every node in the tree gets a distinct path.
    paths = []

    def collect_paths(section):
        if section.heading is not None:
            paths.append(section.heading.ast_path)
        for blk in section.blocks:
            paths.append(blk.ast_path)
            if isinstance(blk, OrderedList):
                paths.extend(item.ast_path for item in blk.items)

    for s in root.sections:
        collect_paths(s)
    assert len(paths) == len(set(paths)), f"ast_path values are not unique: {paths!r}"
