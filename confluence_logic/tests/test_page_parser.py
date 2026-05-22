"""Wave 0 RED test scaffold for PageParser (PROP-V2-02).

Phase 10 — auto-propose-pipeline-quality-redesign-v2.

These tests fail RED until Wave 1 creates
`confluence_logic/agents/page_parser.py`. The failing import on collection
is the contract: Wave 1+ implementations turn these GREEN.
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
    pytest.fail("Wave 0 RED — PageParser implementation pending (Wave 1)")


def test_macro_preserved_as_opaque_body_html():
    """PROP-V2-02: ``ac:structured-macro`` round-trips byte-identical as a Macro node with opaque body_html."""
    pytest.fail("Wave 0 RED — PageParser implementation pending (Wave 1)")


def test_round_trip_text_runs_preserves_bold_italic():
    """PROP-V2-02: text runs with ``<strong>``/``<em>`` survive parse → render round-trip."""
    pytest.fail("Wave 0 RED — PageParser implementation pending (Wave 1)")


def test_section_boundary_split_by_heading_level():
    """PROP-V2-02: PageParser splits the document into sections bounded by Heading nodes by level."""
    pytest.fail("Wave 0 RED — PageParser implementation pending (Wave 1)")
