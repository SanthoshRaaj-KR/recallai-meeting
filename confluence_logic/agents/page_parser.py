"""Confluence storage-format HTML -> typed Pydantic AST for Phase 10 structure-aware drafting (PROP-V2-02).

This module is the parser foundation Phase 10 relies on. The StructureAwareDrafter (Plan 06)
and EditorDispatcher (Plan 04) both operate on the typed ``ASTRoot`` exported here — they
never poke at raw HTML strings. The contract:

  * Sections are bounded by heading tags (``h1``..``h6``). Pre-heading content lives in a
    leading ``Section(heading=None, section_index=0)``.
  * Ordered-list items carry stable 0-based ``index`` values matching the source ``<li>``
    order — this is what makes ``reorder`` ops in the drafter unambiguous.
  * Confluence-specific structures (``ac:structured-macro``, ``<table>``, standalone
    ``<pre>``) are preserved as opaque ``Macro``/``TableNode``/``CodeBlock`` nodes carrying
    their raw inner HTML. The drafter NEVER edits inside them in Phase 10 (Pitfall 1 in
    10-RESEARCH.md).
  * Every node carries an ``ast_path`` so reorder operations can address list items
    unambiguously (e.g. ``section[0].ordered_list[1].item[2]``).
  * ``ASTRoot.raw_html`` round-trips the input verbatim so ``editor_dispatcher`` can do
    whole-section commits (Pattern 4 in 10-RESEARCH.md).

Implementation notes:
  * Uses ``BeautifulSoup(html, "lxml")`` per Pitfall 6 (10-RESEARCH.md lines 953-962).
    ``lxml`` is the only parser BS4 ships that preserves Confluence's XML-namespaced
    ``ac:``/``ri:`` tags as live ``Tag`` objects. If ``lxml`` is unavailable, falls back
    to ``html.parser`` with a warning — reorder ops may then trigger version conflicts.
  * Defensive byte cap (``MAX_PAGE_HTML_BYTES = 2_000_000``) matches the existing
    convention in ``confluence_logic/review/api.py`` and mitigates T-10-02 (DoS).
"""

from __future__ import annotations

import logging
from typing import List, Optional, Union

from bs4 import BeautifulSoup, NavigableString, Tag
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Defensive cap on input HTML — T-10-02 (DoS) mitigation.
MAX_PAGE_HTML_BYTES = 2_000_000

# Heading tag set. Matches the regex used in confluence_logic/utils/html_parser.py
# (``re.compile('^h[1-6]$')``) but as a frozenset for O(1) membership checks during walk.
HEADING_TAGS = frozenset({"h1", "h2", "h3", "h4", "h5", "h6"})


def _is_heading(tag: Tag) -> bool:
    """True iff *tag* is an h1..h6 element."""
    return isinstance(tag, Tag) and tag.name in HEADING_TAGS


def _heading_level(tag: Tag) -> int:
    """Return 1..6 for an h1..h6 tag; raises if called on a non-heading."""
    return int(tag.name[1])


def _is_macro_tag(tag: Tag) -> bool:
    """True iff *tag* is a Confluence storage-format namespaced element (``ac:*`` / ``ri:*``)."""
    if not isinstance(tag, Tag) or not tag.name:
        return False
    return ":" in tag.name or tag.name.startswith("ac:") or tag.name.startswith("ri:")


# ---------------------------------------------------------------------------
# AST node taxonomy (Pydantic)
# ---------------------------------------------------------------------------


class TextRun(BaseModel):
    """A run of inline text with optional formatting.

    Phase 10 keeps formatting flat: a ``<strong><em>X</em></strong>`` collapses to whichever
    flag the innermost matching wrapper sets. Perfect fidelity is deferred (A2 in
    10-RESEARCH.md — drafter never edits inside text runs anyway).
    """

    text: str
    bold: bool = False
    italic: bool = False
    href: Optional[str] = None


class ASTNode(BaseModel):
    """Base class for every block-level AST node. Carries a stable, unique ast_path."""

    kind: str
    ast_path: str


class Heading(ASTNode):
    kind: str = Field(default="heading")
    level: int
    text: str


class Paragraph(ASTNode):
    kind: str = Field(default="paragraph")
    runs: List[TextRun] = Field(default_factory=list)


class ListItem(ASTNode):
    kind: str = Field(default="list_item")
    index: int
    runs: List[TextRun] = Field(default_factory=list)


class OrderedList(ASTNode):
    kind: str = Field(default="ordered_list")
    items: List[ListItem] = Field(default_factory=list)


class UnorderedList(ASTNode):
    kind: str = Field(default="unordered_list")
    items: List[ListItem] = Field(default_factory=list)


class Macro(ASTNode):
    """Confluence ``ac:structured-macro`` (or any other namespaced tag).

    ``body_html`` holds the inner XML *verbatim* — the drafter is forbidden to descend into
    macros in Phase 10 (Pitfall 1 in 10-RESEARCH.md).
    """

    kind: str = Field(default="macro")
    name: str
    body_html: str


class CodeBlock(ASTNode):
    """Standalone ``<pre>``/``<code>`` block. Confluence-stored code macros are routed
    via the ``Macro`` path; this covers plain HTML ``<pre>``."""

    kind: str = Field(default="code_block")
    language: Optional[str] = None
    code: str


class TableNode(ASTNode):
    """Confluence table preserved opaquely. Drafter MUST NOT edit table cells in Phase 10."""

    kind: str = Field(default="table")
    rows_html: str


# A Block is any top-level child of a Section (everything except headings, which are stored on
# the Section itself, and ListItems, which live inside OrderedList/UnorderedList).
Block = Union[Paragraph, OrderedList, UnorderedList, Macro, CodeBlock, TableNode]


class Section(BaseModel):
    """A heading-bounded chunk of the page. ``heading=None`` for pre-heading prose."""

    heading: Optional[Heading] = None
    blocks: List[Block] = Field(default_factory=list)
    section_index: int


class ASTRoot(BaseModel):
    """Root of the parse tree. ``raw_html`` is the verbatim input for round-trip use by
    ``editor_dispatcher``."""

    sections: List[Section] = Field(default_factory=list)
    raw_html: str


# ---------------------------------------------------------------------------
# PageParser
# ---------------------------------------------------------------------------


class PageParser:
    """Confluence storage-format HTML -> :class:`ASTRoot`.

    Stateless; safe to instantiate once per process or per call. Construct soup with
    ``lxml`` (Pitfall 6) so namespaced ``ac:`` tags survive intact.
    """

    def parse(self, html: str) -> ASTRoot:
        """Parse *html* into a typed AST. Returns an ``ASTRoot`` even for empty input."""
        if not isinstance(html, str):
            raise TypeError(f"PageParser.parse expects str, got {type(html).__name__}")

        original_html = html
        # T-10-02: cap absurdly-large pages.
        if len(html.encode("utf-8")) > MAX_PAGE_HTML_BYTES:
            logger.warning(
                "PageParser input exceeds %d bytes — truncating (T-10-02 mitigation)",
                MAX_PAGE_HTML_BYTES,
            )
            # Truncate on a UTF-8 boundary by encoding/decoding with ignore.
            html = html.encode("utf-8")[:MAX_PAGE_HTML_BYTES].decode("utf-8", errors="ignore")

        soup = self._make_soup(html)
        # Confluence storage HTML is fragment-shaped (no <html>/<body>). BS4 with lxml will
        # sometimes wrap it in <html><body>...</body></html> — descend into <body> if present.
        container = soup.body if soup.body else soup

        # Walk the top-level children, splitting into sections at heading tags.
        top_level: List[Tag] = [
            child for child in container.children
            if isinstance(child, Tag) or (isinstance(child, NavigableString) and str(child).strip())
        ]

        sections: List[Section] = []
        current_section_index = 0
        current_heading: Optional[Heading] = None
        current_blocks: List[Block] = []

        def flush_section() -> None:
            nonlocal current_section_index, current_heading, current_blocks
            # Skip an empty leading section (heading=None, no blocks) — keeps the AST clean
            # for inputs that start straight with a heading.
            if current_heading is None and not current_blocks:
                return
            sections.append(
                Section(
                    heading=current_heading,
                    blocks=current_blocks,
                    section_index=current_section_index,
                )
            )
            current_section_index += 1
            current_heading = None
            current_blocks = []

        for child in top_level:
            if isinstance(child, NavigableString):
                # Loose text between block elements becomes a Paragraph with one TextRun.
                text = str(child)
                if not text.strip():
                    continue
                ast_path = self._block_path(current_section_index, len(current_blocks), "paragraph")
                current_blocks.append(Paragraph(ast_path=ast_path, runs=[TextRun(text=text)]))
                continue

            assert isinstance(child, Tag)

            if _is_heading(child):
                # Heading => boundary. Flush any in-progress section and start a fresh one.
                flush_section()
                heading_path = self._heading_path(current_section_index)
                current_heading = Heading(
                    ast_path=heading_path,
                    level=_heading_level(child),
                    text=child.get_text(" ", strip=True),
                )
                continue

            block = self._tag_to_block(child, current_section_index, len(current_blocks))
            current_blocks.append(block)

        # Flush trailing section.
        flush_section()

        return ASTRoot(sections=sections, raw_html=original_html)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_soup(self, html: str) -> BeautifulSoup:
        """Construct a soup, preferring lxml (Pitfall 6) and falling back to html.parser."""
        try:
            return BeautifulSoup(html, "lxml")
        except Exception as exc:  # pragma: no cover - exercised only when lxml missing
            logger.warning(
                "lxml unavailable - reorder may trigger version conflicts (Pitfall 6): %s",
                exc,
            )
            return BeautifulSoup(html, "html.parser")

    def _heading_path(self, section_index: int) -> str:
        return f"section[{section_index}].heading"

    def _block_path(self, section_index: int, block_index: int, kind: str) -> str:
        return f"section[{section_index}].{kind}[{block_index}]"

    def _list_item_path(
        self, section_index: int, block_index: int, list_kind: str, item_index: int
    ) -> str:
        return (
            f"section[{section_index}].{list_kind}[{block_index}].item[{item_index}]"
        )

    def _tag_to_block(self, tag: Tag, section_index: int, block_index: int) -> Block:
        """Dispatch a non-heading top-level tag to its Block subclass."""
        # 1. Confluence namespaced tags (ac:structured-macro, ac:* etc.) -> opaque Macro.
        #    Pitfall 1: NEVER descend into macros — drafter never sees inside them.
        if _is_macro_tag(tag):
            name = tag.get("ac:name") or tag.name.split(":", 1)[-1]
            return Macro(
                ast_path=self._block_path(section_index, block_index, "macro"),
                name=str(name),
                body_html=tag.decode_contents(),
            )

        # 2. Standard HTML block kinds.
        name = (tag.name or "").lower()
        if name == "p":
            return Paragraph(
                ast_path=self._block_path(section_index, block_index, "paragraph"),
                runs=self._extract_runs(tag),
            )
        if name == "ol":
            return OrderedList(
                ast_path=self._block_path(section_index, block_index, "ordered_list"),
                items=self._extract_list_items(
                    tag, section_index, block_index, list_kind="ordered_list"
                ),
            )
        if name == "ul":
            return UnorderedList(
                ast_path=self._block_path(section_index, block_index, "unordered_list"),
                items=self._extract_list_items(
                    tag, section_index, block_index, list_kind="unordered_list"
                ),
            )
        if name == "table":
            return TableNode(
                ast_path=self._block_path(section_index, block_index, "table"),
                rows_html=tag.decode_contents(),
            )
        if name in ("pre", "code"):
            return CodeBlock(
                ast_path=self._block_path(section_index, block_index, "code_block"),
                language=tag.get("data-language") or tag.get("class", [None])[0]
                if isinstance(tag.get("class"), list)
                else tag.get("data-language"),
                code=tag.get_text(),
            )

        # 3. Anything else (div, span at top level, blockquote, etc.) -> Paragraph fallback.
        #    This preserves text content rather than silently dropping it.
        return Paragraph(
            ast_path=self._block_path(section_index, block_index, "paragraph"),
            runs=self._extract_runs(tag),
        )

    def _extract_list_items(
        self,
        list_tag: Tag,
        section_index: int,
        block_index: int,
        list_kind: str,
    ) -> List[ListItem]:
        """Walk direct ``<li>`` children of *list_tag* and produce ListItem nodes with
        sequential 0-based ``index`` values."""
        items: List[ListItem] = []
        # find_all with recursive=False keeps the order intact and avoids picking up nested
        # list items as siblings of their parents.
        for i, li in enumerate(list_tag.find_all("li", recursive=False)):
            items.append(
                ListItem(
                    ast_path=self._list_item_path(section_index, block_index, list_kind, i),
                    index=i,
                    runs=self._extract_runs(li),
                )
            )
        return items

    def _extract_runs(self, tag: Tag) -> List[TextRun]:
        """Walk the inline children of *tag* and produce a list of TextRun objects.

        Rules:
          * NavigableString -> TextRun(text=raw)
          * <strong>/<b>    -> TextRun(text=inner_text, bold=True)
          * <em>/<i>        -> TextRun(text=inner_text, italic=True)
          * <a href=...>    -> TextRun(text=inner_text, href=href)
          * anything else   -> recurse, treating its descendants as flat runs
        """
        runs: List[TextRun] = []
        for child in tag.children:
            if isinstance(child, NavigableString):
                text = str(child)
                # Keep whitespace-only runs around inline formatting (e.g. "This is " before
                # <strong>bold</strong>) so the paragraph reads back correctly.
                if text:
                    runs.append(TextRun(text=text))
                continue
            if not isinstance(child, Tag):
                continue
            name = (child.name or "").lower()
            if name in ("strong", "b"):
                runs.append(TextRun(text=child.get_text(), bold=True))
            elif name in ("em", "i"):
                runs.append(TextRun(text=child.get_text(), italic=True))
            elif name == "a":
                runs.append(
                    TextRun(text=child.get_text(), href=child.get("href"))
                )
            elif name == "br":
                runs.append(TextRun(text="\n"))
            else:
                # Recurse for nested wrappers (span, font, etc.). If recursion produces no
                # runs (e.g. an empty tag), fall back to flat text so we never silently
                # lose content.
                inner = self._extract_runs(child)
                if inner:
                    runs.extend(inner)
                else:
                    text = child.get_text()
                    if text:
                        runs.append(TextRun(text=text))
        return runs


__all__ = [
    "ASTRoot",
    "Section",
    "Heading",
    "Paragraph",
    "OrderedList",
    "UnorderedList",
    "ListItem",
    "Macro",
    "CodeBlock",
    "TableNode",
    "TextRun",
    "PageParser",
    "MAX_PAGE_HTML_BYTES",
    "HEADING_TAGS",
]
