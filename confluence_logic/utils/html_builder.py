import re
from typing import List, Dict, Optional, Union
from ..core.schemas import PageSectionInput


def _looks_like_html(text: str) -> bool:
    return bool(text and "<" in text and ">" in text and re.search(r"<[a-z][a-z0-9]*[\s/>]", text, re.I))


def _inline_md(text: str) -> str:
    """Convert inline markdown (bold, italic, code, links) to HTML inline elements."""
    # Bold: **text** or __text__
    text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"__(.+?)__", r"<strong>\1</strong>", text)
    # Italic: *text* or _text_ (single)
    text = re.sub(r"\*([^*\n]+?)\*", r"<em>\1</em>", text)
    text = re.sub(r"(?<![_])_([^_\n]+?)_(?![_])", r"<em>\1</em>", text)
    # Inline code: `code`
    text = re.sub(r"`([^`]+)`", r"<code>\1</code>", text)
    return text


def markdown_to_html(text: str) -> str:
    """Convert basic markdown to Confluence Storage Format HTML.

    Handles headings (#/##/###), bold (**), italic (*), bullet lists (-/*),
    numbered lists (1.), inline code (`), and plain paragraphs.
    If the text is already HTML, it is returned unchanged.
    """
    if not text or not text.strip():
        return ""
    if _looks_like_html(text):
        return text

    lines = text.splitlines()
    parts: List[str] = []
    list_type: Optional[str] = None  # "ul" or "ol"

    def _close_list() -> None:
        nonlocal list_type
        if list_type:
            parts.append(f"</{list_type}>")
            list_type = None

    for line in lines:
        stripped = line.strip()

        if not stripped:
            _close_list()
            continue

        # Headings
        h3 = re.match(r"^#{3}\s+(.*)", stripped)
        h2 = re.match(r"^#{2}\s+(.*)", stripped)
        h1 = re.match(r"^#\s+(.*)", stripped)
        if h3:
            _close_list()
            parts.append(f"<h3>{_inline_md(h3.group(1))}</h3>")
        elif h2:
            _close_list()
            parts.append(f"<h2>{_inline_md(h2.group(1))}</h2>")
        elif h1:
            _close_list()
            parts.append(f"<h1>{_inline_md(h1.group(1))}</h1>")
        # Unordered list item: - or *
        elif re.match(r"^[-*]\s+", stripped):
            content = re.sub(r"^[-*]\s+", "", stripped)
            if list_type != "ul":
                _close_list()
                parts.append("<ul>")
                list_type = "ul"
            parts.append(f"<li>{_inline_md(content)}</li>")
        # Ordered list item: 1. 2. etc.
        elif re.match(r"^\d+\.\s+", stripped):
            content = re.sub(r"^\d+\.\s+", "", stripped)
            if list_type != "ol":
                _close_list()
                parts.append("<ol>")
                list_type = "ol"
            parts.append(f"<li>{_inline_md(content)}</li>")
        # Horizontal rule
        elif re.match(r"^[-*_]{3,}$", stripped):
            _close_list()
            parts.append("<hr/>")
        # Plain paragraph
        else:
            _close_list()
            parts.append(f"<p>{_inline_md(stripped)}</p>")

    _close_list()
    return "\n".join(parts)


def build_page_html(
    title: str,
    sections: Optional[List[Union[Dict[str, str], PageSectionInput]]] = None,
    body_text: Optional[str] = None,
) -> str:
    """Construct Confluence Storage Format HTML from sections or body_text.

    Both body_text and section content are passed through markdown_to_html so
    agents can write markdown freely — it always renders correctly in Confluence.
    """
    html_out = ""

    if body_text:
        html_out += markdown_to_html(body_text.strip())

    if sections:
        for sec in sections:
            section = sec.model_dump() if isinstance(sec, PageSectionInput) else sec

            if section.get("heading"):
                html_out += f"<h2>{section['heading'].strip()}</h2>"

            content = (section.get("content") or "").strip()
            if content:
                html_out += markdown_to_html(content)

    return html_out
