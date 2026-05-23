"""Phase 10 structured-instruction dispatcher (PROP-V2-06, D-02).

Translates each of the six D-02 instruction shapes (replace, insert_after,
reorder, delete_section, create_section, create_page) to existing
@function_tool primitives in confluence_logic/agents/tools.py.

MUST NOT import editor_agent. MUST NOT call EditorAgent.agent.run(...).
MUST NOT modify any file under editor_agent.py. PROP-V2-06.

The dispatcher is the ONLY place new Phase 10 code interacts with the
apply layer. EditorAgent's existing methods are preserved byte-identical;
this module composes their underlying primitives directly.

Six D-02 shapes:
  - replace        {action, page_id, section_heading, old_text, new_text}
  - insert_after   {action, page_id, section_heading, anchor_text, new_text}
  - reorder        {action, page_id, section_heading, from_index, to_index}
  - delete_section {action, page_id, section_heading}
  - create_section {action, page_id, parent_heading, new_heading, new_content}
  - create_page    {action, parent_page_id, title, content}

Pitfall references (10-RESEARCH.md):
  - Pitfall 3: Always bound BS4 search via get_section_html before find()
  - Pitfall 5: Reorder MUST ignore any LLM-supplied after_content; reconstruct
               from live HTML to defeat token-level fabrication.
  - Pitfall 6: Use BeautifulSoup(..., "lxml") to preserve self-closing macros
               and avoid spurious version conflicts.
"""
from __future__ import annotations

import logging
from typing import Any, Dict

from bs4 import BeautifulSoup

from confluence_logic.agents.tools import (
    commit_delete,
    commit_document_edit,
    create_confluence_page,
    delete_confluence_page,
    fetch_live_page,
    update_page_title,
)
from confluence_logic.connectors.confluence import ConfluenceConnector
from confluence_logic.utils.html_parser import get_section_html

logger = logging.getLogger(__name__)


def _fetch_version(page_id: str) -> int:
    """Best-effort live page-version fetch.

    Returns 1 on any failure; tools.py commit_document_edit handles stale
    version via its existing retry loop (APPLY-02), so a wrong version
    here just costs one extra retry round-trip — never a data error.
    """
    try:
        connector = ConfluenceConnector()
        meta = connector.get_page_metadata(page_id)
        # Confluence REST shape: {"version": {"number": N, ...}, ...}
        version_field = meta.get("version") if isinstance(meta, dict) else None
        if isinstance(version_field, dict):
            return int(version_field.get("number") or 1)
        if isinstance(version_field, (int, float, str)) and str(version_field).strip():
            return int(version_field)
        return 1
    except Exception as exc:  # pragma: no cover - logged + degrades to 1
        logger.warning("_fetch_version failed for %s: %s (defaulting to 1)", page_id, exc)
        return 1


def apply_structured(instruction: Dict[str, Any]) -> Dict[str, Any]:
    """Translate a Phase 10 D-02 structured instruction into tools.py primitives.

    Dispatches on instruction["action"]. Never invokes EditorAgent.agent.run(...).
    Returns the result dict from the underlying tool (or a {success, message}
    dict for the reorder bounds-check error path).
    """
    action = instruction.get("action")
    page_id = instruction.get("page_id") or instruction.get("parent_page_id") or "<n/a>"
    logger.info("apply_structured action=%s page_id=%s", action, page_id)

    if action == "replace":
        return commit_document_edit(
            page_id=instruction["page_id"],
            expected_version=_fetch_version(instruction["page_id"]),
            heading_string=instruction["section_heading"],
            old_block_html=instruction["old_text"],
            new_block_html=instruction["new_text"],
            append=False,
        )

    if action == "insert_after":
        # Maps to commit_document_edit(append=True). Anchor is implicit
        # (end of section). Precise post-anchor placement would map to
        # replace where old=anchor and new=anchor+new_text; the simple
        # append form matches Pattern 4 sketch line 738.
        return commit_document_edit(
            page_id=instruction["page_id"],
            expected_version=_fetch_version(instruction["page_id"]),
            heading_string=instruction["section_heading"],
            new_block_html=instruction["new_text"],
            append=True,
        )

    if action == "delete_section":
        return commit_delete(
            page_id=instruction["page_id"],
            expected_version=_fetch_version(instruction["page_id"]),
            heading_string=instruction["section_heading"],
            delete_entire_section=True,
        )

    if action == "create_section":
        # Append a new <h2> + body under the parent section.
        new_html = (
            f"<h2>{instruction['new_heading']}</h2>"
            + instruction["new_content"]
        )
        return commit_document_edit(
            page_id=instruction["page_id"],
            expected_version=_fetch_version(instruction["page_id"]),
            heading_string=instruction["parent_heading"],
            new_block_html=new_html,
            append=True,
        )

    if action == "create_page":
        # space_key=None falls through to ATLASSIAN_SPACE_KEY env per
        # tools.create_confluence_page default behavior.
        return create_confluence_page(
            title=instruction["title"],
            space_key=instruction.get("space_key"),
            body_text=instruction["content"],
            parent_page_id=instruction.get("parent_page_id"),
        )

    if action == "reorder":
        # Pitfall 5: IGNORE any LLM-supplied after_content. The drafter
        # is bias-prone on reorder ops; the dispatcher reconstructs the
        # after-section HTML purely from live page state.
        if "after_content" in instruction:
            logger.info(
                "apply_structured(reorder): ignoring LLM after_content per Pitfall 5"
            )
        return _apply_reorder(
            page_id=instruction["page_id"],
            section_heading=instruction["section_heading"],
            from_index=instruction["from_index"],
            to_index=instruction["to_index"],
        )

    raise ValueError(f"Unknown structured action: {action}")


def _apply_reorder(
    page_id: str,
    section_heading: str,
    from_index: int,
    to_index: int,
) -> Dict[str, Any]:
    """Swap two <li> entries inside the first <ol> of section_heading.

    Reconstructs after-section HTML from live page HTML so non-moved <li>
    entries are byte-identical to source. Atomic whole-section replacement
    via commit_document_edit. Pitfall 3 enforces bounded BS4 search.
    Pitfall 6 enforces lxml parser to preserve self-closing macros.
    """
    try:
        connector = ConfluenceConnector()
        html = connector.fetch_page_html(page_id)
    except Exception as exc:
        logger.warning("_apply_reorder: fetch_page_html failed for %s: %s", page_id, exc)
        return {"success": False, "message": f"fetch_page_html failed: {exc}"}

    try:
        section_html = get_section_html(html, section_heading)
    except Exception as exc:
        logger.warning(
            "_apply_reorder: get_section_html failed for %s/%s: %s",
            page_id, section_heading, exc,
        )
        return {
            "success": False,
            "message": f"Section '{section_heading}' not found: {exc}",
        }

    # Pitfall 6: lxml preserves self-closing macros (e.g. <ac:structured-macro/>).
    # Pitfall 3: section_html is already bounded — the soup only sees the section.
    soup = BeautifulSoup(section_html, "lxml")
    ol = soup.find("ol")
    if ol is None:
        msg = f"No ordered list in section '{section_heading}'."
        logger.warning("_apply_reorder: %s (page_id=%s)", msg, page_id)
        return {"success": False, "message": msg}

    items = ol.find_all("li", recursive=False)
    n = len(items)
    if (
        from_index < 0
        or to_index < 0
        or from_index >= n
        or to_index >= n
    ):
        msg = "Index out of range."
        logger.warning(
            "_apply_reorder: %s from=%s to=%s len=%s page=%s",
            msg, from_index, to_index, n, page_id,
        )
        return {"success": False, "message": msg}

    if from_index == to_index:
        # No-op reorder — return success without committing.
        logger.info(
            "_apply_reorder: noop from==to=%s page=%s — skipping commit",
            from_index, page_id,
        )
        return {"success": True, "message": "Reorder is a no-op (from_index == to_index)."}

    moved = items[from_index].extract()
    if from_index < to_index:
        items[to_index].insert_after(moved)
    else:
        items[to_index].insert_before(moved)

    # BS4 with lxml may wrap output in <html><body>… — extract only the
    # original section content. We want the bounded-section HTML.
    # The lxml parser wraps fragments; use the soup's body if present.
    body = soup.body
    if body is not None:
        new_section_html = "".join(str(c) for c in body.children)
    else:
        new_section_html = str(soup)

    return commit_document_edit(
        page_id=page_id,
        expected_version=_fetch_version(page_id),
        heading_string=section_heading,
        old_block_html=section_html,
        new_block_html=new_section_html,
        append=False,
    )
