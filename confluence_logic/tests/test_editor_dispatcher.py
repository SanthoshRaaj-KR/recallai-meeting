"""Tests for EditorDispatcher (PROP-V2-06).

Phase 10 — auto-propose-pipeline-quality-redesign-v2.

EditorDispatcher (D-02) is the thin routing layer that takes one of
the six structured instruction shapes and dispatches to the existing
EditorAgent primitives WITHOUT importing or modifying
``editor_agent.py`` directly (scope-lock).

The six D-02 shapes:
  - replace      {action, page_id, section_heading, old_text, new_text}
  - insert_after {action, page_id, section_heading, anchor_text, new_text}
  - reorder      {action, page_id, section_heading, from_index, to_index}
  - delete_section {action, page_id, section_heading}
  - create_section {action, page_id, parent_heading, new_heading, new_content}
  - create_page  {action, parent_page_id, title, content}
"""
from unittest.mock import MagicMock, patch

import pytest

from confluence_logic.agents.editor_dispatcher import apply_structured  # noqa: F401


# Routing inputs for each D-02 shape. Each entry pairs the dispatch input
# instruction with the dispatcher-namespace tool symbol that MUST be called
# and the kwargs we assert on the call.
ROUTING_FIXTURES = {
    "replace": {
        "instruction": {
            "action": "replace",
            "page_id": "p1",
            "section_heading": "Setup",
            "old_text": "old",
            "new_text": "new",
        },
        "patch_target": "confluence_logic.agents.editor_dispatcher.commit_document_edit",
        "expect_kwargs": {
            "page_id": "p1",
            "heading_string": "Setup",
            "old_block_html": "old",
            "new_block_html": "new",
            "append": False,
        },
    },
    "insert_after": {
        "instruction": {
            "action": "insert_after",
            "page_id": "p1",
            "section_heading": "Setup",
            "anchor_text": "anchor",
            "new_text": "appended",
        },
        "patch_target": "confluence_logic.agents.editor_dispatcher.commit_document_edit",
        "expect_kwargs": {
            "page_id": "p1",
            "heading_string": "Setup",
            "new_block_html": "appended",
            "append": True,
        },
    },
    "delete_section": {
        "instruction": {
            "action": "delete_section",
            "page_id": "p1",
            "section_heading": "Old",
        },
        "patch_target": "confluence_logic.agents.editor_dispatcher.commit_delete",
        "expect_kwargs": {
            "page_id": "p1",
            "heading_string": "Old",
            "delete_entire_section": True,
        },
    },
    "create_section": {
        "instruction": {
            "action": "create_section",
            "page_id": "p1",
            "parent_heading": "Parent",
            "new_heading": "New Heading",
            "new_content": "<p>body</p>",
        },
        "patch_target": "confluence_logic.agents.editor_dispatcher.commit_document_edit",
        "expect_kwargs": {
            "page_id": "p1",
            "heading_string": "Parent",
            "append": True,
        },
    },
    "create_page": {
        "instruction": {
            "action": "create_page",
            "title": "T",
            "content": "c",
            "parent_page_id": "pp",
        },
        "patch_target": "confluence_logic.agents.editor_dispatcher.create_confluence_page",
        "expect_kwargs": {
            "title": "T",
            "body_text": "c",
            "parent_page_id": "pp",
        },
    },
}


@pytest.mark.parametrize(
    "action",
    [
        "replace",
        "insert_after",
        "reorder",
        "delete_section",
        "create_section",
        "create_page",
    ],
)
def test_routing_matrix(action):
    """PROP-V2-06: each D-02 instruction shape dispatches to the correct primitive via tools.py."""
    if action == "reorder":
        # Reorder needs a live page fetch + BS4 swap; routed through
        # ConfluenceConnector + commit_document_edit. We assert it routes
        # to commit_document_edit with append=False after the swap.
        synthetic_html = (
            "<h2>Onboarding</h2>"
            "<ol><li>A</li><li>B</li><li>C</li></ol>"
        )
        instruction = {
            "action": "reorder",
            "page_id": "p1",
            "section_heading": "Onboarding",
            "from_index": 1,
            "to_index": 2,
        }
        with patch(
            "confluence_logic.agents.editor_dispatcher.ConfluenceConnector"
        ) as MockConnector, patch(
            "confluence_logic.agents.editor_dispatcher.commit_document_edit"
        ) as mock_commit:
            instance = MockConnector.return_value
            instance.fetch_page_html.return_value = synthetic_html
            instance.get_page_metadata.return_value = {"version": {"number": 5}}
            mock_commit.return_value = {"success": True, "version": 6, "message": "ok"}
            apply_structured(instruction)
            assert mock_commit.call_count == 1, (
                f"reorder must call commit_document_edit exactly once (got {mock_commit.call_count})"
            )
            kwargs = mock_commit.call_args.kwargs
            assert kwargs["page_id"] == "p1"
            assert kwargs["heading_string"] == "Onboarding"
            assert kwargs["append"] is False
        return

    fixture = ROUTING_FIXTURES[action]
    with patch(fixture["patch_target"]) as mock_tool:
        mock_tool.return_value = {"success": True}
        apply_structured(fixture["instruction"])
        assert mock_tool.call_count == 1, (
            f"{action} must call {fixture['patch_target']} exactly once "
            f"(got {mock_tool.call_count})"
        )
        kwargs = mock_tool.call_args.kwargs
        for key, expected in fixture["expect_kwargs"].items():
            assert kwargs.get(key) == expected, (
                f"{action}: expected kwarg {key}={expected!r}, got {kwargs.get(key)!r}"
            )
        if action == "create_section":
            # create_section composes <h2>{new_heading}</h2> + new_content
            new_block = kwargs.get("new_block_html", "")
            assert new_block.startswith("<h2>New Heading</h2>"), (
                f"create_section new_block_html must start with <h2>{{new_heading}}</h2>; got {new_block!r}"
            )
            assert "<p>body</p>" in new_block


def test_reorder_preserves_unmoved_items_byte_identical():
    """PROP-V2-02: reorder produces after-section where non-moved <li>s are byte-identical to before."""
    synthetic_html = (
        "<h2>Onboarding</h2>"
        "<ol><li>A</li><li>B</li><li>C</li></ol>"
    )
    instruction = {
        "action": "reorder",
        "page_id": "p1",
        "section_heading": "Onboarding",
        "from_index": 1,
        "to_index": 2,
    }
    with patch(
        "confluence_logic.agents.editor_dispatcher.ConfluenceConnector"
    ) as MockConnector, patch(
        "confluence_logic.agents.editor_dispatcher.commit_document_edit"
    ) as mock_commit:
        instance = MockConnector.return_value
        instance.fetch_page_html.return_value = synthetic_html
        instance.get_page_metadata.return_value = {"version": {"number": 1}}
        mock_commit.return_value = {"success": True, "version": 2, "message": "ok"}

        apply_structured(instruction)

        new_block = mock_commit.call_args.kwargs["new_block_html"]
        # B and C should have swapped positions; A is byte-identical.
        assert "<li>A</li>" in new_block, "A must remain byte-identical"
        # Order check: A appears before C; C appears before B.
        idx_a = new_block.find("<li>A</li>")
        idx_b = new_block.find("<li>B</li>")
        idx_c = new_block.find("<li>C</li>")
        assert 0 <= idx_a < idx_c < idx_b, (
            f"Expected A→C→B ordering; got positions A={idx_a} C={idx_c} B={idx_b} in {new_block!r}"
        )


def test_reorder_ignores_llm_supplied_after_content():
    """Pitfall 5: dispatcher MUST ignore any LLM-supplied after_content on reorder; reconstruct from live HTML."""
    synthetic_html = (
        "<h2>Onboarding</h2>"
        "<ol><li>A</li><li>B</li><li>C</li></ol>"
    )
    instruction = {
        "action": "reorder",
        "page_id": "p1",
        "section_heading": "Onboarding",
        "from_index": 1,
        "to_index": 2,
        "after_content": "<li>FABRICATED-BY-LLM</li>",
    }
    with patch(
        "confluence_logic.agents.editor_dispatcher.ConfluenceConnector"
    ) as MockConnector, patch(
        "confluence_logic.agents.editor_dispatcher.commit_document_edit"
    ) as mock_commit:
        instance = MockConnector.return_value
        instance.fetch_page_html.return_value = synthetic_html
        instance.get_page_metadata.return_value = {"version": {"number": 1}}
        mock_commit.return_value = {"success": True}

        apply_structured(instruction)
        new_block = mock_commit.call_args.kwargs["new_block_html"]
        assert "FABRICATED-BY-LLM" not in new_block, (
            "Dispatcher must ignore LLM after_content; only A/B/C from live HTML are allowed"
        )


def test_does_not_import_editor_agent_module_directly():
    """PROP-V2-06: static check — dispatcher source MUST NOT contain ``from confluence_logic.agents.editor_agent import`` (scope-lock on locked apply layer)."""
    import ast
    import pathlib

    dispatcher_path = (
        pathlib.Path(__file__).resolve().parent.parent
        / "agents"
        / "editor_dispatcher.py"
    )
    assert dispatcher_path.exists(), (
        f"Dispatcher must exist at {dispatcher_path}"
    )
    source = dispatcher_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    bad_imports = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        and node.module == "confluence_logic.agents.editor_agent"
    ]
    assert not bad_imports, (
        "EditorDispatcher must NOT import editor_agent directly "
        "(scope-lock per D-02; use tools.py @function_tool primitives instead)."
    )
    # Belt-and-braces: also reject bare 'import confluence_logic.agents.editor_agent'
    bad_bare = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        and any(alias.name == "confluence_logic.agents.editor_agent" for alias in node.names)
    ]
    assert not bad_bare, (
        "EditorDispatcher must NOT import editor_agent module under any form."
    )


def test_reorder_invalid_index_returns_error_without_committing():
    """PROP-V2-06: out-of-range index returns {success:False, message:'Index out of range.'} and does NOT call commit_document_edit."""
    synthetic_html = (
        "<h2>Onboarding</h2>"
        "<ol><li>A</li><li>B</li><li>C</li></ol>"
    )
    instruction = {
        "action": "reorder",
        "page_id": "p1",
        "section_heading": "Onboarding",
        "from_index": 5,  # out of range (only 3 items)
        "to_index": 0,
    }
    with patch(
        "confluence_logic.agents.editor_dispatcher.ConfluenceConnector"
    ) as MockConnector, patch(
        "confluence_logic.agents.editor_dispatcher.commit_document_edit"
    ) as mock_commit:
        instance = MockConnector.return_value
        instance.fetch_page_html.return_value = synthetic_html
        instance.get_page_metadata.return_value = {"version": {"number": 1}}

        result = apply_structured(instruction)
        assert isinstance(result, dict)
        assert result.get("success") is False
        assert "Index out of range" in result.get("message", "")
        assert mock_commit.call_count == 0, (
            "Out-of-range reorder must NOT call commit_document_edit "
            f"(got {mock_commit.call_count} calls)"
        )


def test_unknown_action_raises_value_error():
    """Defensive: unknown action raises ValueError; dispatcher never silently passes."""
    with pytest.raises(ValueError, match="Unknown structured action"):
        apply_structured({"action": "this-action-does-not-exist"})
