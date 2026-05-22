"""Wave 0 RED test scaffold for EditorDispatcher (PROP-V2-06).

Phase 10 — auto-propose-pipeline-quality-redesign-v2.

EditorDispatcher (D-02) is the thin routing layer that takes one of
the six structured instruction shapes and dispatches to the existing
EditorAgent primitives WITHOUT importing or modifying
``editor_agent.py`` directly (scope-lock). Wave 1+ creates
`confluence_logic/agents/editor_dispatcher.py`; until then this
import fails RED.

The six D-02 shapes:
  - replace      {action, page_id, section_heading, old_text, new_text}
  - insert_after {action, page_id, section_heading, anchor_text, new_text}
  - reorder      {action, page_id, section_heading, from_index, to_index}
  - delete_section {action, page_id, section_heading}
  - create_section {action, page_id, parent_heading, new_heading, new_content}
  - create_page  {action, parent_page_id, title, content}
"""
import pytest

from confluence_logic.agents.editor_dispatcher import apply_structured  # noqa: F401


# Parametrized over the six D-02 instruction shapes — each row asserts
# the dispatcher routes to the correct editor primitive without
# importing editor_agent directly.
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
    """PROP-V2-06: each of the six D-02 instruction shapes is dispatched to the correct EditorAgent primitive (via tools.py, not editor_agent module import)."""
    pytest.fail(f"Wave 0 RED — EditorDispatcher routing matrix pending (Wave 1); action={action}")


def test_reorder_preserves_unmoved_items_byte_identical():
    """PROP-V2-02: ``apply_structured({action: "reorder", ...})`` produces an after-section where every non-moved ``<li>`` is byte-identical to its before form."""
    pytest.fail("Wave 0 RED — EditorDispatcher reorder primitive pending (Wave 1)")


def test_does_not_import_editor_agent_module_directly():
    """PROP-V2-06: static check — dispatcher source MUST NOT contain ``from confluence_logic.agents.editor_agent import`` (scope-lock on locked apply layer)."""
    import ast
    import pathlib

    dispatcher_path = (
        pathlib.Path(__file__).resolve().parent.parent
        / "agents"
        / "editor_dispatcher.py"
    )
    # Static check: the file does not yet exist in Wave 0; this is the
    # RED gate. Once Wave 1 creates the file, this test will parse the
    # AST and assert no ``from confluence_logic.agents.editor_agent``
    # ImportFrom node is present.
    if not dispatcher_path.exists():
        pytest.fail(
            "Wave 0 RED — editor_dispatcher.py does not exist yet (Wave 1 will create it)."
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


def test_reorder_invalid_index_returns_error_without_committing():
    """PROP-V2-06: when ``from_index`` or ``to_index`` is out of range for the section's ordered list, the dispatcher returns a structured error and never calls commit_document_edit."""
    pytest.fail("Wave 0 RED — EditorDispatcher reorder bounds-check pending (Wave 1)")
