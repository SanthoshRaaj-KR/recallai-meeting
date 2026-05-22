"""Wave 0 RED test scaffold for StructureAwareDrafter (PROP-V2-02, PROP-V2-06).

Phase 10 — auto-propose-pipeline-quality-redesign-v2.

StructureAwareDrafter (D-03 + D-06) receives the ChangeIntent, the parsed
AST of the qualified page, and a transcript window. It emits exactly one
structured operation from the EditorAgent instruction shapes — never
free-form prose. Wave 1 creates
`confluence_logic/agents/structure_aware_drafter.py`; until then this
import fails RED.
"""
import pytest

pytestmark = pytest.mark.asyncio

from confluence_logic.agents.structure_aware_drafter import (  # noqa: F401, E402
    draft_operation,
    StructureAwareDrafterInput,
    StructuredOperation,
)


async def test_reorder_intent_produces_reorder_op():
    """PROP-V2-02: a reorder intent on an ordered list produces ``action="reorder"`` with from_index/to_index — never a regenerated prose block."""
    pytest.fail("Wave 0 RED — StructureAwareDrafter implementation pending (Wave 1)")


async def test_replace_intent_emits_old_text_and_new_text():
    """PROP-V2-02: a replace intent emits ``action="replace"`` with both old_text and new_text set."""
    pytest.fail("Wave 0 RED — StructureAwareDrafter implementation pending (Wave 1)")


async def test_unknown_section_returns_create_section_op():
    """PROP-V2-06: when the intent refers to a section not present in the AST, drafter emits ``action="create_section"`` with explicit parent_heading from the AST."""
    pytest.fail("Wave 0 RED — StructureAwareDrafter implementation pending (Wave 1)")


async def test_drafter_returns_skip_when_grounding_unsatisfiable():
    """PROP-V2-02: when no operation can satisfy grounding (e.g., insert with tokens absent from transcript ∪ page), drafter returns ``{action: "skip", reason: ...}`` (not a card)."""
    pytest.fail("Wave 0 RED — StructureAwareDrafter implementation pending (Wave 1)")


async def test_never_emits_freeform_prose_for_ordered_list():
    """PROP-V2-06: for any intent targeting an ordered list, the drafter MUST NOT emit a freshly-written ordered-list new_text — only ``reorder`` / ``replace`` (single li) / ``insert_after`` ops are allowed."""
    pytest.fail("Wave 0 RED — StructureAwareDrafter implementation pending (Wave 1)")
