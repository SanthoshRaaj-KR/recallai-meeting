"""Tests for StructureAwareDrafter (PROP-V2-02, PROP-V2-06).

Phase 10 — auto-propose-pipeline-quality-redesign-v2 / Plan 06.

StructureAwareDrafter (D-03 + D-06) receives the ChangeIntent, the parsed
AST of the qualified page, and a transcript window. It emits exactly one
structured operation from the EditorAgent instruction shapes — never
free-form prose. The JSON-constrained ``StructuredOperation`` schema
makes the "regenerated procedure" failure mode (Failure 2 in
10-CONTEXT.md) structurally impossible: for a reorder intent, the schema
forces ``action="reorder"`` with from_index/to_index — there is no
``new_text`` payload the LLM can ride to emit a regenerated ``<ol>``.
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any, List, Optional
from unittest.mock import AsyncMock, patch

import pytest

pytestmark = pytest.mark.asyncio

from confluence_logic.agents.fact_extraction_agent import ChangeIntent
from confluence_logic.agents.page_parser import (
    ASTRoot,
    Heading,
    ListItem,
    OrderedList,
    Paragraph,
    Section,
    TextRun,
)
from confluence_logic.agents.structure_aware_drafter import (
    STRUCTURE_AWARE_DRAFTER_PROMPT,
    StructureAwareDrafterInput,
    StructuredOperation,
    draft_operation,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_ordered_procedure_ast() -> ASTRoot:
    """Build an ``ASTRoot`` with a single section "Onboarding" containing an
    ordered list ``[Login, Payment, Dashboard]`` with 0-based indices 0/1/2."""
    heading = Heading(ast_path="section[0].heading", level=2, text="Onboarding")
    ol = OrderedList(
        ast_path="section[0].ordered_list[0]",
        items=[
            ListItem(
                ast_path="section[0].ordered_list[0].item[0]",
                index=0,
                runs=[TextRun(text="Login")],
            ),
            ListItem(
                ast_path="section[0].ordered_list[0].item[1]",
                index=1,
                runs=[TextRun(text="Payment")],
            ),
            ListItem(
                ast_path="section[0].ordered_list[0].item[2]",
                index=2,
                runs=[TextRun(text="Dashboard")],
            ),
        ],
    )
    section = Section(heading=heading, blocks=[ol], section_index=0)
    return ASTRoot(
        sections=[section],
        raw_html=(
            "<h2>Onboarding</h2>"
            "<ol><li>Login</li><li>Payment</li><li>Dashboard</li></ol>"
        ),
    )


def _make_prose_section_ast(section_title: str, body_text: str) -> ASTRoot:
    """Build an ``ASTRoot`` with a single section containing one paragraph."""
    heading = Heading(ast_path="section[0].heading", level=2, text=section_title)
    para = Paragraph(
        ast_path="section[0].paragraph[0]",
        runs=[TextRun(text=body_text)],
    )
    section = Section(heading=heading, blocks=[para], section_index=0)
    return ASTRoot(
        sections=[section],
        raw_html=f"<h2>{section_title}</h2><p>{body_text}</p>",
    )


def _fake_run_result(op: StructuredOperation) -> SimpleNamespace:
    """Wrap a StructuredOperation as a Runner.run result."""
    return SimpleNamespace(final_output=op)


def _patch_runner(op: StructuredOperation):
    """Patch ``Runner.run`` in the drafter module to return *op* once."""
    mock = AsyncMock(return_value=_fake_run_result(op))
    return patch(
        "confluence_logic.agents.structure_aware_drafter.Runner.run", mock
    ), mock


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


async def test_reorder_intent_produces_reorder_op():
    """PROP-V2-02: a reorder intent on an ordered list produces ``action="reorder"``
    with from_index/to_index — never a regenerated prose block."""
    ast = _make_ordered_procedure_ast()
    intent = ChangeIntent(
        instruction="move Login before Payment in the Onboarding flow",
        subject="onboarding step order",
        target_hint="Onboarding",
        action="reorder",
    )
    inp = StructureAwareDrafterInput(
        intent=intent,
        page_ast=ast,
        page_meta={"page_id": "p1", "page_title": "Onboarding Guide"},
        transcript_window="Speaker: in onboarding, Login should come before Payment.",
    )
    # The LLM (mocked) returns the structurally-correct reorder op. The drafter
    # must surface it unchanged (modulo page_id stamping + reorder defense-in-depth).
    op_from_llm = StructuredOperation(
        action="reorder",
        section_heading="Onboarding",
        from_index=1,  # Payment currently at 1
        to_index=0,    # move it before Login (or vice versa — any valid swap)
        change_summary="Reorder onboarding so Login runs before Payment",
    )
    patcher, _ = _patch_runner(op_from_llm)
    with patcher:
        result = await draft_operation(inp)

    assert isinstance(result, StructuredOperation)
    assert result.action == "reorder", (
        f"reorder intent must emit action='reorder' (got {result.action!r}) — "
        "never a regenerated <ol>"
    )
    assert result.from_index is not None and result.to_index is not None, (
        "reorder op must have both from_index and to_index populated"
    )
    assert result.page_id == "p1", "drafter must stamp page_id from page_meta"
    # Defense-in-depth (Pitfall 5): for reorder ops, any LLM-supplied new_text or
    # new_content MUST be cleared at the drafter exit boundary.
    assert result.new_text is None, "reorder MUST NOT carry new_text (Pitfall 5)"
    assert result.new_content is None, "reorder MUST NOT carry new_content (Pitfall 5)"


async def test_replace_intent_emits_old_text_and_new_text():
    """PROP-V2-02: a replace intent emits ``action="replace"`` with both old_text and new_text."""
    ast = _make_prose_section_ast(
        section_title="Frameworks",
        body_text="Our backend stack standardises on React for the frontend layer.",
    )
    intent = ChangeIntent(
        instruction="Replace React with Vue in the Frameworks section",
        subject="frontend framework",
        target_hint="Frameworks",
        action="replace",
        old_value="React",
        new_value="Vue",
    )
    inp = StructureAwareDrafterInput(
        intent=intent,
        page_ast=ast,
        page_meta={"page_id": "p1", "page_title": "Frameworks"},
        transcript_window="We are moving from React to Vue across the frontend.",
    )
    op_from_llm = StructuredOperation(
        action="replace",
        section_heading="Frameworks",
        old_text="React",
        new_text="Vue",
        change_summary="Replace React with Vue in the Frameworks section",
    )
    patcher, _ = _patch_runner(op_from_llm)
    with patcher:
        result = await draft_operation(inp)

    assert result.action == "replace"
    assert result.old_text and "React" in result.old_text
    assert result.new_text and "Vue" in result.new_text
    assert result.page_id == "p1"


async def test_unknown_section_returns_create_section_op():
    """PROP-V2-06: when the intent refers to a section not present in the AST,
    drafter emits ``action="create_section"`` with explicit parent_heading from the AST."""
    ast = _make_prose_section_ast(
        section_title="Overview",
        body_text="This page describes the Acme service.",
    )
    intent = ChangeIntent(
        instruction="Document the new SLA section under the Overview page",
        subject="SLA",
        target_hint="SLA",
        action="add",
    )
    inp = StructureAwareDrafterInput(
        intent=intent,
        page_ast=ast,
        page_meta={"page_id": "p1", "page_title": "Acme Service"},
        transcript_window="We agreed to add an SLA section to the Acme page.",
    )
    op_from_llm = StructuredOperation(
        action="create_section",
        parent_heading="Overview",
        new_heading="SLA",
        new_content="The Acme service SLA is 99.9% uptime measured monthly.",
        change_summary="Create SLA section under Overview",
    )
    patcher, _ = _patch_runner(op_from_llm)
    with patcher:
        result = await draft_operation(inp)

    assert result.action == "create_section"
    assert result.parent_heading == "Overview", (
        "create_section must reference an existing AST heading as parent"
    )
    assert result.new_heading == "SLA"
    assert result.page_id == "p1"


async def test_drafter_returns_skip_when_grounding_unsatisfiable():
    """PROP-V2-02: when grounding cannot be satisfied (e.g., reorder requested
    but no ordered list anywhere on the page), drafter short-circuits to
    ``action="skip"`` BEFORE calling the LLM."""
    # An AST with no ordered list — only prose. A reorder intent cannot ground here.
    ast = _make_prose_section_ast(
        section_title="Background",
        body_text="This page contains only a paragraph and no procedure.",
    )
    intent = ChangeIntent(
        instruction="swap step 2 and step 3 in the Background section",
        subject="background steps",
        target_hint="Background",
        action="reorder",
    )
    inp = StructureAwareDrafterInput(
        intent=intent,
        page_ast=ast,
        page_meta={"page_id": "p1", "page_title": "Background"},
        transcript_window="",
    )
    # The deterministic pre-flight should fire BEFORE Runner.run is invoked.
    # Patch Runner.run so we can assert it was NEVER called.
    skip_sentinel = StructuredOperation(action="skip", reason="should never be returned")
    patcher, mock_run = _patch_runner(skip_sentinel)
    with patcher:
        result = await draft_operation(inp)

    assert result.action == "skip", (
        f"reorder intent against an AST with no OrderedList must skip; "
        f"got action={result.action!r}"
    )
    assert result.reason, "skip ops must carry a reason"
    assert mock_run.await_count == 0, (
        "deterministic pre-flight skip must NOT call the LLM "
        f"(Runner.run was awaited {mock_run.await_count} times)"
    )


async def test_never_emits_freeform_prose_for_ordered_list():
    """PROP-V2-06: for any reorder intent targeting an ordered list, the drafter
    MUST NOT emit a freshly-written ordered-list prose block.

    Two-layer defense:
      1. The ``StructuredOperation`` schema's ``action`` is a Literal — the LLM
         cannot return action="some_freeform_prose".
      2. Even if the LLM tries to ride along ``new_text`` / ``new_content``
         with a regenerated ``<ol>``, the drafter's post-validate (Pitfall 5)
         strips those fields for action="reorder".

    This test exercises layer 2: feed a malicious LLM output that has
    action="reorder" but ALSO a fabricated ``new_text`` with a regenerated
    ``<ol>``. The drafter must drop the prose.
    """
    ast = _make_ordered_procedure_ast()
    intent = ChangeIntent(
        instruction="move Payment to the top of the Onboarding flow",
        subject="onboarding step order",
        target_hint="Onboarding",
        action="reorder",
    )
    inp = StructureAwareDrafterInput(
        intent=intent,
        page_ast=ast,
        page_meta={"page_id": "p1", "page_title": "Onboarding Guide"},
        transcript_window="Payment should come before Login.",
    )

    # Construct a malicious LLM output: technically a valid StructuredOperation
    # with action="reorder" — but the LLM has tried to slip a regenerated <ol> in
    # via new_text. Drafter post-validate MUST clear it.
    malicious = StructuredOperation(
        action="reorder",
        section_heading="Onboarding",
        from_index=1,
        to_index=0,
        new_text=(
            "<ol><li>Enter username</li><li>Click mouse button</li>"
            "<li>Wait for verification</li></ol>"
        ),
        new_content="enter username, click mouse button, wait for verification",
        change_summary="Reorder onboarding (with hallucinated prose)",
    )
    patcher, _ = _patch_runner(malicious)
    with patcher:
        result = await draft_operation(inp)

    assert result.action == "reorder"
    # The two fields the LLM would use to smuggle a regenerated <ol> MUST be None.
    assert result.new_text is None, (
        "Drafter must strip new_text for reorder ops (Pitfall 5 defense-in-depth) "
        f"— got {result.new_text!r}"
    )
    assert result.new_content is None, (
        "Drafter must strip new_content for reorder ops (Pitfall 5 defense-in-depth) "
        f"— got {result.new_content!r}"
    )
    # And of course no "freeform" action ever surfaces — the Literal already
    # enforces this at the schema layer.
    assert result.action in {
        "replace",
        "insert_after",
        "reorder",
        "delete_section",
        "create_section",
        "create_page",
        "skip",
    }


async def test_structured_operation_action_is_literal_constrained():
    """Schema-level guarantee: ``StructuredOperation.action`` is a Pydantic
    Literal — any non-enumerated string raises ValidationError. This is what
    makes "regenerated prose" structurally impossible at the JSON layer."""
    from pydantic import ValidationError

    # Valid actions all construct fine.
    for valid in (
        "replace",
        "insert_after",
        "reorder",
        "delete_section",
        "create_section",
        "create_page",
        "skip",
    ):
        StructuredOperation(action=valid)

    with pytest.raises(ValidationError):
        StructuredOperation(action="freeform_prose")
    with pytest.raises(ValidationError):
        StructuredOperation(action="rewrite_section")


async def test_drafter_uses_gpt5_mini_per_d10():
    """D-10: the structure-aware drafter agent defaults to ``gpt-5-mini``
    (with ``JARVIS_AGENT_MODEL`` as the operator override path).

    NOTE: We assert against the SOURCE default rather than the runtime
    ``_AGENT_MODEL`` value because the user's local ``confluence_logic/.env``
    overrides the env var for cost-control (e.g., ``gpt-4o-mini``). The
    contract per D-10 / CLAUDE.md is: the source default MUST be gpt-5-mini
    (within the GPT-5 ceiling), and the source MUST read the env var so
    deployments can override.
    """
    import inspect

    import confluence_logic.agents.structure_aware_drafter as mod

    source = inspect.getsource(mod)
    # Operator-override env var must be referenced.
    assert "JARVIS_AGENT_MODEL" in source, (
        "Drafter must read JARVIS_AGENT_MODEL env var for operator override (D-10)"
    )
    # Source default must be gpt-5-mini (D-10 + CLAUDE.md GPT-5 ceiling).
    assert 'os.getenv("JARVIS_AGENT_MODEL", "gpt-5-mini")' in source, (
        "Drafter source default MUST be gpt-5-mini per D-10 (within GPT-5 ceiling). "
        "Runtime _AGENT_MODEL may be overridden via env var for local cost-control."
    )
    # And the runtime model must NOT exceed the GPT-5 ceiling (no gpt-5-turbo,
    # gpt-5-large, gpt-6, etc.). Anything in {gpt-4o-mini, gpt-5-mini, gpt-5-nano,
    # gpt-5} is acceptable; reject only over-ceiling models.
    runtime = mod._AGENT_MODEL.lower()
    forbidden_substrings = ("turbo", "gpt-5-large", "gpt-6", "gpt-7")
    for bad in forbidden_substrings:
        assert bad not in runtime, (
            f"Runtime model {mod._AGENT_MODEL!r} exceeds CLAUDE.md GPT-5 ceiling "
            f"(forbidden substring: {bad!r})"
        )


async def test_drafter_falls_back_to_skip_when_runner_returns_invalid_output():
    """When ``Runner.run.final_output`` is not a StructuredOperation and cannot
    be validated as one, drafter returns ``action="skip"`` rather than crashing."""
    ast = _make_prose_section_ast(
        section_title="Frameworks", body_text="We use React."
    )
    intent = ChangeIntent(
        instruction="Replace React with Vue", subject="framework",
        target_hint="Frameworks", action="replace",
        old_value="React", new_value="Vue",
    )
    inp = StructureAwareDrafterInput(
        intent=intent,
        page_ast=ast,
        page_meta={"page_id": "p1", "page_title": "Frameworks"},
        transcript_window="Move from React to Vue.",
    )
    # final_output is a junk object — neither a StructuredOperation nor a
    # dict that validates against the schema.
    junk_result = SimpleNamespace(final_output="not a structured op, just a string")
    with patch(
        "confluence_logic.agents.structure_aware_drafter.Runner.run",
        AsyncMock(return_value=junk_result),
    ):
        result = await draft_operation(inp)

    assert isinstance(result, StructuredOperation)
    assert result.action == "skip"
    assert result.reason
