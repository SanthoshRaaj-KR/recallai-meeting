"""Wave 0 RED test scaffold for GroundingGate (PROP-V2-01).

Phase 10 — auto-propose-pipeline-quality-redesign-v2.

The GroundingGate (D-04) is the hard hallucination gate: it enforces
(1) page_id existence in the user's confluence_page_graph or via live
REST GET, and (2) content-bearing token containment of after_content
in ``{transcript ∪ current_page_content}`` (subset rule per
operation type). Cards failing either check are DROPPED, never
downgraded. Wave 1 creates
`confluence_logic/agents/grounding_gate.py`; until then this import
fails RED.
"""
import pytest

pytestmark = pytest.mark.asyncio

from confluence_logic.agents.grounding_gate import (  # noqa: F401, E402
    check_grounding,
    check_page_existence,
    content_bearing_tokens,
)


async def test_drops_card_when_page_id_missing():
    """PROP-V2-01: a card whose page_id is not in confluence_page_graph AND fails the live REST fallback is DROPPED."""
    pytest.fail("Wave 0 RED — GroundingGate implementation pending (Wave 1)")


async def test_drops_card_when_new_token_introduced():
    """PROP-V2-01: an additive card (replace.new_text / insert / add / create) whose after_content contains a content-bearing token absent from {transcript ∪ page_content} is DROPPED."""
    pytest.fail("Wave 0 RED — GroundingGate implementation pending (Wave 1)")


async def test_drops_card_when_old_text_not_on_page():
    """PROP-V2-01: a replace.old_text / delete card whose old-text tokens are NOT all present in current_page_content is DROPPED."""
    pytest.fail("Wave 0 RED — GroundingGate implementation pending (Wave 1)")


async def test_passes_when_after_content_token_subset_of_transcript_union_page():
    """PROP-V2-01: a card whose after_content tokens are all members of {transcript ∪ page_content} passes the gate unmodified."""
    pytest.fail("Wave 0 RED — GroundingGate implementation pending (Wave 1)")


async def test_reorder_drops_if_new_token_introduced():
    """PROP-V2-01: a reorder op must not introduce ANY new content-bearing token in the moved nodes; if it does, the card is DROPPED."""
    pytest.fail("Wave 0 RED — GroundingGate implementation pending (Wave 1)")


async def test_check_page_existence_falls_back_to_REST_when_graph_misses():
    """PROP-V2-01: ``check_page_existence`` falls back to ConfluenceConnector REST GET when the graph lookup misses; a 200 response satisfies existence."""
    pytest.fail("Wave 0 RED — GroundingGate implementation pending (Wave 1)")


async def test_content_bearing_tokens_keeps_numbers_camelcase_snakecase_uppercase_strips_stopwords():
    """PROP-V2-01: content_bearing_tokens() keeps numbers, camelCase, snake_case, and UPPERCASE identifiers; strips ``the/a/an/of/to/...`` stopwords; preserves hyphen-joined and dot-joined identifiers as single tokens."""
    pytest.fail("Wave 0 RED — GroundingGate implementation pending (Wave 1)")
