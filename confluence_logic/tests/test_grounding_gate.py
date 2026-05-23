"""GREEN tests for GroundingGate (PROP-V2-01, Phase 10 Plan 03).

The GroundingGate (D-04) is the hard hallucination gate. Two checks:
(1) page_id existence in the user's confluence_page_graph or via live
REST GET, and (2) content-bearing token containment of after_content
in ``{transcript ∪ current_page_content}`` per operation type.

Wave 0 RED scaffolds (Plan 10-01) imported these symbols and called
``pytest.fail("Wave 0 RED")``. Wave 1 (Plan 10-03) replaced each body
with concrete assertions covering the seven behaviors from
``10-03-PLAN.md``.
"""
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

pytestmark = pytest.mark.asyncio

from confluence_logic.agents.grounding_gate import (
    check_grounding,
    check_page_existence,
    content_bearing_tokens,
)


async def test_drops_card_when_page_id_missing():
    """PROP-V2-01: a card whose page_id is not in confluence_page_graph AND
    fails the live REST fallback is DROPPED (check_page_existence → False)."""
    fake_connector = MagicMock()
    # REST fallback also misses (raise to simulate 404 / not-found).
    fake_connector.get_page_metadata.side_effect = Exception("404 Not Found")

    with patch(
        "confluence_logic.confluence_page_graph.list_user_confluence_pages",
        new=AsyncMock(return_value=[]),
    ):
        exists = await check_page_existence(
            "p_missing", "user@example.com", connector=fake_connector
        )

    assert exists is False, "expected check_page_existence to return False when both graph and REST miss"


async def test_drops_card_when_new_token_introduced():
    """PROP-V2-01: an additive card whose after_content contains a
    content-bearing token absent from {transcript ∪ page_content} is DROPPED."""
    card = {
        "change_type": "edit",
        "edit_mode": "replace",
        "after_content": "use Kubernetes for scheduling",
        "page_id": "p1",
    }
    result = await check_grounding(
        card,
        transcript_text="we should switch to Vue 3",
        current_page_content="We use React",
    )
    assert result["ok"] is False
    assert "kubernetes" in result["failures"]
    assert "scheduling" in result["failures"]
    assert result["reason"], "expected non-empty reason string on drop"


async def test_drops_card_when_old_text_not_on_page():
    """PROP-V2-01: a replace.old_text / delete card whose old-text tokens
    are NOT all present in current_page_content is DROPPED."""
    card = {
        "change_type": "edit",
        "edit_mode": "replace",
        "before_content": "Python 2.7",
        "after_content": "Python 3.12",
        "page_id": "p1",
    }
    result = await check_grounding(
        card,
        transcript_text="upgrade to 3.12",
        current_page_content="We use Python 3.10",
    )
    assert result["ok"] is False
    # The version token "2.7" is the hallucinated needle that triggers the drop.
    assert "2.7" in result["failures"]
    assert "old_text" in result["reason"] or "current_page_content" in result["reason"]


async def test_passes_when_after_content_token_subset_of_transcript_union_page():
    """PROP-V2-01: a card whose after_content tokens are all members of
    {transcript ∪ page_content} passes the gate unmodified."""
    card = {
        "change_type": "edit",
        "edit_mode": "replace",
        "after_content": "use Vue 3 instead of React",
        "page_id": "p1",
    }
    result = await check_grounding(
        card,
        transcript_text="we should switch to Vue 3",
        current_page_content="We use React for the frontend",
    )
    assert result["ok"] is True
    assert result["failures"] == []


async def test_reorder_drops_if_new_token_introduced():
    """PROP-V2-01: a reorder op must not introduce ANY new content-bearing
    token in the moved nodes; if it does, the card is DROPPED. The
    sibling pass case (just shuffling) must NOT trip the gate."""
    # Pass case — same tokens shuffled.
    pass_card = {
        "operation_type": "reorder",
        "before_content": "Login\nPayment\nDashboard",
        "after_content": "Payment\nLogin\nDashboard",
        "page_id": "p1",
    }
    pass_result = await check_grounding(pass_card, "", "")
    assert pass_result["ok"] is True
    assert pass_result["failures"] == []

    # Fail case — "SignIn" is a fabricated token never present in before.
    fail_card = {
        "operation_type": "reorder",
        "before_content": "Login\nPayment",
        "after_content": "SignIn\nPayment",
        "page_id": "p1",
    }
    fail_result = await check_grounding(fail_card, "", "")
    assert fail_result["ok"] is False
    assert "signin" in fail_result["failures"]
    assert "reorder" in fail_result["reason"]


async def test_check_page_existence_falls_back_to_REST_when_graph_misses():
    """PROP-V2-01: ``check_page_existence`` falls back to ConfluenceConnector
    REST GET when the graph lookup misses; a 200-style response (dict with
    ``id`` key) satisfies existence.

    Also covers the graph-hit short-circuit path: when the graph already
    contains the page_id, REST is not consulted at all and existence is True.
    """
    # Case A: graph HIT — REST must not even be called.
    hit_connector = MagicMock()
    hit_connector.get_page_metadata.side_effect = AssertionError(
        "REST must not be called when graph hits"
    )
    with patch(
        "confluence_logic.confluence_page_graph.list_user_confluence_pages",
        new=AsyncMock(return_value=[{"page_id": "p123", "title": "X"}]),
    ):
        exists_hit = await check_page_existence(
            "p123", "user@example.com", connector=hit_connector
        )
    assert exists_hit is True

    # Case B: graph MISS, REST returns {"id": "p999"} → exists.
    rest_connector = MagicMock()
    rest_connector.get_page_metadata.return_value = {"id": "p999", "title": "Y"}
    with patch(
        "confluence_logic.confluence_page_graph.list_user_confluence_pages",
        new=AsyncMock(return_value=[]),
    ):
        exists_rest = await check_page_existence(
            "p999", "user@example.com", connector=rest_connector
        )
    assert exists_rest is True
    rest_connector.get_page_metadata.assert_called_once_with("p999")


async def test_content_bearing_tokens_keeps_numbers_camelcase_snakecase_uppercase_strips_stopwords():
    """PROP-V2-01: content_bearing_tokens() keeps numbers, camelCase,
    snake_case, and UPPERCASE identifiers; strips ``the/a/an/of/to/...``
    stopwords; preserves hyphen-joined and dot-joined identifiers as
    single tokens."""
    # Numbers + identifier + date — strips "the","is".
    out_1 = content_bearing_tokens("The Q3 deadline is 2024-05-30")
    assert "q3" in out_1
    assert "deadline" in out_1
    assert "2024-05-30" in out_1   # hyphen-joined date stays atomic
    assert "the" not in out_1
    assert "is" not in out_1

    # camelCase / snake_case / UPPER kept (lowercased).
    out_2 = content_bearing_tokens(
        "camelCaseName and SNAKE_CASE and CONST_NAME"
    )
    assert "camelcasename" in out_2
    assert "snake_case" in out_2
    assert "const_name" in out_2
    assert "and" not in out_2     # stopword stripped

    # Dot-joined version stays atomic.
    out_3 = content_bearing_tokens("running v1.2.3 in prod")
    assert "v1.2.3" in out_3      # dot-joined stays one token
    assert "in" not in out_3      # stopword
