"""Tests for the Phase 10 Regenerate endpoint (PROP-V2-05).

Phase 10 — auto-propose-pipeline-quality-redesign-v2.

The Phase 10 regenerate endpoint (D-08) re-runs the structure-aware
drafter against the LIVE Confluence page (forced REST fetch, bypassing
RAG cache for the affected page only) and replaces the proposal in
Supabase ONLY if the new card passes the full Phase 10 grounding gate.
Three contract tests:

  1. test_regenerate_replaces_proposal_in_place — passing gate → in-place
     update_proposal_full call with the new fields.
  2. test_regenerate_invokes_grounding_gate — failing gate → original
     preserved (update_proposal_full NOT called).
  3. test_regenerate_bypasses_graph_cache_for_affected_page_only —
     refresh_page_in_graph called exactly once with this session's
     graph_user_id + this proposal's page_id.

Tests call ``api.regenerate_proposal`` directly (unit-style, no HTTP)
so we can sidestep the bearer-auth + Supabase user-lookup path. The
endpoint's auth gate is exercised by separate integration tests; here
we only pin the gate-and-persist contract.
"""
from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

pytestmark = pytest.mark.asyncio

# Importing the review API module and the Phase 10 helpers; if any of
# these fail to import the test suite itself fails-fast at collection.
from confluence_logic.review import api as review_api
from confluence_logic.agents.grounding_gate import check_grounding  # noqa: F401
from confluence_logic.agents.structure_aware_drafter import (  # noqa: F401
    draft_operation,
)


_SESSION_ID = "sess-regen-01"
_PROPOSAL_ID = "00000000-0000-0000-0000-0000000000aa"
_USER_ID = "user-regen-01"
_PAGE_ID = "page-regen-99"
_BEARER = "Bearer fake-token-xxx"


def _make_original_proposal() -> Dict[str, Any]:
    """Return a Supabase-shaped row for the proposal we are regenerating."""
    return {
        "id": _PROPOSAL_ID,
        "session_id": _SESSION_ID,
        "user_id": _USER_ID,
        "page_id": _PAGE_ID,
        "page_title": "Frameworks",
        "section_heading": "Frontend",
        "change_type": "edit",
        "before_content": "we use React for the frontend",
        "after_content": "we use Vue for the frontend",
        "rationale": "Meeting decided to switch React → Vue",
        "change_summary": "Replace React with Vue on Frameworks > Frontend",
    }


def _make_new_draft(**overrides: Any) -> Dict[str, Any]:
    """Return a drafter-output dict (what _run_intent_drafter returns)."""
    draft = {
        "change_type": "edit",
        "page_title": "Frameworks",
        "section_heading": "Frontend",
        "before_content": "we use React for the frontend",
        "after_content": "we use Vue for the frontend",
        "rationale": "Updated against live page",
        "change_summary": "Replace React with Vue on Frameworks > Frontend",
        "edit_mode": "replace",
        "risk": "safe",
    }
    draft.update(overrides)
    return draft


def _make_qualified_enriched_page() -> Dict[str, Any]:
    """Return what _enrich_page_for_drafter is mocked to return."""
    return {
        "page_id": _PAGE_ID,
        "title": "Frameworks",
        "page_title": "Frameworks",
        "space_key": "ENG",
        "full_content": "we use React for the frontend",
        "available_headings": ["Frontend"],
        "section_content_map": {"Frontend": "we use React for the frontend"},
        "_drafter_ready": True,
    }


async def test_regenerate_replaces_proposal_in_place():
    """PROP-V2-05: on gate pass, update_proposal_full IS called with the new
    card and the SAME proposal_id (in-place replacement, not insert)."""
    original = _make_original_proposal()
    new_draft = _make_new_draft()
    updated_row = {**original, "after_content": new_draft["after_content"]}

    fake_connector = MagicMock()
    fake_connector.fetch_page_html = MagicMock(
        return_value="<p>we use React for the frontend</p>",
    )
    fake_connector.get_page_metadata = MagicMock(return_value={"id": _PAGE_ID})

    with patch.object(
        review_api.supabase_store,
        "user_from_bearer",
        return_value={"id": _USER_ID},
    ), patch.object(
        review_api.supabase_store,
        "get_proposal_with_intent",
        return_value=original,
    ), patch.object(
        review_api.supabase_store,
        "get_history_item",
        return_value={},
    ), patch.object(
        review_api.supabase_store,
        "update_proposal_full",
        return_value=updated_row,
    ) as mock_update, patch.object(
        review_api, "_get_connector", return_value=fake_connector,
    ), patch.object(
        review_api.confluence_page_graph,
        "refresh_page_in_graph",
        new=AsyncMock(return_value=True),
    ), patch.object(
        review_api, "_confluence_graph_user_id", return_value=_USER_ID,
    ), patch.object(
        review_api, "_enrich_page_for_drafter",
        new=AsyncMock(return_value=_make_qualified_enriched_page()),
    ), patch(
        "confluence_logic.agents.page_qualifier._run_page_qualifier",
        new=AsyncMock(return_value={"qualified": True, "page_fit_score": 9}),
    ), patch(
        "confluence_logic.agents.drafter_agent._run_intent_drafter",
        new=AsyncMock(return_value=new_draft),
    ), patch.object(
        review_api, "check_page_existence",
        new=AsyncMock(return_value=True),
    ), patch.object(
        review_api, "check_grounding",
        new=AsyncMock(return_value={"ok": True, "failures": [], "reason": ""}),
    ):
        result = await review_api.regenerate_proposal(
            session_id=_SESSION_ID,
            proposal_id=_PROPOSAL_ID,
            authorization=_BEARER,
        )

    # update_proposal_full MUST have been called with the proposal_id and
    # contain the regenerated fields. This proves in-place replacement
    # (not insert) — the same UUID is patched, not a new row.
    assert mock_update.call_count == 1
    call_args, _ = mock_update.call_args, mock_update.call_args.kwargs
    pid_arg = call_args.args[0]
    fields_arg = call_args.args[1]
    assert pid_arg == _PROPOSAL_ID
    assert fields_arg["after_content"] == new_draft["after_content"]
    assert fields_arg["status"] == "pending"
    # Response surface contract: success path returns a dict that merges the
    # original proposal + updated_fields (regenerate_available is True so
    # the user can retry).
    assert isinstance(result, dict)
    assert result.get("regenerate_available") is True


async def test_regenerate_invokes_grounding_gate():
    """PROP-V2-05: when check_grounding returns ok=False, the original card
    is preserved and update_proposal_full IS NOT called."""
    original = _make_original_proposal()
    new_draft = _make_new_draft(
        # Introduce a token never said in the meeting and not on the page.
        after_content="we use SvelteKit for the frontend",
        change_summary="Replace React with SvelteKit",
    )

    fake_connector = MagicMock()
    fake_connector.fetch_page_html = MagicMock(
        return_value="<p>we use React for the frontend</p>",
    )
    fake_connector.get_page_metadata = MagicMock(return_value={"id": _PAGE_ID})

    with patch.object(
        review_api.supabase_store,
        "user_from_bearer",
        return_value={"id": _USER_ID},
    ), patch.object(
        review_api.supabase_store,
        "get_proposal_with_intent",
        return_value=original,
    ), patch.object(
        review_api.supabase_store,
        "get_history_item",
        return_value={},
    ), patch.object(
        review_api.supabase_store,
        "update_proposal_full",
    ) as mock_update, patch.object(
        review_api, "_get_connector", return_value=fake_connector,
    ), patch.object(
        review_api.confluence_page_graph,
        "refresh_page_in_graph",
        new=AsyncMock(return_value=True),
    ), patch.object(
        review_api, "_confluence_graph_user_id", return_value=_USER_ID,
    ), patch.object(
        review_api, "_enrich_page_for_drafter",
        new=AsyncMock(return_value=_make_qualified_enriched_page()),
    ), patch(
        "confluence_logic.agents.page_qualifier._run_page_qualifier",
        new=AsyncMock(return_value={"qualified": True, "page_fit_score": 9}),
    ), patch(
        "confluence_logic.agents.drafter_agent._run_intent_drafter",
        new=AsyncMock(return_value=new_draft),
    ), patch.object(
        review_api, "check_page_existence",
        new=AsyncMock(return_value=True),
    ), patch.object(
        review_api, "check_grounding",
        new=AsyncMock(return_value={
            "ok": False,
            "failures": ["sveltekit"],
            "reason": "after_content tokens missing from {transcript U page}",
        }),
    ):
        result = await review_api.regenerate_proposal(
            session_id=_SESSION_ID,
            proposal_id=_PROPOSAL_ID,
            authorization=_BEARER,
        )

    # CRITICAL: gate failure must NOT overwrite the original row.
    assert mock_update.call_count == 0, (
        "update_proposal_full was called despite the grounding gate failing — "
        "the regenerate endpoint must preserve the original card on gate failure"
    )
    # Response contract: kept_original=True + grounding_failures surfaces
    # the offending tokens to the UI.
    assert isinstance(result, dict)
    assert result.get("kept_original") is True
    assert result.get("success") is False
    assert "sveltekit" in result.get("grounding_failures", [])


async def test_regenerate_bypasses_graph_cache_for_affected_page_only():
    """PROP-V2-05: regenerate calls refresh_page_in_graph(user, page_id)
    exactly once with THIS proposal's page_id — not for any other page."""
    original = _make_original_proposal()
    new_draft = _make_new_draft()

    fake_connector = MagicMock()
    fake_connector.fetch_page_html = MagicMock(
        return_value="<p>we use React for the frontend</p>",
    )
    fake_connector.get_page_metadata = MagicMock(return_value={"id": _PAGE_ID})

    mock_refresh = AsyncMock(return_value=True)

    with patch.object(
        review_api.supabase_store,
        "user_from_bearer",
        return_value={"id": _USER_ID},
    ), patch.object(
        review_api.supabase_store,
        "get_proposal_with_intent",
        return_value=original,
    ), patch.object(
        review_api.supabase_store,
        "get_history_item",
        return_value={},
    ), patch.object(
        review_api.supabase_store,
        "update_proposal_full",
        return_value={**original, "after_content": new_draft["after_content"]},
    ), patch.object(
        review_api, "_get_connector", return_value=fake_connector,
    ), patch.object(
        review_api.confluence_page_graph,
        "refresh_page_in_graph",
        new=mock_refresh,
    ), patch.object(
        review_api, "_confluence_graph_user_id", return_value=_USER_ID,
    ), patch.object(
        review_api, "_enrich_page_for_drafter",
        new=AsyncMock(return_value=_make_qualified_enriched_page()),
    ), patch(
        "confluence_logic.agents.page_qualifier._run_page_qualifier",
        new=AsyncMock(return_value={"qualified": True, "page_fit_score": 9}),
    ), patch(
        "confluence_logic.agents.drafter_agent._run_intent_drafter",
        new=AsyncMock(return_value=new_draft),
    ), patch.object(
        review_api, "check_page_existence",
        new=AsyncMock(return_value=True),
    ), patch.object(
        review_api, "check_grounding",
        new=AsyncMock(return_value={"ok": True, "failures": [], "reason": ""}),
    ):
        await review_api.regenerate_proposal(
            session_id=_SESSION_ID,
            proposal_id=_PROPOSAL_ID,
            authorization=_BEARER,
        )

    # The graph cache must be bust for THIS page only — exactly one call,
    # exactly this user_id + page_id.
    assert mock_refresh.await_count == 1, (
        "refresh_page_in_graph should be called exactly once per regenerate"
    )
    refresh_args = mock_refresh.await_args
    # Two positional args: (user_id, page_id)
    assert refresh_args.args == (_USER_ID, _PAGE_ID), (
        f"refresh_page_in_graph called with {refresh_args.args!r} but expected "
        f"({_USER_ID!r}, {_PAGE_ID!r})"
    )
