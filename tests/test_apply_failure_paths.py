"""Accept failure paths + regenerate endpoint — Phase 8 / D-07 + D-09.

Three tests:
  1. _direct_apply_change returns error='heading_not_found' when the live page
     has no matching section heading.
  2. _direct_apply_change returns an error mentioning 'not found' when the
     before_content cannot be located in the live HTML.
  3. POST /sessions/{sid}/review/regenerate/{pid} re-drafts against the current
     page and writes back to Supabase via update_proposal_full.
"""
from __future__ import annotations

import asyncio
from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# 1. heading_not_found
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_heading_not_found_returns_specific_error(mock_confluence_connector):
    """_direct_apply_change must return success=False, error='heading_not_found'
    with a 'no longer exists' message when the section is gone from the live page."""
    from confluence_logic.review import api as review_api

    # Live page has a DIFFERENT heading
    mock_confluence_connector.fetch_page_html.return_value = (
        "<h2>Other Section</h2><p>some content</p>"
    )
    mock_confluence_connector.get_page_metadata.return_value = {
        "version": {"number": 3}
    }

    proposal = {
        "change_type": "edit",
        "page_id": "page-1",
        "page_title": "Test Page",
        "section_heading": "Nonexistent Section",
        "before_content": "anything",
        "after_content": "new content",
        "edit_mode": "replace",
        "session_id": "sess-test",
    }

    result = await review_api._direct_apply_change(proposal)

    assert result["success"] is False, f"Expected failure, got: {result}"
    assert result.get("error") == "heading_not_found", (
        f"Expected error='heading_not_found', got error={result.get('error')!r}"
    )
    assert "no longer exists" in (result.get("message") or "").lower(), (
        f"Expected 'no longer exists' in message, got: {result.get('message')!r}"
    )
    mock_confluence_connector.push_update.assert_not_called()


# ---------------------------------------------------------------------------
# 2. text-not-found (targeted-replace fails closed)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_text_not_found_returns_specific_error(mock_confluence_connector, monkeypatch):
    """Targeted-replace strategy that can't locate before_content must FAIL
    CLOSED — return success=False with an error mentioning 'not found' and
    must NOT push an update."""
    from confluence_logic.review import api as review_api

    # Live page contains the heading but DIFFERENT body text — before_content
    # cannot be found on the page.
    mock_confluence_connector.fetch_page_html.return_value = (
        "<h2>Setup</h2><p>Completely different body text, no Python here.</p>"
    )
    mock_confluence_connector.get_page_metadata.return_value = {
        "version": {"number": 5}
    }

    # Stub the LLM-assisted text locator so it can't rescue the lookup.
    monkeypatch.setattr(
        "confluence_logic.review.api._llm_locate_text_on_page",
        AsyncMock(return_value=None),
    )

    proposal = {
        "change_type": "edit",
        "page_id": "page-1",
        "page_title": "Test Page",
        "section_heading": "Setup",
        "before_content": "Python 2 is installed",
        "after_content": "Python 3 is installed",
        "edit_mode": "replace",
        "session_id": "sess-test",
    }

    result = await review_api._direct_apply_change(proposal)

    assert result["success"] is False, f"Expected failure, got: {result}"
    err = (result.get("error") or "").lower()
    assert "not found" in err, (
        f"Expected error containing 'not found', got: {result.get('error')!r}"
    )
    mock_confluence_connector.push_update.assert_not_called()


# ---------------------------------------------------------------------------
# 3. regenerate endpoint replaces in-place
# ---------------------------------------------------------------------------


def test_regenerate_endpoint_replaces_proposal_in_place(
    app, mock_confluence_connector, mock_supabase_store, monkeypatch
):
    """POST /sessions/{sid}/review/regenerate/{pid} must:
      * fetch the live page,
      * re-run the qualifier + drafter,
      * call supabase_store.update_proposal_full,
      * return JSON with status='pending'.
    """
    # Set up the stored proposal that get_proposal_with_intent will return
    stored = {
        "id": "prop-1",
        "session_id": "sess-1",
        "page_id": "pg-1",
        "page_title": "API Runbook",
        "section_heading": "Setup",
        "before_content": "Python 2",
        "after_content": "Python 3",
        "change_type": "edit",
        "rationale": "Upgrade Python",
        "change_summary": "Replace 'Python 2' with 'Python 3' in Setup",
    }
    mock_supabase_store.get_proposal_with_intent.return_value = stored

    # Live page state
    mock_confluence_connector.fetch_page_html.return_value = (
        "<h2>Setup</h2><p>Current text mentions Python 2 here.</p>"
    )
    mock_confluence_connector.get_page_metadata.return_value = {
        "version": {"number": 5},
        "title": "API Runbook",
    }

    # Stub the qualifier + drafter so we don't call the live LLM
    async def _fake_qualifier(_intent, _page):
        return {
            "qualified": True,
            "page_fit_score": 9,
            "old_value_found": True,
            "matched_phrase": "Python 2",
            "why": "test",
        }

    async def _fake_drafter(_intent, _page, _transcript, *, max_page_chars=8000, facts=None, summary_json=None):
        return {
            "change_type": "edit",
            "page_id": "pg-1",
            "page_title": "API Runbook",
            "section_heading": "Setup",
            "before_content": "Python 2",
            "after_content": "Python 3",
            "edit_mode": "replace",
            "rationale": "Re-drafted against current page",
            "change_summary": "Replace 'Python 2' with 'Python 3' in Setup (re-drafted)",
            "risk": "safe",
        }

    monkeypatch.setattr("confluence_logic.agents.page_qualifier._run_page_qualifier", _fake_qualifier)
    monkeypatch.setattr("confluence_logic.agents.drafter_agent._run_intent_drafter", _fake_drafter)

    # update_proposal_full returns the merged row
    mock_supabase_store.update_proposal_full.return_value = {
        "id": "prop-1",
        "status": "pending",
        "change_summary": "Replace 'Python 2' with 'Python 3' in Setup (re-drafted)",
    }

    client = TestClient(app, raise_server_exceptions=False)
    response = client.post(
        "/sessions/sess-1/review/regenerate/prop-1",
        headers={"Authorization": "Bearer fake-token"},
    )

    assert response.status_code == 200, (
        f"Expected 200, got {response.status_code}: {response.text}"
    )

    # Backend must have written the new fields back to Supabase
    assert mock_supabase_store.update_proposal_full.called, (
        "regenerate endpoint did not call supabase_store.update_proposal_full"
    )
    call_args = mock_supabase_store.update_proposal_full.call_args
    proposal_id_arg = call_args.args[0] if call_args.args else call_args.kwargs.get("proposal_id")
    assert proposal_id_arg == "prop-1", (
        f"update_proposal_full called with wrong id: {proposal_id_arg}"
    )

    body = response.json()
    assert body.get("status") == "pending", (
        f"Expected status='pending' in response body, got: {body}"
    )
