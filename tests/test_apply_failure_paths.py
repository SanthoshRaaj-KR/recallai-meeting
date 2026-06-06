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


# ---------------------------------------------------------------------------
# 4. _execute_pipeline_proposal routes single-card Accept through EditorAgent
#    (the same path the batched Accept-all endpoint already uses).
# ---------------------------------------------------------------------------


def _stub_editor_agent(monkeypatch, answer: str):
    """Replace _get_editor_agent() with a stub whose handle_prepared_query
    returns ``answer`` and records the prompt it was given."""
    captured: Dict[str, Any] = {"instruction": None, "calls": 0}

    class _Stub:
        async def handle_prepared_query(self, instruction, **kwargs):
            captured["instruction"] = instruction
            captured["calls"] += 1
            return answer

    monkeypatch.setattr(
        "confluence_logic.review.api._get_editor_agent",
        lambda: _Stub(),
    )
    return captured


@pytest.mark.asyncio
async def test_single_accept_routes_through_editor_agent_by_default(
    monkeypatch, mock_supabase_store
):
    """When JARVIS_PIPELINE_USE_EDITOR_AGENT is on (default), the single-proposal
    Accept handler must delegate to EditorAgent and skip _direct_apply_change."""
    from confluence_logic.review import api as review_api

    monkeypatch.setattr(review_api, "JARVIS_PIPELINE_USE_EDITOR_AGENT", True)

    proposal = {
        "id": "prop-77",
        "session_id": "sess-77",
        "user_id": "user-77",
        "page_id": "page-77",
        "page_title": "Enterprise Customer Feedback",
        "section_heading": "Customer Feedback",
        "before_content": "Microsoft Teams integration is mandatory.",
        "after_content": "Teams integration will be supported.",
        "change_type": "edit",
        "rationale": "Confirm prioritization.",
        "status": "pending",
    }
    mock_supabase_store.get_proposal_by_id = MagicMock(return_value=proposal)
    monkeypatch.setattr(
        "confluence_logic.review.supabase_store.get_proposal_by_id",
        mock_supabase_store.get_proposal_by_id,
    )
    mock_supabase_store.update_proposal_status = MagicMock(return_value=True)
    monkeypatch.setattr(
        "confluence_logic.review.supabase_store.update_proposal_status",
        mock_supabase_store.update_proposal_status,
    )

    captured = _stub_editor_agent(monkeypatch, "Edit applied to Confluence.")

    # _direct_apply_change must NOT be called on the happy path.
    direct_spy = AsyncMock()
    monkeypatch.setattr(review_api, "_direct_apply_change", direct_spy)

    result = await review_api._execute_pipeline_proposal("prop-77", "sess-77")

    assert result["success"] is True, f"Expected success, got: {result}"
    assert captured["calls"] == 1, "EditorAgent.handle_prepared_query was not invoked"
    assert direct_spy.await_count == 0, (
        "_direct_apply_change was called on the happy editor-agent path"
    )
    # The prompt passed to the agent must carry the page title and after_content
    instr = captured["instruction"] or ""
    assert "Enterprise Customer Feedback" in instr, (
        "EditorAgent prompt missing the page title"
    )
    assert "Teams integration will be supported." in instr, (
        "EditorAgent prompt missing the after_content"
    )

    # Status transitions: executing → executed
    statuses = [c.args[1] for c in mock_supabase_store.update_proposal_status.call_args_list]
    assert statuses == ["executing", "executed"], (
        f"Unexpected status transitions: {statuses}"
    )


@pytest.mark.asyncio
async def test_editor_agent_error_falls_back_to_direct_apply(
    monkeypatch, mock_supabase_store
):
    """When the EditorAgent returns an 'ERROR:' answer, the single-proposal
    Accept handler must fall back to _direct_apply_change so we get the best
    of both worlds: agent flexibility AND the REST hardening guarantees."""
    from confluence_logic.review import api as review_api

    monkeypatch.setattr(review_api, "JARVIS_PIPELINE_USE_EDITOR_AGENT", True)

    proposal = {
        "id": "prop-88",
        "session_id": "sess-88",
        "page_id": "page-88",
        "page_title": "API Runbook",
        "section_heading": "Setup",
        "before_content": "Python 2",
        "after_content": "Python 3",
        "change_type": "edit",
        "status": "pending",
    }
    mock_supabase_store.get_proposal_by_id = MagicMock(return_value=proposal)
    monkeypatch.setattr(
        "confluence_logic.review.supabase_store.get_proposal_by_id",
        mock_supabase_store.get_proposal_by_id,
    )
    mock_supabase_store.update_proposal_status = MagicMock(return_value=True)
    monkeypatch.setattr(
        "confluence_logic.review.supabase_store.update_proposal_status",
        mock_supabase_store.update_proposal_status,
    )

    _stub_editor_agent(monkeypatch, "ERROR: The update failed. Some reason.")
    direct_spy = AsyncMock(return_value={"success": True, "message": "Direct path applied."})
    monkeypatch.setattr(review_api, "_direct_apply_change", direct_spy)

    result = await review_api._execute_pipeline_proposal("prop-88", "sess-88")

    assert result["success"] is True, f"Expected fallback success, got: {result}"
    assert direct_spy.await_count == 1, (
        "_direct_apply_change was not invoked as fallback after editor agent ERROR"
    )
    assert result["message"] == "Direct path applied."


@pytest.mark.asyncio
async def test_flag_off_uses_legacy_direct_apply(
    monkeypatch, mock_supabase_store
):
    """When JARVIS_PIPELINE_USE_EDITOR_AGENT is disabled, the legacy direct-REST
    path is used exclusively — no editor agent call."""
    from confluence_logic.review import api as review_api

    monkeypatch.setattr(review_api, "JARVIS_PIPELINE_USE_EDITOR_AGENT", False)

    proposal = {
        "id": "prop-99",
        "session_id": "sess-99",
        "page_id": "page-99",
        "page_title": "Whatever",
        "change_type": "edit",
        "status": "pending",
    }
    mock_supabase_store.get_proposal_by_id = MagicMock(return_value=proposal)
    monkeypatch.setattr(
        "confluence_logic.review.supabase_store.get_proposal_by_id",
        mock_supabase_store.get_proposal_by_id,
    )
    mock_supabase_store.update_proposal_status = MagicMock(return_value=True)
    monkeypatch.setattr(
        "confluence_logic.review.supabase_store.update_proposal_status",
        mock_supabase_store.update_proposal_status,
    )

    captured = _stub_editor_agent(monkeypatch, "should not be reached")
    direct_spy = AsyncMock(return_value={"success": True, "message": "ok"})
    monkeypatch.setattr(review_api, "_direct_apply_change", direct_spy)

    result = await review_api._execute_pipeline_proposal("prop-99", "sess-99")

    assert result["success"] is True
    assert captured["calls"] == 0, "EditorAgent was called despite flag being off"
    assert direct_spy.await_count == 1, "Legacy path was not used"


# ---------------------------------------------------------------------------
# IN-01: additional coverage after the post-review tightening.
# ---------------------------------------------------------------------------


def _wire_proposal(monkeypatch, mock_supabase_store, proposal: Dict[str, Any]) -> MagicMock:
    """Stub supabase_store.get_proposal_by_id + update_proposal_status; return
    a MagicMock whose .call_args_list records every status transition."""
    mock_supabase_store.get_proposal_by_id = MagicMock(return_value=proposal)
    monkeypatch.setattr(
        "confluence_logic.review.supabase_store.get_proposal_by_id",
        mock_supabase_store.get_proposal_by_id,
    )
    status_spy = MagicMock(return_value=True)
    mock_supabase_store.update_proposal_status = status_spy
    monkeypatch.setattr(
        "confluence_logic.review.supabase_store.update_proposal_status",
        status_spy,
    )
    return status_spy


@pytest.mark.asyncio
async def test_executing_status_rejects_double_click(monkeypatch, mock_supabase_store):
    """CR-01: a proposal already in status='executing' must short-circuit so
    a second click never re-enters the path while a prior run is in flight."""
    from confluence_logic.review import api as review_api
    monkeypatch.setattr(review_api, "JARVIS_PIPELINE_USE_EDITOR_AGENT", True)

    _wire_proposal(monkeypatch, mock_supabase_store, {
        "id": "prop-r",
        "session_id": "sess-r",
        "page_id": "p-r",
        "page_title": "Whatever",
        "change_type": "edit",
        "status": "executing",
    })

    result = await review_api._execute_pipeline_proposal("prop-r", "sess-r")
    assert result["success"] is False
    assert "already being applied" in result["message"].lower()


@pytest.mark.asyncio
async def test_exception_mid_flight_transitions_to_failed(monkeypatch, mock_supabase_store):
    """CR-01: if anything between set-executing and the result-check raises,
    the proposal must NOT be stranded in 'executing'. The handler must catch
    and transition to 'failed'."""
    from confluence_logic.review import api as review_api
    monkeypatch.setattr(review_api, "JARVIS_PIPELINE_USE_EDITOR_AGENT", True)

    status_spy = _wire_proposal(monkeypatch, mock_supabase_store, {
        "id": "prop-x",
        "session_id": "sess-x",
        "page_id": "p-x",
        "page_title": "Whatever",
        "change_type": "edit",
        "status": "pending",
    })

    # Force _get_editor_agent to raise — simulates a transient infra failure.
    def _boom():
        raise RuntimeError("simulated infra crash mid-flight")
    monkeypatch.setattr(review_api, "_get_editor_agent", _boom)
    # Also stub direct apply so the fallback doesn't paper over the failure.
    direct_spy = AsyncMock(return_value={"success": False, "message": "direct also failed"})
    monkeypatch.setattr(review_api, "_direct_apply_change", direct_spy)
    # Stub the connector for the starting_version read so it doesn't crash first.
    monkeypatch.setattr(review_api, "_get_connector", lambda: MagicMock(
        get_page_metadata=MagicMock(return_value={"version": {"number": 1}}),
    ))

    result = await review_api._execute_pipeline_proposal("prop-x", "sess-x")

    assert result["success"] is False
    # Status must end in "failed", never stranded in "executing"
    statuses = [c.args[1] for c in status_spy.call_args_list]
    assert statuses[-1] == "failed", (
        f"Status was stranded — full trace: {statuses}"
    )
    assert "executing" in statuses, "Should have transitioned through executing first"


@pytest.mark.asyncio
async def test_version_advance_blocks_direct_apply_fallback(monkeypatch, mock_supabase_store):
    """WR-01: if the page version advanced during the editor agent's run, do
    NOT fall back to _direct_apply_change — that would double-write. Instead
    return a partial_commit_detected error so the user can investigate."""
    from confluence_logic.review import api as review_api
    monkeypatch.setattr(review_api, "JARVIS_PIPELINE_USE_EDITOR_AGENT", True)

    _wire_proposal(monkeypatch, mock_supabase_store, {
        "id": "prop-v",
        "session_id": "sess-v",
        "page_id": "p-v",
        "page_title": "Page",
        "change_type": "edit",
        "before_content": "x",
        "after_content": "y",
        "status": "pending",
    })

    # Connector: starting version=3 BEFORE editor agent; version=4 AFTER (advanced).
    metadata_calls = {"n": 0}
    def _get_meta(_page_id):
        metadata_calls["n"] += 1
        return {"version": {"number": 3 if metadata_calls["n"] == 1 else 4}}
    fake_connector = MagicMock(get_page_metadata=_get_meta)
    monkeypatch.setattr(review_api, "_get_connector", lambda: fake_connector)

    # Editor agent fails — should normally fall back to direct apply
    _stub_editor_agent(monkeypatch, "ERROR: heading not found mid-run")
    direct_spy = AsyncMock(return_value={"success": True, "message": "should NOT be reached"})
    monkeypatch.setattr(review_api, "_direct_apply_change", direct_spy)

    result = await review_api._execute_pipeline_proposal("prop-v", "sess-v")

    # Version advanced → fallback refused → direct apply MUST NOT run
    assert direct_spy.await_count == 0, (
        "Fallback to _direct_apply_change ran despite the version advancing — "
        "this is the double-write hazard the WR-01 fix is supposed to prevent."
    )
    assert result["success"] is False
    assert result["message"].lower().count("page was modified") >= 1 or "partial" in result["message"].lower()


@pytest.mark.asyncio
async def test_compound_failure_both_paths_fail(monkeypatch, mock_supabase_store):
    """IN-01: if both editor agent AND direct apply fail, the response message
    must surface the direct-apply error (most actionable) and status must
    settle on 'failed'."""
    from confluence_logic.review import api as review_api
    monkeypatch.setattr(review_api, "JARVIS_PIPELINE_USE_EDITOR_AGENT", True)

    status_spy = _wire_proposal(monkeypatch, mock_supabase_store, {
        "id": "prop-c",
        "session_id": "sess-c",
        "page_id": "p-c",
        "page_title": "Page",
        "change_type": "edit",
        "before_content": "x",
        "after_content": "y",
        "status": "pending",
    })

    # Connector returns same version on both calls — no version-advance,
    # so fallback IS allowed to run.
    monkeypatch.setattr(review_api, "_get_connector", lambda: MagicMock(
        get_page_metadata=MagicMock(return_value={"version": {"number": 5}}),
    ))

    _stub_editor_agent(monkeypatch, "I couldn't find the section.")
    direct_spy = AsyncMock(return_value={
        "success": False,
        "error": "heading_not_found",
        "message": "Section 'X' no longer exists.",
    })
    monkeypatch.setattr(review_api, "_direct_apply_change", direct_spy)

    result = await review_api._execute_pipeline_proposal("prop-c", "sess-c")

    assert result["success"] is False
    assert direct_spy.await_count == 1, "Fallback should have run when version unchanged"
    # User sees the actionable direct-apply message (not the editor agent's vague answer)
    assert "no longer exists" in result["message"].lower() or "section" in result["message"].lower()
    statuses = [c.args[1] for c in status_spy.call_args_list]
    assert statuses[-1] == "failed"


@pytest.mark.asyncio
async def test_meeting_context_threaded_to_editor_agent(monkeypatch, mock_supabase_store):
    """WR-10: the single-card Accept path must pass meeting_context to the
    editor agent (the same way _execute_single_change does) so the agent can
    disambiguate references like 'the page we were just discussing'."""
    from confluence_logic.review import api as review_api
    monkeypatch.setattr(review_api, "JARVIS_PIPELINE_USE_EDITOR_AGENT", True)

    _wire_proposal(monkeypatch, mock_supabase_store, {
        "id": "prop-m",
        "session_id": "sess-m",
        "page_id": "p-m",
        "page_title": "Page",
        "change_type": "edit",
        "before_content": "x",
        "after_content": "y",
        "status": "pending",
    })
    monkeypatch.setattr(review_api, "_get_connector", lambda: MagicMock(
        get_page_metadata=MagicMock(return_value={"version": {"number": 1}}),
    ))

    captured = {"kwargs": None}

    class _Stub:
        async def handle_prepared_query(self, instruction, **kwargs):
            captured["kwargs"] = kwargs
            return "Done."

    monkeypatch.setattr(review_api, "_get_editor_agent", lambda: _Stub())

    result = await review_api._execute_pipeline_proposal("prop-m", "sess-m")
    assert result["success"] is True
    assert "meeting_context" in (captured["kwargs"] or {}), (
        "meeting_context kwarg not threaded through to handle_prepared_query"
    )


@pytest.mark.asyncio
async def test_editor_agent_partial_success_recognized(monkeypatch, mock_supabase_store):
    """WR-06: the failure detection must catch the various 'I can't / I couldn't'
    phrasings the LLM produces under load — not just the literal 'ERROR:' prefix."""
    from confluence_logic.review import api as review_api
    monkeypatch.setattr(review_api, "JARVIS_PIPELINE_USE_EDITOR_AGENT", True)

    _wire_proposal(monkeypatch, mock_supabase_store, {
        "id": "prop-p",
        "session_id": "sess-p",
        "page_id": "p-p",
        "page_title": "Page",
        "change_type": "edit",
        "before_content": "x",
        "after_content": "y",
        "status": "pending",
    })
    monkeypatch.setattr(review_api, "_get_connector", lambda: MagicMock(
        get_page_metadata=MagicMock(return_value={"version": {"number": 1}}),
    ))

    _stub_editor_agent(monkeypatch, "Unable to apply the change because the heading is gone.")
    direct_spy = AsyncMock(return_value={"success": True, "message": "Direct path applied."})
    monkeypatch.setattr(review_api, "_direct_apply_change", direct_spy)

    result = await review_api._execute_pipeline_proposal("prop-p", "sess-p")

    assert direct_spy.await_count == 1, (
        "Fallback did NOT trigger for an 'Unable to …' answer — the failure "
        "detection still misses common error phrasings (WR-06)."
    )
    assert result["success"] is True
    assert result["message"] == "Direct path applied."
