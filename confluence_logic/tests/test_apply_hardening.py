"""Tests for Phase 5 safe-apply hardening (APPLY-01, APPLY-02, APPLY-03)."""
import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch, call

from confluence_logic.review import api


# ── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_connector():
    """Return a MagicMock ConfluenceConnector with pre-configured stubs."""
    connector = MagicMock()
    connector.fetch_page_html.return_value = "<h2>Goals</h2><p>Content</p>"
    connector.get_page_metadata.return_value = {"version": {"number": 5}, "title": "Test Page"}
    connector.push_update.return_value = True
    return connector


@pytest.fixture
def base_edit_proposal():
    """Return a minimal edit proposal dict."""
    return {
        "change_type": "edit",
        "page_id": "page-123",
        "page_title": "Test Page",
        "section_heading": "Goals",
        "after_content": "Updated content",
        "before_content": "",
        "edit_mode": "append",
        "session_id": "sess-abc",
    }


@pytest.fixture
def base_delete_proposal():
    """Return a minimal delete proposal dict."""
    return {
        "change_type": "delete",
        "page_id": "page-123",
        "page_title": "Test Page",
        "section_heading": "Goals",
        "after_content": "",
        "before_content": "",
        "edit_mode": "append",
        "session_id": "sess-abc",
    }


# ── APPLY-01: Pre-flight heading check ──────────────────────────────────────

@pytest.mark.asyncio
async def test_edit_card_missing_heading_returns_heading_not_found(mock_connector, base_edit_proposal):
    """APPLY-01: edit card with a section heading that does not exist in the live page
    must return success=False with error='heading_not_found' and must NOT call push_update."""
    proposal = {**base_edit_proposal, "section_heading": "Nonexistent Section"}
    mock_connector.fetch_page_html.return_value = "<h2>Goals</h2><p>Content</p>"

    with patch.object(api, "_get_connector", return_value=mock_connector), \
         patch.object(api, "_resolve_page_id", new=AsyncMock(return_value="page-123")):
        result = await api._direct_apply_change(proposal)

    assert result["success"] is False
    assert result.get("error") == "heading_not_found"
    assert "Nonexistent Section" in result.get("message", "")
    mock_connector.push_update.assert_not_called()


@pytest.mark.asyncio
async def test_edit_card_present_heading_proceeds_to_push_update(mock_connector, base_edit_proposal):
    """APPLY-01: edit card whose section_heading is present in the live page must
    call push_update exactly once and return success=True."""
    proposal = {**base_edit_proposal, "section_heading": "Goals", "edit_mode": "append", "before_content": ""}

    with patch.object(api, "_get_connector", return_value=mock_connector), \
         patch.object(api, "_resolve_page_id", new=AsyncMock(return_value="page-123")):
        result = await api._direct_apply_change(proposal)

    assert result["success"] is True
    mock_connector.push_update.assert_called_once()


@pytest.mark.asyncio
async def test_delete_card_missing_heading_returns_heading_not_found(mock_connector, base_delete_proposal):
    """APPLY-01: delete card targeting a section heading that is absent from the live page
    must return success=False with error='heading_not_found' and must NOT call push_update."""
    proposal = {**base_delete_proposal, "section_heading": "Missing Section"}
    mock_connector.fetch_page_html.return_value = "<h2>Goals</h2><p>Content</p>"

    with patch.object(api, "_get_connector", return_value=mock_connector), \
         patch.object(api, "_resolve_page_id", new=AsyncMock(return_value="page-123")):
        result = await api._direct_apply_change(proposal)

    assert result["success"] is False
    assert result.get("error") == "heading_not_found"
    mock_connector.push_update.assert_not_called()


@pytest.mark.asyncio
async def test_delete_card_present_heading_proceeds(mock_connector, base_delete_proposal):
    """APPLY-01: delete card whose section heading exists in the live page must succeed."""
    proposal = {**base_delete_proposal, "section_heading": "Goals"}

    with patch.object(api, "_get_connector", return_value=mock_connector), \
         patch.object(api, "_resolve_page_id", new=AsyncMock(return_value="page-123")), \
         patch("confluence_logic.utils.html_parser.delete_content_in_section", return_value="<p>empty</p>"):
        result = await api._direct_apply_change(proposal)

    assert result["success"] is True


@pytest.mark.asyncio
async def test_create_card_skips_preflight_heading_check(mock_connector, base_edit_proposal):
    """APPLY-01: create cards must NOT be blocked by the pre-flight heading check —
    there is no section to verify for brand-new pages."""
    proposal = {
        **base_edit_proposal,
        "change_type": "create",
        "section_heading": "Goals",
        "after_content": "New content",
    }
    mock_connector.search_pages.return_value = []
    mock_connector.create_page.return_value = {"id": "new-page-id"}

    with patch.object(api, "_get_connector", return_value=mock_connector), \
         patch.object(api, "_generate_page_content", new=AsyncMock(return_value="<p>New content</p>")):
        result = await api._direct_apply_change(proposal)

    assert result["success"] is True
    mock_connector.push_update.assert_not_called()


@pytest.mark.asyncio
async def test_title_card_skips_preflight_heading_check(mock_connector, base_edit_proposal):
    """APPLY-01: title-rename cards must NOT be blocked by the pre-flight heading check —
    they target the page title, not a section heading."""
    proposal = {
        **base_edit_proposal,
        "change_type": "title",
        "after_content": "New Title",
        "section_heading": "Goals",
    }

    with patch.object(api, "_get_connector", return_value=mock_connector), \
         patch.object(api, "_resolve_page_id", new=AsyncMock(return_value="page-123")):
        result = await api._direct_apply_change(proposal)

    assert result["success"] is True
    mock_connector.push_update.assert_called_once()


# ── APPLY-02: Version chain ──────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_version_cache_populated_after_successful_commit(mock_connector, base_edit_proposal):
    """APPLY-02: after a successful commit, _version_cache[(session_id, page_id)]
    must be set to current_version + 1."""
    api._version_cache.clear()

    proposal = {
        **base_edit_proposal,
        "session_id": "sess-version",
        "page_id": "page-v1",
        "edit_mode": "append",
        "section_heading": "Goals",
    }
    mock_connector.get_page_metadata.return_value = {"version": {"number": 3}, "title": "Test Page"}

    with patch.object(api, "_get_connector", return_value=mock_connector), \
         patch.object(api, "_resolve_page_id", new=AsyncMock(return_value="page-v1")):
        result = await api._direct_apply_change(proposal, session_id="sess-version")

    assert result["success"] is True
    assert api._version_cache.get(("sess-version", "page-v1")) == 4


@pytest.mark.asyncio
async def test_second_accept_uses_cached_version_as_expected_version(mock_connector, base_edit_proposal):
    """APPLY-02: when a cached version is present for (session_id, page_id), _direct_apply_change
    must pass that cached version as the expected_version argument to push_update."""
    api._version_cache[("sess-seq", "page-seq")] = 7

    proposal = {
        **base_edit_proposal,
        "session_id": "sess-seq",
        "page_id": "page-seq",
        "section_heading": "Goals",
        "edit_mode": "append",
    }

    with patch.object(api, "_get_connector", return_value=mock_connector), \
         patch.object(api, "_resolve_page_id", new=AsyncMock(return_value="page-seq")):
        await api._direct_apply_change(proposal)

    call_args = mock_connector.push_update.call_args
    # Extract expected_version from positional or keyword args
    if call_args.args and len(call_args.args) >= 3:
        expected_version_passed = call_args.args[2]
    else:
        expected_version_passed = call_args.kwargs.get("expected_version")
    assert expected_version_passed == 7


@pytest.mark.asyncio
async def test_version_conflict_invalidates_cache_and_returns_error(mock_connector, base_edit_proposal):
    """APPLY-02: when push_update raises a version-conflict ValueError, _direct_apply_change
    must return success=False with error='version_conflict' and must remove the stale cache entry."""
    api._version_cache[("sess-conflict", "page-conflict")] = 9
    mock_connector.push_update.side_effect = ValueError("Version Conflict: Expected 9, live is 10")

    proposal = {
        **base_edit_proposal,
        "session_id": "sess-conflict",
        "page_id": "page-conflict",
        "section_heading": "Goals",
        "edit_mode": "append",
    }

    with patch.object(api, "_get_connector", return_value=mock_connector), \
         patch.object(api, "_resolve_page_id", new=AsyncMock(return_value="page-conflict")):
        result = await api._direct_apply_change(proposal)

    assert result["success"] is False
    assert result.get("error") == "version_conflict" or "version_conflict" in str(result)
    assert ("sess-conflict", "page-conflict") not in api._version_cache


# ── APPLY-03: Re-index after commit ─────────────────────────────────────────

@pytest.mark.asyncio
async def test_successful_commit_fires_reindex_create_task(mock_connector, base_edit_proposal):
    """APPLY-03: after a successful commit, _direct_apply_change must call
    asyncio.create_task() to schedule fire-and-forget re-indexing."""
    proposal = {**base_edit_proposal, "section_heading": "Goals", "edit_mode": "append"}

    with patch.object(api, "_get_connector", return_value=mock_connector), \
         patch.object(api, "_resolve_page_id", new=AsyncMock(return_value="page-123")), \
         patch("confluence_logic.review.api.asyncio.create_task") as mock_create_task:
        result = await api._direct_apply_change(proposal)

    assert result["success"] is True
    assert mock_create_task.called is True


@pytest.mark.asyncio
async def test_reindex_failure_does_not_affect_success_response(mock_connector, base_edit_proposal):
    """APPLY-03: if asyncio.create_task() raises (e.g. event loop closed), the exception
    must be swallowed and the accept response must still be success=True."""
    proposal = {**base_edit_proposal, "section_heading": "Goals", "edit_mode": "append"}

    with patch.object(api, "_get_connector", return_value=mock_connector), \
         patch.object(api, "_resolve_page_id", new=AsyncMock(return_value="page-123")), \
         patch("confluence_logic.review.api.asyncio.create_task", side_effect=RuntimeError("event loop is closed")):
        result = await api._direct_apply_change(proposal)

    assert result["success"] is True
