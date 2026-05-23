"""Tests for confluence_logic.connectors.confluence (Phase 10 / PROP-V2-05).

The Phase 10 ProposalCard needs a breadcrumb (Space › Parent › Page) and a
clickable Confluence URL. Both come from the connector's get_page_metadata
when called with expand="ancestors,space". This file pins:

  1. Pre-Phase-10 callers (no expand kwarg) get the legacy JSON shape
     untouched — no ancestors / space keys leak into the dict.
  2. expand="ancestors" surfaces the ancestor list on the returned dict.
  3. expand="ancestors,space" surfaces both ancestors and space metadata.

Module under test:
    confluence_logic.connectors.confluence.ConfluenceConnector.get_page_metadata
"""
from __future__ import annotations

import os
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest

from confluence_logic.connectors.confluence import ConfluenceConnector


@pytest.fixture
def connector(monkeypatch) -> ConfluenceConnector:
    """Build a ConfluenceConnector with fake Atlassian creds.

    The constructor reads three env vars; without them it raises
    ValueError. We patch them locally so the test stays hermetic and
    does not depend on whether the developer has real creds in their
    shell.
    """
    monkeypatch.setenv("ATLASSIAN_USER_EMAIL", "test@example.com")
    monkeypatch.setenv("ATLASSIAN_API_TOKEN", "test-token-xxx")
    monkeypatch.setenv("ATLASSIAN_DOMAIN", "test.atlassian.net")
    return ConfluenceConnector()


def _make_response(payload: Dict[str, Any]) -> MagicMock:
    """Build a MagicMock that quacks like a requests.Response success."""
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = payload
    resp.raise_for_status.return_value = None
    return resp


def test_get_page_metadata_no_expand_returns_legacy_shape(connector: ConfluenceConnector) -> None:
    """Pre-Phase-10 contract: dict contains only the keys Confluence returned.

    No ``ancestors`` / ``space`` keys are added when ``expand`` is omitted,
    so existing callers (e.g., the pipeline's version-conflict path) keep
    seeing exactly the JSON they received before Plan 10-07 landed.
    """
    sample = {
        "id": "p1",
        "title": "Sample Page",
        "type": "page",
        "version": {"number": 7, "minorEdit": False},
    }
    with patch(
        "confluence_logic.connectors.confluence.requests.get",
        return_value=_make_response(sample),
    ) as mocked:
        result = connector.get_page_metadata("p1")

    assert result == sample
    assert "ancestors" not in result
    assert "space" not in result

    # Confirm the connector did NOT pass an expand= param when none was requested.
    _, kwargs = mocked.call_args
    assert kwargs.get("params") in (None, {}), (
        "expected no query params when expand is omitted; got %r"
        % (kwargs.get("params"),)
    )


def test_get_page_metadata_expand_ancestors_returns_ancestor_list(
    connector: ConfluenceConnector,
) -> None:
    """expand="ancestors" surfaces the ancestor list as a top-level key.

    Used by enrich_page_for_drafter to build the breadcrumb shown in the
    ProposalCard header (Plan 10-07 / D-07).
    """
    sample = {
        "id": "p1",
        "title": "Child",
        "ancestors": [{"id": "p0", "title": "Parent", "type": "page"}],
    }
    with patch(
        "confluence_logic.connectors.confluence.requests.get",
        return_value=_make_response(sample),
    ) as mocked:
        result = connector.get_page_metadata("p1", expand="ancestors")

    assert "ancestors" in result, "expand=ancestors must surface ancestors key"
    assert isinstance(result["ancestors"], list)
    assert result["ancestors"][0]["title"] == "Parent"
    # Verify the expand kwarg was forwarded to Confluence as a query param.
    _, kwargs = mocked.call_args
    params = kwargs.get("params") or {}
    assert params.get("expand") == "ancestors"


def test_get_page_metadata_expand_ancestors_space_includes_space_name(
    connector: ConfluenceConnector,
) -> None:
    """expand="ancestors,space" surfaces both expansions on the dict.

    The breadcrumb in D-07 is constructed as [space.name, *[a.title for a in
    ancestors], page.title], so both keys must round-trip.
    """
    sample = {
        "id": "p1",
        "title": "Project Plan",
        "ancestors": [],
        "space": {"key": "ENG", "name": "Engineering", "id": "42"},
    }
    with patch(
        "confluence_logic.connectors.confluence.requests.get",
        return_value=_make_response(sample),
    ) as mocked:
        result = connector.get_page_metadata("p1", expand="ancestors,space")

    assert result["space"]["name"] == "Engineering"
    assert result["space"]["key"] == "ENG"
    assert result["ancestors"] == []
    _, kwargs = mocked.call_args
    params = kwargs.get("params") or {}
    assert params.get("expand") == "ancestors,space"


def test_get_page_metadata_expand_ancestors_missing_in_response_defaults_to_empty_list(
    connector: ConfluenceConnector,
) -> None:
    """Defensive: if Confluence omits the requested expansion, we still
    surface an empty list / dict so downstream callers do not KeyError.

    Real Confluence always echoes the requested expansion (even if empty);
    this test pins the connector's defensive default so a hypothetical
    server-side regression cannot crash the pipeline.
    """
    sample = {"id": "p1", "title": "Orphan"}
    with patch(
        "confluence_logic.connectors.confluence.requests.get",
        return_value=_make_response(sample),
    ):
        result = connector.get_page_metadata("p1", expand="ancestors,space")

    assert result["ancestors"] == []
    assert result["space"] == {}
