from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest

from confluence_logic import confluence_page_graph
from confluence_logic.review import api


def test_sections_from_html_uses_headings_and_content():
    sections = confluence_page_graph._sections_from_html(
        """
        <p>Intro text</p>
        <h1>Roadmap</h1>
        <p>Launch plan details</p>
        <h2>Risks</h2>
        <p>Rollback needed</p>
        <h1>Support</h1>
        <p>Enablement notes</p>
        """
    )

    assert sections[0]["heading"] == "Root"
    assert sections[1]["heading"] == "Roadmap"
    assert "Launch plan details" in sections[1]["text"]
    assert "Rollback needed" in sections[1]["text"]
    assert sections[2]["heading"] == "Risks"
    assert sections[3]["heading"] == "Support"


def test_confluence_graph_user_id_prefers_supabase_user():
    assert api._confluence_graph_user_id({"id": "user-1"}, "session-1") == "supabase:user-1"
    assert api._confluence_graph_user_id(None, "session-1") == "session:session-1"


@pytest.mark.asyncio
async def test_query_user_confluence_graph_scopes_by_user_id():
    record = SimpleNamespace(
        data=lambda: {
            "page_id": "page-1",
            "title": "Roadmap",
            "space_key": "ENG",
            "version": 2,
            "heading": "Launch",
            "text": "Launch moved to Friday",
            "score": 3,
        }
    )
    driver = AsyncMock()
    driver.execute_query = AsyncMock(return_value=([record], None, []))

    with patch.object(confluence_page_graph, "_driver", return_value=driver):
        results = await confluence_page_graph.query_user_confluence_graph("supabase:user-1", "launch Friday")

    assert results[0]["page_id"] == "page-1"
    assert results[0]["source"] == "neo4j_confluence_graph"
    params = driver.execute_query.await_args.args[1]
    assert params["user_id"] == "supabase:user-1"
    assert params["graph_kind"] == "confluence_pages"


@pytest.mark.asyncio
async def test_incremental_write_fetches_only_changed_pages():
    existing_records = [
        SimpleNamespace(data=lambda: {"page_id": "unchanged", "version": 2}),
        SimpleNamespace(data=lambda: {"page_id": "changed", "version": 1}),
    ]

    async def execute_query(cypher, *_args, **_kwargs):
        if "RETURN p.page_id AS page_id, p.version AS version" in cypher:
            return existing_records, None, []
        return [], None, []

    driver = AsyncMock()
    driver.execute_query = AsyncMock(side_effect=execute_query)
    connector = Mock()
    connector.fetch_page_html.return_value = "<h1>Updated</h1><p>New content</p>"

    await confluence_page_graph._write_pages_incremental(
        driver,
        "supabase:user-1",
        connector,
        [
            {"page_id": "unchanged", "title": "Same", "version": 2, "space_key": "ENG", "excerpt": ""},
            {"page_id": "changed", "title": "Changed", "version": 3, "space_key": "ENG", "excerpt": ""},
            {"page_id": "new", "title": "New", "version": 1, "space_key": "ENG", "excerpt": ""},
        ],
    )

    fetched_ids = [call.args[0] for call in connector.fetch_page_html.call_args_list]
    assert fetched_ids == ["changed", "new"]


@pytest.mark.asyncio
async def test_list_user_confluence_pages_scopes_query():
    record = SimpleNamespace(
        data=lambda: {
            "page_id": "page-1",
            "title": "Roadmap",
            "space_key": "ENG",
            "version": 2,
        }
    )
    driver = AsyncMock()
    driver.execute_query = AsyncMock(return_value=([record], None, []))

    with patch.object(confluence_page_graph, "_driver", return_value=driver):
        pages = await confluence_page_graph.list_user_confluence_pages("supabase:user-1", limit=5)

    assert pages[0]["title"] == "Roadmap"
    params = driver.execute_query.await_args.args[1]
    assert params["user_id"] == "supabase:user-1"
    assert params["graph_kind"] == "confluence_pages"
