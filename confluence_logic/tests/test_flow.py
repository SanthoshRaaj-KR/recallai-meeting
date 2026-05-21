import asyncio
import json
import pytest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from confluence_logic.agents.tools import (
    search_workspace_knowledge, fetch_live_page, preview_edit, preview_delete, commit_delete, commit_document_edit, update_page_title, list_workspace_pages, format_page_titles_for_user
)
from confluence_logic.core.schemas import CandidatePage
from confluence_logic.utils.html_parser import delete_content_in_section, edit_block_in_section, get_section_html


def _call(tool, *args, **kwargs):
    """Invoke a FunctionTool synchronously — maps positional args to schema property order."""
    props = list(tool.params_json_schema.get('properties', {}).keys())
    call_args = dict(zip(props, args))
    call_args.update(kwargs)
    return asyncio.run(tool.on_invoke_tool(None, json.dumps(call_args)))

@patch('confluence_logic.agents.tools.get_connector')
@patch('confluence_logic.agents.tools.get_store')
def test_page_selection_disambiguation(mock_get_store, mock_get_connector):
    mock_store = mock_get_store.return_value
    mock_connector = mock_get_connector.return_value
    mock_connector.search_pages.return_value = [
        {"page_id": "1", "title": "Quarterly Goals", "space_key": "DEV", "excerpt": "Live page result"},
    ]
    mock_store.search.return_value = [
        {"metadata": {"page_id": "1", "title": "Quarterly Goals", "heading": "Q1", "text_summary": "Heading: Q1\nExcerpt:", "space_key": "DEV"}},
        {"metadata": {"page_id": "1", "title": "Quarterly Goals", "heading": "Q2", "text_summary": "Heading: Q2\nExcerpt:", "space_key": "DEV"}}
    ]
    resp = _call(search_workspace_knowledge, "Quarterly")
    assert len(resp.candidates) == 1
    assert resp.candidates[0].page_id == "1"
    assert resp.candidates[0].heading in {"Q1", "Q2"}
    assert "Success" in resp.message

@patch('confluence_logic.agents.tools.get_connector')
@patch('confluence_logic.agents.tools.get_store')
def test_page_selection_live_search_without_pinecone(mock_get_store, mock_get_connector):
    mock_store = mock_get_store.return_value
    mock_connector = mock_get_connector.return_value
    mock_connector.search_pages.return_value = [
        {"page_id": "9", "title": "Introduction to Machine Learning", "space_key": "DEV", "excerpt": "ML page"},
    ]
    mock_connector.list_pages.return_value = []
    mock_store.search.side_effect = Exception("pinecone unavailable")

    resp = _call(search_workspace_knowledge, "intro to machine learning")
    assert len(resp.candidates) == 1
    assert resp.candidates[0].page_id == "9"
    assert resp.candidates[0].title == "Introduction to Machine Learning"

@patch('confluence_logic.agents.tools.get_connector')
@patch('confluence_logic.agents.tools.get_store')
def test_page_selection_includes_recent_pages_when_live_search_is_empty(mock_get_store, mock_get_connector):
    mock_store = mock_get_store.return_value
    mock_connector = mock_get_connector.return_value
    mock_connector.search_pages.return_value = []
    mock_connector.list_pages.return_value = [
        {"page_id": "11", "title": "Introduction to Machine Learning", "space_key": "DEV", "excerpt": "Freshly created page"},
        {"page_id": "12", "title": "AI Notes", "space_key": "DEV", "excerpt": "Artificial intelligence"},
    ]
    mock_store.search.side_effect = Exception("pinecone unavailable")

    resp = _call(search_workspace_knowledge, "some vague request")
    assert len(resp.candidates) == 2
    assert {cand.page_id for cand in resp.candidates} == {"11", "12"}

@patch('confluence_logic.agents.tools.get_connector')
def test_live_fetch_and_headings(mock_get_connector):
    mock_connector = mock_get_connector.return_value
    mock_connector.get_page_metadata.return_value = {"version": {"number": 5}}
    mock_connector.fetch_page_html.return_value = "<h1>Doc</h1><h2>Section A</h2><p>text</p><h3>Sub</h3><h2>Section B</h2>"
    
    resp = _call(fetch_live_page, "123", heading_string="Section A")
    assert resp.expected_version == 5
    assert resp.available_headings == ["Doc", "Section A", "Sub", "Section B"]
    assert "<p>text</p><h3>Sub</h3>" == str(resp.section_html).strip()

def test_fragment_replacement():
    html = "<body><h2>Heading 1</h2><p>Old text</p><h2>Heading 2</h2></body>"
    new_html = "<p>New text</p>"
    result = edit_block_in_section(html, "Heading 1", "<p>Old text</p>", new_html)
    assert "Old text" not in result
    assert "New text" in result
    assert "Heading 2" in result
    assert "Heading 1" in result # Anchor should remain intact

def test_disambiguation_error():
    html_dup = "<body><h2>Heading 1</h2><p>A</p><h2>Heading 1</h2><p>B</p></body>"
    with pytest.raises(ValueError, match="Multiple headings match"):
        edit_block_in_section(html_dup, "Heading 1", "", "")

@patch('confluence_logic.agents.tools.get_connector')
def test_version_conflict_handling(mock_get_connector):
    mock_connector = mock_get_connector.return_value
    mock_connector.fetch_page_html.return_value = "<h2>H1</h2><p>test</p>"
    # Simulate a version conflict being thrown by connector on push
    mock_connector.push_update.side_effect = ValueError("Version Conflict: Expected version 4, but live version is 5.")
    
    resp = _call(commit_document_edit, "123", 4, "H1", "<p>test</p>", "<p>new</p>")
    assert resp.success is False
    assert "ConflictError" in resp.message

def test_root_level_blank_edit():
    html_empty = ""
    res1 = edit_block_in_section(html_empty, "Root", "", "<p>Welcome</p>")
    assert "<p>Welcome</p>" in res1
    
    html_populated = "<p>Intro</p><h2>H1</h2>"
    res2 = edit_block_in_section(html_populated, "Root", "<p>Intro</p>", "<p>Brand New Intro</p>")
    assert "Brand New Intro" in res2
    assert "H1" in res2

def test_full_page_replace_removes_existing_sections():
    html = "<p>Intro</p><h2>Applications</h2><p>Old app text</p><h2>Other</h2><p>Keep old? no</p>"
    result = edit_block_in_section(html, "FULL_PAGE", "", "<h2>Donuts</h2><p>Donuts are amazing.</p>")
    assert "Applications" not in result
    assert "Old app text" not in result
    assert "Other" not in result
    assert "Donuts are amazing." in result

def test_root_edit_only_changes_intro_not_full_page():
    html = "<p>Intro</p><h2>Applications</h2><p>Old app text</p>"
    result = edit_block_in_section(html, "Root", "", "<p>New intro</p>")
    assert "New intro" in result
    assert "Applications" in result
    assert "Old app text" in result

def test_plain_text_target_resolves_to_unique_block():
    html = "<h2>Overview</h2><p>14</p><h2>Other</h2><p>Common mistakes</p>"
    result = edit_block_in_section(html, "FULL_PAGE", "14", "<p>Why donuts are awesome</p><ul><li>Point 1</li><li>Point 2</li><li>Point 3</li></ul>")
    assert "<p>14</p>" not in result
    assert "Why donuts are awesome" in result
    assert "Common mistakes" in result

def test_delete_entire_section_removes_heading_and_body():
    html = "<h2>Common mistakes</h2><p>mistake body</p><h2>Keep</h2><p>keep me</p>"
    result = delete_content_in_section(html, "Common mistakes", delete_entire_section=True)
    assert "Common mistakes" not in result
    assert "mistake body" not in result
    assert "Keep" in result
    assert "keep me" in result

def test_delete_unique_visible_text_block_inside_full_page():
    html = "<h2>Overview</h2><p>Delete me</p><p>Keep me</p>"
    result = delete_content_in_section(html, "FULL_PAGE", target_html_or_text="Delete me")
    assert "Delete me" not in result
    assert "Keep me" in result

@patch('confluence_logic.agents.tools.get_connector')
def test_preview_delete(mock_get_connector):
    mock_connector = mock_get_connector.return_value
    mock_connector.fetch_page_html.return_value = "<h2>Common mistakes</h2><p>mistake body</p><h2>Keep</h2><p>keep me</p>"

    resp = _call(preview_delete, "123", "Common mistakes", delete_entire_section=True)
    assert resp.success is True
    assert "Common mistakes" in resp.diff

@patch('confluence_logic.agents.tools._reindex_in_background')
@patch('confluence_logic.agents.tools.get_connector')
def test_commit_delete(mock_get_connector, mock_reindex):
    mock_connector = mock_get_connector.return_value
    mock_connector.fetch_page_html.return_value = "<h2>Common mistakes</h2><p>mistake body</p><h2>Keep</h2><p>keep me</p>"
    mock_connector.push_update.return_value = True

    resp = _call(commit_delete, "123", 4, "Common mistakes", delete_entire_section=True)
    assert resp.success is True
    mock_connector.push_update.assert_called_once()
    pushed_html = mock_connector.push_update.call_args.args[1]
    assert "Common mistakes" not in pushed_html
    assert "Keep" in pushed_html
    mock_reindex.assert_called_once_with("123")

@patch('confluence_logic.agents.tools._reindex_in_background')
@patch('confluence_logic.agents.tools.get_connector')
def test_create_confluence_page_tool(mock_get_connector, mock_reindex):
    from confluence_logic.agents.tools import create_confluence_page
    mock_connector = mock_get_connector.return_value
    mock_connector.create_page.return_value = {"id": "999", "title": "Test Page", "version": {"number": 1}}

    resp = _call(create_confluence_page, "DEV", "Test Page", body_text="Hello World")
    assert resp.success is True
    assert resp.page_id == "999"
    mock_reindex.assert_called_once_with("999")

def test_html_builder():
    from confluence_logic.utils.html_builder import build_page_html
    html = build_page_html(title="Doc", body_text="Test", sections=[{"heading": "Next", "content": "- a\n- b"}])
    assert "<h2>Next</h2>" in html
    assert "<ul>" in html
    assert "<li>a</li>" in html

@patch('confluence_logic.agents.tools.get_connector')
def test_list_workspace_pages(mock_get_connector):
    mock_connector = mock_get_connector.return_value
    mock_connector.list_pages.return_value = [
        {"page_id": "123", "title": "Sample AI Page", "space_key": "DEV", "excerpt": "About AI"},
        {"page_id": "124", "title": "ML Notes", "space_key": "DEV", "excerpt": "About ML"},
    ]

    resp = _call(list_workspace_pages, 10)
    assert len(resp.candidates) == 2
    assert resp.candidates[0].title == "Sample AI Page"
    assert resp.candidates[1].page_id == "124"

def test_format_page_titles_for_user_omits_metadata():
    formatted = format_page_titles_for_user([
        CandidatePage(page_id="1", title="Sample AI Page", space_key="DEV", snippet="About AI"),
        CandidatePage(page_id="2", title="ML Notes", space_key="DEV", snippet="About ML"),
    ])

    assert formatted == "Sample AI Page, ML Notes"
    assert "DEV" not in formatted
    assert "(" not in formatted

def test_format_page_titles_for_user_disambiguates_duplicates_with_heading():
    formatted = format_page_titles_for_user([
        CandidatePage(page_id="1", title="Notes", heading="Quarterly Goals", space_key="DEV", snippet="A"),
        CandidatePage(page_id="2", title="Notes", heading="QBR", space_key="DEV", snippet="B"),
    ])

    assert "Notes - Quarterly Goals" in formatted
    assert "Notes - QBR" in formatted

@patch('confluence_logic.agents.tools._reindex_in_background')
@patch('confluence_logic.agents.tools.get_connector')
def test_update_page_title_tool(mock_get_connector, mock_reindex):
    mock_connector = mock_get_connector.return_value
    mock_connector.fetch_page_html.return_value = "<h2>Overview</h2><p>Hello</p>"
    mock_connector.push_update.return_value = True

    resp = _call(update_page_title, "123", 4, "Why Donuts Are Awesome")
    assert resp.success is True
    mock_connector.push_update.assert_called_once_with(
        "123",
        "<h2>Overview</h2><p>Hello</p>",
        expected_version=4,
        title_override="Why Donuts Are Awesome",
    )
    mock_reindex.assert_called_once_with("123")

@patch("confluence_logic.agents.editor_agent.Runner.run", new_callable=AsyncMock)
def test_handle_prepared_query_bypasses_reframer(mock_runner_run):
    from confluence_logic.agents.editor_agent import EditorAgent

    agent = EditorAgent(model="gpt-5-mini")
    agent.reframer.handle_query = AsyncMock(return_value="ACTION: clarify")
    mock_runner_run.return_value = SimpleNamespace(final_output="Prepared edit completed.")

    result = asyncio.run(agent.handle_prepared_query(
        "Edit the Sample AI Page and add current AI trends.",
        original_query="update sample ai page",
    ))

    assert result == "Prepared edit completed."
    agent.reframer.handle_query.assert_not_awaited()
    mock_runner_run.assert_awaited_once()

@patch("confluence_logic.agents.editor_agent.list_workspace_pages")
def test_handle_prepared_query_lists_titles_cleanly(mock_list_workspace_pages):
    from confluence_logic.agents.editor_agent import EditorAgent

    agent = EditorAgent(model="gpt-5-mini")
    mock_list_workspace_pages.return_value = SimpleNamespace(
        candidates=[
            CandidatePage(page_id="1", title="Sample AI Page", space_key="DEV", snippet="About AI"),
            CandidatePage(page_id="2", title="ML Notes", space_key="DEV", snippet="About ML"),
        ]
    )

    result = asyncio.run(agent.handle_prepared_query("LIST_PAGES", original_query="what pages are available"))
    assert result == "Sample AI Page, ML Notes"


def test_editor_agent_master_exposes_specialist_tools():
    from confluence_logic.agents.editor_agent import EditorAgent

    agent = EditorAgent(model="gpt-5-mini")
    tool_names = {tool.name for tool in agent.agent.tools}

    assert {
        "resolve_request",
        "list_recent_pages",
        "edit_existing_page",
        "delete_from_page",
        "create_new_page",
    }.issubset(tool_names)
