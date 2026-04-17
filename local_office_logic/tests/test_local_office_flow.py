import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from local_office_logic.core.schemas import CandidateArtifact
from local_office_logic.utils.html_builder import build_artifact_html
from local_office_logic.utils.sandbox import sanitize_filename
from local_office_logic.agents.tools import format_artifact_titles_for_user, search_sandbox_artifacts


def test_sanitize_filename_removes_problematic_characters():
    assert sanitize_filename("Q1 Report: Final/Reviewed?") == "Q1 Report Final Reviewed"


def test_build_artifact_html_for_document_contains_title_and_sections():
    html = build_artifact_html(
        title="Quarterly Notes",
        artifact_family="document",
        body_text="Intro line",
        sections=[{"label": "Decisions", "content": "- ship\n- measure"}],
    )
    assert "<h1>Quarterly Notes</h1>" in html
    assert "<h2>Decisions</h2>" in html
    assert "<li>ship</li>" in html


def test_build_artifact_html_for_spreadsheet_contains_table():
    html = build_artifact_html(
        title="Budget",
        artifact_family="spreadsheet",
        sections=[{"label": "Sheet1", "content": "Name,Amount\nOps,10"}],
    )
    assert "<h2>Sheet1</h2>" in html
    assert "<table" in html
    assert "Ops" in html


@patch("local_office_logic.agents.tools.get_connector")
@patch("local_office_logic.agents.tools.get_store")
def test_search_sandbox_artifacts_merges_live_and_semantic(mock_get_store, mock_get_connector):
    mock_connector = mock_get_connector.return_value
    mock_store = mock_get_store.return_value
    mock_connector.search_artifacts.return_value = [
        {
            "artifact_id": "notes.docx",
            "title": "Meeting Notes",
            "relative_path": "notes.docx",
            "artifact_family": "document",
            "file_format": "docx",
        }
    ]
    mock_connector.list_artifacts.return_value = []
    mock_store.search.return_value = [
        {
            "metadata": {
                "artifact_id": "budget.xlsx",
                "title": "Budget Tracker",
                "relative_path": "finance/budget.xlsx",
                "artifact_family": "spreadsheet",
                "file_format": "xlsx",
                "section_label": "Summary",
                "text_summary": "Budget tracker summary",
            }
        }
    ]

    response = search_sandbox_artifacts("budget")
    assert {candidate.artifact_id for candidate in response.candidates} == {"notes.docx", "budget.xlsx"}


def test_format_artifact_titles_for_user_omits_internal_ids():
    formatted = format_artifact_titles_for_user(
        [
            CandidateArtifact(
                artifact_id="reports/q1.docx",
                title="Q1",
                relative_path="reports/q1.docx",
                artifact_family="document",
                file_format="docx",
                snippet="Quarterly report",
            ),
            CandidateArtifact(
                artifact_id="reports/q2.docx",
                title="Q2",
                relative_path="reports/q2.docx",
                artifact_family="document",
                file_format="docx",
                snippet="Quarterly report",
            ),
        ]
    )
    assert "reports/q1.docx" not in formatted
    assert formatted == "Q1, Q2"


@patch("local_office_logic.agents.editor_agent.list_sandbox_artifacts")
def test_handle_prepared_query_lists_titles_cleanly(mock_list_sandbox_artifacts):
    from local_office_logic.agents.editor_agent import EditorAgent

    agent = EditorAgent(model="gpt-5-mini")
    mock_list_sandbox_artifacts.return_value = SimpleNamespace(
        candidates=[
            CandidateArtifact(
                artifact_id="notes.docx",
                title="Meeting Notes",
                relative_path="notes.docx",
                artifact_family="document",
                file_format="docx",
                snippet="Notes",
            ),
            CandidateArtifact(
                artifact_id="budget.xlsx",
                title="Budget",
                relative_path="budget.xlsx",
                artifact_family="spreadsheet",
                file_format="xlsx",
                snippet="Budget",
            ),
        ]
    )

    result = asyncio.run(agent.handle_prepared_query("LIST_ARTIFACTS", original_query="what files are available"))
    assert result == "Meeting Notes, Budget"
