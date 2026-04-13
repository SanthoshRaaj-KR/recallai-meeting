"""Tests for graph_rag — Neo4j Graph RAG module (GRAPHRAG-01)."""
import pytest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch, MagicMock

from confluence_logic import graph_rag


def _mock_openai_response(content: str):
    """Build a mock OpenAI ChatCompletion response."""
    choice = SimpleNamespace(message=SimpleNamespace(content=content))
    return SimpleNamespace(choices=[choice])


@pytest.fixture(autouse=True)
def reset_driver():
    """Reset the module-level driver before each test."""
    graph_rag._driver = None
    yield
    graph_rag._driver = None


@pytest.mark.asyncio
async def test_ingest_calls_execute_query():
    """GRAPHRAG-01: ingest extracts entities and runs MERGE Cypher."""
    mock_driver = AsyncMock()
    mock_driver.execute_query = AsyncMock(return_value=([], None, []))
    entities = {"topics": ["React"], "people": [], "decisions": []}
    with patch.object(graph_rag, "_get_driver", return_value=mock_driver), \
         patch.object(graph_rag, "_extract_entities", return_value=entities):
        await graph_rag.ingest_transcript_entry({"participant": "Alice", "text": "Let's use React", "timestamp": 0})
    mock_driver.execute_query.assert_called()


@pytest.mark.asyncio
async def test_query_context_returns_context_string():
    """GRAPHRAG-01: query returns formatted context when graph has matches."""
    mock_records = [
        SimpleNamespace(data=lambda: {"n": {"name": "React"}, "rel_type": "MENTIONED_BY", "neighbor": {"name": "Alice"}}),
    ]
    mock_driver = AsyncMock()
    mock_driver.execute_query = AsyncMock(return_value=(mock_records, None, []))
    with patch.object(graph_rag, "_get_driver", return_value=mock_driver), \
         patch.object(graph_rag, "_extract_keywords", return_value=["React"]):
        result = await graph_rag.query_context("what about React")
    assert len(result) > 0


@pytest.mark.asyncio
async def test_query_context_no_driver_returns_empty():
    """GRAPHRAG-01: returns empty string when NEO4J_URI not set."""
    with patch.object(graph_rag, "_get_driver", return_value=None):
        result = await graph_rag.query_context("anything")
    assert result == ""


@pytest.mark.asyncio
async def test_ingest_no_driver_returns_silently():
    """GRAPHRAG-01: ingest returns without error when driver is None."""
    with patch.object(graph_rag, "_get_driver", return_value=None):
        await graph_rag.ingest_transcript_entry({"participant": "Alice", "text": "hello", "timestamp": 0})
    # No exception = pass


@pytest.mark.asyncio
async def test_query_context_fallback_on_exception():
    """GRAPHRAG-01: returns empty string when Neo4j raises."""
    mock_driver = AsyncMock()
    mock_driver.execute_query = AsyncMock(side_effect=Exception("Connection refused"))
    with patch.object(graph_rag, "_get_driver", return_value=mock_driver), \
         patch.object(graph_rag, "_extract_keywords", return_value=["test"]):
        result = await graph_rag.query_context("anything")
    assert result == ""


@pytest.mark.asyncio
async def test_extract_entities_returns_dict():
    """GRAPHRAG-01: _extract_entities returns dict with topics/people/decisions."""
    mock_response = _mock_openai_response('{"topics": ["React"], "people": ["Alice"], "decisions": ["use React"]}')
    with patch.object(graph_rag, "_get_client") as mock_client:
        mock_client.return_value.chat.completions.create.return_value = mock_response
        result = await graph_rag._extract_entities("Bob", "Alice said let's use React")
    assert "topics" in result
    assert "people" in result
    assert "decisions" in result
    assert "React" in result["topics"]
