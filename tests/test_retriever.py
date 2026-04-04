"""
Tests for RetrieverAgent — hybrid RAG pipeline wrapper.

All tests mock PineconeClient to avoid Pinecone/OpenAI API calls.
RetrieverAgent.retrieve() is async; tests use @pytest.mark.asyncio.
"""

import pytest
from unittest.mock import MagicMock, patch

from agents.retriever import RetrieverAgent, RetrievalResult


def make_mock_client(reranked_results=None):
    """Return a MagicMock PineconeClient with retrieve() configured."""
    client = MagicMock()
    if reranked_results is None:
        reranked_results = [
            {
                "id": "mtg-001",
                "score": 0.9,
                "metadata": {
                    "channel_id": "C01",
                    "channel_name": "eng-standup",
                    "start_ts": 1711900000,
                    "summary_text": "Sprint planning and Q2 roadmap discussion.",
                },
                "rerank_score": 0.99,
            },
            {
                "id": "mtg-002",
                "score": 0.8,
                "metadata": {
                    "channel_id": "C01",
                    "channel_name": "eng-standup",
                    "start_ts": 1711950000,
                    "summary_text": "Incident retrospective for last week's outage.",
                },
                "rerank_score": 0.87,
            },
        ]
    client.retrieve.return_value = reranked_results
    return client


class TestRetrievalResult:
    def test_retrieval_result_has_required_fields(self):
        result = RetrievalResult(query="what was discussed?", results=[], total_candidates=20, returned_count=0)
        assert result.query == "what was discussed?"
        assert result.results == []
        assert result.total_candidates == 20
        assert result.returned_count == 0

    def test_retrieval_result_results_are_dicts(self):
        result = RetrievalResult(
            query="anything",
            results=[{"id": "m1", "score": 0.9, "metadata": {}, "rerank_score": 0.99}],
            total_candidates=20,
            returned_count=1,
        )
        assert result.results[0]["id"] == "m1"


class TestRetrieverAgentInit:
    def test_init_stores_pinecone_client(self):
        client = MagicMock()
        agent = RetrieverAgent(pinecone_client=client)
        assert agent._client is client

    def test_init_default_top_k_is_20(self):
        agent = RetrieverAgent(pinecone_client=MagicMock())
        assert agent._top_k == 20

    def test_init_default_top_n_is_5(self):
        agent = RetrieverAgent(pinecone_client=MagicMock())
        assert agent._top_n == 5

    def test_init_accepts_custom_top_k_and_top_n(self):
        agent = RetrieverAgent(pinecone_client=MagicMock(), top_k=30, top_n=8)
        assert agent._top_k == 30
        assert agent._top_n == 8


class TestRetrieverAgentRetrieve:
    @pytest.mark.asyncio
    async def test_retrieve_calls_pinecone_client_retrieve(self):
        client = make_mock_client()
        agent = RetrieverAgent(pinecone_client=client)
        await agent.retrieve("what did we decide about the API?")
        client.retrieve.assert_called_once()

    @pytest.mark.asyncio
    async def test_retrieve_passes_query_text(self):
        client = make_mock_client()
        agent = RetrieverAgent(pinecone_client=client)
        await agent.retrieve("sprint planning recap")
        call_kwargs = client.retrieve.call_args[1]
        assert call_kwargs["query_text"] == "sprint planning recap"

    @pytest.mark.asyncio
    async def test_retrieve_passes_channel_id_filter(self):
        client = make_mock_client()
        agent = RetrieverAgent(pinecone_client=client)
        await agent.retrieve("standup summary", channel_id="C_ENG")
        call_kwargs = client.retrieve.call_args[1]
        assert call_kwargs.get("channel_id") == "C_ENG"

    @pytest.mark.asyncio
    async def test_retrieve_passes_date_range_filter(self):
        client = make_mock_client()
        agent = RetrieverAgent(pinecone_client=client)
        await agent.retrieve("last week", start_ts=1711800000, end_ts=1712000000)
        call_kwargs = client.retrieve.call_args[1]
        assert call_kwargs.get("start_ts") == 1711800000
        assert call_kwargs.get("end_ts") == 1712000000

    @pytest.mark.asyncio
    async def test_retrieve_passes_no_filter_when_none(self):
        client = make_mock_client()
        agent = RetrieverAgent(pinecone_client=client)
        await agent.retrieve("generic query")
        call_kwargs = client.retrieve.call_args[1]
        assert call_kwargs.get("channel_id") is None
        assert call_kwargs.get("start_ts") is None
        assert call_kwargs.get("end_ts") is None

    @pytest.mark.asyncio
    async def test_retrieve_passes_top_k_and_top_n(self):
        agent = RetrieverAgent(pinecone_client=make_mock_client(), top_k=30, top_n=8)
        await agent.retrieve("question")
        call_kwargs = agent._client.retrieve.call_args[1]
        assert call_kwargs.get("top_k") == 30
        assert call_kwargs.get("top_n") == 8

    @pytest.mark.asyncio
    async def test_retrieve_returns_retrieval_result_type(self):
        client = make_mock_client()
        agent = RetrieverAgent(pinecone_client=client)
        result = await agent.retrieve("what happened?")
        assert isinstance(result, RetrievalResult)

    @pytest.mark.asyncio
    async def test_retrieve_result_query_matches_input(self):
        client = make_mock_client()
        agent = RetrieverAgent(pinecone_client=client)
        result = await agent.retrieve("sprint recap")
        assert result.query == "sprint recap"

    @pytest.mark.asyncio
    async def test_retrieve_result_contains_client_results(self):
        client = make_mock_client()
        agent = RetrieverAgent(pinecone_client=client)
        result = await agent.retrieve("question")
        assert len(result.results) == 2
        assert result.results[0]["id"] == "mtg-001"
        assert result.results[0]["rerank_score"] == 0.99

    @pytest.mark.asyncio
    async def test_retrieve_returned_count_matches_results_length(self):
        client = make_mock_client()
        agent = RetrieverAgent(pinecone_client=client)
        result = await agent.retrieve("anything")
        assert result.returned_count == 2

    @pytest.mark.asyncio
    async def test_retrieve_total_candidates_equals_top_k(self):
        agent = RetrieverAgent(pinecone_client=make_mock_client(), top_k=20)
        result = await agent.retrieve("anything")
        assert result.total_candidates == 20

    @pytest.mark.asyncio
    async def test_retrieve_empty_results(self):
        client = make_mock_client(reranked_results=[])
        agent = RetrieverAgent(pinecone_client=client)
        result = await agent.retrieve("obscure query")
        assert result.results == []
        assert result.returned_count == 0
