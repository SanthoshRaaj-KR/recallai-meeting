"""
RetrieverAgent — hybrid RAG pipeline wrapper for meeting memory queries.

Wraps PineconeClient.retrieve() (top-20 hybrid query + neural rerank to top-5)
in a simple async interface that returns a structured RetrievalResult.

Design notes:
- RetrieverAgent is a plain Python class, NOT an openai-agents Agent() instance.
  Phase 5 wires it as a registered tool inside the orchestrator.
- PineconeClient.retrieve() is synchronous. Calling it directly from an async
  method is acceptable here — it performs network I/O (Pinecone + OpenAI) that
  is already handled by the SDK's internal HTTP client. Phase 5 can add executor
  offloading if event-loop blocking becomes measurable.
- RetrievalResult is defined here (retrieval layer concern), not in storage/models.py.
"""

from typing import Optional

from pydantic import BaseModel

from storage.pinecone_client import PineconeClient


class RetrievalResult(BaseModel):
    """Structured result from a RetrieverAgent.retrieve() call.

    Fields:
        query: The original natural language query string.
        results: Ordered list of reranked meeting excerpts. Each dict has:
            "id" (str), "score" (float, hybrid similarity score),
            "metadata" (dict, flat Pinecone metadata), "rerank_score" (float).
        total_candidates: Number of candidates fetched from Pinecone before reranking
            (corresponds to top_k passed to PineconeClient.retrieve()).
        returned_count: Actual number of results returned after reranking
            (len(results) — may be less than top_n if fewer candidates exist).
    """

    query: str
    results: list[dict]
    total_candidates: int
    returned_count: int


class RetrieverAgent:
    """Thin wrapper around PineconeClient.retrieve() for the Phase 5 orchestrator.

    Accepts a natural language query with optional channel_id and date-range filters,
    delegates to PineconeClient.retrieve() for hybrid search + neural reranking,
    and returns a RetrievalResult Pydantic model.

    Usage:
        client = PineconeClient(api_key=..., index_name=...)
        agent = RetrieverAgent(pinecone_client=client)
        result = await agent.retrieve("what did we decide about the API design?")
        print(result.results[0]["metadata"]["summary_text"])
    """

    def __init__(
        self,
        pinecone_client: PineconeClient,
        top_k: int = 20,
        top_n: int = 5,
    ):
        """
        Args:
            pinecone_client: Initialized PineconeClient instance.
            top_k: Candidates to fetch from Pinecone before reranking (default: 20).
            top_n: Final results to return after reranking (default: 5).
        """
        self._client = pinecone_client
        self._top_k = top_k
        self._top_n = top_n

    async def retrieve(
        self,
        query_text: str,
        channel_id: Optional[str] = None,
        start_ts: Optional[int] = None,
        end_ts: Optional[int] = None,
    ) -> RetrievalResult:
        """Run the full hybrid RAG retrieval pipeline for a query.

        Calls PineconeClient.retrieve() which:
          1. Runs a hybrid dense+sparse Pinecone query (top_k candidates).
          2. Applies Pinecone's bge-reranker-v2-m3 neural reranker (top_n results).

        Args:
            query_text: Natural language question or topic to search for.
            channel_id: Optional Slack channel ID to scope results ($eq filter).
            start_ts: Optional start of date window as Unix epoch integer ($gte).
            end_ts: Optional end of date window as Unix epoch integer ($lte).

        Returns:
            RetrievalResult with query, reranked results list, total_candidates,
            and returned_count populated.
        """
        reranked = self._client.retrieve(
            query_text=query_text,
            channel_id=channel_id,
            start_ts=start_ts,
            end_ts=end_ts,
            top_k=self._top_k,
            top_n=self._top_n,
        )

        return RetrievalResult(
            query=query_text,
            results=reranked,
            total_candidates=self._top_k,
            returned_count=len(reranked),
        )
