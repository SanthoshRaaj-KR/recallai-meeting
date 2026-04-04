"""
PineconeClient — hybrid vector upsert and query for meeting memory.

Wraps Pinecone SDK v8 to provide index creation (dotproduct metric, dim=1536),
hybrid upsert (dense via OpenAI text-embedding-3-small + sparse via
pinecone-sparse-english-v0), and metadata-filtered hybrid query.

Design constraints:
- metric="dotproduct" is hardcoded — hybrid search requires dotproduct; wrong
  metric requires full re-ingestion to fix (see PITFALLS.md Pitfall 1).
- Sparse vectors use Pinecone hosted inference (pinecone-sparse-english-v0),
  NOT local BM25 — no corpus fitting required (satisfies INFRA-04).
- Metadata stored as flat scalars/lists only — no nested JSON blobs — so
  Pinecone server-side filters work correctly (see PITFALLS.md Pitfall 7).
- input_type="passage" for upsert, input_type="query" for query — per
  Pinecone inference API requirements.
- start_ts filter uses integer Unix epoch with $gte/$lte operators — per
  locked decision (all timestamps are int, never datetime).
"""

import logging
from typing import Optional

from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec

from storage.models import MeetingRecord

logger = logging.getLogger(__name__)

DENSE_MODEL = "text-embedding-3-small"
DENSE_DIMENSION = 1536
SPARSE_MODEL = "pinecone-sparse-english-v0"
INDEX_METRIC = "dotproduct"


class PineconeClient:
    """
    Wrapper around Pinecone SDK v8 for hybrid meeting-memory search.

    Provides:
    - ensure_index_exists(): create the Pinecone index if absent
    - upsert_meeting(record): embed and upsert a MeetingRecord as a hybrid vector
    - query(...): hybrid query with optional channel_id and date range filters
    """

    def __init__(
        self,
        api_key: str,
        index_name: str = "meeting-memory",
        cloud: str = "aws",
        region: str = "us-east-1",
        openai_api_key: Optional[str] = None,
    ):
        """
        Initialize PineconeClient.

        Args:
            api_key: Pinecone API key.
            index_name: Name of the Pinecone index to use.
            cloud: Cloud provider for ServerlessSpec (default: "aws").
            region: Region for ServerlessSpec (default: "us-east-1").
            openai_api_key: Optional OpenAI API key. If not provided, the
                OpenAI client will use the OPENAI_API_KEY environment variable.
        """
        self._pc = Pinecone(api_key=api_key)
        self._openai = OpenAI(api_key=openai_api_key) if openai_api_key else OpenAI()
        self._index_name = index_name
        self._cloud = cloud
        self._region = region
        self._index = None  # lazy-loaded

    def ensure_index_exists(self) -> None:
        """
        Create the Pinecone index if it does not already exist.

        Uses metric="dotproduct" which is required for hybrid (dense + sparse)
        search. Creating with cosine or euclidean will cause hybrid queries to
        fail at runtime — never change this metric.
        """
        existing_names = [idx.name for idx in self._pc.list_indexes()]
        if self._index_name in existing_names:
            logger.info("Pinecone index '%s' already exists", self._index_name)
            return

        self._pc.create_index(
            name=self._index_name,
            dimension=DENSE_DIMENSION,
            metric=INDEX_METRIC,
            vector_type="dense",
            spec=ServerlessSpec(cloud=self._cloud, region=self._region),
        )
        logger.info(
            "Created Pinecone index '%s' (metric=%s, dim=%d)",
            self._index_name,
            INDEX_METRIC,
            DENSE_DIMENSION,
        )

    def _get_index(self):
        """Lazy-load the index handle to avoid unnecessary control-plane calls."""
        if self._index is None:
            self._index = self._pc.Index(self._index_name)
        return self._index

    def _embed_dense(self, text: str) -> list[float]:
        """
        Generate a dense embedding via OpenAI text-embedding-3-small.

        Returns a list[float] of length DENSE_DIMENSION (1536).
        """
        response = self._openai.embeddings.create(model=DENSE_MODEL, input=[text])
        return response.data[0].embedding

    def _embed_sparse(self, text: str, input_type: str = "passage") -> dict:
        """
        Generate a sparse embedding via Pinecone inference (pinecone-sparse-english-v0).

        Uses Pinecone's hosted model — no local BM25 corpus fitting required.

        Args:
            text: Text to encode.
            input_type: "passage" for upsert, "query" for query (per Pinecone docs).

        Returns:
            dict with keys "indices" (list[int]) and "values" (list[float]).

        Note on SDK v8 sparse shape:
            pc.inference.embed() returns an EmbeddingsList.  response[0] is a
            SparseEmbedding object with attributes:
              - sparse_indices: list[int]
              - sparse_values:  list[float]
            (NOT a nested .sparse_values sub-object with .index/.value properties)
        """
        response = self._pc.inference.embed(
            model=SPARSE_MODEL,
            inputs=[text],
            parameters={"input_type": input_type},
        )
        embedding = response[0]
        return {
            "indices": embedding.sparse_indices,
            "values": embedding.sparse_values,
        }

    def upsert_meeting(self, record: MeetingRecord) -> None:
        """
        Upsert a MeetingRecord as a hybrid vector (dense + sparse) with flat metadata.

        The summary_text is used as the source for both dense and sparse embeddings.
        All metadata fields are flat scalars or lists of scalars — no nested objects —
        to ensure Pinecone server-side filtering works correctly.

        Args:
            record: The canonical MeetingRecord to store.
        """
        dense = self._embed_dense(record.summary_text)
        sparse = self._embed_sparse(record.summary_text, input_type="passage")

        # Flat metadata schema — no nested objects (see PITFALLS.md Pitfall 7)
        metadata = {
            "channel_id": record.channel_id,
            "channel_name": record.channel_name,
            "start_ts": record.start_ts,
            "end_ts": record.end_ts,
            "duration_seconds": record.duration_seconds,
            "participants": record.participants,
            "topics_covered": record.topics_covered,
            "decisions": record.decisions,
            "series_name": record.series_name,
            "summary_text": record.summary_text,
            "status": record.status,
        }

        self._get_index().upsert(vectors=[{
            "id": record.meeting_id,
            "values": dense,
            "sparse_values": sparse,
            "metadata": metadata,
        }])
        logger.info("Upserted meeting '%s' to Pinecone index '%s'", record.meeting_id, self._index_name)

    def query(
        self,
        query_text: str,
        channel_id: Optional[str] = None,
        start_ts: Optional[int] = None,
        end_ts: Optional[int] = None,
        top_k: int = 10,
        alpha: float = 0.7,
    ) -> list[dict]:
        """
        Hybrid query against the Pinecone index with optional metadata filters.

        Builds a metadata filter from the provided constraints and issues a hybrid
        query combining dense and sparse vectors. Uses input_type="query" for the
        sparse embedding (not "passage").

        Args:
            query_text: Natural language query to embed.
            channel_id: Optional Slack channel ID to filter by ($eq).
            start_ts: Optional start of date range as Unix epoch integer ($gte).
            end_ts: Optional end of date range as Unix epoch integer ($lte).
            top_k: Number of results to return (default: 10).
            alpha: Float in [0.0, 1.0] controlling dense/sparse balance.
                alpha=1.0 → pure dense (semantic search).
                alpha=0.0 → pure sparse (keyword search).
                Default 0.7 is research-backed for conversational meeting queries.
                Dense values are scaled by alpha; sparse values by (1 - alpha).

        Returns:
            List of dicts, each with "id", "score", and "metadata" keys.
        """
        dense = self._embed_dense(query_text)
        sparse = self._embed_sparse(query_text, input_type="query")

        # Alpha weighting: scale dense by alpha, sparse by (1 - alpha)
        dense = [v * alpha for v in dense]
        sparse = {"indices": sparse["indices"], "values": [v * (1 - alpha) for v in sparse["values"]]}

        # Build metadata filter — flat Pinecone filter operators
        filter_conditions: dict = {}

        if channel_id:
            filter_conditions["channel_id"] = {"$eq": channel_id}

        if start_ts is not None:
            filter_conditions.setdefault("start_ts", {})["$gte"] = start_ts

        if end_ts is not None:
            filter_conditions.setdefault("start_ts", {})["$lte"] = end_ts

        query_kwargs: dict = {
            "vector": dense,
            "sparse_vector": sparse,
            "top_k": top_k,
            "include_metadata": True,
        }
        if filter_conditions:
            query_kwargs["filter"] = filter_conditions

        response = self._get_index().query(**query_kwargs)
        return [
            {"id": m.id, "score": m.score, "metadata": m.metadata}
            for m in response.matches
        ]

    def rerank(
        self,
        query_text: str,
        results: list[dict],
        top_n: int = 5,
    ) -> list[dict]:
        """
        Rerank query results using Pinecone's neural reranker.

        Calls pc.inference.rerank() with model="bge-reranker-v2-m3". Documents are
        built from result metadata summary_text. Returns reranked results with an
        added "rerank_score" field.

        Args:
            query_text: The original query string (used as the reranker's query).
            results: List of dicts from query() — each has "id", "score", "metadata".
            top_n: Number of results to return after reranking (default: 5).

        Returns:
            List of dicts, each with "id", "score", "metadata", "rerank_score" keys,
            ordered by descending rerank_score.
        """
        if not results:
            return []

        documents = [
            {"id": r["id"], "text": r["metadata"].get("summary_text", "")}
            for r in results
        ]

        reranked = self._pc.inference.rerank(
            model="bge-reranker-v2-m3",
            query=query_text,
            documents=documents,
            top_n=top_n,
        )

        output = []
        for item in reranked:
            original = results[item.index]
            output.append({
                "id": original["id"],
                "score": original["score"],
                "metadata": original["metadata"],
                "rerank_score": item.score,
            })
        return output

    def retrieve(
        self,
        query_text: str,
        channel_id: Optional[str] = None,
        start_ts: Optional[int] = None,
        end_ts: Optional[int] = None,
        alpha: float = 0.7,
        top_k: int = 20,
        top_n: int = 5,
    ) -> list[dict]:
        """
        Full hybrid retrieval pipeline: hybrid query then neural rerank.

        Calls query() with top_k=20 to get broad candidates, then rerank() to
        reduce to the top_n most relevant results using Pinecone's neural reranker.

        Args:
            query_text: Natural language query.
            channel_id: Optional Slack channel ID filter ($eq).
            start_ts: Optional start of date range as Unix epoch integer ($gte).
            end_ts: Optional end of date range as Unix epoch integer ($lte).
            alpha: Dense/sparse balance for query() (default: 0.7).
            top_k: Candidates to fetch before reranking (default: 20).
            top_n: Final results to return after reranking (default: 5).

        Returns:
            List of top_n dicts ordered by descending rerank_score, each with
            "id", "score", "metadata", "rerank_score" keys.
        """
        candidates = self.query(
            query_text=query_text,
            channel_id=channel_id,
            start_ts=start_ts,
            end_ts=end_ts,
            top_k=top_k,
            alpha=alpha,
        )
        return self.rerank(query_text, candidates, top_n=top_n)
