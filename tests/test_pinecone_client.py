"""
Tests for PineconeClient — hybrid vector upsert and query wrapper.

Unit tests use unittest.mock.patch to mock Pinecone SDK and OpenAI clients.
The live smoke test requires PINECONE_API_KEY and OPENAI_API_KEY to be set.
"""

import os
from unittest.mock import MagicMock, patch, call
import pytest

from storage.models import MeetingRecord, ActionItem


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_record(**overrides) -> MeetingRecord:
    """Create a minimal MeetingRecord for test use."""
    defaults = {
        "meeting_id": "mtg-001",
        "channel_id": "C01234567",
        "channel_name": "general",
        "start_ts": 1711900000,
        "end_ts": 1711903600,
        "duration_seconds": 3600,
        "participants": ["Alice", "Bob"],
        "summary_text": "Team discussed Q2 roadmap and sprint planning.",
        "topics_covered": ["roadmap", "sprint"],
        "decisions": ["Ship by April 15"],
        "series_name": "weekly-sync",
        "status": "complete",
    }
    defaults.update(overrides)
    return MeetingRecord(**defaults)


def make_sparse_embedding(indices=(1, 42, 100), values=(0.5, 0.3, 0.8)):
    """Create a mock sparse embedding object matching SparseEmbedding SDK shape."""
    emb = MagicMock()
    emb.sparse_indices = list(indices)
    emb.sparse_values = list(values)
    return emb


def make_dense_embedding(dim=1536):
    """Create a mock OpenAI embedding response."""
    mock_resp = MagicMock()
    mock_resp.data = [MagicMock()]
    mock_resp.data[0].embedding = [0.1] * dim
    return mock_resp


# ---------------------------------------------------------------------------
# Unit Tests
# ---------------------------------------------------------------------------


class TestPineconeClientInit:
    """Test __init__ stores configuration correctly."""

    @patch("storage.pinecone_client.Pinecone")
    @patch("storage.pinecone_client.OpenAI")
    def test_init_stores_api_key_and_index_name(self, mock_openai, mock_pinecone):
        from storage.pinecone_client import PineconeClient

        client = PineconeClient(api_key="test-key", index_name="my-index")
        assert client._index_name == "my-index"
        mock_pinecone.assert_called_once_with(api_key="test-key")

    @patch("storage.pinecone_client.Pinecone")
    @patch("storage.pinecone_client.OpenAI")
    def test_init_stores_cloud_and_region(self, mock_openai, mock_pinecone):
        from storage.pinecone_client import PineconeClient

        client = PineconeClient(
            api_key="test-key", cloud="gcp", region="us-central1"
        )
        assert client._cloud == "gcp"
        assert client._region == "us-central1"

    @patch("storage.pinecone_client.Pinecone")
    @patch("storage.pinecone_client.OpenAI")
    def test_init_default_cloud_region(self, mock_openai, mock_pinecone):
        from storage.pinecone_client import PineconeClient

        client = PineconeClient(api_key="test-key")
        assert client._cloud == "aws"
        assert client._region == "us-east-1"

    @patch("storage.pinecone_client.Pinecone")
    @patch("storage.pinecone_client.OpenAI")
    def test_init_openai_with_explicit_key(self, mock_openai, mock_pinecone):
        from storage.pinecone_client import PineconeClient

        PineconeClient(api_key="test-key", openai_api_key="oai-key")
        mock_openai.assert_called_once_with(api_key="oai-key")

    @patch("storage.pinecone_client.Pinecone")
    @patch("storage.pinecone_client.OpenAI")
    def test_init_openai_no_explicit_key(self, mock_openai, mock_pinecone):
        from storage.pinecone_client import PineconeClient

        PineconeClient(api_key="test-key")
        mock_openai.assert_called_once_with()


class TestEnsureIndexExists:
    """Test ensure_index_exists() creates index with dotproduct metric and dim=1536."""

    @patch("storage.pinecone_client.Pinecone")
    @patch("storage.pinecone_client.OpenAI")
    def test_creates_index_with_dotproduct_and_1536(self, mock_openai, mock_pinecone):
        from storage.pinecone_client import PineconeClient

        pc_instance = mock_pinecone.return_value
        # Simulate index does not exist
        pc_instance.list_indexes.return_value = []

        client = PineconeClient(api_key="test-key", index_name="meeting-memory")
        client.ensure_index_exists()

        pc_instance.create_index.assert_called_once()
        call_kwargs = pc_instance.create_index.call_args[1]
        assert call_kwargs["metric"] == "dotproduct"
        assert call_kwargs["dimension"] == 1536
        assert call_kwargs["name"] == "meeting-memory"

    @patch("storage.pinecone_client.Pinecone")
    @patch("storage.pinecone_client.OpenAI")
    def test_creates_index_with_vector_type_dense(self, mock_openai, mock_pinecone):
        from storage.pinecone_client import PineconeClient

        pc_instance = mock_pinecone.return_value
        pc_instance.list_indexes.return_value = []

        client = PineconeClient(api_key="test-key")
        client.ensure_index_exists()

        call_kwargs = pc_instance.create_index.call_args[1]
        assert call_kwargs.get("vector_type") == "dense"

    @patch("storage.pinecone_client.Pinecone")
    @patch("storage.pinecone_client.OpenAI")
    def test_does_not_recreate_existing_index(self, mock_openai, mock_pinecone):
        from storage.pinecone_client import PineconeClient

        pc_instance = mock_pinecone.return_value
        # Simulate index already exists (list_indexes returns objects with .name)
        existing_idx = MagicMock()
        existing_idx.name = "meeting-memory"
        pc_instance.list_indexes.return_value = [existing_idx]

        client = PineconeClient(api_key="test-key", index_name="meeting-memory")
        client.ensure_index_exists()  # Should NOT raise

        pc_instance.create_index.assert_not_called()


class TestEmbedDense:
    """Test _embed_dense() calls OpenAI text-embedding-3-small and returns list[float]."""

    @patch("storage.pinecone_client.Pinecone")
    @patch("storage.pinecone_client.OpenAI")
    def test_embed_dense_calls_openai_with_correct_model(self, mock_openai, mock_pinecone):
        from storage.pinecone_client import PineconeClient, DENSE_MODEL

        openai_instance = mock_openai.return_value
        openai_instance.embeddings.create.return_value = make_dense_embedding()

        client = PineconeClient(api_key="test-key")
        client._embed_dense("some text")

        openai_instance.embeddings.create.assert_called_once_with(
            model=DENSE_MODEL, input=["some text"]
        )

    @patch("storage.pinecone_client.Pinecone")
    @patch("storage.pinecone_client.OpenAI")
    def test_embed_dense_returns_list_of_floats_length_1536(self, mock_openai, mock_pinecone):
        from storage.pinecone_client import PineconeClient

        openai_instance = mock_openai.return_value
        openai_instance.embeddings.create.return_value = make_dense_embedding(1536)

        client = PineconeClient(api_key="test-key")
        result = client._embed_dense("some text")

        assert isinstance(result, list)
        assert len(result) == 1536
        assert all(isinstance(v, float) for v in result)


class TestEmbedSparse:
    """Test _embed_sparse() calls Pinecone inference with pinecone-sparse-english-v0."""

    @patch("storage.pinecone_client.Pinecone")
    @patch("storage.pinecone_client.OpenAI")
    def test_embed_sparse_calls_inference_with_correct_model(self, mock_openai, mock_pinecone):
        from storage.pinecone_client import PineconeClient, SPARSE_MODEL

        pc_instance = mock_pinecone.return_value
        mock_emb = make_sparse_embedding()
        pc_instance.inference.embed.return_value = MagicMock()
        pc_instance.inference.embed.return_value.__getitem__ = lambda self, i: mock_emb

        client = PineconeClient(api_key="test-key")
        client._embed_sparse("some text", input_type="passage")

        pc_instance.inference.embed.assert_called_once_with(
            model=SPARSE_MODEL,
            inputs=["some text"],
            parameters={"input_type": "passage"},
        )

    @patch("storage.pinecone_client.Pinecone")
    @patch("storage.pinecone_client.OpenAI")
    def test_embed_sparse_returns_indices_and_values(self, mock_openai, mock_pinecone):
        from storage.pinecone_client import PineconeClient

        pc_instance = mock_pinecone.return_value
        mock_emb = make_sparse_embedding(indices=[1, 42], values=[0.5, 0.3])
        mock_response = MagicMock()
        mock_response.__getitem__ = lambda self, i: mock_emb
        pc_instance.inference.embed.return_value = mock_response

        client = PineconeClient(api_key="test-key")
        result = client._embed_sparse("some text")

        assert "indices" in result
        assert "values" in result
        assert result["indices"] == [1, 42]
        assert result["values"] == [0.5, 0.3]

    @patch("storage.pinecone_client.Pinecone")
    @patch("storage.pinecone_client.OpenAI")
    def test_embed_sparse_uses_input_type_parameter(self, mock_openai, mock_pinecone):
        from storage.pinecone_client import PineconeClient

        pc_instance = mock_pinecone.return_value
        mock_emb = make_sparse_embedding()
        mock_response = MagicMock()
        mock_response.__getitem__ = lambda self, i: mock_emb
        pc_instance.inference.embed.return_value = mock_response

        client = PineconeClient(api_key="test-key")
        client._embed_sparse("query text", input_type="query")

        call_kwargs = pc_instance.inference.embed.call_args[1]
        assert call_kwargs["parameters"]["input_type"] == "query"


class TestUpsertMeeting:
    """Test upsert_meeting() produces correct vector structure with flat metadata."""

    def _setup_client(self, mock_openai, mock_pinecone):
        from storage.pinecone_client import PineconeClient

        pc_instance = mock_pinecone.return_value
        openai_instance = mock_openai.return_value

        # Mock dense embedding
        openai_instance.embeddings.create.return_value = make_dense_embedding()

        # Mock sparse embedding
        mock_emb = make_sparse_embedding(indices=[1, 2], values=[0.9, 0.7])
        mock_response = MagicMock()
        mock_response.__getitem__ = lambda self, i: mock_emb
        pc_instance.inference.embed.return_value = mock_response

        # Mock index
        mock_index = MagicMock()
        pc_instance.Index.return_value = mock_index

        client = PineconeClient(api_key="test-key")
        return client, pc_instance, mock_index

    @patch("storage.pinecone_client.Pinecone")
    @patch("storage.pinecone_client.OpenAI")
    def test_upsert_calls_index_upsert(self, mock_openai, mock_pinecone):
        client, pc_instance, mock_index = self._setup_client(mock_openai, mock_pinecone)
        record = make_record()

        client.upsert_meeting(record)

        mock_index.upsert.assert_called_once()

    @patch("storage.pinecone_client.Pinecone")
    @patch("storage.pinecone_client.OpenAI")
    def test_upsert_vector_id_is_meeting_id(self, mock_openai, mock_pinecone):
        client, pc_instance, mock_index = self._setup_client(mock_openai, mock_pinecone)
        record = make_record(meeting_id="mtg-abc")

        client.upsert_meeting(record)

        vectors = mock_index.upsert.call_args[1]["vectors"]
        assert vectors[0]["id"] == "mtg-abc"

    @patch("storage.pinecone_client.Pinecone")
    @patch("storage.pinecone_client.OpenAI")
    def test_upsert_has_dense_and_sparse_vectors(self, mock_openai, mock_pinecone):
        client, pc_instance, mock_index = self._setup_client(mock_openai, mock_pinecone)
        record = make_record()

        client.upsert_meeting(record)

        vectors = mock_index.upsert.call_args[1]["vectors"]
        v = vectors[0]
        assert "values" in v          # dense
        assert "sparse_values" in v   # sparse
        assert isinstance(v["values"], list)
        assert "indices" in v["sparse_values"]
        assert "values" in v["sparse_values"]

    @patch("storage.pinecone_client.Pinecone")
    @patch("storage.pinecone_client.OpenAI")
    def test_upsert_metadata_contains_flat_scalars(self, mock_openai, mock_pinecone):
        client, pc_instance, mock_index = self._setup_client(mock_openai, mock_pinecone)
        record = make_record(
            channel_id="C999",
            channel_name="eng",
            start_ts=1711900000,
            end_ts=1711903600,
            participants=["Alice", "Bob"],
            topics_covered=["sprint"],
            decisions=["Ship v2"],
            series_name="weekly",
            summary_text="Sprint review meeting",
            status="complete",
        )

        client.upsert_meeting(record)

        vectors = mock_index.upsert.call_args[1]["vectors"]
        meta = vectors[0]["metadata"]

        assert meta["channel_id"] == "C999"
        assert meta["channel_name"] == "eng"
        assert meta["start_ts"] == 1711900000
        assert meta["end_ts"] == 1711903600
        assert meta["participants"] == ["Alice", "Bob"]
        assert meta["topics_covered"] == ["sprint"]
        assert meta["decisions"] == ["Ship v2"]
        assert meta["series_name"] == "weekly"
        assert meta["summary_text"] == "Sprint review meeting"
        assert meta["status"] == "complete"

    @patch("storage.pinecone_client.Pinecone")
    @patch("storage.pinecone_client.OpenAI")
    def test_upsert_uses_summary_text_for_embeddings(self, mock_openai, mock_pinecone):
        client, pc_instance, mock_index = self._setup_client(mock_openai, mock_pinecone)
        record = make_record(summary_text="Specific summary text for embedding")

        client.upsert_meeting(record)

        # OpenAI was called with the summary text
        openai_instance = mock_openai.return_value
        call_args = openai_instance.embeddings.create.call_args
        assert "Specific summary text for embedding" in call_args[1]["input"]

        # Pinecone inference was called with summary text
        pc_instance = mock_pinecone.return_value
        sparse_call = pc_instance.inference.embed.call_args
        assert "Specific summary text for embedding" in sparse_call[1]["inputs"]


class TestQuery:
    """Test query() builds correct filter and calls index.query."""

    def _setup_client(self, mock_openai, mock_pinecone):
        from storage.pinecone_client import PineconeClient

        pc_instance = mock_pinecone.return_value
        openai_instance = mock_openai.return_value

        openai_instance.embeddings.create.return_value = make_dense_embedding()

        mock_emb = make_sparse_embedding()
        mock_response = MagicMock()
        mock_response.__getitem__ = lambda self, i: mock_emb
        pc_instance.inference.embed.return_value = mock_response

        mock_index = MagicMock()
        mock_index.query.return_value = MagicMock(matches=[])
        pc_instance.Index.return_value = mock_index

        client = PineconeClient(api_key="test-key")
        return client, pc_instance, mock_index

    @patch("storage.pinecone_client.Pinecone")
    @patch("storage.pinecone_client.OpenAI")
    def test_query_calls_index_query(self, mock_openai, mock_pinecone):
        client, pc_instance, mock_index = self._setup_client(mock_openai, mock_pinecone)

        client.query("some query text")
        mock_index.query.assert_called_once()

    @patch("storage.pinecone_client.Pinecone")
    @patch("storage.pinecone_client.OpenAI")
    def test_query_channel_id_filter_uses_eq(self, mock_openai, mock_pinecone):
        client, pc_instance, mock_index = self._setup_client(mock_openai, mock_pinecone)

        client.query("some query", channel_id="C01234567")

        call_kwargs = mock_index.query.call_args[1]
        assert "filter" in call_kwargs
        f = call_kwargs["filter"]
        assert "channel_id" in f
        assert f["channel_id"] == {"$eq": "C01234567"}

    @patch("storage.pinecone_client.Pinecone")
    @patch("storage.pinecone_client.OpenAI")
    def test_query_date_range_filter_uses_gte_lte(self, mock_openai, mock_pinecone):
        client, pc_instance, mock_index = self._setup_client(mock_openai, mock_pinecone)

        client.query("some query", start_ts=1711900000, end_ts=1712000000)

        call_kwargs = mock_index.query.call_args[1]
        f = call_kwargs["filter"]
        assert "start_ts" in f
        assert f["start_ts"]["$gte"] == 1711900000
        assert f["start_ts"]["$lte"] == 1712000000

    @patch("storage.pinecone_client.Pinecone")
    @patch("storage.pinecone_client.OpenAI")
    def test_query_no_filter_when_no_constraints(self, mock_openai, mock_pinecone):
        client, pc_instance, mock_index = self._setup_client(mock_openai, mock_pinecone)

        client.query("generic query")

        call_kwargs = mock_index.query.call_args[1]
        assert "filter" not in call_kwargs

    @patch("storage.pinecone_client.Pinecone")
    @patch("storage.pinecone_client.OpenAI")
    def test_query_includes_channel_and_date_range_together(self, mock_openai, mock_pinecone):
        client, pc_instance, mock_index = self._setup_client(mock_openai, mock_pinecone)

        client.query(
            "some query",
            channel_id="C999",
            start_ts=1711900000,
            end_ts=1712000000,
        )

        call_kwargs = mock_index.query.call_args[1]
        f = call_kwargs["filter"]
        assert "channel_id" in f
        assert "start_ts" in f
        assert f["channel_id"] == {"$eq": "C999"}
        assert f["start_ts"]["$gte"] == 1711900000
        assert f["start_ts"]["$lte"] == 1712000000

    @patch("storage.pinecone_client.Pinecone")
    @patch("storage.pinecone_client.OpenAI")
    def test_query_returns_list_of_dicts(self, mock_openai, mock_pinecone):
        client, pc_instance, mock_index = self._setup_client(mock_openai, mock_pinecone)

        # Simulate Pinecone match objects
        match = MagicMock()
        match.id = "mtg-001"
        match.score = 0.95
        match.metadata = {"channel_id": "C999"}
        mock_index.query.return_value = MagicMock(matches=[match])

        results = client.query("some query")

        assert isinstance(results, list)
        assert len(results) == 1
        assert results[0]["id"] == "mtg-001"
        assert results[0]["score"] == 0.95
        assert results[0]["metadata"]["channel_id"] == "C999"

    @patch("storage.pinecone_client.Pinecone")
    @patch("storage.pinecone_client.OpenAI")
    def test_query_uses_query_input_type_for_sparse(self, mock_openai, mock_pinecone):
        client, pc_instance, mock_index = self._setup_client(mock_openai, mock_pinecone)

        client.query("test query")

        # Second call to inference.embed should use input_type=query
        embed_calls = pc_instance.inference.embed.call_args_list
        assert len(embed_calls) == 1
        call_kwargs = embed_calls[0][1]
        assert call_kwargs["parameters"]["input_type"] == "query"

    @patch("storage.pinecone_client.Pinecone")
    @patch("storage.pinecone_client.OpenAI")
    def test_query_passes_top_k(self, mock_openai, mock_pinecone):
        client, pc_instance, mock_index = self._setup_client(mock_openai, mock_pinecone)

        client.query("some query", top_k=5)

        call_kwargs = mock_index.query.call_args[1]
        assert call_kwargs["top_k"] == 5

    @patch("storage.pinecone_client.Pinecone")
    @patch("storage.pinecone_client.OpenAI")
    def test_query_passes_include_metadata(self, mock_openai, mock_pinecone):
        client, pc_instance, mock_index = self._setup_client(mock_openai, mock_pinecone)

        client.query("some query")

        call_kwargs = mock_index.query.call_args[1]
        assert call_kwargs.get("include_metadata") is True


# ---------------------------------------------------------------------------
# Live smoke test (skipped unless PINECONE_API_KEY is set)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not os.environ.get("PINECONE_API_KEY"),
    reason="PINECONE_API_KEY not set — skipping live smoke test",
)
def test_live_hybrid_smoke():
    """
    Live integration test: create index, upsert a synthetic meeting, query by
    channel_id and date range. Requires PINECONE_API_KEY (and optionally
    OPENAI_API_KEY) to be set in the environment.
    """
    import time

    from storage.pinecone_client import PineconeClient

    api_key = os.environ["PINECONE_API_KEY"]
    openai_api_key = os.environ.get("OPENAI_API_KEY")

    client = PineconeClient(
        api_key=api_key,
        index_name="meeting-memory-test",
        openai_api_key=openai_api_key,
    )

    # 1. Create index
    client.ensure_index_exists()

    # 2. Upsert a synthetic meeting
    record = make_record(
        meeting_id="smoke-test-meeting",
        channel_id="C_SMOKE_TEST",
        summary_text="Smoke test meeting for hybrid search validation. Q2 roadmap sprint planning.",
        start_ts=1711900000,
        end_ts=1711903600,
    )
    client.upsert_meeting(record)

    # Allow time for the upsert to be indexed
    time.sleep(5)

    # 3. Query by channel_id
    results_by_channel = client.query(
        "roadmap sprint planning",
        channel_id="C_SMOKE_TEST",
        top_k=5,
    )
    assert len(results_by_channel) > 0, "Expected at least one result when filtering by channel_id"
    result_ids = [r["id"] for r in results_by_channel]
    assert "smoke-test-meeting" in result_ids, (
        f"Upserted meeting 'smoke-test-meeting' not found in channel results: {result_ids}"
    )

    # 4. Query by date range
    results_by_date = client.query(
        "sprint planning",
        start_ts=1711800000,
        end_ts=1712000000,
        top_k=5,
    )
    assert len(results_by_date) > 0, "Expected at least one result when filtering by date range"

    # 5. Query with both filters
    results_combined = client.query(
        "roadmap",
        channel_id="C_SMOKE_TEST",
        start_ts=1711800000,
        end_ts=1712000000,
        top_k=5,
    )
    assert len(results_combined) > 0, "Expected results with combined channel + date filter"


# ---------------------------------------------------------------------------
# TestQueryAlpha — alpha weighting for dense/sparse balance
# ---------------------------------------------------------------------------


class TestQueryAlpha:
    """Test query() alpha parameter scales dense and sparse vectors correctly."""

    def _setup_client(self, mock_openai, mock_pinecone, dense_value=0.1, sparse_values=(0.5, 0.3, 0.8)):
        from storage.pinecone_client import PineconeClient

        pc_instance = mock_pinecone.return_value
        openai_instance = mock_openai.return_value

        # Mock dense embedding with configurable value
        mock_resp = MagicMock()
        mock_resp.data = [MagicMock()]
        mock_resp.data[0].embedding = [dense_value] * 1536
        openai_instance.embeddings.create.return_value = mock_resp

        # Mock sparse embedding with configurable values
        mock_emb = make_sparse_embedding(values=sparse_values)
        mock_response = MagicMock()
        mock_response.__getitem__ = lambda self, i: mock_emb
        pc_instance.inference.embed.return_value = mock_response

        mock_index = MagicMock()
        mock_index.query.return_value = MagicMock(matches=[])
        pc_instance.Index.return_value = mock_index

        client = PineconeClient(api_key="test-key")
        return client, pc_instance, mock_index

    @patch("storage.pinecone_client.Pinecone")
    @patch("storage.pinecone_client.OpenAI")
    def test_query_alpha_scales_dense_values(self, mock_openai, mock_pinecone):
        client, pc_instance, mock_index = self._setup_client(
            mock_openai, mock_pinecone, dense_value=0.1
        )

        client.query("text", alpha=0.5)

        call_kwargs = mock_index.query.call_args[1]
        vector = call_kwargs["vector"]
        assert all(abs(v - 0.05) < 1e-9 for v in vector), (
            f"Expected all dense values to be 0.05, got: {vector[:3]}"
        )

    @patch("storage.pinecone_client.Pinecone")
    @patch("storage.pinecone_client.OpenAI")
    def test_query_alpha_scales_sparse_values(self, mock_openai, mock_pinecone):
        client, pc_instance, mock_index = self._setup_client(
            mock_openai, mock_pinecone, sparse_values=(0.5, 0.3, 0.8)
        )

        client.query("text", alpha=0.5)

        call_kwargs = mock_index.query.call_args[1]
        sparse_values = call_kwargs["sparse_vector"]["values"]
        expected = [0.25, 0.15, 0.4]
        assert all(abs(a - b) < 1e-9 for a, b in zip(sparse_values, expected)), (
            f"Expected sparse values {expected}, got: {sparse_values}"
        )

    @patch("storage.pinecone_client.Pinecone")
    @patch("storage.pinecone_client.OpenAI")
    def test_query_alpha_default_is_0_7(self, mock_openai, mock_pinecone):
        client, pc_instance, mock_index = self._setup_client(
            mock_openai, mock_pinecone, dense_value=1.0, sparse_values=(1.0,)
        )

        client.query("text")  # no alpha arg — uses default

        call_kwargs = mock_index.query.call_args[1]
        assert abs(call_kwargs["vector"][0] - 0.7) < 1e-9, (
            f"Expected dense[0]=0.7, got {call_kwargs['vector'][0]}"
        )
        assert abs(call_kwargs["sparse_vector"]["values"][0] - 0.3) < 1e-9, (
            f"Expected sparse[0]=0.3, got {call_kwargs['sparse_vector']['values'][0]}"
        )

    @patch("storage.pinecone_client.Pinecone")
    @patch("storage.pinecone_client.OpenAI")
    def test_query_alpha_1_0_pure_dense(self, mock_openai, mock_pinecone):
        client, pc_instance, mock_index = self._setup_client(
            mock_openai, mock_pinecone, dense_value=0.5, sparse_values=(0.9, 0.6)
        )

        client.query("text", alpha=1.0)

        call_kwargs = mock_index.query.call_args[1]
        # Dense scaled by 1.0 — unchanged
        assert all(abs(v - 0.5) < 1e-9 for v in call_kwargs["vector"]), (
            "Expected all dense values to remain 0.5 with alpha=1.0"
        )
        # Sparse scaled by 0.0 — all zero
        assert all(abs(v - 0.0) < 1e-9 for v in call_kwargs["sparse_vector"]["values"]), (
            "Expected all sparse values to be 0.0 with alpha=1.0"
        )

    @patch("storage.pinecone_client.Pinecone")
    @patch("storage.pinecone_client.OpenAI")
    def test_query_alpha_0_0_pure_sparse(self, mock_openai, mock_pinecone):
        client, pc_instance, mock_index = self._setup_client(
            mock_openai, mock_pinecone, dense_value=0.5, sparse_values=(0.9, 0.6)
        )

        client.query("text", alpha=0.0)

        call_kwargs = mock_index.query.call_args[1]
        # Dense scaled by 0.0 — all zero
        assert all(abs(v - 0.0) < 1e-9 for v in call_kwargs["vector"]), (
            "Expected all dense values to be 0.0 with alpha=0.0"
        )
        # Sparse scaled by 1.0 — unchanged
        sparse_values = call_kwargs["sparse_vector"]["values"]
        expected = [0.9, 0.6]
        assert all(abs(a - b) < 1e-9 for a, b in zip(sparse_values, expected)), (
            f"Expected sparse values {expected}, got: {sparse_values}"
        )

    @patch("storage.pinecone_client.Pinecone")
    @patch("storage.pinecone_client.OpenAI")
    def test_query_backward_compatible_no_alpha_arg(self, mock_openai, mock_pinecone):
        client, pc_instance, mock_index = self._setup_client(mock_openai, mock_pinecone)

        client.query("text")  # no alpha arg

        mock_index.query.assert_called_once()


# ---------------------------------------------------------------------------
# TestRerank — neural reranking via Pinecone inference
# ---------------------------------------------------------------------------


class TestRerank:
    """Test rerank() calls Pinecone inference reranker and returns enriched dicts."""

    results = [
        {"id": "mtg-001", "score": 0.9, "metadata": {"summary_text": "Q2 roadmap sprint planning discussion", "channel_id": "C01"}},
        {"id": "mtg-002", "score": 0.8, "metadata": {"summary_text": "API design review session", "channel_id": "C01"}},
        {"id": "mtg-003", "score": 0.75, "metadata": {"summary_text": "Incident retrospective meeting", "channel_id": "C02"}},
    ]

    def _setup_client(self, mock_openai, mock_pinecone):
        from storage.pinecone_client import PineconeClient

        pc_instance = mock_pinecone.return_value
        openai_instance = mock_openai.return_value
        openai_instance.embeddings.create.return_value = make_dense_embedding()

        mock_index = MagicMock()
        pc_instance.Index.return_value = mock_index

        client = PineconeClient(api_key="test-key")
        return client, pc_instance, mock_index

    def _make_rerank_result(self, index, score, doc_id):
        item = MagicMock()
        item.index = index
        item.score = score
        item.document = {"id": doc_id, "text": "..."}
        return item

    @patch("storage.pinecone_client.Pinecone")
    @patch("storage.pinecone_client.OpenAI")
    def test_rerank_calls_inference_rerank_with_correct_model(self, mock_openai, mock_pinecone):
        client, pc_instance, mock_index = self._setup_client(mock_openai, mock_pinecone)

        pc_instance.inference.rerank.return_value = [
            self._make_rerank_result(0, 0.99, "mtg-001"),
            self._make_rerank_result(2, 0.85, "mtg-003"),
        ]

        client.rerank("my query", self.results, top_n=2)

        pc_instance.inference.rerank.assert_called_once()
        call_kwargs = pc_instance.inference.rerank.call_args[1]
        assert call_kwargs["model"] == "bge-reranker-v2-m3"
        assert call_kwargs["query"] == "my query"
        assert call_kwargs["top_n"] == 2

    @patch("storage.pinecone_client.Pinecone")
    @patch("storage.pinecone_client.OpenAI")
    def test_rerank_documents_built_from_summary_text(self, mock_openai, mock_pinecone):
        client, pc_instance, mock_index = self._setup_client(mock_openai, mock_pinecone)

        pc_instance.inference.rerank.return_value = [
            self._make_rerank_result(0, 0.99, "mtg-001"),
            self._make_rerank_result(2, 0.85, "mtg-003"),
        ]

        client.rerank("my query", self.results, top_n=2)

        call_kwargs = pc_instance.inference.rerank.call_args[1]
        documents = call_kwargs["documents"]
        assert isinstance(documents, list)
        assert documents[0] == {"id": "mtg-001", "text": "Q2 roadmap sprint planning discussion"}
        assert documents[1] == {"id": "mtg-002", "text": "API design review session"}

    @patch("storage.pinecone_client.Pinecone")
    @patch("storage.pinecone_client.OpenAI")
    def test_rerank_returns_list_with_rerank_score(self, mock_openai, mock_pinecone):
        client, pc_instance, mock_index = self._setup_client(mock_openai, mock_pinecone)

        pc_instance.inference.rerank.return_value = [
            self._make_rerank_result(0, 0.99, "mtg-001"),
            self._make_rerank_result(2, 0.85, "mtg-003"),
        ]

        returned = client.rerank("my query", self.results, top_n=2)

        assert isinstance(returned, list)
        assert len(returned) == 2
        for item in returned:
            assert "id" in item
            assert "score" in item
            assert "metadata" in item
            assert "rerank_score" in item
        assert returned[0]["rerank_score"] == 0.99
        assert returned[0]["id"] == "mtg-001"
        assert returned[0]["metadata"] == self.results[0]["metadata"]

    @patch("storage.pinecone_client.Pinecone")
    @patch("storage.pinecone_client.OpenAI")
    def test_rerank_default_top_n_is_5(self, mock_openai, mock_pinecone):
        client, pc_instance, mock_index = self._setup_client(mock_openai, mock_pinecone)

        pc_instance.inference.rerank.return_value = []

        client.rerank("query", self.results)  # no top_n arg

        call_kwargs = pc_instance.inference.rerank.call_args[1]
        assert call_kwargs["top_n"] == 5

    @patch("storage.pinecone_client.Pinecone")
    @patch("storage.pinecone_client.OpenAI")
    def test_rerank_empty_results_returns_empty_list(self, mock_openai, mock_pinecone):
        client, pc_instance, mock_index = self._setup_client(mock_openai, mock_pinecone)

        pc_instance.inference.rerank.return_value = []

        returned = client.rerank("query", [], top_n=5)

        assert returned == []


# ---------------------------------------------------------------------------
# TestRetrieve — full pipeline: query then rerank
# ---------------------------------------------------------------------------


class TestRetrieve:
    """Test retrieve() calls query(top_k=20) then rerank(top_n=5) in sequence."""

    def _setup_client(self, mock_openai, mock_pinecone):
        from storage.pinecone_client import PineconeClient

        pc_instance = mock_pinecone.return_value
        openai_instance = mock_openai.return_value
        openai_instance.embeddings.create.return_value = make_dense_embedding()

        mock_index = MagicMock()
        pc_instance.Index.return_value = mock_index

        client = PineconeClient(api_key="test-key")
        return client, pc_instance, mock_index

    @patch("storage.pinecone_client.Pinecone")
    @patch("storage.pinecone_client.OpenAI")
    def test_retrieve_calls_query_with_top_k_20(self, mock_openai, mock_pinecone):
        from storage.pinecone_client import PineconeClient
        from unittest.mock import patch as mock_patch

        client, pc_instance, mock_index = self._setup_client(mock_openai, mock_pinecone)

        with mock_patch.object(PineconeClient, "query", return_value=[]) as mock_query, \
             mock_patch.object(PineconeClient, "rerank", return_value=[]) as mock_rerank:
            client.retrieve("what was discussed?")
            mock_query.assert_called_once_with(
                query_text="what was discussed?", top_k=20,
                channel_id=None, start_ts=None, end_ts=None, alpha=0.7
            )

    @patch("storage.pinecone_client.Pinecone")
    @patch("storage.pinecone_client.OpenAI")
    def test_retrieve_calls_rerank_with_query_results(self, mock_openai, mock_pinecone):
        from storage.pinecone_client import PineconeClient
        from unittest.mock import patch as mock_patch

        client, pc_instance, mock_index = self._setup_client(mock_openai, mock_pinecone)
        fake_results = [{"id": "m1", "score": 0.9, "metadata": {}}] * 20

        with mock_patch.object(PineconeClient, "query", return_value=fake_results) as mock_query, \
             mock_patch.object(PineconeClient, "rerank", return_value=fake_results[:5]) as mock_rerank:
            client.retrieve("question")
            mock_rerank.assert_called_once_with("question", fake_results, top_n=5)

    @patch("storage.pinecone_client.Pinecone")
    @patch("storage.pinecone_client.OpenAI")
    def test_retrieve_passes_filters_through_to_query(self, mock_openai, mock_pinecone):
        from storage.pinecone_client import PineconeClient
        from unittest.mock import patch as mock_patch

        client, pc_instance, mock_index = self._setup_client(mock_openai, mock_pinecone)

        with mock_patch.object(PineconeClient, "query", return_value=[]) as mock_query, \
             mock_patch.object(PineconeClient, "rerank", return_value=[]) as mock_rerank:
            client.retrieve("query", channel_id="C999", start_ts=100, end_ts=200)
            call_kwargs = mock_query.call_args[1]
            assert call_kwargs["channel_id"] == "C999"
            assert call_kwargs["start_ts"] == 100
            assert call_kwargs["end_ts"] == 200

    @patch("storage.pinecone_client.Pinecone")
    @patch("storage.pinecone_client.OpenAI")
    def test_retrieve_returns_reranked_results(self, mock_openai, mock_pinecone):
        from storage.pinecone_client import PineconeClient
        from unittest.mock import patch as mock_patch

        client, pc_instance, mock_index = self._setup_client(mock_openai, mock_pinecone)
        reranked = [{"id": "m1", "score": 0.9, "metadata": {}, "rerank_score": 0.99}]

        with mock_patch.object(PineconeClient, "query", return_value=[{"id": "m1", "score": 0.9, "metadata": {}}]), \
             mock_patch.object(PineconeClient, "rerank", return_value=reranked):
            result = client.retrieve("q")
            assert result == reranked
