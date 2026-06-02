"""Smoke tests for the full pipeline run — PIPE-01.

All tests fail with ImportError until Wave 4 implements pipeline/run.py.
The import is deferred into each test body so pytest can collect without errors.
"""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, patch


@pytest.mark.asyncio
async def test_pipeline_smoke():
    """PIPE-01: Full pipeline: transcript -> intents -> proposals (smoke test with mocks)."""
    from pipeline.run import run_pipeline, PipelineConfig  # ImportError until Wave 4

    transcript = (
        "We're changing the data retention policy to 5 years. "
        "Also the access control process now needs two approvals."
    )
    config = PipelineConfig(
        session_id="smoke-test-001",
        doc_folder="tests/fixtures",
        use_embeddings=False,   # avoid OpenAI API calls in tests
        skip_contextual=True,   # avoid GPT-4o-mini calls at index time
    )
    proposals = await run_pipeline(transcript=transcript, config=config)
    assert isinstance(proposals, list)
    # With mocked agents, smoke test only checks the pipeline returns without error


@pytest.mark.asyncio
async def test_pipeline_sse_stages():
    """Pipeline emits all 8 SSE stage names in order."""
    from pipeline.run import PIPELINE_STAGES  # ImportError until Wave 4

    expected = [
        "transcript_source",
        "intent_extraction",
        "rag_indexing",
        "rag_retrieval",
        "evaluation",
        "drafting",
        "verification",
        "ready_for_review",
    ]
    assert PIPELINE_STAGES == expected


def test_pipeline_config_defaults():
    """PipelineConfig has sensible defaults."""
    from pipeline.run import PipelineConfig  # ImportError until Wave 4

    config = PipelineConfig(session_id="x", doc_folder="/tmp")
    assert config.top_k == 3
    assert config.relevance_threshold == 0.7
    assert config.use_embeddings is True
    assert config.rerank is True
    assert config.contextual_retrieval is True
