import pytest

from confluence_logic import classifier


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "query",
    [
        "what is the fix?",
        "how do we fix the problem?",
        "how should we handle it?",
        "should we do that?",
        "is that a good idea?",
    ],
)
async def test_context_dependent_questions_are_meeting_opinion(query):
    assert await classifier.classify_intent(query) == "meeting_opinion"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "query",
    [
        "What is the weather like today in LA?",
        "what is the stock price of nvidia?",
        "what is money?",
        "what does P99 latency mean?",
        "explain retrieval augmented generation",
    ],
)
async def test_standalone_questions_remain_general(query):
    assert await classifier.classify_intent(query) == "general"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "query",
    [
        "when is SOC2 coming?",
        "when is our SOC2 audit?",
        "what does the roadmap say about the launch?",
        "when is the product launch?",
        "who owns the security page?",
    ],
)
async def test_workspace_factual_queries_route_to_confluence(query):
    """Workspace-specific read queries must route to 'confluence', not 'general'."""
    result = await classifier.classify_intent(query)
    assert result == "confluence", (
        f"Expected 'confluence' for workspace query {query!r}, got {result!r}"
    )
