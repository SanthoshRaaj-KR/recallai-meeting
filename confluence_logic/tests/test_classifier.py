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
