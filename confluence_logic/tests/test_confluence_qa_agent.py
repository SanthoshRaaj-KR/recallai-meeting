"""Tests for ConfluenceQAAgent — Confluence Document Q&A Agent (QA-01, QA-02, QA-03, QA-04)."""
import asyncio
import json
import pytest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from confluence_logic.agents.confluence_qa_agent import (
    ConfluenceQAAgent,
    search_confluence_pages,
    get_full_page_content,
    list_confluence_pages,
)


def _call(tool, *args, **kwargs):
    """Invoke a FunctionTool synchronously — maps positional args to schema property order."""
    props = list(tool.params_json_schema.get('properties', {}).keys())
    call_args = dict(zip(props, args))
    call_args.update(kwargs)
    return asyncio.run(tool.on_invoke_tool(None, json.dumps(call_args)))


# ── QA-01: Factual question answered from Pinecone ──────────────────────────

@pytest.mark.asyncio
async def test_qa_returns_answer_from_pinecone():
    """QA-01: A factual Confluence question answered by the agent returns a correct spoken
    response. Verifies that: (a) the agent returns text mentioning the topic, and
    (b) the synthesis step uses model='gpt-4o-mini' per D-10."""
    mock_runner_result = SimpleNamespace(
        final_output="SOC2 is planned for Q3 2025 per the Security Roadmap page."
    )
    mock_oai_client = MagicMock()
    mock_oai_client.chat.completions.create.return_value.choices = [
        SimpleNamespace(message=SimpleNamespace(content="SOC2 is coming in Q3 2025."))
    ]

    with patch("confluence_logic.agents.confluence_qa_agent.Runner") as mock_runner_cls, \
         patch("confluence_logic.agents.confluence_qa_agent.get_store") as mock_get_store, \
         patch("confluence_logic.agents.confluence_qa_agent._get_openai_client") as mock_get_oai, \
         patch("confluence_logic.agents.confluence_qa_agent.confluence_page_graph") as mock_graph:

        mock_runner_cls.run = AsyncMock(return_value=mock_runner_result)
        mock_get_store.return_value.search.return_value = [
            {
                "metadata": {
                    "page_id": "p99",
                    "title": "Security Roadmap",
                    "heading": "SOC2 Timeline",
                    "text_summary": "SOC2 is planned for Q3 2025.",
                    "space_key": "SEC",
                }
            }
        ]
        mock_graph.ensure_user_confluence_graph = AsyncMock(return_value=True)
        mock_graph.get_current_graph_user_id.return_value = "user-123"
        mock_get_oai.return_value = mock_oai_client

        answer = await ConfluenceQAAgent().run("when is SOC2 coming", "user-123")

    assert "SOC2" in answer or "Q3" in answer, (
        f"Expected answer to mention SOC2 or Q3, got: {answer!r}"
    )
    # Synthesis step must use gpt-4o-mini (D-10 / QA-03)
    create_kwargs = mock_oai_client.chat.completions.create.call_args.kwargs
    assert create_kwargs.get("model") == "gpt-4o-mini", (
        f"Synthesis model should be 'gpt-4o-mini', got {create_kwargs.get('model')!r}"
    )


# ── QA-02: Fallback to live REST when Pinecone and Neo4j return empty ───────

@pytest.mark.asyncio
async def test_qa_fallback_to_rest_when_pinecone_empty():
    """QA-02: When Pinecone index has no relevant chunks and Neo4j graph returns nothing,
    the agent falls back to live Confluence REST and still returns a non-empty answer."""
    mock_runner_result = SimpleNamespace(
        final_output="SOC2 audit is scheduled for Q3 2025 (from REST fallback)."
    )
    mock_oai_client = MagicMock()
    mock_oai_client.chat.completions.create.return_value.choices = [
        SimpleNamespace(message=SimpleNamespace(content="SOC2 is planned for Q3 based on the Security page."))
    ]
    mock_connector = MagicMock()
    mock_connector.search_pages.return_value = [
        {"page_id": "p1", "title": "Security Page", "excerpt": "SOC2 audit is scheduled for Q3 2025"}
    ]

    with patch("confluence_logic.agents.confluence_qa_agent.Runner") as mock_runner_cls, \
         patch("confluence_logic.agents.confluence_qa_agent.get_store") as mock_get_store, \
         patch("confluence_logic.agents.confluence_qa_agent._get_openai_client") as mock_get_oai, \
         patch("confluence_logic.agents.confluence_qa_agent.confluence_page_graph") as mock_graph, \
         patch("confluence_logic.agents.confluence_qa_agent.get_connector") as mock_get_connector:

        # Pinecone returns empty — triggers fallback
        mock_get_store.return_value.search.return_value = []
        # Neo4j returns empty — triggers REST fallback
        mock_graph.query_user_confluence_graph = AsyncMock(return_value=[])
        mock_graph.get_current_graph_user_id.return_value = "user-123"
        mock_graph.ensure_user_confluence_graph = AsyncMock(return_value=True)
        # REST fallback returns a result
        mock_get_connector.return_value = mock_connector
        mock_runner_cls.run = AsyncMock(return_value=mock_runner_result)
        mock_get_oai.return_value = mock_oai_client

        answer = await ConfluenceQAAgent().run("when is SOC2 coming", "user-123")

    assert answer, "Expected a non-empty answer from the REST fallback path"
    assert answer not in ("I don't know", "No relevant"), (
        f"Agent should not give up when REST fallback is available, got: {answer!r}"
    )


# ── QA-03: Model split — gpt-5-mini for tools, gpt-4o-mini for synthesis ────

@pytest.mark.asyncio
async def test_qa_model_split_tools_gpt5mini_synthesis_gpt4omini():
    """QA-03: Verifies the two-model split required by D-09/D-10.
    - agent.agent.model must be 'gpt-5-mini' (tool orchestration)
    - The synthesis chat.completions.create call must use model='gpt-4o-mini'
    """
    agent = ConfluenceQAAgent()

    # Structural assertion: agent SDK object must use gpt-5-mini for tool orchestration
    assert agent.agent.model == "gpt-5-mini", (
        f"Tool orchestration model should be 'gpt-5-mini', got {agent.agent.model!r}"
    )

    mock_runner_result = SimpleNamespace(final_output="The roadmap covers Q3 SOC2 timeline.")
    mock_oai_client = MagicMock()
    mock_oai_client.chat.completions.create.return_value.choices = [
        SimpleNamespace(message=SimpleNamespace(content="The SOC2 timeline is Q3 2025."))
    ]

    with patch("confluence_logic.agents.confluence_qa_agent.Runner") as mock_runner_cls, \
         patch("confluence_logic.agents.confluence_qa_agent.get_store") as mock_get_store, \
         patch("confluence_logic.agents.confluence_qa_agent._get_openai_client") as mock_get_oai, \
         patch("confluence_logic.agents.confluence_qa_agent.confluence_page_graph") as mock_graph:

        mock_runner_cls.run = AsyncMock(return_value=mock_runner_result)
        mock_get_store.return_value.search.return_value = [
            {
                "metadata": {
                    "page_id": "p1",
                    "title": "Roadmap",
                    "heading": "Timeline",
                    "text_summary": "Q3 SOC2",
                    "space_key": "ENG",
                }
            }
        ]
        mock_graph.ensure_user_confluence_graph = AsyncMock(return_value=True)
        mock_graph.get_current_graph_user_id.return_value = "u1"
        mock_get_oai.return_value = mock_oai_client

        answer = await agent.run("what is the SOC2 timeline", "u1")

    # Synthesis call must use gpt-4o-mini — NOT gpt-5-mini
    create_call = mock_oai_client.chat.completions.create.call_args
    assert create_call is not None, "Expected synthesis (chat.completions.create) to be called"
    synthesis_model = create_call.kwargs.get("model")
    assert synthesis_model == "gpt-4o-mini", (
        f"Synthesis model should be 'gpt-4o-mini', got {synthesis_model!r}"
    )

    # Runner.run call should not override the model (it uses the Agent's gpt-5-mini)
    runner_call = mock_runner_cls.run.call_args
    runner_kwargs = runner_call.kwargs if runner_call else {}
    # The call should not pass a model override that would be gpt-4o-mini
    if "model" in runner_kwargs:
        assert runner_kwargs["model"] == "gpt-5-mini", (
            f"Runner.run model override, if present, must be 'gpt-5-mini', got {runner_kwargs['model']!r}"
        )


# ── QA-04: Read gate blocks mutation queries ─────────────────────────────────

def test_qa_read_gate_blocks_mutations():
    """QA-04: Mutation queries ('update the SOC2 page', 'add a section') must NOT pass
    the _is_confluence_read_query() gate — they are never routed to the Q&A agent.
    This supplements the existing coverage in test_jarvis_agentic.py::
    test_confluence_read_query_detection_excludes_mutations.
    """
    import confluence_logic.jarvis_agentic as ja

    # Mutation queries must be blocked
    assert not ja._is_confluence_read_query("update the SOC2 page with new findings"), \
        "Mutation query 'update ...' should NOT pass the read gate"
    assert not ja._is_confluence_read_query("add a section to the roadmap page"), \
        "Mutation query 'add ...' should NOT pass the read gate"
    assert not ja._is_confluence_read_query("delete the security section"), \
        "Mutation query 'delete ...' should NOT pass the read gate"

    # Read queries must pass
    assert ja._is_confluence_read_query("when is SOC2 coming"), \
        "Read query 'when is SOC2 coming' should pass the read gate"
    assert ja._is_confluence_read_query("what does the roadmap page say about security"), \
        "Read query 'what does ...' should pass the read gate"
    assert ja._is_confluence_read_query("list the Confluence pages"), \
        "Read query 'list ...' should pass the read gate"
