"""ConfluenceQAAgent — Pinecone-first retrieval agent for Confluence Q&A.

Implements QA-01 through QA-04:
  - QA-01: Factual questions answered from Pinecone semantic search
  - QA-02: REST fallback when Pinecone and Neo4j return empty
  - QA-03: gpt-5-mini for tool orchestration, gpt-4o-mini for synthesis
  - QA-04: Read gate (not in this file — lives in jarvis_agentic.py)

Design decisions:
  - D-01: Pinecone is the primary retrieval path (score >= 0.3 threshold)
  - D-02: Neo4j confluence_page_graph is secondary retrieval
  - D-03: Live Confluence REST is the final fallback — never "I don't know" when page exists
  - D-09: Agent uses gpt-5-mini for tool orchestration
  - D-10: Separate gpt-4o-mini synthesis call converts raw context to spoken answer
"""
import asyncio
import concurrent.futures
import json
import logging
import os

from agents import Agent, Runner, function_tool
from ..db.vector_store import PineconeStore
from ..connectors.confluence import ConfluenceConnector
from confluence_logic import confluence_page_graph
from .tools import _run_async_blocking

logger = logging.getLogger(__name__)

# ── Lazy singletons (NOT instantiated at module import time — anti-pattern) ──

_store = None
_connector = None
_openai_client = None


def get_store():
    global _store
    if _store is None:
        _store = PineconeStore()
    return _store


def get_connector():
    global _connector
    if _connector is None:
        _connector = ConfluenceConnector()
    return _connector


def _get_openai_client():
    global _openai_client
    if _openai_client is None:
        from openai import OpenAI
        _openai_client = OpenAI()
    return _openai_client


# ── Private formatting helpers (module-level) ────────────────────────────────

def _format_pinecone_results(matches: list) -> str:
    items = []
    for m in matches:
        meta = m.get("metadata", {})
        items.append({
            "page_id": meta.get("page_id"),
            "title": meta.get("title"),
            "heading": meta.get("heading"),
            "content": meta.get("text_summary") or meta.get("markdown_content", ""),
            "score": m.get("score"),
            "source": "pinecone",
        })
    return json.dumps(items, ensure_ascii=False)


def _format_graph_results(results: list) -> str:
    items = []
    for r in results:
        items.append({
            "page_id": r.get("page_id"),
            "title": r.get("title"),
            "heading": r.get("heading"),
            "content": r.get("relevant_content", ""),
            "score": r.get("score"),
            "source": "neo4j_graph",
        })
    return json.dumps(items, ensure_ascii=False)


def _format_rest_results(results: list) -> str:
    items = []
    for r in results:
        items.append({
            "page_id": r.get("page_id"),
            "title": r.get("title"),
            "content": r.get("excerpt", ""),
            "source": "confluence_rest",
        })
    return json.dumps(items, ensure_ascii=False)


# ── @function_tool functions (module-level — NOT class methods) ───────────────

@function_tool
def search_confluence_pages(query: str) -> str:
    """Searches Confluence for page/section content relevant to the query.
    Tries Pinecone semantic search first, then Neo4j confluence page graph, then live Confluence REST API.
    Returns a JSON-formatted string of retrieved context for the LLM to reason over."""
    try:
        # Primary: Pinecone semantic search (score >= 0.3 per Pitfall 4 / D-01)
        store = get_store()
        raw_matches = store.search(query, top_k=8)
        matches = [m for m in (raw_matches or []) if m.get("score", 0) >= 0.3]
        if matches:
            return _format_pinecone_results(matches)

        # Secondary: Neo4j confluence page graph (D-02) — 1.2s budget to avoid blocking the agent.
        # Without a timeout, a cold Neo4j connection adds 2-3s (seen in production traces).
        user_id = confluence_page_graph.get_current_graph_user_id()
        if user_id:
            try:
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as _ex:
                    _future = _ex.submit(
                        _run_async_blocking,
                        confluence_page_graph.query_user_confluence_graph(user_id, query, limit=8),
                    )
                    graph_results = _future.result(timeout=1.2)
                if graph_results:
                    return _format_graph_results(graph_results)
            except concurrent.futures.TimeoutError:
                logger.warning("Neo4j graph search timed out (>1.2s) — skipping to REST fallback")
            except Exception as graph_exc:
                logger.warning("Neo4j confluence graph search failed: %s", graph_exc)

        # Fallback: live Confluence REST API (D-03 — never say I don't know when REST has results)
        try:
            rest_results = get_connector().search_pages(query, limit=5)
            if rest_results:
                return _format_rest_results(rest_results)
        except Exception as rest_exc:
            logger.warning("REST fallback search failed: %s", rest_exc)

        return "No relevant Confluence content found."
    except Exception as e:
        logger.error("search_confluence_pages failed: %s", e)
        return f"Search error: {e}"


@function_tool
def get_full_page_content(page_id: str) -> str:
    """Fetches the full HTML content of a Confluence page by ID.
    Use this when search results show a relevant page but the excerpt is insufficient."""
    try:
        html = get_connector().fetch_page_html(page_id)
        return html or "No content found for this page."
    except Exception as e:
        logger.error("get_full_page_content failed for page %s: %s", page_id, e)
        return f"Error fetching page {page_id}: {e}"


@function_tool
def list_confluence_pages(limit: int = 10) -> str:
    """Lists available Confluence pages from the user's graph scope.
    Returns a JSON array of {page_id, title, space_key} objects."""
    try:
        user_id = confluence_page_graph.get_current_graph_user_id()
        if not user_id:
            return "No user context available for page listing."
        pages = _run_async_blocking(
            confluence_page_graph.list_user_confluence_pages(user_id, limit=limit)
        )
        if not pages:
            return "No pages found in the confluence graph."
        return json.dumps([
            {"page_id": p.get("page_id"), "title": p.get("title"), "space_key": p.get("space_key")}
            for p in pages
        ], ensure_ascii=False)
    except Exception as e:
        logger.error("list_confluence_pages failed: %s", e)
        return f"Error listing pages: {e}"


# ── ConfluenceQAAgent class ───────────────────────────────────────────────────

# Use JARVIS_AGENT_MODEL env var; fall back to gpt-4o-mini when env var is absent or invalid.
# Set JARVIS_AGENT_MODEL=gpt-5-mini once that model becomes available on the API.
_DEFAULT_AGENT_MODEL = os.environ.get("JARVIS_AGENT_MODEL", "gpt-4o-mini")


class ConfluenceQAAgent:
    def __init__(self, model: str = _DEFAULT_AGENT_MODEL):
        self.model = model
        self.agent = Agent(
            name="Jarvis Confluence QA",
            model=model,
            instructions=(
                "You answer questions about Confluence workspace content. "
                "Call search_confluence_pages exactly ONCE. "
                "Return the relevant section text from the search result as your final output. "
                "Do NOT call get_full_page_content — the search excerpt is sufficient. "
                "Keep your final answer under 100 words."
            ),
            tools=[search_confluence_pages, get_full_page_content, list_confluence_pages],
        )

    async def run(self, query: str, graph_user_id: str) -> str:
        """Run the agent and return a spoken-quality answer string.

        Step 1: Pre-warm the Neo4j confluence graph (D-07 pattern from jarvis_agentic.py:458-460).
        Step 2: Run the Agent with gpt-5-mini for tool orchestration (D-09).
        Step 3: Synthesize the spoken answer with a separate gpt-4o-mini call (D-10).
        """
        # Step 1: Pre-warm Neo4j graph with timeout — best-effort; search_confluence_pages will
        # query the graph on demand if pre-warm is skipped.
        try:
            await asyncio.wait_for(
                confluence_page_graph.ensure_user_confluence_graph(graph_user_id), timeout=0.7
            )
        except Exception:
            logger.debug("Graph pre-warm skipped for %s — will query on demand", graph_user_id)

        # Step 2: Run Agent — tool orchestration and retrieval (D-09)
        # max_turns=2: at most 1 tool call + 1 final answer, preventing slow multi-round iterations.
        try:
            result = await Runner.run(self.agent, query, max_turns=2)
            if hasattr(result, 'final_output'):
                raw_context = result.final_output.strip()
            else:
                raw_context = str(result)
        except Exception as exc:
            logger.error("ConfluenceQAAgent runner failed: %s", exc)
            return "I could not retrieve the Confluence context right now."

        if not raw_context:
            return "I could not find a relevant Confluence page for that."

        # Step 3: Synthesis (gpt-4o-mini) — converts raw context to a spoken answer (D-10).
        # max_tokens=120 keeps inference fast for spoken delivery (2-3 sentences is sufficient).
        try:
            response = await asyncio.to_thread(
                lambda: _get_openai_client().chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "Answer the question from the retrieved Confluence context in 2-3 sentences. "
                                "Be direct and spoken-quality. If context is insufficient, say so briefly."
                            ),
                        },
                        {
                            "role": "user",
                            "content": f"Q: {query}\n\nContext:\n{raw_context[:2000]}",
                        },
                    ],
                    max_tokens=120,
                    temperature=0.2,
                )
            )
            return (response.choices[0].message.content or "").strip() or \
                   "I could not answer that from the Confluence graph."
        except Exception as exc:
            logger.error("ConfluenceQAAgent synthesis failed: %s", exc)
            return raw_context  # degrade gracefully to raw context if synthesis fails
