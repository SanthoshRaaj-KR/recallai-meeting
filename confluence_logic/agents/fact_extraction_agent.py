"""FactExtractionAgent — extracts structured facts from meeting transcripts (RETR-02)."""

import asyncio
import logging
import os
from typing import Any, Dict, List, Optional, Union

from agents import Agent, AgentOutputSchema, Runner
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

JARVIS_FACT_INPUT_MAX_CHARS = int(os.getenv("JARVIS_FACT_INPUT_MAX_CHARS", "60000"))
JARVIS_FACT_CHUNK_CHARS = int(os.getenv("JARVIS_FACT_CHUNK_CHARS", "55000"))
JARVIS_FACT_CHUNK_OVERLAP = int(os.getenv("JARVIS_FACT_CHUNK_OVERLAP", "2000"))
JARVIS_AGENT_MODEL = os.getenv("JARVIS_AGENT_MODEL", "gpt-5-mini").strip()
JARVIS_PIPELINE_MAX_PAGES = int(os.getenv("JARVIS_PIPELINE_MAX_PAGES", "20"))

# ---------------------------------------------------------------------------
# Output schema
# ---------------------------------------------------------------------------


class ExtractedFacts(BaseModel):
    """Structured facts extracted from a meeting transcript."""

    decisions: List[str] = []
    action_items: List[str] = []
    new_requirements: List[str] = []
    owners: Dict[str, str] = {}      # task_description -> owner_name
    deadlines: Dict[str, str] = {}   # task_description -> deadline_string
    doc_worthy_updates: List[str] = []
    query_terms: List[str] = []
    mentioned_page_titles: List[str] = []  # exact Confluence page/doc names spoken in the meeting
    content_phrases: List[str] = []        # specific strings to find verbatim in page content (e.g. "OpenAI Agents SDK")


# ---------------------------------------------------------------------------
# Agent prompt
# ---------------------------------------------------------------------------

FACT_EXTRACTION_PROMPT = (
    "You are a meeting intelligence assistant. Read the meeting transcript carefully and extract "
    "structured facts that are relevant for updating documentation.\n\n"
    "Extract the following:\n"
    "1. decisions: Key decisions made during the meeting (e.g. 'Approved migration to PostgreSQL')\n"
    "2. action_items: Tasks assigned or committed to, with owner and deadline if mentioned "
    "(e.g. 'Ben will prepare the rollout checklist by Friday')\n"
    "3. new_requirements: New or changed requirements discussed (e.g. 'API must support pagination')\n"
    "4. owners: A JSON object mapping task description to owner name "
    "(e.g. {\"Prepare rollout checklist\": \"Ben\", \"Write API docs\": \"Asha\"}). "
    "Use {} if none mentioned.\n"
    "5. deadlines: A JSON object mapping task description to deadline string "
    "(e.g. {\"Prepare rollout checklist\": \"next Friday\", \"API migration\": \"2024-Q2\"}). "
    "Use {} if none mentioned.\n"
    "6. doc_worthy_updates: Items worth adding or updating in documentation — new processes, "
    "changed procedures, architectural decisions, API changes, launch timelines, configuration changes, "
    "and anything else that should be reflected in team knowledge bases\n"
    "7. query_terms: 3-8 short search query terms (keywords or phrases) that would retrieve the "
    "most relevant Confluence pages to update based on the meeting content\n"
    "8. mentioned_page_titles: Exact names of Confluence pages, documents, wikis, or runbooks that "
    "participants explicitly referred to during the meeting — capture the full title as spoken. "
    "Examples: 'HR Onboarding 2024', 'API Runbook v2', 'Deployment Checklist', 'Architecture Overview'. "
    "Only include names that sound like document/page titles, not generic topics. "
    "Use [] if no specific document names were mentioned.\n"
    "9. content_phrases: Specific strings or names that participants said should be FOUND and REPLACED "
    "inside existing page content — the OLD values that literally appear in documentation right now. "
    "Examples: if the meeting says 'replace Python 2 with Python 3 everywhere' capture 'Python 2'; "
    "if they say 'change all references to the old API endpoint' capture the endpoint string; "
    "if they say 'rename the team lead in all docs from Alice to Bob' capture 'Alice'. "
    "Only include concrete strings that would appear verbatim in existing pages. "
    "Use [] if no specific content replacements were discussed.\n\n"
    "IMPORTANT: Businesses rely on these facts for documentation — do not omit items. "
    "Be thorough and complete. Return valid JSON matching the ExtractedFacts schema with all fields. "
    "For list fields (decisions, action_items, new_requirements, doc_worthy_updates, query_terms, mentioned_page_titles, content_phrases): "
    "return an empty list [] if no items. "
    "For dict fields (owners, deadlines): return an empty object {} if none mentioned. "
    "Return JSON only — no markdown, no explanation."
)

# ---------------------------------------------------------------------------
# Module-level singleton agent (constructed at import time)
# ---------------------------------------------------------------------------

_fact_agent = Agent(
    name="FactExtractionAgent",
    model=JARVIS_AGENT_MODEL,
    instructions=FACT_EXTRACTION_PROMPT,
    output_type=AgentOutputSchema(ExtractedFacts, strict_json_schema=False),
)

# ---------------------------------------------------------------------------
# Core extraction function
# ---------------------------------------------------------------------------


def _transcript_to_text(transcript: Union[str, List[Any]]) -> str:
    """Convert a transcript (str or list of turn dicts) to a plain text string."""
    if isinstance(transcript, str):
        return transcript
    lines = []
    for turn in transcript:
        if isinstance(turn, dict):
            participant = turn.get("participant") or turn.get("speaker") or "Unknown"
            text = turn.get("text") or turn.get("content") or ""
            lines.append(f"{participant}: {text}")
        else:
            lines.append(str(turn))
    return "\n".join(lines)


def _merge_facts(chunks: List[ExtractedFacts]) -> ExtractedFacts:
    """Merge facts extracted from multiple transcript chunks, deduplicating by value."""
    decisions: List[str] = []
    action_items: List[str] = []
    new_requirements: List[str] = []
    doc_worthy_updates: List[str] = []
    query_terms: List[str] = []
    mentioned_page_titles: List[str] = []
    content_phrases: List[str] = []
    owners: Dict[str, str] = {}
    deadlines: Dict[str, str] = {}

    seen_decisions: set = set()
    seen_actions: set = set()
    seen_requirements: set = set()
    seen_doc: set = set()
    seen_terms: set = set()
    seen_page_titles: set = set()
    seen_phrases: set = set()

    for chunk in chunks:
        for item in chunk.decisions:
            key = item.strip().lower()
            if key and key not in seen_decisions:
                seen_decisions.add(key)
                decisions.append(item)
        for item in chunk.action_items:
            key = item.strip().lower()
            if key and key not in seen_actions:
                seen_actions.add(key)
                action_items.append(item)
        for item in chunk.new_requirements:
            key = item.strip().lower()
            if key and key not in seen_requirements:
                seen_requirements.add(key)
                new_requirements.append(item)
        for item in chunk.doc_worthy_updates:
            key = item.strip().lower()
            if key and key not in seen_doc:
                seen_doc.add(key)
                doc_worthy_updates.append(item)
        for item in chunk.query_terms:
            key = item.strip().lower()
            if key and key not in seen_terms:
                seen_terms.add(key)
                query_terms.append(item)
        for item in chunk.mentioned_page_titles:
            key = item.strip().lower()
            if key and key not in seen_page_titles:
                seen_page_titles.add(key)
                mentioned_page_titles.append(item)
        for item in chunk.content_phrases:
            key = item.strip().lower()
            if key and key not in seen_phrases:
                seen_phrases.add(key)
                content_phrases.append(item)
        owners.update(chunk.owners)
        deadlines.update(chunk.deadlines)

    return ExtractedFacts(
        decisions=decisions,
        action_items=action_items,
        new_requirements=new_requirements,
        doc_worthy_updates=doc_worthy_updates,
        query_terms=query_terms[:12],
        mentioned_page_titles=mentioned_page_titles[:20],
        content_phrases=content_phrases[:20],
        owners=owners,
        deadlines=deadlines,
    )


async def _extract_chunk(text: str) -> ExtractedFacts:
    """Run fact extraction on a single text chunk."""
    result = await Runner.run(_fact_agent, text)
    if isinstance(result.final_output, ExtractedFacts):
        return result.final_output
    return ExtractedFacts.model_validate(result.final_output)


async def _run_fact_extraction(
    transcript_text: Optional[str] = None,
    *,
    transcript: Optional[Union[str, List[Any]]] = None,
) -> ExtractedFacts:
    """Extract structured facts from the full transcript with no data loss.

    For transcripts longer than JARVIS_FACT_CHUNK_CHARS, splits into overlapping
    chunks and runs extraction in parallel on each, then merges results.
    A 100-minute meeting and a 10-minute meeting both get full coverage.
    """
    try:
        if transcript_text is not None:
            text = transcript_text
        elif transcript is not None:
            text = _transcript_to_text(transcript)
        else:
            logger.warning("_run_fact_extraction called with no transcript input")
            return ExtractedFacts()

        # Short transcript — single call, no chunking needed
        if len(text) <= JARVIS_FACT_CHUNK_CHARS:
            return await _extract_chunk(text)

        # Long transcript — split into overlapping chunks and extract in parallel
        chunks: List[str] = []
        start = 0
        while start < len(text):
            end = start + JARVIS_FACT_CHUNK_CHARS
            chunks.append(text[start:end])
            if end >= len(text):
                break
            # Overlap: next chunk starts JARVIS_FACT_CHUNK_OVERLAP chars before end
            # so decisions/actions spanning a chunk boundary are captured
            start = end - JARVIS_FACT_CHUNK_OVERLAP

        logger.info(
            "Fact extraction: transcript %d chars → %d chunks of ~%d chars each",
            len(text), len(chunks), JARVIS_FACT_CHUNK_CHARS,
        )
        chunk_results = await asyncio.gather(
            *[_extract_chunk(c) for c in chunks],
            return_exceptions=True,
        )
        valid: List[ExtractedFacts] = []
        for i, r in enumerate(chunk_results):
            if isinstance(r, Exception):
                logger.warning("Fact extraction chunk %d failed (non-fatal): %s", i, r)
            else:
                valid.append(r)

        return _merge_facts(valid) if valid else ExtractedFacts()

    except Exception as exc:
        logger.warning("Fact extraction failed, using empty facts: %s", exc)
        return ExtractedFacts()


# ---------------------------------------------------------------------------
# PineconeStore singleton
# ---------------------------------------------------------------------------

from confluence_logic.db.vector_store import PineconeStore  # noqa: E402

_store: Optional[PineconeStore] = None


def _get_store() -> PineconeStore:
    """Lazily construct a module-level PineconeStore singleton."""
    global _store
    if _store is None:
        _store = PineconeStore()
    return _store


# ---------------------------------------------------------------------------
# Merged RAG retrieval
# ---------------------------------------------------------------------------


async def _merged_rag_retrieval(
    graph_user_id: Optional[str] = None,
    query_terms: Optional[List[str]] = None,
    *,
    user_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Query Neo4j and Pinecone in parallel and merge results by page_id.

    Both sources are always queried regardless of whether either returns results (RETR-01).
    Pinecone.search is synchronous and must be wrapped in asyncio.to_thread.

    Accepts either graph_user_id (positional) or user_id (keyword) for the user identifier.
    """
    from confluence_logic import confluence_page_graph  # deferred to avoid circular imports

    # Resolve user identifier
    resolved_user_id = graph_user_id or user_id or ""
    resolved_terms = query_terms or []

    if not resolved_terms:
        return []

    # Limit to 3 query terms to control concurrency fan-out (T-02-03-02)
    terms = resolved_terms[:3]

    neo4j_coros = [
        confluence_page_graph.query_user_confluence_graph(resolved_user_id, q, limit=8)
        for q in terms
    ]
    pinecone_coros = [
        asyncio.to_thread(_get_store().search, q, 8)
        for q in terms
    ]

    all_results = await asyncio.gather(
        *neo4j_coros, *pinecone_coros,
        return_exceptions=True,
    )

    # Merge by page_id — prefer result with relevant_content; prefer higher score on tie
    pages_by_id: Dict[str, Dict[str, Any]] = {}
    for result in all_results:
        if isinstance(result, Exception):
            logger.warning("RAG source returned error (non-fatal): %s", result)
            continue
        for item in (result or []):
            if not isinstance(item, dict):
                continue
            page_id = item.get("page_id") or item.get("id") or ""
            if not page_id:
                continue
            existing = pages_by_id.get(page_id)
            if existing is None:
                pages_by_id[page_id] = item
            else:
                # Prefer the entry with relevant_content; break tie by higher score
                existing_score = float(existing.get("score") or 0)
                new_score = float(item.get("score") or 0)
                existing_has_content = bool(existing.get("relevant_content"))
                new_has_content = bool(item.get("relevant_content"))
                if new_has_content and not existing_has_content:
                    pages_by_id[page_id] = item
                elif new_has_content and existing_has_content and new_score > existing_score:
                    pages_by_id[page_id] = item
                elif not existing_has_content and new_score > existing_score:
                    pages_by_id[page_id] = item

    sorted_pages = sorted(
        pages_by_id.values(),
        key=lambda p: -float(p.get("score") or 0),
    )
    return sorted_pages[:JARVIS_PIPELINE_MAX_PAGES]
