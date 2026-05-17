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


class ChangeIntent(BaseModel):
    """A single structured documentation change extracted from the meeting.

    Each change_intent represents ONE thing that must be updated in Confluence.
    The pipeline uses these to drive per-intent retrieval and per-(intent, page) drafting,
    so subtle/contextual changes never get lost in a single sweep.
    """
    instruction: str = ""        # human-readable: "Reduce Akshat's gym plan to 9 weeks"
    subject: str = ""            # what is being changed: "Akshat's gym plan", "OpenAI Agents SDK"
    target_hint: str = ""        # where it likely lives: "Akshat plan page", "framework docs"
    old_value: str = ""          # specific old string on the page (empty if unknown)
    new_value: str = ""          # specific new value to write
    action: str = "replace"      # replace | add | remove | rename | create
    rationale: str = ""          # why: "Akshat is busy", "migration to Claude SDK"
    verbatim_content: str = ""   # NEW: for add/create actions, exact quoted content from transcript


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
    change_intents: List[ChangeIntent] = []  # structured per-change extraction — primary signal for the drafter


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
    "Use [] if no specific content replacements were discussed.\n"
    "10. change_intents: The MOST IMPORTANT field. A structured list of every distinct change that must be "
    "made to documentation. EACH change spoken about in the meeting must produce ONE intent — no skipping.\n"
    "    Each ChangeIntent has these fields:\n"
    "    - instruction: human-readable description of what to change. Example: "
    "\"Reduce Akshat's gym plan duration from the documented value to 9 weeks because he is busy\"\n"
    "    - subject: WHAT is being changed. Example: \"Akshat's gym plan\", \"OpenAI Agents SDK usage\", "
    "\"team lead role\", \"deployment process for service X\"\n"
    "    - target_hint: WHERE on Confluence this likely lives. Use a short phrase that would match page titles "
    "or content. Example: \"Akshat plan\", \"framework SDK\", \"team roster\", \"deployment runbook\". Empty string if unsure.\n"
    "    - old_value: the SPECIFIC old string that probably appears in the page TODAY and needs replacing. "
    "Example: \"12 months\", \"OpenAI Agents SDK\", \"Alice\". Empty string if not explicitly stated "
    "or if the new value is purely additive.\n"
    "    - new_value: the SPECIFIC new value to apply. Example: \"9 weeks\", \"Claude SDK\", \"Bob\". "
    "Empty string only if action is remove/delete.\n"
    "    - action: one of \"replace\", \"add\", \"remove\", \"rename\", \"create\". "
    "Use \"replace\" for most edits, \"rename\" for page-title changes, \"create\" for new pages, "
    "\"remove\" for deletions, \"add\" for purely-additive new sections/bullets.\n"
    "    - rationale: WHY this change. Example: \"Akshat is now busy with other commitments\", "
    "\"Team migrated to Claude SDK\". Empty if no reason was given.\n"
    "    EXAMPLES (technical / business scenarios):\n"
    "    A. Transcript: \"We're migrating from PostgreSQL 13 to PostgreSQL 16 across all services next quarter.\"\n"
    "       → {instruction: \"Update PostgreSQL version references from 13 to 16\", "
    "subject: \"PostgreSQL version\", target_hint: \"PostgreSQL, database, infrastructure\", "
    "old_value: \"PostgreSQL 13\", new_value: \"PostgreSQL 16\", action: \"replace\", "
    "rationale: \"Quarterly database upgrade\"}\n"
    "    B. Transcript: \"The on-call rotation owner for the payments service changed — it's now Priya, was Marcus.\"\n"
    "       → {instruction: \"Update payments service on-call owner from Marcus to Priya\", "
    "subject: \"payments on-call owner\", target_hint: \"payments runbook, on-call rotation\", "
    "old_value: \"Marcus\", new_value: \"Priya\", action: \"replace\", "
    "rationale: \"Ownership handover\"}\n"
    "    C. Transcript: \"Deployment SLA is being tightened from 99.9% to 99.95% across all production services.\"\n"
    "       → {instruction: \"Update production deployment SLA from 99.9% to 99.95%\", "
    "subject: \"production SLA\", target_hint: \"SLA, deployment, production services\", "
    "old_value: \"99.9%\", new_value: \"99.95%\", action: \"replace\", "
    "rationale: \"Tightened reliability target\"}\n"
    "    D. Transcript: \"We need a new architecture overview page for the Inventory Service we just launched.\"\n"
    "       → {instruction: \"Create a new Confluence page documenting the Inventory Service architecture\", "
    "subject: \"Inventory Service architecture\", target_hint: \"Inventory Service\", "
    "old_value: \"\", new_value: \"\", action: \"create\", rationale: \"New service launched\"}\n"
    "    E. Transcript: \"The legacy /v1 checkout endpoint is deprecated; clients should move to /v2/checkout.\"\n"
    "       → {instruction: \"Replace /v1/checkout endpoint references with /v2/checkout\", "
    "subject: \"checkout API endpoint\", target_hint: \"checkout API, endpoints\", "
    "old_value: \"/v1/checkout\", new_value: \"/v2/checkout\", action: \"replace\", "
    "rationale: \"v1 deprecation\"}\n"
    "    F. Transcript: \"We've reduced the order fulfillment SLA from 48 hours to 24 hours because the new "
    "warehouse routing went live.\"\n"
    "       → {instruction: \"Update order fulfillment SLA from 48h to 24h\", "
    "subject: \"order fulfillment SLA\", target_hint: \"fulfillment, order processing\", "
    "old_value: \"48 hours\", new_value: \"24 hours\", action: \"replace\", "
    "rationale: \"New warehouse routing reduces processing time\"}\n"
    "    EXPLICIT CREATE RULE — phrase-triggered creates:\n"
    "    If the transcript contains any of the phrases: 'create a page', 'create a new page', "
    "'make a page', 'make a new page', 'new page for', 'set up a page', 'create a Confluence page', "
    "referring to a subject S, you MUST produce exactly ONE ChangeIntent with action='create' "
    "and subject=S — even if S also appears elsewhere in existing documentation. "
    "The user's explicit instruction overrides retrieval. Do not classify these as 'edit' or 'add'. "
    "If the user lists specific items to put on the new page (bullets, names, metrics), copy them "
    "EXACTLY into verbatim_content per CRITICAL RULE C below.\n"
    "    CRITICAL RULE A — FINAL STATE ONLY:\n"
    "    Extract the NET FINAL agreed state, not intermediate positions. "
    "If the group first proposes X and then reverts or revises to Y, produce ONE intent for Y. "
    "If the final decision is 'keep as-is / no change', produce ZERO intents for that topic. "
    "NEVER produce two intents for the same topic with different new_values — that means you captured "
    "an intermediate step that was overruled.\n"
    "    CRITICAL RULE B — NO DUPLICATES:\n"
    "    Each distinct change must appear exactly once. If the same update was mentioned at multiple "
    "points in the meeting (e.g. someone reminded the group of a decision made earlier), extract it "
    "only once with the most complete information available.\n"
    "    CRITICAL RULE C — verbatim_content for add/create:\n"
    "    For action='add' or action='create' intents where participants named SPECIFIC items to add "
    "(a list of concerns, features, names, metrics, etc.): copy the EXACT items from the transcript "
    "into `verbatim_content`. This must be a verbatim quote — do NOT paraphrase or summarize. "
    "Example: meeting says 'we have three problems: A is too slow, B crashes on mobile, C loses user data' "
    "→ verbatim_content='A is too slow, B crashes on mobile, C loses user data'. "
    "SECOND EXAMPLE: meeting says 'create a pros and cons page about bots: A is fast, "
    "B is cheap, C is expensive, D is slow' → "
    "{instruction: 'Create a pros and cons page about bots', subject: 'bots pros and cons', "
    "target_hint: 'bots', old_value: '', new_value: '', action: 'create', "
    "rationale: 'User requested a new pros/cons doc', "
    "verbatim_content: 'A is fast, B is cheap, C is expensive, D is slow'}\n"
    "    Leave verbatim_content empty ('') for replace/remove/rename actions or when no specific list was named.\n"
    "    Use [] only if no documentation changes were discussed.\n"
    "    Be EXHAUSTIVE — if a meeting decision implies a documentation update, capture an intent. "
    "Capture both explicit changes (numbers, names, versions) and contextual ones "
    "(process changes, ownership changes, scope changes).\n\n"
    "IMPORTANT: Businesses rely on these facts for documentation — do not omit items. "
    "Be thorough and complete. Return valid JSON matching the ExtractedFacts schema with all fields. "
    "For list fields (decisions, action_items, new_requirements, doc_worthy_updates, query_terms, "
    "mentioned_page_titles, content_phrases, change_intents): return an empty list [] if no items. "
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

    # Dedup by (normalized_subject, action) keeping LAST occurrence.
    # "Last" = most recent in transcript order = final agreed state.
    # This prevents contradictory intents (e.g. "change Q3 to Q1" then "Q3 is fine")
    # from both passing through — the final state wins.
    def _norm(text: str) -> str:
        import re as _re
        t = _re.sub(r"[^\w\s]", " ", (text or "").lower())
        return _re.sub(r"\s+", " ", t).strip()[:60]

    intent_key_order: list = []          # insertion-ordered unique keys
    intent_last: dict = {}               # key -> last ChangeIntent seen

    for chunk in chunks:
        for intent in chunk.change_intents:
            key = (_norm(intent.subject), intent.action.strip().lower())
            if not all(key):
                continue
            if key not in intent_last:
                intent_key_order.append(key)
            intent_last[key] = intent    # LAST wins — final state of the discussion

    change_intents = [intent_last[k] for k in intent_key_order]

    return ExtractedFacts(
        decisions=decisions,
        action_items=action_items,
        new_requirements=new_requirements,
        doc_worthy_updates=doc_worthy_updates,
        query_terms=query_terms[:12],
        mentioned_page_titles=mentioned_page_titles[:20],
        content_phrases=content_phrases[:20],
        change_intents=change_intents[:30],
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


def _normalize_pinecone_match(match: Any) -> Optional[Dict[str, Any]]:
    """Flatten a raw Pinecone match into the shared {page_id, title, ...} shape.

    Pinecone returns matches where the chunk id is at the top level (e.g. 'pageX_3')
    and the real page_id, title, heading, content all live under 'metadata'. Without
    this normalization, _merged_rag_retrieval was grouping chunks by chunk-id instead
    of page-id — causing duplicates and "wrong page" selection downstream.

    Returns None if the match has no real page_id (can't be merged safely).
    """
    if match is None:
        return None
    # Pinecone match objects may be dicts or have attribute-style access; handle both
    if isinstance(match, dict):
        metadata = match.get("metadata") or {}
        score = match.get("score")
        chunk_id = match.get("id") or ""
    else:
        metadata = getattr(match, "metadata", None) or {}
        score = getattr(match, "score", None)
        chunk_id = getattr(match, "id", "") or ""

    if not isinstance(metadata, dict):
        return None

    page_id = (metadata.get("page_id") or "").strip()
    if not page_id:
        return None

    return {
        "page_id": page_id,
        "title": metadata.get("title") or "",
        "space_key": metadata.get("space_key") or "",
        "heading": metadata.get("heading") or None,
        "relevant_content": (
            metadata.get("markdown_content")
            or metadata.get("text_summary")
            or ""
        ),
        "score": float(score) if score is not None else 0.0,
        "source": "pinecone_rag",
        "_chunk_id": chunk_id,  # kept for debug only
    }


def _normalize_rag_item(item: Any) -> Optional[Dict[str, Any]]:
    """Normalize a single RAG result (from Neo4j or Pinecone) to the shared shape.

    Neo4j results come back already shaped {page_id, title, score, relevant_content, ...}.
    Pinecone matches need the flatten step above. Anything that doesn't yield a real
    page_id is dropped here so downstream code never has to guess.
    """
    if item is None:
        return None
    if isinstance(item, dict) and item.get("page_id"):
        # Already in our shape (Neo4j path)
        return item
    # Otherwise assume it's a Pinecone match — flatten it
    return _normalize_pinecone_match(item)


async def _merged_rag_retrieval(
    graph_user_id: Optional[str] = None,
    query_terms: Optional[List[str]] = None,
    *,
    user_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Query Neo4j and Pinecone in parallel and merge results by REAL page_id.

    Both sources are always queried regardless of whether either returns results (RETR-01).
    Pinecone.search is synchronous and must be wrapped in asyncio.to_thread.

    All matches are normalized through _normalize_rag_item so chunk IDs never end up
    being treated as page IDs. Multiple chunks of the same page collapse to one entry
    keyed by real page_id; the chunk with the best score (and longest content) wins.
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

    # Merge by REAL page_id. Multiple Pinecone chunks from the same page collapse to one,
    # keeping the highest-score / longest-content entry so the drafter gets the best snippet.
    pages_by_id: Dict[str, Dict[str, Any]] = {}
    for result in all_results:
        if isinstance(result, Exception):
            logger.warning("RAG source returned error (non-fatal): %s", result)
            continue
        for raw in (result or []):
            item = _normalize_rag_item(raw)
            if not item:
                continue
            page_id = item.get("page_id") or ""
            if not page_id:
                continue
            existing = pages_by_id.get(page_id)
            if existing is None:
                pages_by_id[page_id] = item
            else:
                existing_score = float(existing.get("score") or 0)
                new_score = float(item.get("score") or 0)
                existing_content_len = len(existing.get("relevant_content") or "")
                new_content_len = len(item.get("relevant_content") or "")
                # Prefer: longer content first, then higher score on near-ties
                if new_content_len > existing_content_len * 1.1:
                    pages_by_id[page_id] = item
                elif new_score > existing_score and new_content_len >= existing_content_len * 0.8:
                    pages_by_id[page_id] = item

    sorted_pages = sorted(
        pages_by_id.values(),
        key=lambda p: -float(p.get("score") or 0),
    )
    return sorted_pages[:JARVIS_PIPELINE_MAX_PAGES]
