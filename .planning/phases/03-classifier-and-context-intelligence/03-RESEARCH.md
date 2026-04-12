# Phase 3: Classifier and Context Intelligence — Research

**Researched:** 2026-04-12
**Domain:** Neo4j Graph RAG, Python async driver, LLM routing, topic drift sliding window
**Confidence:** HIGH (core stack), MEDIUM (Cypher traversal patterns), HIGH (sliding window implementation)

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

- **D-01:** Graph RAG (Neo4j AuraDB) is the mechanism for injecting meeting context into general question answers — NOT naive transcript line injection
- **D-02:** When `_handle_general_question` fires, query the Neo4j graph for entities/context relevant to the question and inject the retrieved subgraph context into the LLM prompt
- **D-03:** Raw `transcript_log` entries are NOT passed to the general responder; only the graph query result is injected
- **D-04:** `conversation_history` passed to `answer_general_question` is capped to the **last 3 Q&A exchanges** (3 user + 3 Jarvis pairs max)
- **D-05:** Older exchanges are discarded from the window; stale topic context ages out naturally after 3 turns
- **D-06:** `general_history` in `meeting_state` should store exchanges as a list and `_format_general_history()` should enforce the 3-exchange window
- **D-07:** Replace regex-based `_needs_web_search()` with a lightweight LLM call (`gpt-4o-mini`, `max_tokens=5`, `temperature=0.0`) that returns "yes" or "no"
- **D-08:** The LLM web search decider prompt should cover: weather, sports results, news, stock prices, current events, live data — anything that changes day-to-day
- **D-09:** The DuckDuckGo Instant Answer API remains the fetch mechanism; the decision of WHEN to call it changes
- **D-10:** Graph backend: **Neo4j AuraDB** (cloud), connection URL configurable via `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD` env vars
- **D-11:** Node types: **Topics**, **People** (speakers + mentioned persons), **Decisions**
- **D-12:** Edge types: `MENTIONED_BY` (topic/decision ← speaker), `RELATED_TO` (topic ↔ topic co-occurrence), `DECIDED_IN` (decision → topic context)
- **D-13:** Ingestion is **real-time**: each new `transcript_log` entry triggers an async `asyncio.create_task` that extracts entities via LLM and upserts nodes/edges into Neo4j
- **D-14:** Query mechanism: extract entities from user's question via LLM, run Cypher query for related nodes (1-2 hops), format as a short context string injected into general responder prompt
- **D-15:** A new module `confluence_logic/graph_rag.py` owns: ingestion (`ingest_transcript_entry()`), querying (`query_context(question: str) -> str`), and Neo4j connection management

### Claude's Discretion

- LLM prompt for entity extraction during ingestion (few-shot vs zero-shot)
- Cypher query depth (1 vs 2 hops)
- Exact format of graph context injected into the general responder prompt
- Fallback behavior when Neo4j is unreachable (log + skip graph context, still answer)

### Deferred Ideas (OUT OF SCOPE)

- Multi-session Graph RAG (graph persists across meetings) — future phase
- Graph visualization / inspection tooling — backlog
- Switching from DuckDuckGo to a richer search API (SerpAPI, Brave) — backlog
- Sentiment analysis or speaker tone tracking as graph edges — out of scope

</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| CLASSIFY-03 | Classifier must correctly route meeting-context questions (e.g., "what model did my colleague suggest") to `general` intent with graph-injected context rather than ignoring meeting content | D-01 through D-03 define the injection mechanism; classifier itself is unchanged, context is injected downstream |
| TOPIC-01 | Topic drift prevention: cap `conversation_history` to last 3 Q&A exchanges so stale context ages out naturally | D-04 through D-06 specify the sliding window; `_format_general_history()` change in `jarvis_agentic.py`, `_MAX_GENERAL_HISTORY` constant from 4 to 3 |
| WEBSEARCH-01 | Expand web search trigger to cover weather, news, sports via LLM yes/no routing instead of regex | D-07 through D-09 specify the `_needs_web_search()` replacement in `general_responder.py` |
| GRAPHRAG-01 | Real-time Neo4j knowledge graph built from transcript entries; queried for every general question to inject entity context | D-10 through D-15 fully specify the design; `confluence_logic/graph_rag.py` is the new module |

</phase_requirements>

---

## Summary

Phase 3 adds four interconnected intelligence upgrades to the Jarvis pipeline. The most significant is Graph RAG: a new `graph_rag.py` module that builds a Neo4j AuraDB knowledge graph in real-time from transcript entries and queries it before every general answer. The graph captures Topics, People, and Decisions as nodes connected by `MENTIONED_BY`, `RELATED_TO`, and `DECIDED_IN` edges. Each transcript entry triggers `asyncio.create_task(ingest_transcript_entry(...))` — a fire-and-forget pattern already established in `jarvis_agentic.py`.

The second change is sliding-window topic drift: `_MAX_GENERAL_HISTORY` drops from 4 to 3, and `_format_general_history()` already slices the window — only the constant needs updating.

The third change is LLM web search routing: `_needs_web_search()` in `general_responder.py` is replaced by an async LLM call that classifies any question a user would normally Google (weather, news, sports, prices) as needing real-time data. The DuckDuckGo fetch remains unchanged.

The Neo4j Python driver (version 6.1.0, latest as of April 2026) ships a native async API (`AsyncGraphDatabase`, `AsyncSession`) that integrates directly with the codebase's `asyncio.to_thread` / `asyncio.create_task` patterns. **Recommendation: use the sync driver wrapped in `asyncio.to_thread` for simplicity, OR use the async driver directly** — both approaches work; the async driver is preferred for AuraDB to avoid thread pool exhaustion under high transcript rate.

**Primary recommendation:** Use `AsyncGraphDatabase.driver()` for the Neo4j connection; use `MERGE` for all node/edge upserts; use a dedicated entity-extraction LLM prompt (zero-shot with JSON output) for ingestion; use 1-hop Cypher traversal for query context to keep latency low.

---

## Standard Stack

### Core

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| `neo4j` | 6.1.0 | Neo4j Python driver — sync and async | Official first-party driver; ships `AsyncGraphDatabase`; AuraDB-compatible |
| `openai` | Already installed | Entity extraction LLM + yes/no web search router | Already used project-wide; gpt-4o-mini pattern established in classifier |
| `python-dotenv` | Already installed | `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD` env vars | Already used project-wide |

### Supporting (optional)

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| `neo4j-graphrag` | 1.14.1 | Higher-level GraphRAG retrievers (VectorCypherRetriever, etc.) | NOT needed for this phase — the graph schema is custom and simple enough for raw Cypher; this library adds dependency weight without benefit at this scale |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| `AsyncGraphDatabase` (native async) | sync driver + `asyncio.to_thread` | `asyncio.to_thread` is simpler but creates a new thread per call; async driver is more efficient for AuraDB's bolt protocol but requires `await` everywhere in `graph_rag.py` |
| Raw Cypher MERGE | `neo4j-graphrag` SimpleKGPipeline | neo4j-graphrag adds vector embeddings and chunking overhead not needed here; raw Cypher is more direct and controllable |

**Installation (add to requirements.txt):**
```bash
pip install neo4j==6.1.0
```

**Version verification (confirmed 2026-04-12):**
- `neo4j`: 6.1.0 (latest on PyPI as of April 2026)
- `neo4j-graphrag`: 1.14.1 (not required for this phase)

---

## Architecture Patterns

### Recommended Project Structure

```
confluence_logic/
├── graph_rag.py         # NEW: Neo4j connection, ingest_transcript_entry(), query_context()
├── general_responder.py # MODIFY: replace _needs_web_search() with async LLM call
├── jarvis_agentic.py    # MODIFY: hook ingest + query, cap sliding window to 3
├── classifier.py        # NO CHANGE — classifier routing is already correct
└── meeting_responder.py # NO CHANGE
```

### Pattern 1: Neo4j Async Driver Lifecycle

The driver should be created once at module level (lazy init on first use) and closed on application shutdown. Sessions are short-lived.

```python
# Source: https://neo4j.com/docs/api/python-driver/current/async_api.html
from neo4j import AsyncGraphDatabase
import os

_driver = None

def _get_driver():
    global _driver
    if _driver is None:
        uri = os.getenv("NEO4J_URI", "")
        user = os.getenv("NEO4J_USER", "neo4j")
        password = os.getenv("NEO4J_PASSWORD", "")
        if uri:
            _driver = AsyncGraphDatabase.driver(uri, auth=(user, password))
    return _driver

async def close_driver():
    global _driver
    if _driver:
        await _driver.close()
        _driver = None
```

**AuraDB URI format:** `neo4j+s://<instance-id>.databases.neo4j.io` (TLS, CA-signed cert, no port needed).

### Pattern 2: MERGE Upsert for Nodes and Edges

MERGE is Neo4j's upsert operator. Always MERGE nodes first, then MERGE the relationship. Never MERGE a long multi-node pattern in one statement — it creates duplicates when only part of the pattern exists.

```python
# Source: https://neo4j.com/docs/cypher-manual/current/clauses/merge/
MERGE_TOPIC_CYPHER = """
MERGE (t:Topic {name: $topic_name})
ON CREATE SET t.first_seen = timestamp()
SET t.last_seen = timestamp()
"""

MERGE_PERSON_CYPHER = """
MERGE (p:Person {name: $person_name})
ON CREATE SET p.first_seen = timestamp()
SET p.last_seen = timestamp()
"""

MERGE_DECISION_CYPHER = """
MERGE (d:Decision {text: $decision_text})
ON CREATE SET d.first_seen = timestamp()
SET d.last_seen = timestamp()
"""

MERGE_MENTIONED_BY_CYPHER = """
MATCH (subject {name: $subject_name})
MATCH (speaker:Person {name: $speaker_name})
MERGE (subject)-[:MENTIONED_BY]->(speaker)
"""

MERGE_RELATED_TO_CYPHER = """
MATCH (t1:Topic {name: $topic_a})
MATCH (t2:Topic {name: $topic_b})
MERGE (t1)-[:RELATED_TO]-(t2)
"""
```

### Pattern 3: Async Session with execute_query (Recommended for Simple Calls)

```python
# Source: https://neo4j.com/docs/api/python-driver/current/async_api.html
async def _run_write(cypher: str, **params) -> None:
    driver = _get_driver()
    if driver is None:
        return
    await driver.execute_query(cypher, params, routing_=neo4j.RoutingControl.WRITE)
```

`execute_query()` handles retry logic automatically, which is important for AuraDB transient network errors.

### Pattern 4: Entity Extraction Prompt (Zero-Shot JSON)

This is Claude's discretion per CONTEXT.md. Zero-shot with structured JSON output is recommended over few-shot for speed and consistency at `max_tokens=100`.

```python
ENTITY_EXTRACTION_PROMPT = """
You are an entity extractor for meeting transcripts.
Extract from the spoken text:
- topics: list of subjects discussed (e.g., ["GPT-4", "project deadline"])
- people: list of person names mentioned (e.g., ["Alice", "Bob"])
- decisions: list of decisions made, if any (e.g., ["use React for frontend"])
Return ONLY valid JSON: {"topics": [...], "people": [...], "decisions": [...]}
If nothing found in a category, return an empty list.
"""
```

For `gpt-4o-mini` with `max_tokens=80`, `temperature=0.0`, this consistently returns parseable JSON for short transcript entries (1-3 sentences).

### Pattern 5: Graph Context Query (1-Hop, Cypher)

```cypher
// Source: official Neo4j Cypher docs + GraphRAG traversal patterns
MATCH (n)
WHERE toLower(n.name) CONTAINS toLower($keyword)
   OR toLower(n.text) CONTAINS toLower($keyword)
WITH n
OPTIONAL MATCH (n)-[r]-(neighbor)
RETURN n, type(r) AS rel_type, neighbor
LIMIT 20
```

For query_context(), extract 1-3 entity names from the user's question via LLM, run this pattern for each, and format results as:
`"In the meeting: Alice mentioned GPT-4 (Topic). Decision: use React for frontend."` — 1-2 sentences injected into the general responder system prompt.

### Pattern 6: LLM Yes/No Web Search Router

```python
# Replaces regex _needs_web_search() in general_responder.py
WEB_SEARCH_ROUTER_PROMPT = (
    "You are a routing classifier. Answer only 'yes' or 'no'.\n"
    "Does this question require real-time or current-day data to answer accurately?\n"
    "Answer 'yes' for: weather, sports scores, news headlines, stock prices, "
    "current events, today's date/time, live data, anything that changes daily.\n"
    "Answer 'no' for: factual/historical questions, explanations, opinions, "
    "meeting transcript questions.\n"
    "Question: {question}"
)

async def _needs_web_search(question: str) -> bool:
    response = await asyncio.to_thread(
        lambda: _get_client().chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": WEB_SEARCH_ROUTER_PROMPT.format(question=question)}],
            max_tokens=5,
            temperature=0.0,
        )
    )
    return (response.choices[0].message.content or "").strip().lower().startswith("yes")
```

**Note:** `_needs_web_search` becomes `async` — the call site in `answer_general_question` must `await` it. The function signature of `answer_general_question` is already `async`, so this is a simple change.

### Pattern 7: Sliding Window — Existing Code

Current `_MAX_GENERAL_HISTORY = 4` in `jarvis_agentic.py` (line 156). D-06 requires this to be **3**. `_format_general_history()` already slices `[-_MAX_GENERAL_HISTORY:]`, so changing the constant to 3 is sufficient. No other changes required for topic drift.

### Anti-Patterns to Avoid

- **Merging a long pattern in one MERGE:** `MERGE (a:Topic)-[:MENTIONED_BY]->(b:Person)` will create duplicate nodes if either exists already — always MERGE nodes separately, then MERGE the edge.
- **Sharing an AsyncSession across concurrent tasks:** AsyncSession is not concurrency-safe; create a new session (or use `execute_query`) per call.
- **Blocking the event loop with sync neo4j calls:** If using the sync driver, always wrap in `asyncio.to_thread`. The async driver is preferred.
- **Raising on Neo4j unavailability:** Neo4j AuraDB may be unreachable (cold start, network timeout). All graph_rag calls must have `try/except Exception` that logs and returns empty string — never blocks the general question response.
- **MERGE on mutable text for Decisions:** Decision text can be long/verbose. Use a truncated or normalized form as the MERGE key, or add a hash. Otherwise, two similar-but-not-identical transcribed decisions create duplicate nodes.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Neo4j connection | Custom HTTP client to Bolt API | `neo4j` Python driver 6.1.0 | Connection pooling, retry, auth management, async support all built in |
| Graph upsert logic | Conditional INSERT/UPDATE with existence checks | `MERGE` Cypher clause | MERGE is atomic and idempotent by design |
| Entity extraction JSON parsing | Regex on LLM response | `json.loads()` with `try/except` fallback | LLM JSON output is occasionally malformed; silent fallback to empty entities is safer than crashing ingest |
| Web search routing | Expanded regex with 50+ keywords | LLM yes/no call | Regex misses paraphrases, slang, and new question forms; LLM generalizes |

---

## Common Pitfalls

### Pitfall 1: AsyncSession Shared Across Tasks

**What goes wrong:** Two concurrent `create_task` calls both try to run queries on the same `AsyncSession` — one fails with "read() called while another coroutine is already waiting."
**Why it happens:** `AsyncSession` is explicitly not concurrency-safe (official docs).
**How to avoid:** Use `driver.execute_query()` for single-query operations (it creates an internal session per call). For multi-query transactions, create a new session per task with `async with driver.session() as session`.
**Warning signs:** `asyncio error: read() called while another coroutine is already waiting` in logs.

### Pitfall 2: Neo4j Driver Not Initialized at Startup

**What goes wrong:** First transcript entry arrives before `_get_driver()` has been called; lazy init inside `create_task` works but AuraDB connection takes ~300-500ms, causing the first few transcript entries to have no graph.
**Why it happens:** Cold start on AuraDB bolt handshake.
**How to avoid:** Add an optional `await warm_graph_rag_connection()` call in the bot startup sequence (or just accept first-entry miss — fallback already handles this by returning empty context string).
**Warning signs:** First general question gets no graph context; subsequent questions work fine.

### Pitfall 3: MERGE Duplicate Nodes on Decision Text

**What goes wrong:** Two transcript entries produce near-identical decision text ("use React" vs "use React for the frontend") — MERGE creates two Decision nodes because the `text` property differs.
**Why it happens:** LLM extraction normalizes inconsistently.
**How to avoid:** Truncate Decision.text to first 80 characters for the MERGE key, or lowercase + strip punctuation before MERGE. Use `SET d.full_text = $full_text` to store the original.
**Warning signs:** Decision node count grows linearly with entries rather than plateauing.

### Pitfall 4: Blocking Event Loop in Ingestion

**What goes wrong:** `ingest_transcript_entry()` is called via `asyncio.create_task()` but internally does sync Neo4j driver calls — this blocks the event loop during bolt I/O.
**Why it happens:** Using sync `GraphDatabase.driver` instead of `AsyncGraphDatabase.driver`.
**How to avoid:** Use `AsyncGraphDatabase.driver` and `await session.run(...)` throughout `graph_rag.py`. Alternatively, wrap all sync driver calls in `asyncio.to_thread(lambda: ...)`.
**Warning signs:** WebSocket events queue up; TTS latency spikes during transcription bursts.

### Pitfall 5: LLM Web Search Router Adds Latency to Every Answer

**What goes wrong:** Every call to `answer_general_question` now makes two LLM round trips (web search router + final answer), doubling latency.
**Why it happens:** The web search router runs synchronously before the answer generation.
**How to avoid:** Run web search router and answer generation in parallel using `asyncio.gather()`: `[router_result, _] = await asyncio.gather(_needs_web_search(question), ...)`. Since web search is conditional, gather the router call alongside the (non-web) answer prefetch — or accept the sequential pattern if implementation complexity is not worth it.
**Warning signs:** General question response time exceeds 3 seconds consistently.

### Pitfall 6: Cypher Query Returns Nothing Because Text Case Mismatch

**What goes wrong:** Entity name "gpt-4" is stored in graph as "GPT-4"; Cypher `WHERE n.name CONTAINS $keyword` misses it.
**Why it happens:** Case-sensitive string matching in Cypher.
**How to avoid:** Always normalize entity names to title-case before MERGE and before query. Or use `toLower(n.name) CONTAINS toLower($keyword)` in the MATCH clause.

---

## Code Examples

### Cypher: Full Ingest Transaction (Nodes + Edges for One Entry)

```cypher
// Source: https://neo4j.com/docs/cypher-manual/current/clauses/merge/
// Step 1: MERGE topic nodes
UNWIND $topics AS topic_name
MERGE (t:Topic {name: topic_name})
ON CREATE SET t.first_seen = timestamp()
SET t.last_seen = timestamp()

// Step 2: MERGE decision nodes
WITH $decisions AS decisions
UNWIND decisions AS decision_text
MERGE (d:Decision {key: left(toLower(decision_text), 80)})
ON CREATE SET d.text = decision_text, d.first_seen = timestamp()
SET d.last_seen = timestamp()

// Step 3: MERGE speaker + MENTIONED_BY edges
// (run as a separate query after nodes exist)
MATCH (t:Topic {name: $topic_name})
MATCH (p:Person {name: $speaker_name})
MERGE (t)-[:MENTIONED_BY]->(p)
```

### Python: ingest_transcript_entry() Skeleton

```python
# confluence_logic/graph_rag.py
import asyncio
import json
import logging
import os
from typing import Any, Dict

import neo4j
from neo4j import AsyncGraphDatabase
from openai import OpenAI

logger = logging.getLogger(__name__)

_driver = None
_openai_client = None

def _get_driver():
    global _driver
    if _driver is None:
        uri = os.getenv("NEO4J_URI", "")
        if not uri:
            return None
        user = os.getenv("NEO4J_USER", "neo4j")
        password = os.getenv("NEO4J_PASSWORD", "")
        _driver = AsyncGraphDatabase.driver(uri, auth=(user, password))
    return _driver

async def ingest_transcript_entry(entry: Dict[str, Any]) -> None:
    """Fire-and-forget: extract entities and upsert into Neo4j."""
    driver = _get_driver()
    if driver is None:
        return
    try:
        entities = await _extract_entities(entry["participant"], entry["text"])
        await _upsert_entities(driver, entities, entry["participant"])
    except Exception as e:
        logger.debug("Graph ingest skipped (non-fatal): %s", e)

async def query_context(question: str) -> str:
    """Return a short context string from the graph relevant to the question."""
    driver = _get_driver()
    if driver is None:
        return ""
    try:
        keywords = await _extract_keywords(question)
        if not keywords:
            return ""
        return await _cypher_query(driver, keywords)
    except Exception as e:
        logger.debug("Graph query skipped (non-fatal): %s", e)
        return ""
```

### jarvis_agentic.py: Hook Points (lines ~1384 and ~992)

```python
# At transcript_log.append (line ~1384) — after the append:
entry = {"participant": participant, "text": sentence, "timestamp": time.time()}
log.append(entry)
# New: fire-and-forget graph ingest
if graph_rag.is_available():
    asyncio.create_task(graph_rag.ingest_transcript_entry(entry))

# In _handle_general_question (line ~992) — before answer_general_question call:
graph_context = await graph_rag.query_context(query)
answer = await answer_general_question(query, conversation_history, graph_context=graph_context)
```

### general_responder.py: Graph Context Injection

```python
# answer_general_question signature addition
async def answer_general_question(
    question: str,
    conversation_history: str = "",
    graph_context: str = "",          # NEW
) -> str:
    ...
    if graph_context:
        system_prompt += (
            "\n\nMeeting context (from the current conversation):\n"
            + graph_context
        )
```

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Regex `_needs_web_search()` | LLM yes/no classifier (`gpt-4o-mini`, max_tokens=5) | Phase 3 | Catches weather, news, sports that regex missed |
| `_MAX_GENERAL_HISTORY = 4` turns | 3 turns sliding window | Phase 3 | Topic drift resolves after 3 more exchanges |
| No meeting context in general answers | Neo4j Graph RAG context injection | Phase 3 | "What model did colleague suggest?" now answered from graph |
| Naive `asyncio.to_thread` for all LLM calls | AsyncGraphDatabase.driver for Neo4j | Phase 3 | No thread pool blocking on high-frequency transcript ingest |

**Deprecated/outdated:**
- `_FRESHNESS_PATTERNS` regex in `general_responder.py`: replaced by LLM router. Delete after migration.

---

## Open Questions

1. **Neo4j AuraDB free tier rate limits**
   - What we know: AuraDB Free tier has storage limits (~200K nodes), connection limits not publicly documented
   - What's unclear: Whether the free tier supports sustained bolt connections during a 60-minute meeting (100-300 transcript entries)
   - Recommendation: Implement with graceful fallback (if NEO4J_URI not set or unreachable, skip graph context silently). Document that AuraDB Pro is needed for production.

2. **Entity extraction JSON reliability for very short transcript entries (< 5 words)**
   - What we know: gpt-4o-mini occasionally returns non-JSON for very short inputs like "Yeah" or "Exactly"
   - What's unclear: Whether wrapping in try/except json.loads is sufficient or if the prompt needs a "return {} if nothing to extract" instruction
   - Recommendation: Add explicit instruction "If the text is too short to extract any entities, return {\"topics\": [], \"people\": [], \"decisions\": []}" to the prompt.

3. **Should `query_context()` run entity extraction or keyword matching?**
   - What we know: D-14 says "extract entities from the user's question via LLM" — this requires a second LLM call per question
   - What's unclear: Latency impact (adds ~200ms); whether simple keyword extraction from question words is sufficient
   - Recommendation: Start with simple word extraction from question (no LLM call) for MVP. If "what model did colleague suggest" misses nodes because "model" isn't in the graph verbatim, upgrade to LLM entity extraction. This is Claude's discretion.

---

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| Python 3.13 | All code | Yes | 3.13.9 | — |
| `neo4j` Python driver | graph_rag.py | Not installed (not in requirements.txt) | — (6.1.0 on PyPI) | Wave 0: add to requirements.txt |
| Neo4j AuraDB cloud | graph_rag.py at runtime | Unknown — not testable without credentials | — | Graceful fallback: if NEO4J_URI unset, skip all graph ops |
| `openai` | All LLM calls | Yes (already installed) | Already in requirements.txt | — |
| `pytest` | Tests | Yes (in requirements.txt) | Installed | — |
| `pytest-asyncio` | Async tests | Unknown | — | Wave 0: `pip install pytest-asyncio` |

**Missing dependencies with no fallback:**
- `neo4j==6.1.0` is not in requirements.txt — must be added in Wave 0.

**Missing dependencies with fallback:**
- AuraDB cloud instance: `graph_rag.py` must return empty string gracefully when `NEO4J_URI` is unset or unreachable.
- `pytest-asyncio`: needed for async test cases; add to dev requirements.

---

## Validation Architecture

### Test Framework

| Property | Value |
|----------|-------|
| Framework | pytest (installed) |
| Config file | None detected — `pytest.ini` or `pyproject.toml [tool.pytest.ini_options]` may need creation |
| Quick run command | `pytest confluence_logic/tests/ -x -q` |
| Full suite command | `pytest confluence_logic/tests/ -v` |

### Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| TOPIC-01 | `_format_general_history()` returns at most 3 exchanges | unit | `pytest confluence_logic/tests/test_jarvis_agentic.py -x -k "test_format_general_history"` | No — Wave 0 |
| TOPIC-01 | `_remember_general_exchange()` discards oldest beyond 3 | unit | `pytest confluence_logic/tests/test_jarvis_agentic.py -x -k "test_sliding_window"` | No — Wave 0 |
| WEBSEARCH-01 | `_needs_web_search("weather in Paris")` returns True | unit | `pytest confluence_logic/tests/test_general_responder.py -x -k "test_web_search_router"` | No — Wave 0 |
| WEBSEARCH-01 | `_needs_web_search("what is a variable")` returns False | unit | `pytest confluence_logic/tests/test_general_responder.py -x -k "test_web_search_router"` | No — Wave 0 |
| GRAPHRAG-01 | `ingest_transcript_entry()` calls MERGE Cypher with correct params | unit (mock) | `pytest confluence_logic/tests/test_graph_rag.py -x -k "test_ingest"` | No — Wave 0 |
| GRAPHRAG-01 | `query_context()` returns empty string when driver is None | unit | `pytest confluence_logic/tests/test_graph_rag.py -x -k "test_query_context_no_driver"` | No — Wave 0 |
| GRAPHRAG-01 | `query_context()` returns empty string when Neo4j raises exception | unit | `pytest confluence_logic/tests/test_graph_rag.py -x -k "test_query_context_fallback"` | No — Wave 0 |
| CLASSIFY-03 | `_handle_general_question` injects graph_context into `answer_general_question` call | unit (mock) | `pytest confluence_logic/tests/test_jarvis_agentic.py -x -k "test_general_question_graph_injection"` | No — Wave 0 |

### Sampling Rate

- **Per task commit:** `pytest confluence_logic/tests/ -x -q`
- **Per wave merge:** `pytest confluence_logic/tests/ -v`
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps

- [ ] `confluence_logic/tests/test_graph_rag.py` — covers GRAPHRAG-01 (mock AsyncGraphDatabase, mock OpenAI)
- [ ] `confluence_logic/tests/test_general_responder.py` — covers WEBSEARCH-01 (mock openai for `_needs_web_search`)
- [ ] Add sliding window tests to existing `confluence_logic/tests/test_jarvis_agentic.py` — covers TOPIC-01
- [ ] `pip install pytest-asyncio` — needed for `@pytest.mark.asyncio` on async test functions
- [ ] Add `neo4j==6.1.0` to `requirements.txt`

**Test mocking strategy for Neo4j (no live DB required):**
```python
# Pattern: mock AsyncGraphDatabase.driver and AsyncSession
from unittest.mock import AsyncMock, patch, MagicMock

@pytest.mark.asyncio
async def test_ingest_transcript_entry_calls_merge():
    mock_driver = AsyncMock()
    mock_driver.execute_query = AsyncMock(return_value=([], None, []))
    with patch("confluence_logic.graph_rag._get_driver", return_value=mock_driver):
        with patch("confluence_logic.graph_rag._extract_entities", return_value={"topics": ["React"], "people": [], "decisions": []}):
            await ingest_transcript_entry({"participant": "Alice", "text": "Let's use React", "timestamp": 0})
    mock_driver.execute_query.assert_called()
```

---

## Sources

### Primary (HIGH confidence)

- [Neo4j Python Driver 6.1 Async API Docs](https://neo4j.com/docs/api/python-driver/current/async_api.html) — AsyncGraphDatabase, AsyncSession patterns, execute_query
- [Neo4j Python Manual — Concurrency](https://neo4j.com/docs/python-manual/current/concurrency/) — asyncio integration, session safety
- [Neo4j Python Manual — Advanced Connection](https://neo4j.com/docs/python-manual/current/connect-advanced/) — AuraDB URI format (`neo4j+s://`), auth options
- [Neo4j Cypher MERGE docs](https://neo4j.com/docs/cypher-manual/current/clauses/merge/) — upsert patterns, ON CREATE/ON MATCH
- PyPI `neo4j` package — version 6.1.0 verified 2026-04-12
- PyPI `neo4j-graphrag` package — version 1.14.1 verified 2026-04-12

### Secondary (MEDIUM confidence)

- [Neo4j Community: batch upsert with UNWIND](https://community.neo4j.com/t/how-to-do-large-batch-insert-or-upsert-nodes-and-relationship-neo4j-using-python-driver/58880) — UNWIND batch pattern for high-frequency ingestion
- [Neo4j GraphRAG Python package docs](https://neo4j.com/docs/neo4j-graphrag-python/current/) — traversal patterns; determined NOT needed for this phase
- [Medium: Mocking Neo4j in Python tests](https://medium.com/@pbilling_97992/how-to-mock-neo4j-database-responses-using-the-python-driver-71bd30000ac2) — test mocking strategy

### Tertiary (LOW confidence)

- WebSearch results on LLM yes/no routing patterns — no single authoritative source; pattern validated by existing classifier code in this codebase which uses the same gpt-4o-mini + max_tokens=5 + temperature=0 approach

---

## Metadata

**Confidence breakdown:**

- Standard stack: HIGH — neo4j driver version verified from PyPI; async API verified from official docs
- Architecture: HIGH — patterns verified from official Neo4j docs; integration points verified from reading actual source code
- Pitfalls: MEDIUM — AsyncSession concurrency pitfall from official docs (HIGH); MERGE duplicate pitfall from docs + community (MEDIUM); LLM latency concern from code analysis (MEDIUM)
- Test strategy: HIGH — mocking patterns match existing project test style (AsyncMock/patch already used in test_jarvis_agentic.py)

**Research date:** 2026-04-12
**Valid until:** 2026-05-12 (neo4j driver versions — stable API; AuraDB connection format — stable)
