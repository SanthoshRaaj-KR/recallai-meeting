# Phase 3: Classifier and Context Intelligence - Context

**Gathered:** 2026-04-12
**Status:** Ready for planning

<domain>
## Phase Boundary

Fix four interrelated intelligence issues in the Jarvis pipeline:
1. General questions that reference meeting content (e.g., "what model did my colleague suggest") get no transcript context today — Graph RAG fixes this
2. Topic drift — stale Q&A history biases answers when user shifts topics (e.g., solar panels → F1 race)
3. Web search gaps — weather, sports, news don't trigger the existing DuckDuckGo search
4. Graph RAG implementation — Neo4j AuraDB knowledge graph over the meeting transcript, ingested in real-time, queried for every general question

This phase does NOT change the confluence editing pipeline, WAV caching, or the summarize/opinion handlers from Phase 2.

</domain>

<decisions>
## Implementation Decisions

### Meeting Context Injection via Graph RAG
- **D-01:** Graph RAG (Neo4j) is the mechanism for injecting meeting context into general question answers — NOT naive transcript line injection
- **D-02:** When `_handle_general_question` fires, query the Neo4j graph for entities/context relevant to the question and inject the retrieved subgraph context into the LLM prompt
- **D-03:** Raw `transcript_log` entries are NOT passed to the general responder; only the graph query result is injected

### Topic Drift — Sliding Window
- **D-04:** `conversation_history` passed to `answer_general_question` is capped to the **last 3 Q&A exchanges** (3 user + 3 Jarvis pairs max)
- **D-05:** Older exchanges are discarded from the window; stale topic context ages out naturally after 3 turns
- **D-06:** `general_history` in `meeting_state` should store exchanges as a list and `_format_general_history()` should enforce the 3-exchange window

### Web Search — LLM Routing
- **D-07:** Replace the regex-based `_needs_web_search()` with a lightweight LLM call (`gpt-4o-mini`, `max_tokens=5`, `temperature=0.0`) that returns "yes" or "no" for whether real-time data is needed
- **D-08:** The LLM web search decider prompt should cover: weather, sports results, news, stock prices, current events, live data — anything that changes day-to-day
- **D-09:** The DuckDuckGo Instant Answer API remains the fetch mechanism; the decision of WHEN to call it changes

### Graph RAG — Neo4j AuraDB
- **D-10:** Graph backend: **Neo4j AuraDB** (cloud), connection URL configurable via `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD` env vars
- **D-11:** Node types: **Topics**, **People** (speakers + mentioned persons), **Decisions**
- **D-12:** Edge types: `MENTIONED_BY` (topic/decision ← speaker), `RELATED_TO` (topic ↔ topic co-occurrence), `DECIDED_IN` (decision → topic context)
- **D-13:** Ingestion is **real-time**: each new `transcript_log` entry triggers an async `asyncio.create_task` that extracts entities via LLM and upserts nodes/edges into Neo4j — graph is always current
- **D-14:** Query mechanism: extract entities from the user's question via LLM, then run a Cypher query to retrieve related nodes (1-2 hops), format as a short context string injected into the general responder prompt
- **D-15:** A new module `confluence_logic/graph_rag.py` owns: ingestion (`ingest_transcript_entry()`), querying (`query_context(question: str) -> str`), and Neo4j connection management

### Claude's Discretion
- LLM prompt for entity extraction during ingestion (few-shot vs zero-shot)
- Cypher query depth (1 vs 2 hops)
- Exact format of graph context injected into the general responder prompt
- Fallback behavior when Neo4j is unreachable (log + skip graph context, still answer)

</decisions>

<specifics>
## Specific Ideas

- "What model did my colleague suggest?" — this is the canonical test case. The colleague mentioned a model in the transcript; the graph should have a node for that model/decision and the Cypher query for "model" should retrieve it.
- For web search: the user expected Jarvis to "do a simple Chrome search" for weather in Paris — the LLM decider should catch any question a person would normally Google
- Topic drift example: solar panels → F1 race winner. After 3 more Q&A turns, Jarvis should no longer mention solar panels when talking about F1.

</specifics>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Core Pipeline
- `confluence_logic/jarvis_agentic.py` — Main handler: `_handle_general_question()`, `meeting_state["general_history"]`, `meeting_state["transcript_log"]`, `_format_general_history()`
- `confluence_logic/classifier.py` — Intent classifier: fast-path heuristics + LLM fallback; do not break existing 4-intent routing

### General Responder
- `confluence_logic/general_responder.py` — `answer_general_question(question, conversation_history)`, `_needs_web_search()`, `_quick_web_search()`, `_history_to_messages()` — all touch points for this phase

### Graph RAG (new)
- `confluence_logic/graph_rag.py` — **New file to create**: owns Neo4j connection, `ingest_transcript_entry()`, `query_context()`

### Prior Phase Context
- `.planning/phases/01-intelligent-question-classification-and-conversational-response/01-CONTEXT.md` — General responder design decisions from Phase 1
- `.planning/phases/02-meeting-transcript-access-with-summarization-and-opinion-generation/02-VERIFICATION.md` — `transcript_log` schema: list of dicts with `participant` and `text` keys

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `general_responder._quick_web_search()`: DuckDuckGo fetch stays as-is; only the trigger logic (`_needs_web_search`) changes
- `general_responder._history_to_messages()`: Already converts history string → proper OpenAI message objects; sliding window should feed into this same function
- `meeting_state["transcript_log"]`: list of `{"participant": str, "text": str}` dicts — input to graph ingestion

### Established Patterns
- Fire-and-forget async tasks via `asyncio.create_task()` — use this for real-time graph ingestion on each transcript entry
- `asyncio.to_thread()` for blocking I/O (OpenAI calls, HTTP) — use for Neo4j writes and LLM entity extraction
- Env-var configuration via `os.getenv()` with defaults — follow this for `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD`

### Integration Points
- `jarvis_agentic.py`: Where transcript entries are appended to `transcript_log` — this is where to hook the real-time ingestion `create_task`
- `_handle_general_question()` (line ~987): Where to call `graph_rag.query_context(query)` and inject result into the `answer_general_question` call
- `_format_general_history()` (line ~169): Enforce 3-exchange sliding window here

</code_context>

<deferred>
## Deferred Ideas

- Multi-session Graph RAG (graph persists across meetings) — future phase
- Graph visualization / inspection tooling — backlog
- Switching from DuckDuckGo to a richer search API (SerpAPI, Brave) — backlog
- Sentiment analysis or speaker tone tracking as graph edges — out of scope

</deferred>

---

*Phase: 03-classifier-and-context-intelligence*
*Context gathered: 2026-04-12*
