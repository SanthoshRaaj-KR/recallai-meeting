---
phase: 03-classifier-and-context-intelligence
plan: 02
subsystem: graph-rag
tags: [neo4j, graph-rag, entity-extraction, cypher, asyncio, meeting-context, gpt-4o-mini]

dependency_graph:
  requires:
    - phase: 03-01
      provides: "neo4j==6.1.0 in requirements.txt, pytest-asyncio installed, general_responder.py with async _needs_web_search"
    - phase: 02-02
      provides: "transcript_log schema: {participant, text, timestamp} entries in meeting_state"
  provides:
    - "confluence_logic/graph_rag.py: Neo4j async driver lifecycle, _extract_entities, ingest_transcript_entry, query_context"
    - "Real-time graph ingestion of transcript entries via asyncio.create_task"
    - "Meeting-context injection into general question answers via graph_context parameter"
  affects:
    - "general_responder.answer_general_question signature now accepts graph_context kwarg"
    - "jarvis_agentic._handle_general_question now queries and injects graph context"

tech-stack:
  added: []
  patterns:
    - "Fire-and-forget async graph ingestion via asyncio.create_task(graph_rag.ingest_transcript_entry(entry))"
    - "Lazy singleton Neo4j AsyncDriver with graceful fallback when NEO4J_URI unset"
    - "Zero-shot JSON entity extraction via gpt-4o-mini for ingestion"
    - "Simple keyword extraction (no LLM) for query_context MVP — no latency added"
    - "MERGE-based Neo4j upserts: nodes first, then edges (avoids duplicate pattern MERGE pitfall)"
    - "All graph ops wrapped in try/except — non-fatal, pipeline never blocked"

key-files:
  created:
    - confluence_logic/graph_rag.py
    - confluence_logic/tests/test_graph_rag.py
  modified:
    - confluence_logic/jarvis_agentic.py
    - confluence_logic/general_responder.py
    - confluence_logic/tests/test_jarvis_agentic.py

key-decisions:
  - "Simple word-based keyword extraction for query_context (no LLM call) — zero latency, adequate for MVP; upgrade to LLM entity extraction if entity name mismatch becomes a problem"
  - "Decision node MERGE key is truncated text[:80].lower() — prevents duplicate nodes from near-identical decision text (per RESEARCH Pitfall 3)"
  - "graph_context injected into system_prompt (not user message) — keeps meeting facts in model instruction context"

requirements-completed: [GRAPHRAG-01, CLASSIFY-03]

duration: 8min
completed: 2026-04-13
---

# Phase 03 Plan 02: Graph RAG Module and Pipeline Wiring Summary

**Neo4j-backed real-time Graph RAG over meeting transcripts: entity extraction on ingest (gpt-4o-mini), MERGE-based upsert of Topic/Person/Decision nodes, 1-hop Cypher query for meeting context injection into general question answers.**

## Performance

- **Duration:** ~8 min
- **Started:** 2026-04-13T08:56:03Z
- **Completed:** 2026-04-13T09:04:00Z
- **Tasks:** 2 (Task 1 TDD with 2 commits: RED + GREEN; Task 2 integration)
- **Files modified:** 5

## Accomplishments

- Created `graph_rag.py` module: lazy `AsyncGraphDatabase` driver singleton with graceful fallback when `NEO4J_URI` is unset; `_extract_entities` via gpt-4o-mini zero-shot JSON; `ingest_transcript_entry` with MERGE upserts for Topic/Person/Decision nodes and MENTIONED_BY/RELATED_TO/DECIDED_IN edges; `query_context` with simple keyword extraction and 1-hop Cypher traversal
- Wired real-time ingestion into `jarvis_agentic.py`: `asyncio.create_task(graph_rag.ingest_transcript_entry(log[-1]))` fires after every transcript_log.append (fire-and-forget, non-blocking)
- Wired query injection into `_handle_general_question`: `graph_context = await graph_rag.query_context(query)` called before `answer_general_question`, result injected as Meeting context section in system prompt (per D-01, D-02, D-03 — raw transcript_log NOT passed)

## Task Commits

Each task was committed atomically:

1. **Task 1 RED: Failing tests for Graph RAG module** - `689207c` (test)
2. **Task 1 GREEN: graph_rag.py module implementation** - `6225a62` (feat)
3. **Task 2: Wire graph_rag into jarvis_agentic.py and general_responder.py** - `831de3d` (feat)

## Files Created/Modified

- `confluence_logic/graph_rag.py` — New module: Neo4j async driver lifecycle, entity extraction, MERGE ingestion, Cypher query context
- `confluence_logic/tests/test_graph_rag.py` — 6 async unit tests with mocked Neo4j driver and OpenAI (TDD RED/GREEN)
- `confluence_logic/jarvis_agentic.py` — Added `from confluence_logic import graph_rag` import; ingest create_task hook after transcript_log.append; query_context call + graph_context kwarg in _handle_general_question
- `confluence_logic/general_responder.py` — Added `graph_context: str = ""` parameter to `answer_general_question`; injects as "Meeting context" section in system_prompt when non-empty
- `confluence_logic/tests/test_jarvis_agentic.py` — Added `test_handle_general_question_injects_graph_context` (CLASSIFY-03)

## Decisions Made

- **Simple keyword extraction for query_context:** Per research Open Question 3, started with word-based keyword extraction (no LLM call, zero latency) instead of the D-14 LLM entity extraction. The stop-word filter handles common question words. Upgrade path documented if entity name mismatch causes misses.
- **Decision node key:** Truncated `text[:80].lower()` used as MERGE key for Decision nodes — prevents duplicate nodes from near-identical LLM-transcribed decision text (RESEARCH Pitfall 3).
- **graph_context injected into system_prompt:** Meeting context string prepended to system_prompt (not as user message) — keeps meeting facts in model instruction context so they influence the entire response.

## Deviations from Plan

None — plan executed exactly as written. The plan included the complete implementation code for both graph_rag.py and all wiring changes.

## Issues Encountered

- **git stash interference:** Verification runs required stash operations that conflicted with pycache files. Resolved by checking out pycache files before stash pop. No code was lost — all changes correctly in place.
- **Pre-existing test failures:** 7 tests in test_jarvis_agentic.py (handle_spoken_request variants, SWITCH_ACK) and 1 in test_flow.py were pre-existing failures predating this plan, confirmed by reverting changes and running same tests. All new plan-required tests pass.

## User Setup Required

**External services require manual configuration.** Neo4j AuraDB connection details must be set as environment variables for graph ingestion to activate:

- `NEO4J_URI` — Connection URI from Neo4j AuraDB Console (format: `neo4j+s://<id>.databases.neo4j.io`)
- `NEO4J_USER` — Username (default: `neo4j`)
- `NEO4J_PASSWORD` — Password shown once at AuraDB instance creation

If these are unset, the Graph RAG module silently skips all graph operations — general questions still answer correctly without meeting context. The module is fully graceful on missing credentials or Neo4j unavailability.

## Known Stubs

None — all changes are wired and functional. Graph operations gracefully no-op when `NEO4J_URI` is unset (intended behavior per D-10 and must_haves).

## Next Phase Readiness

- Phase 3 is now complete: sliding window (03-01), LLM web search router (03-01), and Graph RAG with pipeline wiring (03-02) all delivered
- All GRAPHRAG-01, CLASSIFY-03, TOPIC-01, WEBSEARCH-01 requirements fulfilled
- Graph ingestion will populate from first transcript entry once NEO4J_URI is configured; cold-start latency (~300ms AuraDB handshake) means first entry may miss graph, subsequent queries benefit

## Self-Check: PASSED

### Files Verified
- [x] `confluence_logic/graph_rag.py` — exists
- [x] `confluence_logic/tests/test_graph_rag.py` — exists
- [x] `.planning/phases/03-classifier-and-context-intelligence/03-02-SUMMARY.md` — exists

### Commits Verified
- [x] `689207c` — RED phase tests
- [x] `6225a62` — graph_rag.py implementation
- [x] `831de3d` — pipeline wiring

### Key Content Verified
- [x] AsyncGraphDatabase, MERGE Topic/Person/Decision, MENTIONED_BY, RELATED_TO, DECIDED_IN, toLower in Cypher
- [x] All 6 test_graph_rag.py tests pass

---
*Phase: 03-classifier-and-context-intelligence*
*Completed: 2026-04-13*
