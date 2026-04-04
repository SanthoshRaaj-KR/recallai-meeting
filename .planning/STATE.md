---
gsd_state_version: 1.0
milestone: v1.0
milestone_name: milestone
status: executing
stopped_at: Completed 07-01-PLAN.md
last_updated: "2026-04-04T18:50:07.242Z"
last_activity: 2026-04-04
progress:
  total_phases: 7
  completed_phases: 6
  total_plans: 17
  completed_plans: 15
  percent: 0
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-04-04)

**Core value:** Institutional memory for teams — every decision, action item, and discussion from every meeting is instantly queryable
**Current focus:** Phase 07 — fully-agentic-meeting-pipeline-redesign

## Current Position

Phase: 07 (fully-agentic-meeting-pipeline-redesign) — EXECUTING
Plan: 2 of 3
Status: Ready to execute
Last activity: 2026-04-04

Progress: [░░░░░░░░░░] 0%

## Performance Metrics

**Velocity:**

- Total plans completed: 0
- Average duration: -
- Total execution time: 0 hours

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| - | - | - | - |

**Recent Trend:**

- Last 5 plans: none yet
- Trend: -

*Updated after each plan completion*
| Phase 01-prerequisite-refactor P01 | 2 | 2 tasks | 5 files |
| Phase 01-prerequisite-refactor P02 | 5 | 2 tasks | 2 files |
| Phase 02-storage-foundation P01 | 2 | 1 tasks | 6 files |
| Phase 02-storage-foundation P02 | 5 minutes | 1 tasks | 3 files |
| Phase 03-ingestion-pipeline P03-01 | 6 minutes | 2 tasks | 3 files |
| Phase 03-ingestion-pipeline P03-02 | 15 minutes | 2 tasks | 3 files |
| Phase 04-retrieval-hybrid-rag P04-01 | 2 minutes | 2 tasks | 2 files |
| Phase 04-retrieval-hybrid-rag P04-02 | 3 minutes | 2 tasks | 2 files |
| Phase 05-orchestration-answer-path P05-01 | 3 minutes | 4 tasks | 4 files |
| Phase 05 P05-02 | 3 minutes | 2 tasks | 2 files |
| Phase 05-orchestration-answer-path P05-03 | 3 minutes | 2 tasks | 2 files |
| Phase 06-budget-model-switch-and-streaming-llm-to-tts-pipeline P06-01 | 1 | 1 tasks | 1 files |
| Phase 06 P02 | 2 minutes | 1 tasks | 2 files |
| Phase 06 P03 | 4 minutes | 1 tasks | 2 files |
| Phase 07-fully-agentic-meeting-pipeline-redesign P07-01 | 2 | 4 tasks | 4 files |

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.
Recent decisions affecting current work:

- [Pre-phase]: Use agents-as-tools pattern (NOT handoffs) for RAG pipeline — Orchestrator must retain conversation control to merge retrieval + answer outputs
- [Pre-phase]: One vector per meeting at summary level — never store raw transcript chunks as individual vectors
- [Pre-phase]: All timestamps stored as Unix epoch integers in Pinecone metadata — required for $gte/$lte date range filters
- [Pre-phase]: Pinecone index must be created with metric="dotproduct" — wrong metric requires full re-ingestion to fix
- [Pre-phase]: INFRA-01 (asyncio.Lock for meeting_state) is a hard prerequisite before any agent code is wired
- [Phase 01]: pyaudio removed from requirements.txt — unused, causes macOS build failures
- [Phase 01]: All asyncio.Lock reads also protected — prevents torn reads in concurrent contexts
- [Phase 01]: get_health_snapshot acquires lock once for atomic snapshot — avoids TOCTOU in /health endpoint
- [Phase Phase 01]: _sync_set_state helper used for main() sync-to-async bridge — uvicorn loop not yet running when main() sets state
- [Phase Phase 01]: asyncio.to_thread used for speak() calls in handle_query — keeps blocking HTTP off the event loop
- [Phase Phase 01]: asyncio.create_task used for handle_query dispatch — fire-and-forget from websocket_endpoint without blocking receive loop
- [Phase 02-storage-foundation]: All timestamp fields (start_ts, end_ts, summarized_at) are Python int — no datetime objects in model, enforced by Pydantic type annotation
- [Phase 02-storage-foundation]: MetadataStore raises FileNotFoundError on read of nonexistent path — explicit error, not None sentinel
- [Phase 02-storage-foundation]: Sparse vector attributes on SparseEmbedding SDK v8 are sparse_indices and sparse_values directly on embedding object — not a nested sub-object
- [Phase 02-storage-foundation]: input_type=passage for upsert, input_type=query for query in pinecone-sparse-english-v0 inference calls
- [Phase 03-ingestion-pipeline]: conftest.py extends openai-agents SDK agents.__path__ to include local agents/ directory — resolves namespace conflict without __init__.py
- [Phase 03-ingestion-pipeline]: SummarizerAgent raises ValueError on blank ActionItem.owner — enforces attribution at runtime, not just in prompt
- [Phase 03-ingestion-pipeline]: pinecone_client and slack_app initialized behind env var guards — app starts without PINECONE_API_KEY or SLACK_BOT_TOKEN set
- [Phase 03-ingestion-pipeline]: asyncio.create_task() used in disconnect handler so WebSocket teardown is not blocked by ingestion pipeline
- [Phase 04-retrieval-hybrid-rag]: alpha=0.7 default used for dense/sparse balance in query() — research-backed for conversational meeting queries
- [Phase 04-retrieval-hybrid-rag]: bge-reranker-v2-m3 model used for neural reranking via pc.inference.rerank — Pinecone hosted, no local model required
- [Phase 04-retrieval-hybrid-rag]: retrieve() pipeline fixed at top_k=20 candidates reranked to top_n=5 by default — both overridable
- [Phase Phase 04-retrieval-hybrid-rag]: RetrieverAgent is a plain Python class (not openai-agents Agent() instance) — Phase 5 wires it as a registered tool
- [Phase Phase 04-retrieval-hybrid-rag]: RetrievalResult defined in agents/retriever.py (retrieval layer concern), not storage/models.py
- [Phase 05]: DateResolutionAgent is a plain Python class — not an openai-agents Agent() instance — consistent with RetrieverAgent pattern from Phase 4
- [Phase 05]: AnswerAgent uses Agent(output_type=AnswerOutput) for structured LLM output — no manual JSON parsing
- [Phase 05]: OrchestratorAgent is a plain Python class (NOT openai-agents Agent() instance) — consistent with RetrieverAgent pattern from Phase 4
- [Phase 05]: Disambiguation triggered ONLY when date in query AND retriever returns >1 meeting — prevents false positives on broad queries
- [Phase 05]: _handle_ask and _handle_message_disambig defined at module level so they are importable in tests regardless of SLACK_BOT_TOKEN
- [Phase 05]: _handle_memory_query returns OrchestratorResult with 'not configured' answer when orchestrator is None — no exception raised
- [Phase 06]: OrchestratorAgent tests patch AsyncOpenAI to prevent OPENAI_API_KEY requirement at test time
- [Phase 06]: _ABBREVS frozenset used for abbreviation-aware sentence splitting — prevents false sentence boundary detection on Dr/Mr/Mrs etc.
- [Phase 06]: finditer-based sentence splitter chosen over re.split lookbehind — Python re does not support variable-width lookbehinds
- [Phase 06]: _run_stream inner function collected entirely via asyncio.to_thread before token processing — avoids async/thread boundary complexity while still freeing event loop during generation
- [Phase 06]: Fallback to speak_chunked on _stream_llm_and_speak exception — guarantees voice output even if streaming fails
- [Phase 07-01]: MeetingWriterAgent is a plain Python class (no LLM) — consistent with RetrieverAgent pattern from Phase 4
- [Phase 07-01]: .md file path: {base_dir}/{channel_id}/{date_str}_{meeting_id}.md (discretion decision)
- [Phase 07-01]: upsert_index stores entries as json.loads(entry.model_dump_json()) dicts in a JSON array

### Roadmap Evolution

- Phase 6 added: Budget model switch and streaming LLM-to-TTS pipeline
- Phase 7 added: Fully agentic meeting pipeline redesign

### Pending Todos

None yet.

### Blockers/Concerns

- [Phase 5 design spike needed]: The clarification loop state machine interaction with jarvis.py's WebSocket async handler is not fully specified — plan a design spike before Phase 5 begins
- [Phase 4 calibration]: alpha=0.7 default is research-backed but not empirically validated for this corpus — plan calibration exercise after first 10 real meetings are indexed

## Session Continuity

Last session: 2026-04-04T18:50:07.238Z
Stopped at: Completed 07-01-PLAN.md
Resume file: None
