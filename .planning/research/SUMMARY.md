# Project Research Summary

**Project:** RecallAI Meeting Memory Bot — Hybrid RAG Milestone
**Domain:** Meeting intelligence with multi-agent hybrid RAG retrieval
**Researched:** 2026-04-04
**Confidence:** HIGH

## Executive Summary

This milestone adds institutional memory to an existing Python/FastAPI/Slack bot (`jarvis.py`) by introducing a hybrid RAG pipeline, multi-agent orchestration via the OpenAI Agents SDK, and persistent meeting storage across Pinecone and local JSON files. The recommended approach is a clean two-path architecture: an ingestion path that captures meeting transcripts, summarizes them with GPT-4o-mini via a `SummarizerAgent`, and writes structured records to both disk and Pinecone; and a query path that routes natural-language questions through an `OrchestratorAgent` using agents-as-tools to drive a `RetrieverAgent` (hybrid Pinecone search) and `AnswerAgent` (synthesis). The build adds five new Python dependencies (pinecone, pinecone-text, dateparser, aiofiles, pytest-asyncio), pins all existing unpinned deps, and extends `jarvis.py` at two integration points only.

The key architectural decisions are well-supported by official documentation. Pinecone's hybrid search requires `metric=dotproduct` at index creation, one vector per meeting at summary level (not chunk level), and a flat scalar metadata schema with UNIX timestamp integers for date filtering. The OpenAI Agents SDK's agents-as-tools pattern — not handoffs — is the correct mechanism for the RAG pipeline because the Orchestrator must synthesize outputs from both Retriever and Answer rather than ceding conversational control. The neural sparse model `pinecone-sparse-english-v0` outperforms fitted BM25 by 23–44% on retrieval benchmarks and eliminates the corpus-management overhead of `BM25Encoder`.

The dominant risks are silent configuration failures (wrong Pinecone index metric, ISO date strings instead of UNIX integers, micro-chunk vectors), race conditions in the existing global `meeting_state` dict, and context window explosion across multi-agent handoffs. All three are preventable at design time. A prerequisite refactor of `meeting_state` to use `asyncio.Lock` must happen before any agent code is wired in. The codebase currently has no tests; establishing pytest patterns in parallel with Phase 1 is essential to catch silent failures early.

---

## Key Findings

### Recommended Stack

The project builds on an existing Python/FastAPI/OpenAI stack. All new dependencies are well-matched: Pinecone 8.1.1 (with serverless indexes and integrated inference), `dateparser` 1.4.0 (200+ language locales, handles relative expressions), `aiofiles` 25.1.0 (non-blocking file I/O for asyncio), and `pydantic` 2.12.5 (already a FastAPI transitive dep — pin explicitly). The `openai-agents` package is already in `requirements.txt` but unpinned; pin to `0.13.4`. All existing deps must also be pinned. Do not add `sentence-transformers`, `rank_bm25` (unmaintained since 2022), or LangChain.

**Core technologies:**
- `openai-agents==0.13.4`: Multi-agent orchestration — already in stack; provides Agent, Runner, handoff() primitives; built-in tracing
- `pinecone==8.1.1`: Vector DB client — managed scaling, hybrid search in single index, integrated sparse inference
- `pinecone-text==0.11.0`: BM25Encoder for sparse vectors — needed only if local BM25 is chosen; prefer hosted `pinecone-sparse-english-v0` instead
- `dateparser==1.4.0`: Natural language date resolution — handles "last Wednesday", timezone-aware output, Python 3.10+ compatible
- `aiofiles==25.1.0`: Async file I/O — prevents event loop stall from blocking disk writes in FastAPI async handlers
- `pydantic==2.12.5`: Schema validation and serialization — single canonical model for both JSON files and Pinecone upserts
- `pytest==9.0.2` + `pytest-asyncio==1.3.0`: Test infrastructure — no tests currently exist; establish patterns now

See `.planning/research/STACK.md` for complete version list, environment variables, and alternatives considered.

### Expected Features

The features research identifies a clear critical path: Pinecone storage with metadata is the blocker for all retrieval features. Action item tracking across meetings is the most complex feature (requires persistent item identity) and should be explicitly deferred. Cross-meeting trend detection requires historical data to be meaningful — it is a second-iteration feature, not MVP.

**Must have (table stakes):**
- End-of-meeting summary generation (auto on meeting end + `/summarize` manual trigger)
- Action item extraction per meeting — users rank this as the #1 use case
- Exact-date lookup ("what was decided last Tuesday?") — failing this destroys trust
- Pinecone storage with full flat metadata schema
- Natural language date resolution with disambiguation
- Persist meeting summaries to both JSON on disk and Pinecone

**Should have (competitive differentiators):**
- Hybrid RAG (semantic + keyword) — catches both exact-term and conceptual queries that single-mode search misses
- Multi-agent query routing with handoffs — enables complex multi-hop queries
- Recurring series awareness via channel_id grouping
- Disambiguation clarification when multiple meetings match

**Defer to v2+:**
- Cross-meeting trend detection — requires corpus depth; first useful after 20+ meetings
- Action item status tracking across meetings — requires persistent item identity system
- Action item push to external task managers (Asana, Jira, Linear) — separate product surface

See `.planning/research/FEATURES.md` for the full query type taxonomy and disambiguation pattern specification.

### Architecture Approach

The system has two independent paths sharing a Pinecone index and JSON metadata store. Ingestion: `Transcript → SummarizerAgent → JSON file + Pinecone upsert`. Query: `User query → OrchestratorAgent → [RetrieverAgent as tool → Pinecone hybrid search] → [AnswerAgent as tool] → spoken/Slack response`. The `OrchestratorAgent` uses agents-as-tools (not handoffs) for the Retriever and Answer so it retains control to merge outputs and handle clarification loops. The hybrid query pipeline runs at `alpha=0.7` (semantic-heavy) by default, adjusting per query type classification. Reranking via Pinecone's `bge-reranker-v2-m3` reduces top-20 candidates to top-5–8 before synthesis.

**Major components:**
1. `MetadataStore` (`storage/metadata_store.py`) — async JSON read/write using aiofiles; resolves date expressions to UNIX timestamp ranges
2. `PineconeClient` (`storage/pinecone_client.py`) — wraps Pinecone SDK for upsert and hybrid query; handles alpha weighting
3. `SummarizerAgent` (`agents/summarizer.py`) — transcript → structured Pydantic MeetingRecord; writes to both disk and Pinecone
4. `OrchestratorAgent` (`agents/orchestrator.py`) — query classification, date resolution, delegates to Retriever and Answer as tools
5. `RetrieverAgent` (`agents/retriever.py`) — encodes dense + sparse query vectors, executes hybrid Pinecone search with metadata filters
6. `AnswerAgent` (`agents/answer.py`) — synthesizes 2–5 sentence speech-friendly response from retrieved chunks

See `.planning/research/ARCHITECTURE.md` for full data flow diagrams, metadata schemas, and scalability considerations.

### Critical Pitfalls

1. **Wrong Pinecone index metric** — Creating the index with `cosine` instead of `dotproduct` silently succeeds at upsert time but fails at hybrid query time. Fix: always pass `metric="dotproduct"` explicitly; run a hybrid query smoke test immediately after index creation.

2. **ISO date strings in Pinecone metadata** — `$gte`/`$lte` date range filters require integer UNIX timestamps. ISO strings silently return wrong results. Fix: store all timestamps as epoch integers at ingestion time; never convert at query time.

3. **Micro-chunk vectors** — Storing each Recall.ai WebSocket chunk (2–5 words) as a separate vector produces semantically empty embeddings. Fix: store one vector per meeting embedding the full `summary_text`; raw chunks are only for summarization, then discard.

4. **Global `meeting_state` race condition** — The existing module-level `meeting_state` dict has no synchronization. Async agent runs racing with the WebSocket transcript handler will corrupt state. Fix: encapsulate in a class with `asyncio.Lock` before wiring any agents; this is a prerequisite, not optional.

5. **Context window explosion across handoffs** — Each handoff passes the full prior conversation history. A three-agent chain with a clarification loop can accumulate 15K+ tokens, causing silent truncation. Fix: apply `input_filter` on handoffs to pass only the resolved query parameters as a typed `ResolvedQuery` object; set `max_turns` explicitly per `Runner.run()` call; catch `MaxTurnsExceeded` and return a user-facing degradation message.

See `.planning/research/PITFALLS.md` for the full list of 16 pitfalls with phase-specific warnings.

---

## Implications for Roadmap

Based on the dependency graph in FEATURES.md and the build order in ARCHITECTURE.md, five phases are recommended. Pinecone storage is the critical blocker; everything else follows from it.

### Phase 0: Prerequisite Refactor + Test Infrastructure
**Rationale:** The existing `meeting_state` global dict has documented race conditions (CONCERNS.md). Async agents will corrupt state if this is not fixed first. Tests are also entirely absent — establishing pytest patterns in Phase 0 prevents silent failures in all subsequent phases.
**Delivers:** Thread-safe `meeting_state` encapsulation with `asyncio.Lock`; pytest + pytest-asyncio test harness; all existing deps pinned in requirements.txt; new deps added
**Avoids:** Pitfall 12 (global mutable dict conflict), Pitfall 1 (wrong index metric — caught by smoke test established here)
**Research flag:** None needed — established refactoring patterns; pytest docs are standard

### Phase 1: Storage Foundation
**Rationale:** Pinecone storage is the critical path. Nothing in the retrieval pipeline can be built or tested until meeting records can be written and queried. Building storage in isolation with synthetic data allows early validation of the metadata schema and index configuration.
**Delivers:** `MetadataStore` (async JSON read/write), `PineconeClient` (upsert + hybrid query + rerank), Pinecone index creation script, verified flat metadata schema with UNIX timestamps
**Uses:** pinecone==8.1.1, aiofiles==25.1.0, pydantic==2.12.5
**Avoids:** Pitfall 1 (wrong index metric), Pitfall 2 (stale BM25 corpus), Pitfall 7 (nested JSON in Pinecone), Pitfall 15 (mutable channel_name as key), Pitfall 14 (JSON/Pinecone schema divergence)
**Research flag:** None needed — Pinecone hybrid search is well-documented in official docs

### Phase 2: Summarizer Agent + Ingestion Pipeline
**Rationale:** The query pipeline is useless without data. Building and testing the summarizer second ensures real meeting records populate Pinecone for all subsequent phases. This phase also resolves the two-phase prompting requirement (speaker attribution before summarization) documented in PITFALLS.
**Delivers:** `SummarizerAgent` with structured Pydantic output, `/summarize` slash command wiring, auto-trigger on WebSocket disconnect, JSON + Pinecone dual write
**Addresses:** Summary generation (table stakes), action item extraction per meeting, speaker attribution
**Avoids:** Pitfall 5 (monolithic transcript prompt), Pitfall 9 (micro-chunk vectors), Pitfall 16 (partial transcript trigger)
**Research flag:** None needed — GPT-4o-mini structured output is well-established

### Phase 3: Retriever Agent + Hybrid RAG Pipeline
**Rationale:** Retrieval quality must be validated in isolation before wiring the full agent pipeline. Bugs in alpha weighting or metadata filtering are easier to diagnose with a standalone test harness than inside an orchestration chain.
**Delivers:** `RetrieverAgent` with dense + sparse encoding, alpha-weighted hybrid Pinecone query, metadata date/channel filtering, Pinecone reranking, measurable retrieval relevance
**Uses:** pinecone-sparse-english-v0 (hosted neural sparse model — no BM25 corpus fitting needed)
**Avoids:** Pitfall 2 (stale BM25 corpus — avoided by using hosted sparse model), Pitfall 6 (fixed alpha for all query types), Pitfall 4 (timezone-naive date ranges)
**Research flag:** May need brief research into `alpha` tuning for this specific meeting corpus — the 0.3/0.7 split is a starting point, not a verified optimum

### Phase 4: Orchestrator Agent + Answer Agent + End-to-End Query Path
**Rationale:** Orchestration and synthesis are relatively thin once retrieval works correctly. This phase wires the agents-as-tools chain and integrates into `jarvis.py`'s `handle_query` fork. Natural language date resolution with dateparser and the clarification loop are also completed here.
**Delivers:** Full end-to-end: "Hey Jarvis, what did we decide last week?" → spoken answer; query type classification; date resolution; disambiguation clarification for ambiguous queries
**Uses:** openai-agents==0.13.4, dateparser==1.4.0
**Implements:** OrchestratorAgent, AnswerAgent, date resolution with `dateparser`, agents-as-tools pattern
**Avoids:** Pitfall 3 (handoffs instead of agents-as-tools), Pitfall 4 (timezone-naive dates), Pitfall 8 (unhandled MaxTurnsExceeded), Pitfall 11 (over-asking clarifications), Pitfall 13 (prompt injection through transcript)
**Research flag:** The clarification loop state machine (speak question, await next WebSocket chunk) interacts with the existing async handler — may need a brief design spike to confirm interaction with `jarvis.py`'s WebSocket state machine

### Phase 5: Cross-Meeting Features + Series Awareness
**Rationale:** Cross-meeting trend detection and recurring series grouping require a meaningful corpus and working single-meeting retrieval. These are differentiators, not table stakes, so deferring them prevents scope creep in earlier phases.
**Delivers:** Recurring meeting series grouping by channel_id, "latest in series" shortcut, cross-meeting trend detection queries
**Addresses:** Hybrid RAG full capability, cross-meeting trend detection, recurring series awareness
**Avoids:** Out-of-scope features (action item status tracking across meetings, external task manager push — these remain deferred)
**Research flag:** Cross-meeting trend detection via topic clustering is not well-documented for this specific architecture — needs deeper research or a discovery phase before implementation

### Phase Ordering Rationale

- **Storage before agents:** Pinecone upsert and query paths must be verified before any agent invokes them. Building agents against an untested storage layer produces compound failures that are hard to attribute.
- **Ingestion before retrieval:** The retriever needs real data to evaluate quality. Synthetic data is useful for smoke tests but not for validating hybrid alpha tuning or reranking.
- **Retrieval before orchestration:** The orchestrator is thin — it classifies and delegates. If the retrieval it delegates to is broken, debugging orchestrator behavior is moot.
- **Single-meeting features before cross-meeting features:** Cross-meeting trends require history. Building trend detection before there is meaningful history produces nothing testable or demonstrable.
- **Phase 0 prerequisite:** The race condition in `meeting_state` is not a "nice to fix" — it is a prerequisite for correctness. Every subsequent phase depends on it.

### Research Flags

Phases likely needing deeper research during planning:
- **Phase 4 (Clarification Loop):** The interaction between the OrchestratorAgent's clarification dialogue and `jarvis.py`'s WebSocket state machine is not fully specified. The WebSocket handler is a persistent async loop; interrupting it to await a clarification response from Slack requires careful state management. A design spike before implementation is recommended.
- **Phase 5 (Cross-Meeting Trends):** Topic clustering across embeddings is not covered in the existing research. The retrieval strategy (aggregation query vs. multi-hop retrieval vs. topic vector clustering) needs a brief research phase before Phase 5 begins.

Phases with standard patterns (skip research-phase):
- **Phase 0 (Refactor):** Standard asyncio lock pattern; pytest setup is fully documented
- **Phase 1 (Storage):** Pinecone hybrid search docs are comprehensive and high-confidence
- **Phase 2 (Summarizer):** Structured output with GPT-4o-mini is well-established
- **Phase 3 (Retriever):** Covered in full detail by STACK.md and ARCHITECTURE.md

---

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | HIGH | All versions PyPI-verified; official SDK docs consulted for openai-agents and Pinecone |
| Features | MEDIUM-HIGH | Table stakes confirmed against competitive landscape and ACM/IEEE research; cross-meeting trends are under-specified |
| Architecture | HIGH | Based on official OpenAI Agents SDK and Pinecone docs; agents-as-tools pattern is normative per SDK docs |
| Pitfalls | HIGH | Most pitfalls sourced from official docs and peer-reviewed research; first-party codebase audit (CONCERNS.md) cited for meeting_state race condition |

**Overall confidence:** HIGH

### Gaps to Address

- **Hybrid alpha calibration:** The `alpha=0.7` default and `0.3/0.7` per-query-type split are research-backed starting points, not empirically validated for this corpus. Plan a brief calibration exercise after the first 10 real meetings are indexed.
- **Pinecone sparse model vs. local BM25 choice:** STACK.md recommends `pinecone-sparse-english-v0` (hosted neural sparse). `pinecone-text BM25Encoder` remains in the dependency list as a fallback. This choice should be finalized in Phase 1 — `pinecone-text` may be removable if the hosted model meets all requirements.
- **Clarification loop / WebSocket state machine interaction:** Not fully specified. `jarvis.py`'s async loop was not designed for interruption-style dialogue. This needs a design decision before Phase 4 begins.
- **Cross-meeting trend retrieval strategy:** Not covered in current research. Topic clustering, multi-hop retrieval, and embedding aggregation are distinct approaches with different trade-offs for this domain. Research required before Phase 5.
- **`series_name` inference:** Research recommends using `channel_id` as the primary series key (immutable) and `channel_name` as display-only. However, `series_name` as a human-readable label still needs to be populated. Whether this comes from the channel name at meeting time or from an LLM inference step is unresolved.

---

## Sources

### Primary (HIGH confidence)
- [OpenAI Agents SDK documentation](https://openai.github.io/openai-agents-python/) — multi-agent patterns, handoffs, agents-as-tools, MaxTurnsExceeded
- [Pinecone Docs — Hybrid Search](https://docs.pinecone.io/guides/search/hybrid-search) — dotproduct requirement, sparse_values format, alpha weighting
- [Pinecone Docs — Filter by Metadata](https://docs.pinecone.io/guides/search/filter-by-metadata) — $gte/$lte with UNIX integers
- [Pinecone Docs — Encode Sparse Vectors](https://docs.pinecone.io/guides/data/encode-sparse-vectors) — hosted sparse model setup
- [arxiv: Summaries, Highlights, and Action Items](https://arxiv.org/html/2307.15793v3) — peer-reviewed research on meeting recap feature priorities
- [arxiv: What's Wrong? Refining Meeting Summaries with LLM Feedback](https://arxiv.org/abs/2407.11919) — peer-reviewed; speaker attribution failure modes

### Secondary (MEDIUM confidence)
- [IEEE: Mitigating Retrieval Errors via Ambiguity Detection](https://ieeexplore.ieee.org/document/11225289/) — disambiguation pattern specification
- [Superlinked: Optimizing RAG with Hybrid Search and Reranking](https://superlinked.com/vectorhub/articles/optimizing-rag-with-hybrid-search-reranking) — hybrid search best practices; corroborated against Pinecone official docs
- [Luna.ai: Contextual RAG for Meeting Notes](https://withluna.ai/blog/contextual-rag-product-meeting-notes-slack) — practitioner experience
- [Pinecone Community — BM25 sparse encoding](https://community.pinecone.io/t/bm25-sparse-encoding-for-hybrid-search/6390) — consistent with official docs
- [AssemblyAI: Top Meeting Intelligence Platforms 2026](https://www.assemblyai.com/blog/meeting-intelligence-platforms) — competitive landscape

### Tertiary (LOW confidence)
- [DEV Community: RAG Retrieval Performance Enhancement](https://dev.to/jamesli/rag-retrieval-performance-enhancement-practices-detailed-explanation-of-hybrid-retrieval-and-self-query-techniques-59ja) — single source, practical hybrid search guide; needs validation

---
*Research completed: 2026-04-04*
*Ready for roadmap: yes*
