# Roadmap: RecallAI Meeting Bot — Memory Milestone

## Overview

This milestone adds institutional memory to the existing Jarvis bot by building a hybrid RAG pipeline on top of Pinecone and the OpenAI Agents SDK. The build follows a strict dependency order: fix the race condition that would corrupt any agent work (Phase 1), establish the storage layer that everything else depends on (Phase 2), build ingestion to populate that storage (Phase 3), build retrieval to query it (Phase 4), then wire the full orchestration and answer path (Phase 5). Each phase delivers a independently testable, working capability before the next begins.

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

Decimal phases appear between their surrounding integers in numeric order.

- [x] **Phase 1: Prerequisite Refactor** - Thread-safe meeting state and test infrastructure that unblocks all agent work (completed 2026-04-04)
- [x] **Phase 2: Storage Foundation** - Pinecone index + MetadataStore + PineconeClient with verified schema (completed 2026-04-04)
- [x] **Phase 3: Ingestion Pipeline** - SummarizerAgent that writes real meeting records to disk and Pinecone (completed 2026-04-04)
- [ ] **Phase 4: Retrieval + Hybrid RAG** - RetrieverAgent with dense + sparse + metadata filter search
- [ ] **Phase 5: Orchestration + Answer Path** - Full end-to-end query pipeline: natural language question to spoken answer

## Phase Details

### Phase 1: Prerequisite Refactor
**Goal**: The existing bot is safe for async agent work — no race conditions can corrupt meeting state
**Depends on**: Nothing (first phase)
**Requirements**: INFRA-01
**Success Criteria** (what must be TRUE):
  1. `meeting_state` is encapsulated in a class using `asyncio.Lock`; direct dict mutations are eliminated from `jarvis.py`
  2. A concurrent test simulating simultaneous WebSocket transcript writes and agent reads produces no state corruption
  3. pytest + pytest-asyncio harness runs with at least one passing async test; CI pattern is established
  4. All existing dependencies in requirements.txt are pinned to exact versions; new deps (pinecone, aiofiles, dateparser, pydantic, pytest-asyncio) are added
**Plans**: 2 plans
Plans:
- [x] 01-01-PLAN.md — MeetingState class with asyncio.Lock + pinned dependencies + pytest config
- [x] 01-02-PLAN.md — Refactor jarvis.py to use MeetingState + concurrent integration tests

### Phase 2: Storage Foundation
**Goal**: A meeting record can be written as JSON to disk and as a vector to Pinecone, then queried by channel and date
**Depends on**: Phase 1
**Requirements**: INFRA-02, INFRA-03, INFRA-04
**Success Criteria** (what must be TRUE):
  1. Pinecone index exists with `metric="dotproduct"` and a hybrid smoke test (dense + sparse query) returns results without error
  2. A synthetic meeting record written via `MetadataStore` appears as a JSON file on disk with all required fields (meeting_id, timestamp as Unix epoch int, duration_seconds, channel_id, channel_name, participants, summary_text, topics_covered, action_items, decisions, series_name, recurrence_pattern)
  3. The same record upserted via `PineconeClient` is queryable by `channel_id` filter and `start_ts` date range filter using `$gte`/`$lte` operators
  4. A Pydantic model is the single canonical schema that drives both the JSON write and the Pinecone upsert — no schema divergence possible
**Plans**: 2 plans
Plans:
- [x] 02-01-PLAN.md — MeetingRecord Pydantic model + MetadataStore (JSON read/write to disk)
- [x] 02-02-PLAN.md — PineconeClient (index creation + hybrid upsert + filtered query)

### Phase 3: Ingestion Pipeline
**Goal**: At the end of any meeting, a structured summary is automatically stored to both disk and Pinecone with full metadata and speaker attribution
**Depends on**: Phase 2
**Requirements**: INGEST-01, INGEST-02, INGEST-03, INGEST-04, AGENT-02
**Success Criteria** (what must be TRUE):
  1. When a meeting ends (Recall.ai disconnect event), a JSON metadata file appears in `meetings/` and a vector record appears in Pinecone within 30 seconds — without any manual action
  2. User can type `/summarize` in Slack during or after a meeting and receive a structured recap (decisions, topics, action items, participants) posted to the channel
  3. Every action item in the summary identifies the specific participant who committed to it (not "someone" or left blank)
  4. Summary output is structured (not free text): decisions list, topics list, action items list with owner/task fields, participant list
  5. Summaries generated from incomplete transcripts are tagged `status: partial`; only `status: complete` summaries are indexed for search by default
**Plans**: 2 plans
Plans:
- [x] 03-01-PLAN.md — SummarizerAgent with structured output and action item attribution
- [x] 03-02-PLAN.md — Ingestion wiring: auto-trigger on disconnect + /summarize Slack command

### Phase 4: Retrieval + Hybrid RAG
**Goal**: Given a query with optional date and channel filters, the system returns ranked, relevant meeting excerpts with measurable retrieval quality
**Depends on**: Phase 3
**Requirements**: RETR-01, AGENT-03
**Success Criteria** (what must be TRUE):
  1. A query against real meeting data (from Phase 3) returns results that blend dense semantic vectors and sparse keyword vectors in a single Pinecone query with alpha weighting applied
  2. A channel-scoped query (e.g., only meetings from `#eng-standup`) correctly excludes meetings from other channels via `channel_id` metadata filter
  3. A date-range query (Unix epoch `$gte`/`$lte` filter) returns only meetings within the specified window — verified by checking result timestamps
  4. Pinecone reranker reduces the top-20 candidates to a ranked top-5 to top-8 list that can be inspected for relevance
**Plans**: 2 plans
Plans:
- [x] 04-01-PLAN.md — Alpha-weighted hybrid query + Pinecone reranker (extends PineconeClient)
- [ ] 04-02-PLAN.md — RetrieverAgent wrapping PineconeClient.retrieve() with RetrievalResult output

### Phase 5: Orchestration + Answer Path
**Goal**: Users can ask natural language questions about past meetings via Slack or voice and receive accurate, attributed answers
**Depends on**: Phase 4
**Requirements**: RETR-02, RETR-03, QUERY-01, QUERY-02, QUERY-03, QUERY-04, AGENT-01, AGENT-04, AGENT-05
**Success Criteria** (what must be TRUE):
  1. User asks "what did we decide about X?" and receives a precise answer with the meeting date and channel as source attribution
  2. User asks "what happened in last Monday's standup?" and receives a structured recap identifying the meeting by natural language date — date parsed correctly to UTC range using dateparser
  3. When a natural language date matches more than one meeting, the bot presents a disambiguation list (meeting title, channel, date) and waits for user selection before answering
  4. User asks "has this topic come up before?" and receives a response synthesizing relevant instances across meeting history
  5. User asks "what did I commit to last week?" and receives a list of action items attributed to them with meeting context
**Plans**: TBD

## Progress

**Execution Order:**
Phases execute in numeric order: 1 → 2 → 3 → 4 → 5

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Prerequisite Refactor | 2/2 | Complete    | 2026-04-04 |
| 2. Storage Foundation | 2/2 | Complete    | 2026-04-04 |
| 3. Ingestion Pipeline | 2/2 | Complete   | 2026-04-04 |
| 4. Retrieval + Hybrid RAG | 1/2 | In Progress|  |
| 5. Orchestration + Answer Path | 0/TBD | Not started | - |
