# Requirements — RecallAI Meeting Bot (Memory Milestone)

**Version:** v1
**Generated:** 2026-04-04
**Status:** Approved

---

## v1 Requirements

### Infrastructure (INFRA)

- [x] **INFRA-01**: Bot uses asyncio.Lock to protect shared meeting state, preventing race conditions when agent runs overlap with WebSocket transcript handlers
- [x] **INFRA-02**: Pinecone index is created with `metric="dotproduct"` and supports both dense (`values`) and sparse (`sparse_values`) vectors in a single index
- [x] **INFRA-03**: Each meeting produces a JSON metadata file containing: meeting_id, timestamp (Unix epoch int), duration_seconds, channel_id, channel_name, participants, summary_text, topics_covered, action_items, decisions, series_name, recurrence_pattern
- [x] **INFRA-04**: Sparse vectorization uses Pinecone's `pinecone-sparse-english-v0` inference model (no local corpus fitting required)

### Ingestion (INGEST)

- [x] **INGEST-01**: Bot automatically generates and stores a meeting summary when a meeting ends (triggered by Recall.ai end event or timeout heuristic)
- [x] **INGEST-02**: User can trigger summary generation at any time via `/summarize` Slack slash command
- [x] **INGEST-03**: Summary extraction produces structured output: decisions made, topics discussed, action items, participant list
- [x] **INGEST-04**: Each action item in the summary is attributed to the specific participant who committed to it

### Retrieval (RETR)

- [ ] **RETR-01**: Query pipeline executes hybrid search combining dense semantic vectors, sparse BM25 vectors, and Pinecone metadata filters in a single query
- [ ] **RETR-02**: System parses natural language date expressions ("last Wednesday", "two weeks ago", "last standup") into exact UTC date ranges using dateparser
- [ ] **RETR-03**: When a date query matches more than one meeting, bot presents a disambiguation list showing meeting titles, channels, and dates — user selects which meeting to query

### Query Types (QUERY)

- [ ] **QUERY-01**: User can ask about a specific event or decision from a meeting ("What did we decide about the API rate limits?") and receive a precise answer with source attribution
- [ ] **QUERY-02**: User can request a general summary of a specific meeting ("What happened in last Monday's standup?") and receive a structured recap
- [ ] **QUERY-03**: User can ask cross-meeting questions ("Has the deployment pipeline issue come up before?") and receive a response synthesizing relevant instances across meeting history
- [ ] **QUERY-04**: User can ask about their own action items ("What did I commit to last week?") and receive a list of attributed action items with meeting context

### Agents (AGENT)

- [ ] **AGENT-01**: An Orchestrator Agent classifies incoming queries (live meeting vs. memory query vs. action item query) and delegates to specialist agents using `Agent.as_tool()` pattern
- [x] **AGENT-02**: A Summarizer Agent processes meeting transcripts and produces structured meeting summaries, storing results to both Pinecone and the JSON metadata file
- [ ] **AGENT-03**: A Retriever Agent executes the hybrid RAG pipeline (dense + sparse + metadata filters) and returns ranked, relevant meeting context
- [ ] **AGENT-04**: An Answer Agent synthesizes retrieved context into a natural language response formatted for Slack
- [ ] **AGENT-05**: A Date Resolution Agent parses natural language date expressions into UTC date ranges and triggers a clarification loop when the expression is ambiguous or returns no results

---

## v2 Requirements (Deferred)

- Recurring meeting series detection beyond channel grouping (auto-detect from title patterns or attendee overlap)
- Multi-meeting summary-of-summaries for long-running series
- Pinecone cold start mitigation / keep-alive strategy (depends on measuring production latency)
- BM25 encoder refit cadence automation (manual for v1)
- Cross-series trend visualization (Slack-formatted charts)

---

## Out of Scope

- Calendar integrations (Google Cal, Outlook) — Slack channel identity is sufficient
- Video/screen recording — transcript only
- Multi-workspace Slack support — single workspace
- Semantic chunking at sub-meeting level — one vector per meeting summary (whole-meeting embeddings)
- Fine-tuned summarization model — GPT-4o-mini structured extraction is sufficient

---

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| INFRA-01 | Phase 1 | Complete |
| INFRA-02 | Phase 2 | Complete |
| INFRA-03 | Phase 2 | Complete |
| INFRA-04 | Phase 2 | Complete |
| INGEST-01 | Phase 3 | Complete |
| INGEST-02 | Phase 3 | Complete |
| INGEST-03 | Phase 3 | Complete |
| INGEST-04 | Phase 3 | Complete |
| RETR-01 | Phase 4 | Pending |
| RETR-02 | Phase 5 | Pending |
| RETR-03 | Phase 5 | Pending |
| QUERY-01 | Phase 5 | Pending |
| QUERY-02 | Phase 5 | Pending |
| QUERY-03 | Phase 5 | Pending |
| QUERY-04 | Phase 5 | Pending |
| AGENT-01 | Phase 5 | Pending |
| AGENT-02 | Phase 3 | Complete |
| AGENT-03 | Phase 4 | Pending |
| AGENT-04 | Phase 5 | Pending |
| AGENT-05 | Phase 5 | Pending |
