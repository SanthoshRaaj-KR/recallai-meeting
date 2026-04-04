# RecallAI Meeting Bot

## What This Is

A Slack meeting bot that transcribes real-time conversations, facilitates interviews, and — the new milestone — answers questions about previous meetings using a hybrid RAG pipeline. Every meeting generates a structured summary stored in Pinecone with full metadata. Users can ask natural-language questions like "what was decided last Wednesday?" or "has this topic come up before?" and the agentic system retrieves and synthesizes answers across meeting history.

## Core Value

Institutional memory for teams — every decision, action item, and discussion point from every meeting is instantly queryable.

## Requirements

### Validated

- ✓ Real-time meeting transcription via Recall.ai WebSocket integration — existing
- ✓ Slack bot framework with event handling and slash commands — existing
- ✓ Sentence completion agent using OpenAI GPT-4o-mini — existing
- ✓ Interview facilitator agent — existing
- ✓ FastAPI + Uvicorn HTTP/WebSocket server — existing
- ✓ asyncio.Lock-protected MeetingState class — no race conditions (INFRA-01, Validated in Phase 1: Prerequisite Refactor)
- ✓ pytest + pytest-asyncio test harness — 13 tests passing including concurrent stress tests (Validated in Phase 1: Prerequisite Refactor)
- ✓ `MeetingRecord` Pydantic model — single canonical schema driving JSON + Pinecone, 65 tests passing (INFRA-03, Validated in Phase 2: Storage Foundation)
- ✓ `MetadataStore` — async JSON read/write at `meetings/{channel_id}/{meeting_id}.json` (INFRA-03, Validated in Phase 2: Storage Foundation)
- ✓ `PineconeClient` — hybrid upsert (dense + sparse via `pinecone-sparse-english-v0`) + filtered query by `channel_id`/`start_ts` (INFRA-02, INFRA-04, Validated in Phase 2: Storage Foundation)

### Active

*(All milestone requirements validated — see Validated section)*

### Validated (continued)

- ✓ Meeting summary generation at end of each meeting (auto-detect + `/summarize` manual trigger) — (Validated in Phase 3: Summarization Pipeline)
- ✓ Pinecone storage for meeting summaries + embeddings — (Validated in Phase 3: Summarization Pipeline)
- ✓ JSON metadata file per meeting (Validated in Phase 2: Storage Foundation)
- ✓ Fully agentic architecture using OpenAI Agents SDK with multi-agent handoffs — `SummarizerAgent`, `RetrieverAgent`, `OrchestratorAgent`, `AnswerAgent`, `DateResolutionAgent` (Validated in Phase 5: Orchestration Answer Path)
- ✓ Hybrid RAG pipeline: semantic (Pinecone) + keyword (BM25) + metadata filters — `RetrieverAgent` with reranking (Validated in Phase 4: Retriever Agent)
- ✓ Natural language date resolution — `DateResolutionAgent` converts NL expressions to UTC epoch ranges, OrchestratorAgent triggers disambiguation when >1 meeting matches (Validated in Phase 5: Orchestration Answer Path)
- ✓ Query types: decision, summary, cross_meeting, action_items — all handled by `AnswerAgent` (Validated in Phase 5: Orchestration Answer Path)
- ✓ `/ask` Slack slash command routes to OrchestratorAgent with disambiguation flow (Validated in Phase 5: Orchestration Answer Path)

### Out of Scope

- Calendar integrations (Google Cal, Outlook) — not needed, Slack channel = meeting identity
- Video recording or screen capture — audio/transcript only
- Multi-workspace Slack support — single workspace for now

## Context

- **Existing bot**: `jarvis.py` — FastAPI + Recall.ai WebSocket + `/ask` slash command + OrchestratorAgent integration
- **Current architecture**: Multi-agent pipeline: `SummarizerAgent` → Pinecone → `RetrieverAgent` → `OrchestratorAgent` → `DateResolutionAgent` + `AnswerAgent`
- **Meeting data source**: Recall.ai provides real-time transcript chunks via WebSocket
- **Slack integration**: Bot joins huddles, listens to transcripts, responds in-channel; `/ask` routes memory queries through full orchestration pipeline
- **Test harness**: pytest + pytest-asyncio, 181 tests passing (Phase 5 complete — full milestone delivered)

## Constraints

- **Tech stack**: Python — no language switch
- **Vector DB**: Pinecone — chosen for managed scaling and hybrid search support
- **Agentic framework**: OpenAI Agents SDK (not LangChain/LlamaIndex) — explicitly requested
- **Meeting identity**: Slack channel ID as series identifier — no external calendar dependency
- **Metadata persistence**: JSON file on disk per meeting + Pinecone vector record

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Pinecone for vector storage | Managed scaling, native hybrid search support | — Pending |
| OpenAI Agents SDK multi-agent handoffs | Explicit requirement; better separation of concerns than single agent | — Pending |
| Slack channel = meeting series | Simple, no external deps; already have channel context | — Pending |
| Hybrid RAG: semantic + BM25 + metadata | Full coverage: meaning + exact terms + date/channel filters | — Pending |
| Date resolution: exact range + clarify if ambiguous | Balance UX (no false positives) vs friction (don't over-ask) | — Pending |
| JSON metadata file per meeting | Auditable, portable, no DB dependency for metadata access | — Pending |

## Evolution

This document evolves at phase transitions and milestone boundaries.

**After each phase transition** (via `/gsd:transition`):
1. Requirements invalidated? → Move to Out of Scope with reason
2. Requirements validated? → Move to Validated with phase reference
3. New requirements emerged? → Add to Active
4. Decisions to log? → Add to Key Decisions
5. "What This Is" still accurate? → Update if drifted

**After each milestone** (via `/gsd:complete-milestone`):
1. Full review of all sections
2. Core Value check — still the right priority?
3. Audit Out of Scope — reasons still valid?
4. Update Context with current state

---
*Last updated: 2026-04-04 after Phase 5: Orchestration Answer Path (milestone complete)*
