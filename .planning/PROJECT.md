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

### Active

- [ ] Meeting summary generation at end of each meeting (auto-detect + `/summarize` manual trigger)
- [ ] Pinecone storage for meeting summaries + embeddings
- [ ] JSON metadata file per meeting (meeting_id, timestamp, duration, channel_id, channel_name, participants, summary_text, topics_covered, series_name, recurrence_pattern)
- [ ] Fully agentic architecture using OpenAI Agents SDK with multi-agent handoffs
- [ ] Hybrid RAG pipeline: semantic (Pinecone) + keyword (BM25) + metadata filters
- [ ] Natural language date resolution ("last Wednesday" → exact date range → clarify if ambiguous)
- [ ] Recurring meeting series: Slack channel = meeting series identity
- [ ] Query types: exact event lookup, general summary, cross-meeting trends, action item tracking

### Out of Scope

- Calendar integrations (Google Cal, Outlook) — not needed, Slack channel = meeting identity
- Video recording or screen capture — audio/transcript only
- Multi-workspace Slack support — single workspace for now

## Context

- **Existing bot**: `jarvis.py` — monolithic Python file with FastAPI, Recall.ai WebSocket, OpenAI tool calling, gTTS audio playback
- **Current architecture**: Single-agent with tool use; moving to multi-agent handoffs with OpenAI Agents SDK
- **Meeting data source**: Recall.ai provides real-time transcript chunks via WebSocket
- **Slack integration**: Bot joins huddles, listens to transcripts, responds in-channel
- **No tests currently**: Codebase has no test suite; new code should establish testing patterns

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
*Last updated: 2026-04-04 after initialization*
