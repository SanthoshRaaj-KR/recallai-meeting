# Jarvis — Meeting Intelligence Platform

## What This Is

An AI-powered meeting assistant that joins meetings via a Recall.ai bot, captures transcripts in real-time, and at the end of each meeting generates proposed Confluence page changes based on what was discussed. Proposed changes are presented as review cards in the sync-sage-bot UI where users can accept or reject each one individually — only accepted changes are applied to Confluence.

## Core Value

Users never have to manually update Confluence after a meeting — the system proposes the right changes to the right pages, and the user just approves or rejects.

## Requirements

### Validated

- ✓ Recall.ai bot joins meetings via URL and captures live transcript — existing
- ✓ Post-meeting transcript stored and accessible for analysis — existing
- ✓ Confluence page ingestion pipeline (chunking → Pinecone + Neo4j graph) — existing
- ✓ Graph RAG combining Pinecone vector search + Neo4j relationship traversal — existing
- ✓ Basic proposal cards structure (change_type, page, before, after) — existing
- ✓ Supabase persistence for meeting history and proposals — existing
- ✓ sync-sage-bot UI (Vite/React/Radix UI/TanStack Query) with auth — existing
- ✓ FastAPI backend with WebSocket and TTS during meetings — existing
- ✓ Meeting summary page (MeetingSummary.tsx) as post-meeting destination — existing

### Active

- [ ] "Generate Confluence Changes" button on MeetingSummary page that triggers the proposal pipeline
- [ ] Structured meeting fact extraction from transcript (decisions, action items, new/changed requirements, owners, deadlines, doc-worthy updates)
- [ ] Automated candidate Confluence page retrieval via RAG — no user input required
- [ ] Worker agents draft proposed edits and new pages per affected section
- [ ] Verifier/critic agent validates each proposal against transcript evidence and current Confluence content
- [ ] Proposal cards returned with: change_type, target page/section, before content, after content, rationale, transcript evidence snippets, confidence score, risk level
- [ ] Review UI in sync-sage-bot showing proposal cards with accept/reject per card
- [ ] Safe Confluence apply: fetch latest version, verify section anchor still matches, handle version conflicts
- [ ] Re-index changed Confluence pages into Pinecone + Neo4j after accepted changes are applied
- [ ] Pipeline progress indicator in UI (proposal generation can take up to 20 minutes)

### Out of Scope

- Auto-editing Confluence without explicit user approval — safety requirement
- User specifying Confluence page names or candidates — system must infer from RAG
- Live full-page scan after button click — use pre-indexed graph/RAG
- local_office_logic improvements — focus on Confluence connector only
- LLM models above GPT-5 — cost and availability constraint
- Authentication system changes — auth already exists in sync-sage-bot
- review-ui (Next.js) improvements — sync-sage-bot is the primary UI

## Context

- **Backend:** Python FastAPI (`confluence_logic/jarvis_agentic.py`), OpenAI Agents SDK for multi-agent orchestration
- **Database:** Supabase (Postgres) for meeting history + proposals, Pinecone for vector search, Neo4j AuraDB for graph traversal
- **UI:** `sync-sage-bot/` — Vite/React SPA with Radix UI, shadcn/ui, TanStack Query, Zod, Supabase auth; has `MeetingSummary.tsx` already as the post-meeting page
- **Existing proposal infrastructure:** `confluence_logic/agents/proposed_changes_agent.py` and `confluence_logic/review/` (API + Supabase store) provide a starting skeleton but need significant expansion for the multi-agent verifier approach
- **Graph infrastructure:** `confluence_logic/confluence_page_graph.py` and `graph_rag.py` are the foundation; need to ensure pre-indexed and scoped per user session
- **Model strategy:** GPT-5 mini / GPT-5.4 nano for worker/extraction agents; GPT-5 for orchestration and final verification only

## Constraints

- **Model ceiling:** GPT-5 maximum — no GPT-5 turbo or above
- **Model preference:** GPT-5 mini / GPT-5.4 nano wherever sufficient; GPT-5 only for orchestration and hard verification
- **Latency:** Pipeline may take up to 20 minutes — must show progress; do not time out
- **Safety:** No Confluence edits without explicit per-card user approval
- **RAG-first:** All Confluence page lookup must use pre-indexed graph/RAG, not live API scans
- **Focus scope:** Confluence connector only; local_office_logic is out of scope

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Multi-agent with verifier/critic | Accuracy over speed; verifier catches hallucinated changes before user sees them | — Pending |
| GPT-5 mini/nano for workers, GPT-5 for orchestration | Cost control while preserving quality at critical decision points | — Pending |
| Pre-indexed graph/RAG, not live scan | 20-minute budget is for agent reasoning, not live API traversal | — Pending |
| sync-sage-bot as primary UI (not review-ui) | sync-sage-bot already has auth, full component library, MeetingSummary page | — Pending |
| Per-card accept/reject (not bulk approve) | Prevents accidentally applying bad changes alongside good ones | — Pending |

## Evolution

This document evolves at phase transitions and milestone boundaries.

**After each phase transition** (via `/gsd-transition`):
1. Requirements invalidated? → Move to Out of Scope with reason
2. Requirements validated? → Move to Validated with phase reference
3. New requirements emerged? → Add to Active
4. Decisions to log? → Add to Key Decisions
5. "What This Is" still accurate? → Update if drifted

**After each milestone** (via `/gsd-complete-milestone`):
1. Full review of all sections
2. Core Value check — still the right priority?
3. Audit Out of Scope — reasons still valid?
4. Update Context with current state

---
*Last updated: 2026-05-11 after initialization*
