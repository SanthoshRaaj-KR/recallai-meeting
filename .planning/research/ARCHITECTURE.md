# Architecture Patterns: Meeting Memory with Hybrid RAG

**Domain:** Multi-agent meeting memory system on top of existing Python/FastAPI Slack bot
**Researched:** 2026-04-04
**Confidence:** HIGH (OpenAI Agents SDK official docs, Pinecone official docs)

---

## Recommended Architecture

The system has two modes of operation that share the same storage layer:

**Ingestion path** (triggered at meeting end or via `/summarize`):
Transcript → Summarizer Agent → JSON metadata file + Pinecone upsert

**Query path** (triggered by wake word + memory question or Slack command):
User query → Orchestrator Agent → [Retrieval Agent → Pinecone hybrid search] → Answer Agent → spoken/Slack response

These two paths are independent and share only Pinecone and the JSON metadata store.

---

## Component Boundaries

### Component Map

| Component | File | Responsibility | Talks To |
|-----------|------|---------------|----------|
| `jarvis.py` (existing) | `jarvis.py` | Wake-word detection, WebSocket, speech output, agentic query dispatch | All components |
| `SummarizerAgent` | `agents/summarizer.py` | Consume raw transcript, produce structured meeting summary, write JSON metadata file, upsert to Pinecone | Pinecone, `metadata_store.py` |
| `OrchestratorAgent` | `agents/orchestrator.py` | Classify query type, resolve natural language dates, delegate to Retrieval + Answer agents | `RetrieverAgent` (as tool), `AnswerAgent` (as tool) |
| `RetrieverAgent` | `agents/retriever.py` | Execute hybrid search (semantic + BM25 + metadata filter), return ranked chunks | Pinecone |
| `AnswerAgent` | `agents/answer.py` | Synthesize final natural-language answer from retrieved chunks + conversation context | None (synthesis only) |
| `MetadataStore` | `storage/metadata_store.py` | Read/write JSON metadata files per meeting, resolve date expressions to UNIX timestamp ranges | Disk |
| `PineconeClient` | `storage/pinecone_client.py` | Wrapper around Pinecone SDK: upsert vectors, hybrid query, rerank | Pinecone cloud |
| `BM25Encoder` | `storage/bm25_encoder.py` | Fit corpus of meeting texts, encode sparse vectors at query time | Used by `PineconeClient` |

### What Stays in `jarvis.py`

Only the entry point wiring changes. `jarvis.py` gains:
- A call to `SummarizerAgent` when meeting ends (WebSocket disconnect or `/summarize` slash command)
- A route into `OrchestratorAgent` when wake-word query is memory-related (new query classifier step before `handle_query`)

Nothing else moves. The existing tool loop, speak(), and WebSocket handler remain intact.

---

## Agent Boundaries Detail

### Summarizer Agent

**When invoked:** (1) on WebSocket disconnect / meeting end, (2) on `/summarize` Slack slash command.

**Input:** Raw `transcript_log` list of `{participant, text, timestamp}` dicts plus `channel_id`, `channel_name`, `meeting_url`.

**Output:** Structured meeting record (see Metadata Schema section below). Writes JSON file to disk AND upserts embedding to Pinecone.

**Why a dedicated agent:** Summarization requires multi-step prompting (extract topics, identify decisions, identify action items, write summary paragraph). Encapsulating this in one agent keeps the prompt focused and allows independent testing.

**Pattern:** Single agent with structured output (OpenAI Agents SDK `output_type` with Pydantic model). No handoffs or sub-agents — it is a pure transformation.

### Orchestrator Agent

**When invoked:** When a user query is classified as memory-related (queries about past meetings, previous decisions, recurring topics).

**Input:** Raw user query string + current Slack `channel_id`.

**Output:** Spoken or Slack-posted answer.

**Responsibilities:**
1. Query classification: is this a memory question or a live-meeting question? If live-meeting, route to existing `handle_query`. If memory, proceed.
2. Date resolution: parse natural language date expressions ("last Wednesday", "two weeks ago", "Q1") → UNIX timestamp range. If ambiguous (e.g., "last meeting" could mean two different dates), ask clarifying question before retrieval.
3. Delegate to `RetrieverAgent` and `AnswerAgent` using **agents-as-tools pattern** (not handoffs).

**Why agents-as-tools, not handoffs:** The Orchestrator must combine outputs from both Retrieval and Answer, then deliver a synthesized response. Handoffs transfer conversational ownership; agents-as-tools keep the Orchestrator in control so it can merge results and handle clarification loops. Official SDK docs confirm: "Use agents as tools when a specialist should help with a bounded subtask but should not take over the user-facing conversation."

**Date resolution strategy:**
- "last Wednesday" → compute exact date → confirm with user only if query date is ambiguous (e.g., "meeting" matches 2 series on that day)
- "what did we decide about X" with no date → no date filter, broader semantic search
- "recent meetings" → last 30 days filter
- If ambiguous → ask one clarifying question via speak() or Slack message, do not proceed to retrieval

### Retriever Agent

**When invoked:** Called as a tool by Orchestrator Agent.

**Input:** Structured retrieval request: `{query_text, channel_id (optional), start_ts (optional), end_ts (optional), top_k}`.

**Output:** List of ranked `{meeting_id, summary_excerpt, date, channel_name, relevance_score}` chunks.

**Responsibilities:** Execute hybrid search (see Hybrid RAG Pipeline section). This agent is essentially a smart wrapper around the Pinecone hybrid query — its value is that the LLM can decide how to weight the query and whether to add metadata filters.

**Pattern:** Primarily tool-call-based (calls `pinecone_hybrid_search` as a function tool). Could be a plain Python function callable instead of a full Agent, but modeling it as an Agent allows future expansion (e.g., query expansion, multi-hop retrieval).

### Answer Agent

**When invoked:** Called as a tool by Orchestrator Agent, after retrieval results are available.

**Input:** User's original query + list of ranked meeting chunks from RetrieverAgent.

**Output:** Final answer string, 2-5 sentences, optimized for speech (no markdown, no bullet points for the spoken path; structured for Slack message path).

**Responsibilities:** Synthesize a coherent answer from retrieved chunks. Detect if retrieved context is insufficient and flag "no relevant meeting history found" rather than hallucinating.

**Pattern:** Single OpenAI call with retrieved context injected into prompt. No tool use. Explicit instruction to refuse to answer if context is absent.

---

## Hybrid RAG Pipeline

### Full Pipeline Flow

```
User query
    |
    v
[1. Query Analysis] — OrchestratorAgent
    - Classify: memory vs live
    - Resolve dates → UNIX timestamps
    - Extract channel filter (default: current channel)
    |
    v
[2. Query Encoding] — RetrieverAgent
    - Dense: text-embedding-3-small (1536 dims, or llama-text-embed-v2)
    - Sparse: pinecone-sparse-english-v0 OR BM25Encoder.encode_queries()
    |
    v
[3. Parallel Hybrid Query] — Pinecone single dotproduct index
    - Apply alpha weighting: alpha=0.7 (semantic-heavy for meeting queries)
    - Apply metadata filters: channel_id, date range (UNIX timestamp $gte/$lte)
    - top_k=20 candidates
    |
    v
[4. Reranking] — Pinecone Inference reranker (bge-reranker-v2-m3)
    - Input: top 20 candidates + original query text
    - Output: top 5-8 reranked results
    |
    v
[5. Answer Synthesis] — AnswerAgent
    - Inject top results as context
    - Generate concise, speech-friendly answer
    |
    v
Spoken response (speak()) or Slack message
```

### Alpha Weighting Rationale

Meeting memory queries fall into two types:

| Query type | Example | Recommended alpha |
|------------|---------|------------------|
| Conceptual / topic | "what did we discuss about the roadmap?" | 0.8 (semantic-heavy) |
| Exact phrase / name | "who mentioned the Jenkins pipeline?" | 0.3 (BM25-heavy) |
| Date-anchored | "what happened last Tuesday?" | metadata filter handles date; 0.6 default |

The Orchestrator can adjust alpha based on query type classification. Default: 0.7.

### BM25 Encoder Lifecycle

The `BM25Encoder` must be fit on the corpus before use. Two approaches:

**Option A (simpler):** Use Pinecone's hosted `pinecone-sparse-english-v0` model via `pc.inference.embed()`. No local fitting needed. This is the recommended approach per Pinecone docs as of 2025.

**Option B (local BM25):** Use `pinecone_text.sparse.BM25Encoder`, fit on all meeting summaries at startup, re-fit when new meetings are added. More control over tokenization but requires corpus management.

Recommendation: Start with Option A (hosted sparse model). Switch to Option B only if token-level control is needed for domain-specific terms.

---

## Meeting Metadata Schema

### JSON File (per meeting, on disk)

```json
{
  "meeting_id": "uuid4-string",
  "channel_id": "C01234567",
  "channel_name": "weekly-eng-sync",
  "series_name": "Weekly Engineering Sync",
  "meeting_url": "https://meet.google.com/abc-xyz",
  "start_ts": 1712000000,
  "end_ts": 1712003600,
  "duration_seconds": 3600,
  "participants": ["Alice", "Bob", "Charlie"],
  "summary_text": "Free-form paragraph summary of the meeting...",
  "topics_covered": ["roadmap Q2", "Jenkins pipeline", "hiring"],
  "decisions": ["Ship v2.1 by April 15", "Defer Jenkins migration to Q3"],
  "action_items": [
    {"owner": "Alice", "task": "Write RFC for v2.1", "due": "2024-04-10"}
  ],
  "recurrence_pattern": "weekly",
  "raw_transcript_chars": 14200,
  "summarized_at": 1712003700
}
```

### Pinecone Vector Record Schema

```
{
  "id": "<meeting_id>",
  "values": [<1536-dim dense embedding of summary_text>],
  "sparse_values": {"indices": [...], "values": [...]},
  "metadata": {
    "channel_id": "C01234567",
    "channel_name": "weekly-eng-sync",
    "series_name": "Weekly Engineering Sync",
    "start_ts": 1712000000,           // UNIX timestamp integer — required for $gte/$lte
    "end_ts": 1712003600,
    "participants": ["Alice", "Bob"],  // stored as list for $in filter
    "topics_covered": ["roadmap Q2", "Jenkins pipeline"],
    "decisions": ["Ship v2.1 by April 15"],
    "summary_text": "Free-form paragraph..."  // stored for reranker document input
  }
}
```

### Pinecone Index Configuration

```
index_type: dense (with sparse_values in record)
metric: dotproduct  -- REQUIRED for hybrid search (cosine does not support sparse)
dimension: 1536     -- for text-embedding-3-small; or 1024 for llama-text-embed-v2
cloud: aws
region: us-east-1
```

### Date Filter Pattern

```python
# At query time — natural language date resolved by OrchestratorAgent
filter = {
    "channel_id": {"$eq": "C01234567"},
    "start_ts": {"$gte": 1711900000, "$lte": 1712000000}
}
```

Never store dates as ISO strings in Pinecone metadata. The `$gte`/`$lte` operators require integers. Store `start_ts` and `end_ts` as UNIX timestamps everywhere.

---

## Data Flow

### Ingestion Flow

```
Recall.ai WebSocket
    → transcript_log (in-memory)
    → Meeting ends (WebSocket disconnect OR /summarize command)
    → SummarizerAgent.run(transcript_log, channel_id, channel_name)
        → GPT-4o: extract summary, topics, decisions, action items
        → Returns structured MeetingRecord (Pydantic)
    → MetadataStore.write(meeting_id, record)   → disk: meetings/<meeting_id>.json
    → PineconeClient.upsert(meeting_id, record) → Pinecone index
```

### Query Flow

```
Wake word + memory query (or Slack /memory command)
    → OrchestratorAgent.run(query, channel_id)
        → Classify query type (memory vs live)
        → Resolve date expressions → (start_ts, end_ts) or None
        → If ambiguous: speak clarification question, await next input
        → Invoke RetrieverAgent.as_tool(query, channel_id, start_ts, end_ts)
            → Encode dense + sparse query vectors
            → Pinecone hybrid query (alpha-weighted, metadata filtered)
            → Pinecone rerank top 20 → top 8
            → Return list of MeetingChunk objects
        → Invoke AnswerAgent.as_tool(query, chunks)
            → Synthesize 2-5 sentence answer
            → Return answer string
    → speak(answer, bot_id) OR post to Slack channel
```

### State Boundaries

| State | Location | Owner | Lifetime |
|-------|----------|-------|---------|
| `transcript_log` | `meeting_state` dict in `jarvis.py` | WebSocket handler | Single meeting session |
| `meeting_id` → JSON | `meetings/` directory | `MetadataStore` | Permanent |
| Dense + sparse vectors | Pinecone | `PineconeClient` | Permanent |
| BM25 corpus (if local) | `BM25Encoder` instance | `PineconeClient` init | Process lifetime, rebuilt on startup |
| `channel_id` context | passed from WebSocket handler | `jarvis.py` | Per query |

---

## Suggested Build Order

Dependencies drive this order. Each phase produces a working, independently testable artifact.

### Phase 1: Storage Foundation
**Build:** `MetadataStore` (JSON read/write) + `PineconeClient` (upsert + basic query) + Pinecone index creation script.

**Why first:** Everything else depends on data being in Pinecone. Building storage in isolation allows testing with synthetic data before any agents exist.

**Deliverable:** Can write a meeting record as JSON and query Pinecone by channel_id and date range.

### Phase 2: Summarizer Agent
**Build:** `SummarizerAgent` with structured Pydantic output schema + integration into `jarvis.py` WebSocket disconnect handler.

**Why second:** The query pipeline is useless without meeting data. Building this first ensures real data populates Pinecone for all subsequent testing.

**Depends on:** Phase 1 (MetadataStore + PineconeClient write path).

**Deliverable:** At end of any meeting, a JSON file appears on disk and a vector record appears in Pinecone.

### Phase 3: Retriever Agent + Hybrid Search
**Build:** `RetrieverAgent` with BM25 sparse encoding + dense embedding + hybrid Pinecone query + reranking. Test in isolation with synthetic queries against data from Phase 2.

**Why third:** Cleanest to validate the retrieval quality before wiring to the full agent pipeline. Retrieval bugs are easier to debug in isolation.

**Depends on:** Phase 1 (PineconeClient read path) + Phase 2 (real data in Pinecone).

**Deliverable:** Given a query + optional date/channel filter, returns ranked meeting excerpts with measurable relevance.

### Phase 4: Orchestrator Agent + Answer Agent
**Build:** `OrchestratorAgent` (query classification + date resolution) + `AnswerAgent` (synthesis). Wire agents-as-tools chain. Integrate into `jarvis.py` `handle_query` path via classifier fork.

**Why fourth:** Depends on retrieval being correct. The orchestrator and answer agent are relatively thin once retrieval works.

**Depends on:** Phase 3 (RetrieverAgent as tool).

**Deliverable:** Full end-to-end: "Hey Jarvis, what did we decide last week?" → spoken answer.

### Phase 5: Natural Language Date Resolution + Clarification Loop
**Build:** Date parser (regex + LLM fallback), ambiguity detection, clarification dialogue. This is isolated enough to be built alongside Phase 4 but tested separately.

**Note:** Clarification loops (speak question, await next chunk) require careful integration with the WebSocket state machine. Build last to avoid disrupting the working pipeline.

**Depends on:** Phase 4 (Orchestrator wired up).

**Deliverable:** "What happened last Tuesday?" resolves correctly. Ambiguous queries ask one clarifying question before proceeding.

---

## Anti-Patterns to Avoid

### Anti-Pattern 1: Handoffs for the RAG Pipeline
**What goes wrong:** Using handoffs (instead of agents-as-tools) for Retriever and Answer agents means the OrchestratorAgent loses control mid-query. The Retriever agent becomes the "active agent" and cannot hand results back to the Orchestrator.

**Why it happens:** Handoffs look simpler initially. The SDK's triage examples use handoffs for routing, which appears analogous to query routing.

**Instead:** Use `Agent.as_tool()` for Retriever and Answer agents. The Orchestrator calls both as tools and synthesizes the final answer. Handoffs are appropriate only for routing to entirely different conversational flows (e.g., handing off to a completely different persona).

### Anti-Pattern 2: Storing ISO Date Strings in Pinecone Metadata
**What goes wrong:** `$gte` and `$lte` operators require numeric values. ISO strings ("2024-04-01") cannot be range-filtered. Queries like "meetings in the last 30 days" silently return wrong results.

**Instead:** Always store `start_ts` and `end_ts` as integer UNIX timestamps. Convert at ingestion time, not at query time.

### Anti-Pattern 3: Fitting BM25 Encoder at Query Time
**What goes wrong:** Fitting BM25 on every query forces re-scanning all meeting texts per request. As meeting history grows, this becomes a blocking bottleneck.

**Instead:** Prefer Pinecone's hosted `pinecone-sparse-english-v0` sparse encoder (stateless, no fitting). If using local BM25Encoder, fit once at startup and persist the fitted model to disk using `BM25Encoder.dump()`. Re-fit only when new meetings are added.

### Anti-Pattern 4: One Pinecone Record Per Transcript Chunk
**What goes wrong:** Chunking raw transcript lines into many small records (one per sentence or speaker turn) bloats the index with low-signal fragments. Meeting-level summaries have much higher signal-to-noise for retrieval.

**Instead:** Store one vector per meeting, embedding the full `summary_text`. If cross-meeting topic search proves insufficient, add a second index of chunked summaries — but this is a later optimization, not the baseline design.

### Anti-Pattern 5: Re-embedding the Entire Corpus on Restart
**What goes wrong:** Without persisting the BM25 corpus, every restart triggers a full re-fit scan of all JSON metadata files. With 500+ meeting records this takes tens of seconds.

**Instead:** Use `BM25Encoder.dump("bm25_corpus.json")` / `.load()` pattern. Persist fitted encoder alongside meeting metadata files.

---

## Scalability Considerations

| Concern | At 100 meetings | At 10K meetings |
|---------|----------------|----------------|
| Pinecone query latency | <100ms (serverless auto-scales) | <100ms (same) |
| BM25 fit time (local) | ~1s on startup | ~30s+ → switch to hosted sparse model |
| JSON metadata scan | Negligible | Add SQLite index for date/channel lookups |
| Reranking candidates | top_k=20 is fine | top_k=20 still fine; reduce if latency degrades |
| Embedding cost at ingest | 1 API call per meeting | Same; cost scales linearly |

---

## Sources

- [OpenAI Agents SDK — Agent Orchestration](https://openai.github.io/openai-agents-python/multi_agent/) — HIGH confidence
- [OpenAI Agents SDK — Handoffs](https://openai.github.io/openai-agents-python/handoffs/) — HIGH confidence
- [Pinecone Docs — Hybrid Search](https://docs.pinecone.io/guides/search/hybrid-search) — HIGH confidence
- [Pinecone Docs — Filter by Metadata](https://docs.pinecone.io/guides/search/filter-by-metadata) — HIGH confidence
- [Pinecone Docs — Encode Sparse Vectors](https://docs.pinecone.io/guides/data/encode-sparse-vectors) — HIGH confidence
- [Pinecone Community — Date Filtering with UNIX Timestamps](https://community.pinecone.io/t/filtering-on-date-metadata/913) — MEDIUM confidence (community thread, consistent with official docs)
- [OpenAI Cookbook — Multi-Agent Portfolio Collaboration](https://cookbook.openai.com/examples/agents_sdk/multi-agent-portfolio-collaboration/multi_agent_portfolio_collaboration) — MEDIUM confidence (example, not normative spec)
- [Superlinked — Optimizing RAG with Hybrid Search and Reranking](https://superlinked.com/vectorhub/articles/optimizing-rag-with-hybrid-search-reranking) — MEDIUM confidence (verified against Pinecone official docs)

---

*Research date: 2026-04-04*
