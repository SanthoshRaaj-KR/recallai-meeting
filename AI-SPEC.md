# AI-SPEC.md — Jarvis Meeting Intelligence Platform

**System type:** Hybrid — Real-Time Conversational Voice Agent + RAG Knowledge Retrieval  
**Framework:** LiveKit Agents SDK ~1.5 (live agent) + OpenAI Agents SDK ~0.1 (post-meeting review pipeline)  
**Model provider:** OpenAI (embeddings, memory compaction, review) + Cerebras (live LLM inference via `gpt-oss-120b`)  
**Vector store:** Pinecone 8.x integrated inference index (`llama-text-embed-v2`)  
**Graph DB:** Neo4j AuraDB (page relationship traversal — not yet active in `my-agent`)

---

## Section 1 — System Overview

*(Populated by product spec — not managed here)*

## Section 2 — Architecture

*(Populated by product spec — not managed here)*

---

## Section 3 — Framework Quick Reference

### 3.1 Installation

```bash
# my-agent (LiveKit Agents runtime)
uv add "livekit-agents[assemblyai,cerebras,deepgram,mcp,silero,turn-detector]~=1.5"
uv add livekit-plugins-ai-coustics~=0.2 livekit-plugins-cartesia>=1.5.12

# Post-meeting review agents (OpenAI Agents SDK)
uv add "openai-agents>=0.1" "openai>=1.0"

# Vector store
uv add "pinecone>=8.0"          # SDK 8.x — required for integrated-inference search()

# Dev / test
uv add --dev pytest pytest-asyncio ruff
```

Download required model files before first run:

```bash
uv run python src/agent.py download-files
```

### 3.2 Key Imports

```python
# Live voice agent
from livekit.agents import (
    Agent, AgentServer, AgentSession, JobContext, JobProcess,
    ModelSettings, StopResponse, cli, llm, mcp, room_io,
    stt as lk_stt,
)
from livekit import rtc
from livekit.plugins import cartesia, cerebras, deepgram, silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel

# Post-meeting review pipeline (OpenAI Agents SDK)
from agents import Agent as OAIAgent, Runner, function_tool

# Pinecone SDK 8.x (integrated inference path)
from pinecone import Pinecone
```

### 3.3 Entry Point Pattern — Hybrid RAG Voice Agent

The live meeting agent follows the LiveKit Agents pattern: `AgentServer` → `@server.rtc_session` decorator → `AgentSession` wiring STT/LLM/TTS. RAG is injected in `on_user_turn_completed`, not in the LLM system prompt, so stale context never leaks between turns.

```python
# src/agent.py  (abridged — see full file for production detail)
import asyncio
from livekit.agents import Agent, AgentServer, AgentSession, JobContext, JobProcess, StopResponse, llm
from livekit.plugins import cartesia, cerebras, deepgram, silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel
from .confluence_rag import ConfluenceLiveRAG

server = AgentServer()

def prewarm(proc: JobProcess):
    # Load VAD weights once per worker process — not per session.
    proc.userdata["vad"] = silero.VAD.load()

server.setup_fnc = prewarm


class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            llm=cerebras.LLM(model="gpt-oss-120b"),  # ~150ms to first token
            instructions="You are Jarvis, a voice meeting assistant...",
        )
        self._rag = ConfluenceLiveRAG()
        self._transcript: list[str] = []

    async def on_enter(self) -> None:
        # Warm up Pinecone TCP + TLS during opening greeting — not on first query.
        await asyncio.to_thread(self._rag.warmup)

    async def on_user_turn_completed(
        self,
        turn_ctx: llm.ChatContext,
        new_message: llm.ChatMessage,
    ) -> None:
        raw = new_message.text_content or ""
        self._transcript.append(raw)

        if not self._is_wake_word(raw):
            raise StopResponse()   # suppress LLM — no wake word

        query = self._extract_query(raw)
        # RAG runs in a thread so it never blocks the asyncio event loop.
        hits = await asyncio.to_thread(self._rag.search, query)
        rag_ctx = self._rag.format_context(hits)

        # Insert context as a system message just before the user turn.
        if rag_ctx:
            user_idx = turn_ctx.index_by_id(new_message.id)
            turn_ctx.items.insert(
                user_idx if user_idx is not None else len(turn_ctx.items),
                llm.ChatMessage(role="system", content=[
                    "[Confluence Knowledge]\n" + rag_ctx
                ]),
            )
        await self.update_chat_ctx(turn_ctx)


@server.rtc_session(agent_name="my-agent")
async def my_agent(ctx: JobContext):
    turn_detector = MultilingualModel()
    session = AgentSession(
        stt=deepgram.STT(model="nova-3", language="en", keyterm=["Jarvis", "Hey Jarvis"]),
        tts=cartesia.TTS(model="sonic-3", voice="9626c31c-bec5-4cca-baa8-f8ba9e84c8bc"),
        turn_detection=turn_detector,
        vad=ctx.proc.userdata["vad"],
        preemptive_generation=False,  # disabled: wake-word gate rewrites every message
    )
    await ctx.connect()
    await session.start(agent=Assistant(), room=ctx.room)


if __name__ == "__main__":
    from livekit.agents import cli
    cli.run_app(server)
```

### 3.4 Core Abstractions

| Abstraction | Class / Function | Purpose in Jarvis |
|---|---|---|
| `Agent` | `livekit.agents.Agent` | Base class for `Assistant`; owns STT/LLM/TTS node overrides, wake-word gate, RAG injection via `on_user_turn_completed` |
| `AgentSession` | `livekit.agents.AgentSession` | Wires together STT, LLM, TTS, VAD, turn detection into a single pipeline; one per connected room |
| `ChatContext` / `ChatMessage` | `livekit.agents.llm` | Mutable conversation window injected each turn; RAG and transcript snapshots are inserted as `role="system"` messages, then cleared after each LLM call |
| `ConfluenceLiveRAG` | `src/confluence_rag.py` | Stateful Pinecone client; `search()` is synchronous (blocking), always called via `asyncio.to_thread()`; caches results for 120 s per session to avoid redundant Pinecone round-trips |
| `TranscriptCompactor` | `src/memory_compaction.py` | Maintains a rolling LLM-compacted summary of utterances that fall outside the recent-transcript sliding window; runs in a dedicated `ThreadPoolExecutor(max_workers=1)` to serialise OpenAI calls off the event loop |
| `ConfluenceVectorIndex` | `src/review_pipeline/rag.py` | Post-meeting review retriever; wraps Pinecone integrated index with BM25 blend (`MY_AGENT_BM25_BLEND`, default 0.25), multi-query RRF fusion, and Pinecone reranking (`bge-reranker-v2-m3`) |
| `PineconeHybridIndex` | `src/review_pipeline/confluence_pipeline/retrieval.py` | Local-doc pipeline retriever; two Pinecone indexes (dense via `llama-text-embed-v2` or `text-embedding-3-small`, sparse via `pinecone-sparse-english-v0`), RRF-fused, reranked |

### 3.5 Pinecone Index Schema

**Integrated inference index** (used by `ConfluenceLiveRAG` and `ConfluenceVectorIndex`):

```
Index name:   MY_AGENT_RAG_INDEX  (default: "confluence-review-rag-v2")
Namespace:    MY_AGENT_RAG_NAMESPACE  (default: "confluence-review")
Embed model:  llama-text-embed-v2  (server-side; field_map.text = "chunk_text")

Record schema per chunk:
  _id           str     "{page_id}:{chunk_order}"  — e.g. "12345:3"
  chunk_text    str     What gets embedded: "Title: X\nHeading: Y\nSpace: Z\n{text}"
                        (capped at MY_AGENT_RAG_METADATA_CHARS = 5000 chars)
  text          str     Raw display text returned to LLM (capped at 800 chars per hit)
  page_id       str     Confluence page ID
  title         str     Page title
  space_key     str     Confluence space key (e.g. "ENG")
  heading       str     Section heading (e.g. "Deployment Process")
  section_order int     Position of section in page (for ordering related chunks)
  chunk_order   int     Global chunk index within page
  version       int     Confluence page version number (for freshness checks)
  content_hash  str     SHA-256 of page content (skip re-embed when unchanged)
  chunk_count   int     Total chunks for this page (used in stale-chunk cleanup)
```

**Creation command** (run once, or guarded by `MY_AGENT_RAG_CREATE_INDEX=1`):

```python
from pinecone import Pinecone

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
pc.create_index_for_model(
    name="confluence-review-rag-v2",
    cloud="aws",
    region="us-east-1",
    embed={
        "model": "llama-text-embed-v2",
        "field_map": {"text": "chunk_text"},
    },
)
```

**Hybrid sparse index** (used by `PineconeHybridIndex` in the local-doc pipeline):

```
Dense index:   MY_AGENT_LDOC_DENSE_INDEX   (default: "confluence-corpus-dense")
               - Model: text-embedding-3-small (1536-dim, OpenAI external) OR
                        llama-text-embed-v2 (integrated, MY_AGENT_LDOC_DENSE_BACKEND=pinecone)
Sparse index:  MY_AGENT_LDOC_SPARSE_INDEX  (default: "confluence-corpus-sparse")
               - Model: pinecone-sparse-english-v0 (integrated, BM25-equivalent lexical)
Namespace:     MY_AGENT_LDOC_NAMESPACE     (default: "smarthub")
Fusion:        Reciprocal Rank Fusion (k=60) over dense + sparse result sets
BM25 blend:    MY_AGENT_BM25_BLEND=0.25 (also applied post-RRF in ConfluenceVectorIndex)
Reranker:      bge-reranker-v2-m3 via pc.inference.rerank()
```

### 3.6 Chunking Strategy

Chunks are produced by `chunk_page()` in `src/review_pipeline/rag.py`:

- **Primary split**: Confluence sections delimited by heading tags (`<h1>`–`<h6>`) extracted from raw HTML.  One chunk per section heading.
- **Overflow split**: Sections exceeding `MY_AGENT_RAG_CHUNK_WORDS` words (default: 350 words ≈ ~450 tokens) are split by paragraph boundaries first, then by sliding word windows (overlap: 50 words).
- **Table chunks**: Each `<table>` row is also emitted as a separate mini-chunk with column-header associations preserved (`"Header1: value1 | Header2: value2"`).  This enables exact numeric lookups (SLA times, sprint capacities, version numbers) that prose chunking buries.
- **Embedding text prefix**: Every chunk's `chunk_text` field is prefixed with `Title: {title}\nHeading: {heading}\nSpace: {space_key}\n` before embedding.  This gives the embedding model page-level context for generic headings like "Overview" or "Background".
- **Contextual enrichment** (optional, `MY_AGENT_RAG_CONTEXTUAL_ENRICHMENT=1`): A 1–2 sentence GPT-4o-mini context summary is prepended to `chunk_text` at upsert time (Anthropic contextual retrieval technique; reduces retrieval failure 35–67% for generic-heading sections).

### 3.7 Folder Structure

```
my-agent/
  src/
    agent.py                      # LiveKit Agents entrypoint; Assistant + JarvisCallAssistant
    confluence_rag.py             # ConfluenceLiveRAG — in-meeting Pinecone retriever
    memory_compaction.py          # TranscriptCompactor — LLM-compacted sliding memory
    session_store.py              # SQLite session persistence (meetings + transcripts)
    bot_service.py                # Recall.ai bot lifecycle REST wrapper
    recall_bridge.py              # HTTP bridge: Recall webhook -> LiveKit room dispatch
    review_pipeline/
      rag.py                      # ConfluenceVectorIndex — post-meeting hybrid retriever
      pipeline.py                 # Post-meeting review pipeline orchestrator
      models.py                   # PageCandidate, VectorSearchHit, PageChunk dataclasses
      text_utils.py               # html_to_text, extract_sections, normalize_ws
      confluence_pipeline/
        chunker.py                # Markdown/text section chunker (ChunkRecord)
        retrieval.py              # PineconeHybridIndex — dense+sparse+RRF+rerank
        pipeline.py               # Local-doc pipeline orchestrator
        editor.py                 # LLM-driven section editor (OpenAI Agents SDK)
        verifier.py               # Edit verifier agent
        models.py                 # ChunkRecord and pipeline dataclasses
```

### 3.8 Known Pitfalls

1. **Pinecone SDK 8.x `search()` signature changed from SDK 5.x.** The `top_k` and `inputs` parameters must be nested inside a `query={"top_k": N, "inputs": {"text": q}}` dict. Passing `top_k=` at the top level raises a `TypeError`. The `_score` attribute (not `score`) carries the similarity score on `Hit` objects. Both quirks are worked around in `ConfluenceLiveRAG.search()` and `ConfluenceVectorIndex.search()`.

2. **Calling blocking Pinecone I/O directly on the asyncio event loop stalls the audio pipeline.** `ConfluenceLiveRAG.search()` is synchronous. If called with `await rag.search(query)` (wrong) rather than `await asyncio.to_thread(rag.search, query)` (correct), the STT and TTS pipelines freeze until Pinecone responds (~100–400 ms), causing audible glitches. The same applies to the `TranscriptCompactor`: its OpenAI calls run in a dedicated `ThreadPoolExecutor(max_workers=1)` via `_run_in_compactor()`.

3. **`preemptive_generation=False` is mandatory with a wake-word gate.** LiveKit Agents begins generating an LLM reply speculatively as soon as the turn detector fires. When `on_user_turn_completed` either raises `StopResponse()` (no wake word) or rewrites `new_message.content` (strips the wake word), preemptive output is discarded — but if preemptive generation is enabled, a partial audio glitch plays before the discard. Set `preemptive_generation=False` in `AgentSession`.

4. **Stale RAG context bleeding between turns.** The `_last_rag_context` field is set during RAG retrieval and consumed (injected as a system message) in `_refresh_transcript_in_ctx()`. It must be explicitly cleared after injection (`self._last_rag_context = ""`). If not cleared, a wake-word-less utterance that skips retrieval will re-inject the previous turn's Confluence context into the next LLM call.

5. **Pinecone integrated index TPM ceiling during bulk upsert.** `llama-text-embed-v2` has a tokens-per-minute ceiling (250k TPM on the free tier). Upserting 1000+ Confluence pages without pacing causes HTTP 429 errors and silent data loss. The `_upsert_batch_with_backoff()` method in `PineconeHybridIndex` and the `_UPSERT_BATCH=40` / `_UPSERT_PACE_S=1.5` env vars implement exponential backoff. Always pace bulk ingestion; never upsert the full corpus in a single loop.

### 3.9 Sources

- LiveKit Agents build nodes / RAG: https://docs.livekit.io/agents/build/nodes
- LiveKit Agents external data (RAG): https://docs.livekit.io/agents/build/external-data
- LiveKit Agents chat context: https://docs.livekit.io/agents/logic/chat-context
- LiveKit Agents testing: https://docs.livekit.io/agents/start/testing/
- LiveKit Agents workflows (handoffs/tasks): https://docs.livekit.io/agents/build/workflows/
- Pinecone integrated index creation: https://github.com/pinecone-io/python-sdk/blob/main/docs/how-to/integrated-records.md
- Pinecone SDK 8.x `search()` reference: https://github.com/pinecone-io/python-sdk/blob/main/docs/reference/sync-index.md
- Pinecone `upsert_records` reference: https://github.com/pinecone-io/python-sdk/blob/main/docs/reference/grpc.md
- Anthropic contextual retrieval: https://www.anthropic.com/news/contextual-retrieval

---

## Section 4 — Implementation Guidance

### 4.1 Model Configuration

| Role | Model | Notes |
|---|---|---|
| Live meeting LLM | `cerebras/gpt-oss-120b` | ~150 ms time-to-first-token via Cerebras inference; used in `Agent.__init__()` via `cerebras.LLM(model="gpt-oss-120b")` |
| Memory compaction | `gpt-4o-mini` | Summarises overflowed transcript utterances; env `MY_AGENT_MEMORY_MODEL` or `JARVIS_MEMORY_MODEL` |
| Contextual chunk enrichment | `gpt-4o-mini` | Prepends 1–2 sentence context to each chunk at upsert time; only called when `MY_AGENT_RAG_CONTEXTUAL_ENRICHMENT=1` |
| Post-meeting review agents | `gpt-4o-mini` (routing/classification) + GPT-5 (hard verification) | Env `MY_AGENT_REVIEW_MODEL`; do not use GPT-5 turbo — model ceiling is GPT-5 |
| Embeddings | `llama-text-embed-v2` (server-side, Pinecone integrated) or `text-embedding-3-small` (OpenAI external) | Pinecone integrated avoids a separate OpenAI API call per chunk; external is cheaper for high-volume batch upserts |

**Key model parameters** (set explicitly in every production LLM call):

```python
# Memory compaction (gpt-4o-mini path)
response = client.chat.completions.create(
    model="gpt-4o-mini",
    max_tokens=2000,       # never unbounded in production
    temperature=0.0,       # deterministic compaction
    messages=[...]
)

# GPT-5 / o-series models use max_completion_tokens instead of max_tokens
if model.startswith(("gpt-5", "o1", "o3", "o4")):
    kwargs["max_completion_tokens"] = 2000
else:
    kwargs["max_tokens"] = 2000
    kwargs["temperature"] = 0.0
```

### 4.2 Real-Time RAG Pipeline (Core Pattern)

```
Wake word detected (STT interim) → "Yes?" ack played immediately (stt_node)
             |
     on_user_turn_completed fires (final STT transcript)
             |
      ┌──────────────────────────────────────────────┐
      │  asyncio.to_thread(rag.search, enriched_q)   │  ← non-blocking; Pinecone ~100-250ms
      │  Cache hit → return immediately (~0ms)        │
      └──────────────────────────────────────────────┘
             |
      Build enriched query (question + topic_hint + transcript context)
      Pinecone integrated search() → top-K chunks
      format_context() → "[Page Title > Section]\n{text}" blocks
             |
      Insert as role="system" ChatMessage before user turn
      call update_chat_ctx(turn_ctx)
             |
      LLM (Cerebras gpt-oss-120b) → streaming text → Cartesia TTS → audio
             |
      Clear _last_rag_context to prevent bleed into next turn
```

**Enriched query construction** (implemented in `ConfluenceLiveRAG.build_search_query()`):

```python
def build_search_query(
    self,
    question: str,
    recent_transcript: list[str],
    topic_hint: str = "",          # from compacted memory; only used for vague questions
    context_lines: int = 12,
) -> str:
    # 1. Clean the question: remove filler words, collapse whitespace
    q = _FILLER_RE.sub(" ", question).strip()

    # 2. Harvest recent transcript for contextual signal
    context_parts = [
        _SPEAKER_PREFIX_RE.sub("", line)     # strip "Alice: " prefixes
        for line in recent_transcript[-context_lines:]
        if not re.match(r"(?i)^jarvis\s*:", line)   # exclude Jarvis's own replies
        and len(line.split()) >= 4                   # skip back-channel noise
    ]

    # 3. topic_hint only when the question is too short to carry topical signal
    parts = [q]
    if topic_hint and len(q.split()) < 5:
        parts.append(topic_hint.strip())
    parts.extend(context_parts)

    combined = " ".join(parts)
    return combined[:350]    # JARVIS_CONFLUENCE_RAG_MAX_QUERY_CHARS
```

**Why this approach works for meeting Q&A:** ASR output for short spoken questions is terse and often ambiguous ("What does that mean?", "Who owns it?"). Appending the last 12 transcript lines gives the embedding model the surrounding conversational context — the meeting's current topic — even when the question itself contains no topical signal. The 350-character cap prevents dilution of the embedding with unrelated prior context.

### 4.3 Latency Budget Breakdown

Target end-to-end latency: 3–5 seconds from wake word to first audio syllable.

| Stage | Budget | How Achieved |
|---|---|---|
| VAD end-of-speech detection | 100–300 ms | Silero VAD; `MultilingualModel` turn detector reduces false cut-offs |
| STT (Deepgram Nova-3) | 150–400 ms | Deepgram hosted; `keyterm=["Jarvis","Hey Jarvis"]` boosts recognition accuracy; partial-wake ack fires at ~200 ms via interim transcript in `stt_node` |
| Pinecone retrieval (`asyncio.to_thread`) | 80–250 ms | Integrated inference: single round-trip (no separate embed call); cache hit is ~0 ms; TCP pre-warmed in `on_enter()` |
| Query enrichment + cache check | <5 ms | In-memory; runs on event loop before dispatching to thread |
| ChatContext assembly | <2 ms | In-memory deque slicing |
| LLM TTFT (Cerebras gpt-oss-120b) | 120–200 ms | Cerebras inference is the fastest available for this model class |
| TTS TTFB (Cartesia sonic-3) | 80–150 ms | Cartesia streaming; first audio chunk arrives before LLM finishes |
| **Total (p50)** | **~1.5–2.5 s** | Within target |
| **Total (p95, Pinecone cold + LLM retry)** | **~4.5 s** | Pre-warm in `on_enter()` eliminates cold-start for Pinecone; first turn always served from warm connection |

**Critical path:** Pinecone retrieval and LLM TTFT are in parallel with ChatContext assembly. Because RAG is dispatched via `asyncio.to_thread()` before `_refresh_transcript_in_ctx()` is called, the Pinecone network round-trip overlaps with other synchronous work in the event loop.

### 4.4 Incremental Ingestion

Implemented in `ConfluenceVectorIndex.sync_index()`:

```
Algorithm (O(pages/100) Pinecone calls + O(changed) embed calls):

1. Batch-fetch chunk :0 metadata for every page (100 IDs per fetch)
   → extract stored version number + content_hash

2. Compare against Confluence REST listing (live version.number)
   → stale = new pages + pages where live_version > stored_version

3. Re-embed only stale pages via upsert_page()
   → skips fresh pages entirely (0 embed calls for unchanged content)

4. Purge orphan chunks (pages deleted from Confluence)
   → list() all chunk IDs, filter those whose page_id is not in the live set
   → delete() in batches of 1000
```

**Recommended trigger strategy** — prefer webhook over polling:

- Configure a Confluence webhook for `page_updated` / `page_created` / `page_deleted` events pointing at `/api/reindex` on the review pipeline server.
- Debounce rapid edits (a page saved 10 times in 30 s should trigger one re-embed, not 10). Implement with a per-page `asyncio.Task` that waits 30 s before calling `upsert_page()`, cancelling and restarting if another event arrives.
- As a fallback, run `sync_index()` on a 15-minute schedule. With 1000+ pages, a full sync takes ~2–3 min (dominated by the batch-fetch round-trips, not embedding, since most pages are fresh).

```python
# Debounced webhook handler pattern
_pending_reindex: dict[str, asyncio.Task] = {}

async def handle_page_updated(page_id: str):
    if page_id in _pending_reindex:
        _pending_reindex[page_id].cancel()
    async def _delayed():
        await asyncio.sleep(30)
        page = fetch_page_fn(page_id)
        await asyncio.to_thread(vector_index.upsert_page, page)
        _pending_reindex.pop(page_id, None)
    _pending_reindex[page_id] = asyncio.create_task(_delayed())
```

### 4.5 Graph RAG Expansion (Neo4j — not yet active in `my-agent`)

When Neo4j is available, graph traversal should augment, not replace, Pinecone vector search. The recommended decision logic:

```
Query type                              Action
─────────────────────────────────────────────────────────────
Short factual question                  Pinecone only (faster; ~150ms saved)
"Related pages" / "parent topic"        Neo4j traversal: MATCH (p)-[:CHILD_OF*1..2]->(parent)
"Everything about X" / broad topic      Pinecone → expand hits via Neo4j neighbours
Post-meeting proposal retrieval         Neo4j + Pinecone (RRF fusion)
```

**Recommended Neo4j schema for Confluence:**

```cypher
// Nodes
(:Page {page_id, title, space_key, version, content_hash, indexed_at})
(:Section {section_id, heading, section_order, chunk_id_prefix})

// Relationships
(:Page)-[:HAS_SECTION]->(:Section)
(:Page)-[:CHILD_OF]->(:Page)          // Confluence child pages
(:Page)-[:LINKS_TO]->(:Page)          // Inline page links extracted from HTML
(:Section)-[:NEXT_SECTION]->(:Section) // sequential navigation
```

**Expansion query** — given a vector-search hit, fetch its sibling sections and parent page for additional context:

```cypher
MATCH (p:Page {page_id: $page_id})-[:HAS_SECTION]->(s:Section)
OPTIONAL MATCH (p)-[:CHILD_OF]->(parent:Page)
RETURN p, collect(s) AS sections, parent
LIMIT 1
```

---

## Section 4b — AI Systems Best Practices

### 4b.1 Structured Outputs with Pydantic

All inter-component data contracts use Pydantic models. For the post-meeting review pipeline, the LLM must produce structured change proposals that the UI renders as review cards.

**Output schema for change proposals** (from `src/review_pipeline/models.py`):

```python
from pydantic import BaseModel, Field
from typing import Literal

class ProposedChange(BaseModel):
    change_type: Literal["create", "edit", "delete", "title"]
    page_id: str
    page_title: str
    section_heading: str | None = None
    before_content: str | None = None
    after_content: str | None = None
    rationale: str = Field(
        ...,
        description="One sentence explaining why this change was proposed.",
    )

class ProposalBatch(BaseModel):
    changes: list[ProposedChange]
    meeting_summary: str = Field(default="", max_length=2000)
```

**Integration with OpenAI Agents SDK** (used in the review pipeline):

```python
from agents import Agent, Runner
from pydantic import BaseModel

class ProposalBatch(BaseModel):
    changes: list[ProposedChange]

review_agent = Agent(
    name="ProposedChangesAgent",
    instructions="Analyse the meeting transcript and propose Confluence changes...",
    output_type=ProposalBatch,   # SDK enforces JSON schema and retries
    model="gpt-4o-mini",
)

result = await Runner.run(
    review_agent,
    input=f"Transcript:\n{transcript}\n\nRelevant pages:\n{rag_context}",
)
batch = result.final_output   # type: ProposalBatch — SDK guarantees this
```

**Retry logic:** The OpenAI Agents SDK retries JSON parse failures up to 3 times by default when `output_type` is set. For direct `client.chat.completions.create()` calls (memory compaction, contextual enrichment), use manual retry with exponential backoff:

```python
import time

def call_with_retry(client, model: str, messages: list, max_tokens: int, max_attempts: int = 3) -> str:
    delay = 1.0
    for attempt in range(max_attempts):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=0.0,
                response_format={"type": "json_object"},  # force JSON mode
            )
            return resp.choices[0].message.content or ""
        except Exception as exc:
            is_parse = "json" in str(exc).lower() or "parse" in str(exc).lower()
            is_rate = "429" in str(exc)
            if attempt < max_attempts - 1 and (is_parse or is_rate):
                logger.warning("LLM call attempt %d failed (%s); retrying in %.1fs", attempt + 1, exc, delay)
                time.sleep(delay)
                delay = min(delay * 2, 10.0)
                continue
            logger.error("LLM call failed after %d attempts: %s", max_attempts, exc)
            return ""
    return ""
```

**When to surface errors to the user:** After 3 failed retries, log at `ERROR` level, return a graceful fallback (empty proposal list / empty context string), and never raise an exception that would crash the audio pipeline. The meeting bot must keep running even if the review agent fails.

### 4b.2 Async-First Design

This codebase is asyncio-native. Every component that touches I/O must be designed for this environment.

**How async works in LiveKit Agents:**

LiveKit Agents runs a single-threaded asyncio event loop per worker process. The `AgentSession` pipeline (STT audio frames, LLM token streaming, TTS audio frames) all run as async generators or coroutines on this loop. Any blocking call on the loop stalls audio I/O.

**The one common mistake:**

```python
# WRONG — blocks the event loop during Pinecone query (~100-400ms)
hits = self._rag.search(query)

# WRONG — asyncio.run() inside an already-running event loop raises RuntimeError
hits = asyncio.run(self._async_rag_search(query))

# CORRECT — offload the blocking sync call to a thread pool
hits = await asyncio.to_thread(self._rag.search, query)
```

**Stream vs await for different use cases:**

```python
# Stream: for TTS output — first audio plays before LLM finishes; critical for UX
async for frame in Agent.default.tts_node(self, text_stream, model_settings):
    yield frame   # audio frames stream to the room immediately

# Await: for structured output that must be complete before use
result = await Runner.run(review_agent, input=context)
batch = result.final_output   # type: ProposalBatch — fully parsed, validated

# Thread executor: for blocking I/O that lacks async wrappers
# Use asyncio.to_thread() for single calls; ThreadPoolExecutor for serialised multi-call sequences
compacted = await loop.run_in_executor(self._compactor_executor, self._compactor.memory_text)
```

**Thread executor pattern for compactor serialisation:**

The `TranscriptCompactor` is called from two concurrent paths: the event loop (`on_user_turn_completed`) and `tts_node` (which runs as a background async generator). Sharing mutable state between these paths requires serialisation. Solution: `ThreadPoolExecutor(max_workers=1)` ensures compactor calls are processed sequentially without locks.

```python
self._compactor_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="compactor")

async def _run_in_compactor(self, fn, *args):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(self._compactor_executor, fn, *args)
```

### 4b.3 Prompt Engineering Discipline

**System vs user prompt separation:**

- The system prompt (`_build_instructions()`) is static per session. It defines the agent's identity, output format rules (no markdown, 1–3 sentences, TTS-friendly prose), and source reliability hierarchy (transcript > Confluence > general knowledge).
- RAG context and transcript snapshots are injected as dynamic `role="system"` messages in `on_user_turn_completed`, just before the user's message. They are cleared after each turn via `_last_rag_context = ""`.
- Never embed dynamic context (transcript text, retrieved chunks) in the static system prompt — it makes every token in the static prompt more expensive and causes stale content to persist across turns.

**Few-shot examples:**

For the review pipeline agents (editor, verifier), inline few-shot examples in the system prompt are sufficient for this use case — the output schema is strict (Pydantic) and the task is well-defined. For the live meeting agent, no few-shot examples are needed in the system prompt; the output format rules (`plain text, 1–3 sentences`) are enforced by instruction, not by examples.

**Always set `max_tokens` explicitly:**

```python
# Memory compaction
max_tokens=2000   # a compacted memory block is never longer than this

# Contextual chunk enrichment (upsert-time)
max_tokens=120    # 1–2 sentences only

# Review agent via OpenAI Agents SDK — set on the Agent
Agent(model="gpt-4o-mini", model_settings=ModelSettings(max_tokens=4000), ...)
```

Never leave `max_tokens` unset in production. An unset `max_tokens` allows the model to generate until its context limit, which can cause unexpected cost spikes and response delays for the live TTS pipeline.

**Wake-word query cleaning before embedding:**

```python
# Strip filler words before sending to Pinecone — they dilute the embedding
_FILLER_RE = re.compile(
    r"\b(um+|uh+|hmm+|yeah|yep|okay|ok|right|like|so|just|you know|"
    r"actually|basically|literally|well|anyway|alright|kind of|sort of)\b",
    re.IGNORECASE,
)
clean_query = re.sub(r"\s+", " ", _FILLER_RE.sub(" ", question)).strip()
```

**HyDE (Hypothetical Document Embeddings) verdict for this use case:**

HyDE adds one extra LLM call (generate a hypothetical answer, embed that instead of the question) and ~200–400 ms of latency. For a live voice agent with a 3–5 s end-to-end budget, this is too expensive unless retrieval recall is demonstrably poor. The transcript-enriched query (question + last 12 transcript lines) already provides contextual signal equivalent to what HyDE achieves by generating a hypothetical answer. **Do not enable HyDE for the live path.** It can be evaluated for the post-meeting review pipeline where latency is not critical.

### 4b.4 Context Window Management

**Live agent — sliding window + compaction:**

The live meeting transcript grows unboundedly. Three separate budgets are enforced per LLM call:

```
_CHAT_HISTORY_WINDOW = 1           # max Q&A pairs in chat history (~750 tokens)
_RECENT_TRANSCRIPT_TOKENS = 4000   # recent raw transcript (~100 utterances)
_COMPACTED_MEMORY_TOKENS = 6000    # LLM-compacted older context
```

These are enforced in `_refresh_transcript_in_ctx()` via `turn_ctx.truncate(max_items=_CHAT_HISTORY_WINDOW)` and character-budget slicing. The total context sent to the LLM per turn is bounded at ~10,000 tokens + system prompt + RAG context.

**RAG context budget — top-K and per-chunk truncation:**

```python
_TOP_K_DEFAULT = 10        # JARVIS_CONFLUENCE_RAG_TOP_K
_MAX_CHUNK_CHARS = 800     # JARVIS_CONFLUENCE_RAG_MAX_CHUNK_CHARS
# Per-page deduplication: only the top-scoring chunk per page is included
# (format_context() suppresses duplicate page_ids)
```

At default settings, RAG context adds at most ~8,000 characters (~2,000 tokens) to each LLM call. If the total context exceeds the model's context window, reduce `_TOP_K_DEFAULT` or `_MAX_CHUNK_CHARS` first before touching the transcript budgets.

**Post-meeting review — reranking as context selection:**

For the review pipeline, retrieved chunks go through:
1. Multi-query vector search at `top_k_per_query=25`
2. RRF fusion → top-50 candidates
3. Pinecone reranking (`bge-reranker-v2-m3`) → top-8 final chunks
4. Per-page diversity cap: no single page takes more than `ceil(top_n/2)` slots

This means at most ~8 chunks × 2,000 chars = ~16,000 chars of Confluence context are included in the review LLM call. For GPT-4o-mini's 128K context window this is well within budget; for any model with a 8K context, reduce `top_n` to 4.

**Autonomous/compaction:** When `TranscriptCompactor` detects that `_staged` has accumulated `chunk_size=24` utterances beyond the window, it calls `force_compact()` which makes a blocking OpenAI call to rewrite the staged block into the running memory. This keeps the compacted memory below `max_memory_chars=8000` characters regardless of meeting length.

### 4b.5 Cost and Latency Budget

**Per-meeting cost estimate (1-hour meeting, 1000-page index):**

| Component | Volume | Model | Est. cost |
|---|---|---|---|
| Live LLM (Cerebras gpt-oss-120b) | ~20 wake-word queries × 5K tokens avg | Cerebras pricing | ~$0.05 |
| Memory compaction (gpt-4o-mini) | ~15 compaction cycles × 2K tokens | $0.00015/1K in, $0.00060/1K out | ~$0.02 |
| Pinecone retrieval | ~20 queries × 1 RU each | 1 RU = ~$0.00001 | <$0.001 |
| STT (Deepgram Nova-3) | 60 min | $0.0043/min | $0.26 |
| TTS (Cartesia sonic-3) | ~600 words spoken | Cartesia pricing | ~$0.01 |
| Post-meeting review (gpt-4o-mini) | 1 pipeline run × ~30K tokens | $0.00015/1K in | ~$0.01 |
| **Total per meeting** | | | **~$0.35–0.50** |

**Caching strategy:**

1. **Embedding cache** (Pinecone integrated index): Pinecone caches server-side embedding computations — identical query text strings within a session hit a Pinecone internal cache (not user-controllable but effectively free).

2. **In-session result cache** (`_CACHE_TTL_S=120, _CACHE_MAX=20` in `ConfluenceLiveRAG`): Identical enriched query strings within 2 minutes return cached hits without a Pinecone call. This handles repeated follow-up questions on the same topic during a meeting.

3. **Freshness-based upsert skip** (content hash + version check in `_indexed_page_is_fresh()`): At ingestion time, each page's SHA-256 content hash and Confluence version number are stored in Pinecone metadata. Re-indexing skips pages whose hash and version match — 0 embed calls for unchanged pages.

4. **Cheaper models for sub-tasks:**
   - Memory compaction: `gpt-4o-mini` (not GPT-5). Compaction is a summarisation task; model quality above gpt-4o-mini does not meaningfully improve memory quality.
   - Intent classification (classifier.py): `gpt-4o-mini` with a short system prompt and `max_tokens=20`.
   - Contextual chunk enrichment: `gpt-4o-mini` with `max_tokens=120`.
   - Reserve GPT-5 for the hard verification step in the post-meeting review pipeline (checking that proposed Confluence edits are factually grounded in the transcript).

**Latency optimisation checklist:**
- [ ] Pinecone TCP pre-warmed in `on_enter()` via throwaway search (1 RU cost)
- [ ] `asyncio.to_thread()` for all Pinecone and OpenAI calls in the live path
- [ ] `TranscriptCompactor` running in `ThreadPoolExecutor(max_workers=1)` — never blocks event loop
- [ ] `preemptive_generation=False` to prevent audio glitches on wake-word rewrites
- [ ] Deepgram `keyterm=["Jarvis","Hey Jarvis"]` for faster accurate STT on the wake word
- [ ] Partial-wake ack ("Yes?") fires on STT interim transcript (~200 ms before VAD silence), buying time for RAG retrieval
- [ ] Result cache hit eliminates Pinecone round-trip for repeated questions (~100–250 ms saved)
