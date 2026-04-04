# Technology Stack

**Project:** RecallAI Meeting Memory Bot — Hybrid RAG Milestone
**Researched:** 2026-04-04
**Context:** Adding hybrid RAG, multi-agent architecture, and meeting memory to an existing Python/FastAPI/Slack bot (jarvis.py)

---

## Existing Stack (Do Not Replace)

These are already installed and working. The new milestone builds on top of them.

| Technology | Version (pinned) | Role |
|------------|-----------------|------|
| Python | 3.10+ | Language (constraint — no switch) |
| FastAPI | unpinned | HTTP/WebSocket server |
| Uvicorn | unpinned | ASGI runtime |
| openai | unpinned | Chat completions, embeddings |
| openai-agents | unpinned | Agent framework (installed, needs activation) |
| python-dotenv | unpinned | Env var loading |
| requests | unpinned | Recall.ai REST calls |
| gTTS / pyaudio | unpinned | TTS audio output |
| websockets | unpinned | WebSocket support |

**Action required:** Pin all existing dependencies in requirements.txt. Unpinned deps in production are a maintenance hazard, and the new libs have version constraints that may collide silently.

---

## Recommended Stack — New Additions

### Agentic Framework

| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| openai-agents | 0.13.4 | Multi-agent orchestration with handoffs | Already in requirements.txt; latest as of 2026-04-01; produced by OpenAI, not LangChain; explicit project constraint. Uses three primitives: Agent, Runner, handoff(). Built-in tracing to OpenAI dashboard. Zero extra abstraction cost. |

**Confidence:** HIGH — verified via PyPI (released 2026-04-01).

**What it gives you:**
- `Agent(name=..., instructions=..., tools=[...])` — define specialist agents
- `handoff(agent)` — triage agent delegates to specialist mid-conversation
- `Agent.as_tool(agent)` — manager agent invokes specialist as a callable tool (does not hand off control)
- `Runner.run_sync()` / `await Runner.run()` — synchronous and async execution paths
- Built-in guardrails: `input_guardrail`, `output_guardrail` decorators
- Tracing: enabled by default, visible in OpenAI dashboard; disable via `OPENAI_AGENTS_DISABLE_TRACING=1`

**Pattern for this project:** Triage agent receives Slack query → handoff to DateResolutionAgent (resolves "last Wednesday") → handoff to RAGRetrievalAgent (queries Pinecone) → returns to SynthesisAgent (final answer). Use `Agent.as_tool()` for the summarization sub-task since it is a bounded, non-conversational subtask.

**Do not use:** LangChain Agents, LlamaIndex query engines, AutoGen — explicitly out of scope per PROJECT.md constraint.

---

### Vector Storage and Hybrid Search

| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| pinecone | 8.1.1 | Vector database client + inference API | Latest as of 2026-04-02; supports serverless indexes, integrated inference (embed + sparse), hybrid search in a single index. Managed scaling — no self-hosting. |
| pinecone-text | 0.11.0 | BM25Encoder for sparse vector generation | Official Pinecone library; integrates directly with Pinecone index upsert format; `hybrid_convex_scale()` for alpha-weighted merging. Released 2025-08-11. |

**Confidence:** HIGH — both verified via PyPI with recent release dates.

**Hybrid search architecture decision:**

Use **Pinecone's integrated inference** (`pc.inference.embed`) with `pinecone-sparse-english-v0` for sparse vectors, NOT `rank_bm25` or `pinecone-text BM25Encoder`. Rationale:

- `pinecone-sparse-english-v0` is a neural sparse model (DeepImpact architecture). It outperforms BM25 by up to 44% NDCG@10 on TREC. It understands context, not just term frequency.
- `BM25Encoder` from `pinecone-text` works but requires you to fit the model on your corpus at index-build time. For a growing meeting corpus, maintaining a fitted BM25 model adds state management overhead (must refit or use stale values).
- `rank_bm25` (latest: 0.2.2, last released 2022-02-16) is unmaintained. Do not use it.

**Index setup:**

```python
# Single index, dotproduct metric — the ONLY combination that supports hybrid search
pc.create_index(
    name="meeting-memory",
    dimension=1536,           # text-embedding-3-small output dimension
    metric="dotproduct",      # required for hybrid; not cosine
    vector_type="dense",
    spec=ServerlessSpec(cloud="aws", region="us-east-1")
)
```

**Upsert pattern:**

```python
# Dense: pc.inference.embed(model="text-embedding-3-small", inputs=[text])
# Sparse: pc.inference.embed(model="pinecone-sparse-english-v0", inputs=[text], parameters={"input_type": "passage"})
index.upsert(vectors=[{
    "id": meeting_id,
    "values": dense_vector,
    "sparse_values": {"indices": [...], "values": [...]},
    "metadata": {"channel_id": ..., "date": ..., "series_name": ..., "participants": [...]}
}])
```

**Query pattern — alpha weighting:**

```python
# alpha=1.0 → pure semantic; alpha=0.0 → pure keyword; alpha=0.5 → balanced
# Start with alpha=0.7 for meeting memory (semantic similarity is dominant signal)
index.query(
    vector=dense_query_vector,
    sparse_vector=sparse_query_vector,
    top_k=10,
    filter={"channel_id": {"$eq": channel_id}}   # metadata filter
)
```

**Do not use:** LangChain `PineconeHybridSearchRetriever` — it adds LangChain as a dependency and conflicts with the OpenAI Agents SDK requirement. Do the hybrid query directly via the Pinecone SDK.

---

### Embeddings

| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| openai (text-embedding-3-small) | via existing openai SDK | Dense vector generation | Already in stack; 1536 dimensions; cost-effective; strong MTEB retrieval scores; same vendor as the LLM, no additional API key or client needed. |

**Confidence:** MEDIUM — dimensions and availability verified via OpenAI docs. MTEB ranking is based on training data knowledge, not live verification.

**Do not use:** `sentence-transformers` locally — adds a heavy dependency (PyTorch) for no gain when OpenAI embeddings are already available and paid for.

---

### Natural Language Date Parsing

| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| dateparser | 1.4.0 | Resolve "last Wednesday", "two weeks ago", "the standup on March 3rd" to Python datetime objects | Production/Stable; 200+ language locales; handles relative expressions, timezone abbreviations, fuzzy text; proven on 100M+ web pages; released 2026-03-26; Python 3.10+ compatible. |

**Confidence:** HIGH — verified via PyPI with recent release date.

**Usage pattern:**

```python
import dateparser

parsed = dateparser.parse(
    "last Wednesday",
    settings={"RETURN_AS_TIMEZONE_AWARE": True, "PREFER_DAY_OF_MONTH": "first"}
)
# Returns timezone-aware datetime or None if unparseable
```

**Clarification strategy when `None` is returned:** The DateResolutionAgent should respond with an explicit disambiguation prompt to the user ("I couldn't determine what date you meant. Did you mean [date A] or [date B]?") rather than silently failing or returning all meetings.

**Do not use:** `parsedatetime` — older, less maintained, lower language coverage, requires more boilerplate. `python-dateutil` handles structured date strings well but is weak on natural language relative expressions ("last Wednesday" is ambiguous without it).

---

### Metadata Persistence (JSON files per meeting)

| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| aiofiles | 25.1.0 | Async JSON file I/O | FastAPI/uvicorn runs on asyncio; blocking file I/O inside an async handler causes event loop stall. aiofiles wraps file ops in a thread pool with async interface. Production/Stable; Python 3.9+; released 2025-10-09. |
| pydantic | 2.12.5 | Meeting metadata schema validation + serialization | Already a FastAPI dependency; v2 is 5-50x faster than v1 with Rust core; `model.model_dump_json()` serializes directly to JSON string; `model.model_validate(dict)` deserializes back; field validators for data integrity. |
| pathlib (stdlib) | stdlib | File path management | No dependency required; `Path.mkdir(parents=True, exist_ok=True)` for directory creation; compose `meetings/{channel_id}/{meeting_id}.json` paths cleanly. |

**Confidence:** HIGH for aiofiles and pathlib. HIGH for pydantic (version verified via PyPI).

**File layout:**

```
meetings/
  {channel_id}/
    {meeting_id}.json     # one file per meeting
```

**Pydantic schema for the JSON record:**

```python
from pydantic import BaseModel, Field
from datetime import datetime

class MeetingMetadata(BaseModel):
    meeting_id: str
    channel_id: str
    channel_name: str
    series_name: str
    recurrence_pattern: str | None = None
    start_time: datetime
    end_time: datetime | None = None
    duration_seconds: int | None = None
    participants: list[str] = Field(default_factory=list)
    summary_text: str
    topics_covered: list[str] = Field(default_factory=list)
    action_items: list[str] = Field(default_factory=list)
    pinecone_vector_id: str   # foreign key to Pinecone record
```

**Do not use:** SQLite or any relational DB for meeting metadata — JSON files are explicitly required per PROJECT.md for auditability and portability. Do not add a database dependency for this milestone.

---

### Meeting Summarization

| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| openai (gpt-4o-mini) | via existing openai SDK | Summarize transcript into structured summary + topics + action items | Already in stack; existing `OPENAI_MODEL` env var; cost-effective for summarization tasks; can be prompted to return structured JSON matching `MeetingMetadata`. |

**Confidence:** HIGH — uses existing SDK, no new dependency.

**Summarization should be triggered two ways:**
1. Auto-detect: when Recall.ai signals meeting end (bot_leave event or transcript stream closes)
2. Manual: Slack slash command `/summarize` (explicitly listed in PROJECT.md Active requirements)

The SummaryAgent should produce output structured as a Pydantic model (validated before persistence) — do not persist raw LLM text directly.

---

### Testing (establishing patterns — no tests currently)

| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| pytest | 9.0.2 | Test runner | Standard Python testing; latest stable released 2025-12-06; requires Python 3.10+ matching existing constraint. |
| pytest-asyncio | 1.3.0 | Async test support | FastAPI and the new agent code are async; pytest-asyncio allows `async def test_*` functions with full asyncio support; integrates with FastAPI's `AsyncClient`. |
| httpx | latest stable | Async HTTP client for FastAPI test client | FastAPI's recommended async test client (replaces Starlette TestClient for async tests); enables testing of endpoints without spinning up a real server. |

**Confidence:** MEDIUM — pytest and pytest-asyncio versions verified via PyPI. httpx version not verified (it is a FastAPI transitive dependency, already installed).

---

## Complete New Dependency List

Add to requirements.txt:

```
# Vector storage
pinecone==8.1.1
pinecone-text==0.11.0

# Date parsing
dateparser==1.4.0

# Async file I/O
aiofiles==25.1.0

# Data validation (may already be present as FastAPI transitive dep — pin explicitly)
pydantic==2.12.5

# Testing
pytest==9.0.2
pytest-asyncio==1.3.0
httpx
```

**Note on openai-agents:** Already in requirements.txt as `openai-agents` (unpinned). Pin it: `openai-agents==0.13.4`.

---

## Alternatives Considered

| Category | Recommended | Alternative | Why Not |
|----------|-------------|-------------|---------|
| Agentic framework | openai-agents 0.13.4 | LangChain Agents | Explicit project constraint to use OpenAI Agents SDK; LangChain adds heavy dependencies and a different abstraction model |
| Agentic framework | openai-agents 0.13.4 | LlamaIndex | Same — project constraint; LlamaIndex is optimized for document pipelines, not conversational agents |
| Sparse vectors | pinecone-sparse-english-v0 (neural) | rank_bm25 | rank_bm25 is unmaintained (last release 2022); produces raw scores, not Pinecone-compatible sparse vector format; would require custom integration code |
| Sparse vectors | pinecone-sparse-english-v0 (neural) | pinecone-text BM25Encoder | BM25Encoder requires corpus fitting at index time; neural sparse model outperforms BM25 by 23-44% on retrieval benchmarks and requires no corpus fitting |
| Dense embeddings | text-embedding-3-small (OpenAI) | sentence-transformers (local) | sentence-transformers requires PyTorch (~2GB); no benefit over OpenAI when the API key is already in use and cost is negligible at meeting-scale volumes |
| Date parsing | dateparser 1.4.0 | parsedatetime | parsedatetime has lower language coverage, less active maintenance, more verbose API for the same result |
| Date parsing | dateparser 1.4.0 | python-dateutil | dateutil handles structured dates well but is weak on natural language relative expressions like "last Wednesday" |
| Metadata storage | JSON files + pydantic | SQLite | PROJECT.md explicitly requires JSON files for auditability and portability; SQLite adds a dependency and migration story |
| Metadata storage | JSON files + pydantic | PostgreSQL/any DB | Same as SQLite — explicitly out of scope per requirements |
| Async file I/O | aiofiles | stdlib open() in asyncio | Blocking `open()` inside async handlers blocks the event loop; aiofiles is the established solution for non-blocking file ops in asyncio apps |

---

## Environment Variables to Add

```bash
# Pinecone
PINECONE_API_KEY=
PINECONE_INDEX_NAME=meeting-memory
PINECONE_CLOUD=aws
PINECONE_REGION=us-east-1

# Meeting metadata storage
MEETING_DATA_DIR=./meetings

# RAG tuning
HYBRID_ALPHA=0.7              # 0.0=keyword-only, 1.0=semantic-only
RAG_TOP_K=10                  # number of Pinecone results to retrieve before reranking
```

---

## Confidence Assessment

| Area | Confidence | Basis |
|------|------------|-------|
| openai-agents version (0.13.4) | HIGH | PyPI verified 2026-04-01 release |
| openai-agents API patterns | MEDIUM | Official docs fetched; SDK is ~1 year old so patterns are stable but evolving |
| pinecone version (8.1.1) | HIGH | PyPI verified 2026-04-02 release |
| Pinecone hybrid search (dotproduct + sparse_values) | HIGH | Official Pinecone docs fetched directly |
| pinecone-sparse-english-v0 superiority over BM25 | MEDIUM | Pinecone blog claims verified; independent benchmark not checked |
| pinecone-text version (0.11.0) | HIGH | PyPI verified; note: Beta status |
| dateparser version (1.4.0) | HIGH | PyPI verified 2026-03-26 release |
| text-embedding-3-small at 1536 dims | HIGH | OpenAI docs confirm; widely used |
| aiofiles version (25.1.0) | HIGH | PyPI verified 2025-10-09 release |
| pydantic version (2.12.5) | HIGH | PyPI verified |
| pytest version (9.0.2) | HIGH | PyPI verified 2025-12-06 release |
| pytest-asyncio version (1.3.0) | HIGH | PyPI verified 2025-11-10 release |
| rank_bm25 deprecation | HIGH | Last release 2022-02-16, confirmed unmaintained |

---

## Sources

- [openai-agents PyPI](https://pypi.org/project/openai-agents/)
- [OpenAI Agents SDK documentation](https://openai.github.io/openai-agents-python/)
- [OpenAI Agents SDK — Multi-agent orchestration](https://openai.github.io/openai-agents-python/multi_agent/)
- [OpenAI Agents SDK — Handoffs](https://openai.github.io/openai-agents-python/handoffs/)
- [pinecone PyPI](https://pypi.org/project/pinecone/)
- [Pinecone hybrid search docs](https://docs.pinecone.io/guides/search/hybrid-search)
- [Pinecone encode sparse vectors](https://docs.pinecone.io/guides/data/encode-sparse-vectors)
- [pinecone-sparse-english-v0 announcement](https://www.pinecone.io/learn/learn-pinecone-sparse/)
- [pinecone-text PyPI](https://pypi.org/project/pinecone-text/)
- [dateparser PyPI](https://pypi.org/project/dateparser/)
- [aiofiles PyPI](https://pypi.org/project/aiofiles/)
- [pydantic PyPI](https://pypi.org/project/pydantic/)
- [pytest PyPI](https://pypi.org/project/pytest/)
- [pytest-asyncio PyPI](https://pypi.org/project/pytest-asyncio/)
- [rank-bm25 PyPI](https://pypi.org/project/rank-bm25/)
- [FastAPI async testing guide](https://fastapi.tiangolo.com/advanced/async-tests/)
