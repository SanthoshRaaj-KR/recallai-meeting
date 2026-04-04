---
phase: 02-storage-foundation
plan: "02"
subsystem: storage
tags: [pinecone, hybrid-search, vector-database, openai-embeddings, tdd]
dependency_graph:
  requires: [02-01]
  provides: [PineconeClient]
  affects: [Phase-3-ingestion, Phase-4-retrieval]
tech_stack:
  added: []
  patterns: [pinecone-sdk-v8-hybrid, openai-embeddings, dotproduct-metric, flat-metadata-schema, tdd-red-green]
key_files:
  created:
    - storage/pinecone_client.py
    - tests/test_pinecone_client.py
  modified:
    - .env.example
decisions:
  - "Sparse vector attributes on SparseEmbedding SDK v8 object are sparse_indices (list[int]) and sparse_values (list[float]) as direct attributes — not a nested .sparse_values sub-object with .index/.value — plan pseudocode was incorrect; implementation adapted to real SDK shape"
  - "metric=dotproduct hardcoded in ensure_index_exists — wrong metric requires full re-ingestion, no config override allowed"
  - "Sparse embeddings use pinecone-sparse-english-v0 via pc.inference.embed() — no local BM25 fitting required (satisfies INFRA-04)"
  - "input_type=passage for upsert, input_type=query for query — per Pinecone inference API"
  - "Flat metadata schema only — channel_id, start_ts, participants, decisions, etc. as top-level scalars for server-side Pinecone filtering"
metrics:
  duration: "5 minutes"
  completed_date: "2026-04-04"
  tasks_completed: 1
  files_created: 2
  files_modified: 1
---

# Phase 2 Plan 2: PineconeClient Hybrid Vector Storage Summary

**One-liner:** PineconeClient wrapping Pinecone SDK v8 with dotproduct index creation, hybrid upsert (dense via text-embedding-3-small + sparse via pinecone-sparse-english-v0 hosted inference), and channel_id/$gte/$lte metadata-filtered query

## What Was Built

### storage/pinecone_client.py

`PineconeClient` class with four public methods:

- `ensure_index_exists()` — creates index with `metric="dotproduct"`, `dimension=1536`, `vector_type="dense"` in ServerlessSpec; idempotent (skips if name already in `list_indexes()`)
- `_embed_dense(text)` — calls `openai.embeddings.create(model="text-embedding-3-small")`, returns `list[float]` of length 1536
- `_embed_sparse(text, input_type)` — calls `pc.inference.embed(model="pinecone-sparse-english-v0")`, returns `{"indices": list[int], "values": list[float]}`
- `upsert_meeting(record: MeetingRecord)` — generates dense+sparse from `summary_text`, stores flat metadata dict, calls `index.upsert()`
- `query(query_text, channel_id, start_ts, end_ts, top_k)` — builds filter with `{"channel_id": {"$eq": ...}}` and `{"start_ts": {"$gte": ..., "$lte": ...}}`, calls `index.query()` with both vector types

Key design constraints enforced:
- `metric="dotproduct"` hardcoded — hybrid search breaks silently with other metrics (Pitfall 1)
- Metadata is flat scalars and lists only — no nested objects (Pitfall 7)
- Pinecone hosted sparse inference used — no corpus fitting risk (Pitfall 2 / INFRA-04)
- `input_type` differs between upsert (passage) and query (query) — per Pinecone docs

### tests/test_pinecone_client.py

28 test functions across 6 test classes:

- `TestPineconeClientInit` (5 tests) — api_key stored, cloud/region stored, defaults, OpenAI key handling
- `TestEnsureIndexExists` (3 tests) — dotproduct metric, dimension=1536, vector_type=dense, no-op if exists
- `TestEmbedDense` (2 tests) — OpenAI model, return length 1536
- `TestEmbedSparse` (3 tests) — SPARSE_MODEL used, indices/values dict returned, input_type parameter
- `TestUpsertMeeting` (5 tests) — upsert called, id=meeting_id, dense+sparse structure, flat metadata, summary_text source
- `TestQuery` (9 tests) — calls index.query, channel_id $eq, date range $gte/$lte, no filter without constraints, combined filter, returns list[dict], query input_type, top_k, include_metadata
- `test_live_hybrid_smoke` (1 test) — gated on `PINECONE_API_KEY`, runs full create/upsert/query cycle

### .env.example

Added four Pinecone environment variables:
- `PINECONE_API_KEY`
- `PINECONE_INDEX_NAME`
- `PINECONE_CLOUD`
- `PINECONE_REGION`

## Verification Results

```
python -m pytest tests/test_pinecone_client.py -v -k "not live"
27 passed, 1 deselected in 0.28s

python -m pytest tests/ -v -k "not live" --tb=short
65 passed, 1 deselected in 0.33s

python -c "from storage.pinecone_client import PineconeClient; print('OK')" -> OK
grep 'dotproduct' storage/pinecone_client.py -> found (INDEX_METRIC = "dotproduct")
grep 'pinecone-sparse-english-v0' storage/pinecone_client.py -> found (SPARSE_MODEL = ...)
```

## TDD Commits

| Phase | Commit | Description |
|-------|--------|-------------|
| RED   | 9fdbc95 | test(02-02): add failing tests for PineconeClient hybrid upsert and query |
| GREEN | 95aea66 | feat(02-02): implement PineconeClient with hybrid upsert and filtered query |

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Corrected sparse embedding attribute names from plan pseudocode**
- **Found during:** Task 1 implementation
- **Issue:** Plan's implementation template iterated over `embedding.sparse_values` as a list of objects with `.index` and `.value` attributes. The actual Pinecone SDK v8 `SparseEmbedding` model has `sparse_indices: list[int]` and `sparse_values: list[float]` as direct attributes on the embedding object.
- **Fix:** Changed `_embed_sparse()` to access `embedding.sparse_indices` and `embedding.sparse_values` directly, verified against `/opt/anaconda3/envs/meetagents/lib/python3.11/site-packages/pinecone/core/openapi/inference/model/sparse_embedding.py`
- **Files modified:** `storage/pinecone_client.py` (GREEN commit), `tests/test_pinecone_client.py` (RED commit — mocks use `mock_emb.sparse_indices` and `mock_emb.sparse_values`)
- **Commit:** 95aea66

## Known Stubs

None — PineconeClient is fully wired. Dense embeddings call OpenAI API, sparse embeddings call Pinecone inference API, all methods interact with real SDK objects (mocked in tests). No placeholder data or hardcoded empty returns.

## Self-Check: PASSED

- storage/pinecone_client.py: FOUND
- tests/test_pinecone_client.py: FOUND
- .env.example (PINECONE_API_KEY): FOUND
- Commit 9fdbc95: FOUND
- Commit 95aea66: FOUND
- 27 unit tests: PASS (0 failures)
- All 65 tests (no regressions): PASS
