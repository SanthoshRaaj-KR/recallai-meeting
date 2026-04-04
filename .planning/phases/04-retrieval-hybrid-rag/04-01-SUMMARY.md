---
phase: 04-retrieval-hybrid-rag
plan: 04-01
subsystem: storage/retrieval
tags: [pinecone, hybrid-search, alpha-weighting, reranking, tdd]
dependency_graph:
  requires: [storage/pinecone_client.py (Phase 02-02)]
  provides: [PineconeClient.query(alpha), PineconeClient.rerank(), PineconeClient.retrieve()]
  affects: [agents/retrieval_agent.py (Phase 04-02)]
tech_stack:
  added: []
  patterns: [alpha-weighted hybrid search, Pinecone neural reranking, TDD red-green-commit]
key_files:
  created: []
  modified:
    - storage/pinecone_client.py
    - tests/test_pinecone_client.py
decisions:
  - "alpha=0.7 default is research-backed for conversational meeting queries (dense/sparse balance)"
  - "rerank uses bge-reranker-v2-m3 model via pc.inference.rerank (Pinecone hosted)"
  - "retrieve() pipeline fixed at top_k=20 candidates → reranked to top_n=5 — calibration deferred post real-meeting data"
metrics:
  duration: "2 minutes"
  completed: "2026-04-04"
  tasks_completed: 2
  files_modified: 2
requirements_satisfied: [RETR-01]
---

# Phase 04 Plan 01: Alpha-Weighted Hybrid Query + Pinecone Reranker Summary

## One-liner

Alpha-weighted hybrid query (dense*alpha, sparse*(1-alpha)) with `bge-reranker-v2-m3` neural reranking via `query()`, `rerank()`, and `retrieve()` methods on `PineconeClient`.

## What Was Built

Extended `PineconeClient` in `storage/pinecone_client.py` with three changes:

1. **`query()` extended with `alpha: float = 0.7`** — backward-compatible new parameter. Dense embedding values are scaled by `alpha` and sparse values by `(1 - alpha)` before being passed to `index.query()`. Default of 0.7 preserves existing caller behavior.

2. **`rerank()` method added** — calls `self._pc.inference.rerank(model="bge-reranker-v2-m3", ...)` with documents built from `result["metadata"]["summary_text"]`. Returns list of dicts with `"id"`, `"score"`, `"metadata"`, `"rerank_score"` keys.

3. **`retrieve()` convenience method added** — full two-stage pipeline: `query(top_k=20)` followed by `rerank(top_n=5)`. All optional filters (`channel_id`, `start_ts`, `end_ts`) and `alpha` are forwarded to `query()`.

## TDD Discipline

- **RED commit** (`4ccfd72`): 15 new failing tests added across `TestQueryAlpha` (6), `TestRerank` (5), `TestRetrieve` (4). All failed with `TypeError: unexpected keyword argument 'alpha'` or `AttributeError`.
- **GREEN commit** (`463b2c3`): Implementation added — all 42 tests pass (27 pre-existing + 15 new), 1 skipped (live smoke).

## Task Commits

| Task | Description | Commit | Files |
|------|-------------|--------|-------|
| 1 (RED) | Failing tests for alpha, rerank, retrieve | `4ccfd72` | `tests/test_pinecone_client.py` |
| 2 (GREEN) | Implement alpha weighting, rerank, retrieve | `463b2c3` | `storage/pinecone_client.py` |

## Decisions Made

- **alpha=0.7 default** — research-backed for conversational/meeting corpus hybrid queries. Not yet empirically validated against real data; calibration planned post first 10 meetings indexed.
- **bge-reranker-v2-m3** — Pinecone's recommended cross-encoder reranker; no local model required.
- **top_k=20 → top_n=5 fixed defaults in retrieve()** — broad candidate pool before neural reranking; both overridable by caller.
- **Backward compatibility preserved** — `query()` callers that pass no `alpha` get 0.7 transparently; `jarvis.py` requires no changes.

## Verification Results

```
pytest tests/test_pinecone_client.py -v
42 passed, 1 skipped in 0.18s
```

All classes: `TestPineconeClientInit`, `TestEnsureIndexExists`, `TestEmbedDense`, `TestEmbedSparse`, `TestUpsertMeeting`, `TestQuery`, `TestQueryAlpha`, `TestRerank`, `TestRetrieve` — all pass.

## Deviations from Plan

None — plan executed exactly as written. TDD red-green-commit discipline followed.

## Known Stubs

None. All methods are fully implemented with real Pinecone SDK calls (mocked in tests).

## Self-Check: PASSED
