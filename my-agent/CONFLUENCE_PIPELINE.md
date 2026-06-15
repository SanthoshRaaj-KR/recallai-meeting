# Local-doc proposal logic → Confluence (my-agent, Pinecone-hybrid)

The proven **local_doc_change** proposal logic now runs **natively inside
`my-agent`** to generate Confluence change proposals after a meeting. The
proposal agents (intent extraction → evaluation → editing → verification) are
**vendored byte-for-byte** from the confluence branch; only the **retrieval**
layer is swapped to **Pinecone-native** embeddings — nothing runs on a local
vector store, so it scales with the product.

## Where the code lives

```
my-agent/src/review_pipeline/confluence_pipeline/
  models.py            # ChunkRecord, LocalDocIntent, ConfluenceProposal (vendored)
  llm_runtime.py       # guarded_run throttle/retry          (vendored, unchanged)
  intent_extraction.py # IntentExtractionAgent                (vendored, unchanged)
  evaluation.py        # EvaluationAgent                      (vendored, unchanged)
  editor.py            # LocalDocEditorAgent                  (vendored, unchanged)
  verifier.py          # VerifierAgent                        (vendored, unchanged)
  structural.py        # classify_kind / is_cross_cutting …   (vendored, unchanged)
  chunker.py           # markdown section chunker (no docling)        [new]
  retrieval.py         # PineconeHybridIndex (dense+sparse+rerank)    [new]
  pipeline.py          # ported run.py edit path over Pinecone        [new]
  reindex.py           # one-shot corpus → Pinecone indexer           [new]
confluence_proposal_adapter.py  # maps ConfluenceProposal → my-agent Proposal (+page_id)
```

`local_doc_change/` is **not** imported or modified at runtime (the confluence
branch remains the source of truth for the agents).

## Retrieval: Pinecone built-in embeddings, no local vector store

| Stage | Model | Where |
|-------|-------|-------|
| dense embedding  | `llama-text-embed-v2`        | Pinecone integrated inference (server-side) |
| sparse embedding | `pinecone-sparse-english-v0` | Pinecone integrated inference (the managed BM25/lexical half) |
| fusion           | Reciprocal Rank Fusion        | in `retrieval.py` |
| rerank           | `bge-reranker-v2-m3`          | Pinecone inference |

Two integrated indexes (`confluence-corpus-dense`, `confluence-corpus-sparse`)
in namespace `smarthub`. If the sparse index can't be created (e.g. the Pinecone
project's serverless-index cap is reached), retrieval **degrades gracefully to
dense + rerank**.

## Data flow

```
meeting transcript
  → confluence_pipeline.propose(retriever = PineconeHybridIndex)
      IntentExtractionAgent → per-intent hybrid retrieve (dense+sparse+rerank)
      → EvaluationAgent (≥0.70, + field-label / phrase-overlap recall floors)
      → LocalDocEditorAgent → VerifierAgent → drop no-ops / unfulfilled
  → ConfluenceProposal[]   (source_chunk.source_path = corpus filename)
  → map filename → Confluence page_id via local_doc_change/corpus_page_map.json
  → my-agent Proposal[]  (only pages with a real page_id survive)
  → session_store → review cards (sync-sage-bot) → accept → Confluence
```

## Config (env, `my-agent/.env.local`)

| Var | Default | Meaning |
|-----|---------|---------|
| `MY_AGENT_USE_LOCAL_DOC_PIPELINE` | `1` | `0` falls back to the built-in `ProposalPipeline.run`. |
| `MY_AGENT_LDOC_DENSE_INDEX`  | `confluence-corpus-dense`  | dense integrated index |
| `MY_AGENT_LDOC_SPARSE_INDEX` | `confluence-corpus-sparse` | sparse integrated index (optional) |
| `MY_AGENT_LDOC_NAMESPACE`    | `smarthub`                 | Pinecone namespace |
| `MY_AGENT_LDOC_PAGE_MAP`     | `../local_doc_change/corpus_page_map.json` | filename → page_id map |
| `MY_AGENT_LDOC_CREATE_INDEX` | `1` | auto-create the integrated indexes if missing |

## Index the corpus (one-time / on corpus change)

```bash
cd my-agent
PYTHONPATH=src ./.venv/Scripts/python.exe -m review_pipeline.confluence_pipeline.reindex \
    ../local_doc_change/stress_corpus/docs \
    ../local_doc_change/corpus_page_map.json
```
Chunks every markdown/text file, server-side-embeds them into the dense (and
sparse, if available) indexes, tagging each chunk with its Confluence `page_id`.

## Test the proposal logic (no bot needed)

```bash
cd my-agent
PYTHONIOENCODING=utf-8 PYTHONPATH=src ./.venv/Scripts/python.exe _native_test.py
```
Runs the user's SmartHub transcript over the combined corpus and prints extracted
intents, each proposal (page + section + diff), and any intent that produced no
card. Or via HTTP on the confluence service (port 8001):
`POST /review/test/propose  {"transcript": "..."}`.

## Known limitations / follow-ups

- **Sparse half needs a free Pinecone index slot.** The project is at the
  serverless-index cap (5); until a slot is freed (delete an obsolete index) or
  the plan is upgraded, retrieval runs **dense + rerank** (still strong on exact
  terms via the cross-encoder reranker).
- **Renames / corpus-wide removals** are not ported (they need the whole-corpus
  chunk list); **edits and in-section additions** are covered.
- **Write-back fidelity for tables:** proposal before/after are markdown; the
  accept→Confluence apply step is the existing engineer's mechanism.
