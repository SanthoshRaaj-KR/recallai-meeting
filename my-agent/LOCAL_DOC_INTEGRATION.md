# Local-doc pipeline → Confluence integration (my-agent)

Runs the proven **local_doc_change** proposal pipeline against **Confluence**
pages after a meeting, inside `my-agent`. The local-doc pipeline logic is
**unchanged** — only an I/O adapter was added.

## What changed (3 new files + 1 wiring edit)

| File | Role |
|------|------|
| `local_doc_change/run_json.py` | **New, logic-free.** JSON in/out wrapper around the unchanged `run_pipeline`, so it can be invoked from another venv. |
| `my-agent/src/review_pipeline/confluence_proposal_adapter.py` | **New.** Picks meeting-relevant Confluence pages (reusing the existing RAG), materializes them as markdown, runs `run_json.py` as a subprocess in `local_doc_change/.venv`, maps each `LocalDocProposal` → my-agent `Proposal`. |
| `my-agent/src/recall_bridge.py` | **Edited.** Added `_propose()` which routes both the post-meeting job and the manual `/propose` endpoint through the adapter (toggle below). Nothing else changed. |
| `my-agent/_adapter_smoke.py` | **New.** Offline end-to-end test (stubs live Confluence). |

`local_doc_change/pipeline/`, `agents_local/`, `rag/` are **not modified**.

## Data flow
```
meeting transcript
  → ProposalPipeline._extract_meeting + Confluence RAG   (pick relevant pages)
  → fetch pages, write them as page_<id>.md in a temp folder
  → subprocess: local_doc_change/.venv python run_json.py   (UNCHANGED pipeline)
  → LocalDocProposal[]  → map page_<id>.md back to page_id
  → my-agent Proposal[]  → session_store → review cards → accept → Confluence
```

## Config (env, in `my-agent/.env.local`)
| Var | Default | Meaning |
|-----|---------|---------|
| `MY_AGENT_USE_LOCAL_DOC_PIPELINE` | `1` | `0` falls back to the original `ProposalPipeline.run`. |
| `LDOC_DIR` | sibling `local_doc_change/` | Pipeline location. |
| `LDOC_PYTHON` | `local_doc_change/.venv/Scripts/python.exe` | Interpreter with docling/faiss. |
| `MY_AGENT_LDOC_MAX_PAGES` | `16` | Max candidate pages materialized per meeting. |
| `MY_AGENT_LDOC_TIMEOUT` | `900` | Subprocess timeout (s). |

## How to test

### 1. Offline end-to-end (no Confluence creds needed) — proves the new code path
```bash
cd my-agent
./.venv/Scripts/python.exe _adapter_smoke.py
```
Expect: `RESULT: 3 proposal(s), 0 mis-mapped -> PASS`. This stubs the live RAG /
page-fetch and exercises materialize → subprocess → map against 3 real SmartHub
pages.

### 2. Runner only (the unchanged pipeline via its venv)
```bash
cd local_doc_change
# build a request.json with {transcript, doc_folder, session_id}; then:
LDOC_VECTOR_DB=faiss ./.venv/Scripts/python.exe run_json.py request.json out.json
```

### 3. Live (your Confluence + a real/simulated meeting)
Requires `my-agent/.env.local` with `ATLASSIAN_*`, `PINECONE_API_KEY` (the
Confluence review index the prev builder populated), and `OPENAI_API_KEY`.
1. Start the confluence service: `uv run uvicorn src.recall_bridge:app --port 8001`
2. Create/seed a session with a transcript (via the bot flow, or POST a session
   with `transcript`).
3. Trigger proposals:
   `POST /sessions/{id}/review/changes/propose` (body `{}`), **or** run a meeting
   so the post-meeting job fires automatically.
4. `GET /sessions/{id}/review/changes` → cards now come from the local-doc
   pipeline (`source: "local-doc-pipeline"`), each carrying a real `page_id`.

## Verified working on port 8001 (this session)

Config added to `my-agent/.env.local` (full cred set copied from `local_doc_change/.env.local`), plus:
```
MY_AGENT_RAG_INDEX=confluence-review-rag-v2     # the integrated-inference index (llama-text-embed-v2, 8747 vectors)
MY_AGENT_RAG_NAMESPACE=confluence-review
```
…and `pinecone` installed into the venv: `uv pip install --python .venv/Scripts/python.exe pinecone`.

Run + test:
```bash
cd my-agent
# 1. start the confluence service on 8001
./.venv/Scripts/python.exe -m uvicorn src.recall_bridge:app --host 127.0.0.1 --port 8001
# 2. seed a session (separate shell)
./.venv/Scripts/python.exe -c "import sys; sys.path.insert(0,'src'); from dotenv import load_dotenv; load_dotenv('.env.local'); import session_store as ss; ss.upsert('test-8001', {'session_id':'test-8001','status':'ended','transcript':[{'speaker':'P','text':'Change the Managed Threat Hunting add-on from 5 to 9 dollars per device per year.'}],'changes':[]})"
# 3. trigger proposals through the local-doc adapter
curl -s -X POST http://127.0.0.1:8001/sessions/test-8001/review/changes/propose -H "Content-Type: application/json" -d '{}'
```
Result observed: `HTTP 200`, `generated_count: 2`, proposals on **real page_id 49217537 ('Pricing & Plans')** with `source: "local-doc-pipeline"`.

Quick offline check (no creds): `./.venv/Scripts/python.exe _live_adapter_test.py` → `PASS`.

## RAG index note
The code default `confluence-review-rag` is the OLD 1536-dim OpenAI-embedding index (286 vectors) and rejects integrated-inference search. The prev builder's current knowledge is **`confluence-review-rag-v2`** (1024-dim, Pinecone `llama-text-embed-v2`, 8747 vectors) — set via `MY_AGENT_RAG_INDEX`.

## Known limitations (follow-ups, not blockers)
- **Live Confluence search REST = 403** (token/account access). Page *search* is served by the RAG instead; page *fetch-by-id* works via Rovo MCP. **Write-back** on accept still needs a Confluence-capable token.
- **Flattened RAG content:** the v2 index stores section text without markdown table structure, so proposal before/after lose table formatting and the editor is less surgical on tables than when reading real markdown. Sourcing full pages (when Confluence is reachable) restores fidelity.
- **Write-back fidelity for tables:** proposal `before/after` are markdown (from
  the materialized page). The existing accept→`update_page` path replaces text in
  Confluence **storage HTML** — fine for prose edits; table-structured sections
  may not match exactly on accept. Generation/mapping is correct; the apply step
  for tables needs an HTML-aware pass.
- **Deployment:** the subprocess uses `local_doc_change/.venv`. For container
  deploys, ship `local_doc_change` + its venv (or point `LDOC_PYTHON` at an env
  that has docling/faiss).
- **Recall is bounded by candidate selection:** only RAG-retrieved pages are
  materialized; raise `MY_AGENT_LDOC_MAX_PAGES` if changes target pages the RAG
  doesn't surface.
