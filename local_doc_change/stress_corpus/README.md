# Local-Doc Pipeline — 100-Document Scale Corpus

A synthetic, mixed-format corpus for stress-testing the local-document change
pipeline at scale (RAG retrieval, per-format chunking/write-back, precision).

## Contents

- **`docs/`** — 100 generated documents, 25 each of `.md` / `.txt` / `.docx` /
  `.odt`, 15–80 pages each (~4,800 chunks total, ~48 sections/doc). Realistic
  multi-section policy/handbook prose for 8 domains (security, HR, engineering,
  support, product, finance, IT, data governance).
- **`manifest.json`** — ground truth. Every "anchor" records the file, format,
  the exact chunked section heading, the value to change, and a ready-made
  meeting transcript (both a doc-named and a *blind* variant). Also records the
  corpus-wide rename test (a shared brand planted across 14 docs).
- **`generate_corpus.py`** — deterministic (seeded) generator. Re-run to
  regenerate identical docs.
- **`scale_test.py`** — the test battery.

## Anchors (ground-truth test cases)

- **16 edit anchors** — a globally-unique value (e.g. *cold-storage archival
  window = 47 days*) planted in a middle paragraph of one section, spread evenly
  across all 4 formats. Tested with **blind** transcripts (no document named) to
  stress pure-content retrieval.
- **4 remove-named anchors** — a uniquely-named section (e.g. *Legacy Fax Intake
  Procedure*) to delete, one per format.
- **1 corpus-wide rename** — `Vantcorex Robotics → Helios Automata` across the 14
  brand documents.

## Running

```bash
# from local_doc_change/ with OPENAI_API_KEY set
python stress_corpus/generate_corpus.py     # (re)generate docs + manifest
python stress_corpus/scale_test.py          # build index + run battery
# progress streams to stress_corpus/scale_progress.txt
```

The first index build embeds ~4,800 chunks (batched, ~50 API calls) and is
cached on disk; subsequent runs reuse it.

## What it measures

- **Index at scale** — chunk count + build time across all formats.
- **RAG needle hit** — for each anchor, did the change land on the exact right
  file + section among 100 docs, with the value applied and no spurious edits to
  the other 99?
- **Removals** — did the correct unique section get a `delete_section`?
- **Rename coverage** — how many brand docs renamed, and were any non-brand docs
  wrongly touched?
- **Per-format apply-to-disk** — md / txt / docx / odt write-back on temp copies.
