"""Scale test: 100-doc mixed-format corpus, needle-in-haystack RAG + precision.

Measures (writes progress to scale_progress.txt with flush so it can be tailed):
  - index build time + chunk count over 100 docs (RAG at scale)
  - EDIT anchors via BLIND transcripts (no document named — pure content
    retrieval): did the change land on the exact right file+section among 100
    docs? was the value changed correctly? any spurious change to another doc?
  - REMOVE-named anchors: did the right unique section get a delete_section?
  - corpus-wide RENAME: coverage across brand docs + non-brand docs untouched
  - per-format apply-to-disk (md / txt / docx / odt) on temp copies
"""
from __future__ import annotations

import json, os, re, shutil, sys, tempfile, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / "../confluence_logic/.env")

from rag.indexer import build_index
from pipeline.run import run_pipeline, PipelineConfig
from pipeline.safe_apply import SafeApply
import openai

HERE = Path("stress_corpus")
DOCS = str(HERE / "docs")
MAN = json.loads((HERE / "manifest.json").read_text(encoding="utf-8"))
PROG = open(HERE / "scale_progress.txt", "w", encoding="utf-8", buffering=1)

def log(msg):
    PROG.write(msg + "\n"); PROG.flush()
    print(msg, flush=True)

def base(p): return os.path.basename(p).replace("\\", "/")
def num(s):
    m = re.search(r"[\d][\d,\.]*", s or ""); return m.group(0) if m else None

async def pipe(transcript):
    cfg = PipelineConfig(session_id="scale", doc_folder=DOCS, use_embeddings=True,
                         rerank=False, contextual_retrieval=False)
    return await run_pipeline(transcript, cfg)


async def main():
    import asyncio
    log("="*70)
    log("SCALE TEST — 100 documents, mixed formats")
    log("="*70)

    # ── Index build (RAG at scale) ────────────────────────────────────────────
    t0 = time.time()
    client = openai.AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    index = build_index(DOCS, use_embeddings=True, contextual_retrieval=False, openai_client=client)
    dt = time.time() - t0
    from collections import Counter
    fmt_chunks = Counter(c.source_format for c in index.chunks)
    docs_seen = len({c.source_path for c in index.chunks})
    log(f"\nINDEX: {len(index.chunks)} chunks from {docs_seen} docs in {dt:.1f}s")
    log(f"       chunks by format: {dict(fmt_chunks)}")
    log(f"       avg chunks/doc: {len(index.chunks)/max(1,docs_seen):.1f}")

    edit_anchors = [a for a in MAN["anchors"] if a["kind"] == "edit"]
    rem_anchors = [a for a in MAN["anchors"] if a["kind"] == "remove_named"]

    # ── EDIT anchors — BLIND needle test ──────────────────────────────────────
    log("\n" + "-"*70)
    log(f"EDIT needle test (BLIND transcripts, no doc named) — {len(edit_anchors)} anchors")
    log("-"*70)
    hit = correct = spurious_runs = 0
    per_fmt = Counter(); per_fmt_hit = Counter()
    apply_samples = {}  # fmt -> (file, heading, after_content)
    for a in edit_anchors:
        ps = await pipe(a["transcript_blind"])
        want_file = base(a["file"]); want_head = a["section_heading"]
        target = [p for p in ps if base(p.source_chunk.source_path) == want_file
                  and p.source_chunk.section_heading == want_head]
        others = [p for p in ps if not (base(p.source_chunk.source_path) == want_file
                  and p.source_chunk.section_heading == want_head)]
        rag = bool(target)
        ok = rag and (num(a["new_value"]) or "") in (target[0].after_content or "") \
             and (num(a["old_value"]) or "X") not in (target[0].after_content or "")
        per_fmt[a["fmt"]] += 1
        if rag: hit += 1; per_fmt_hit[a["fmt"]] += 1
        if ok: correct += 1
        if others: spurious_runs += 1
        if rag and a["fmt"] not in apply_samples:
            apply_samples[a["fmt"]] = (target[0].source_chunk.source_path, want_head,
                                       target[0].after_content)
        log(f"  [{a['fmt']:4s}] {'HIT ' if rag else 'MISS'} "
            f"{'val-ok' if ok else 'val-? '} extra={len(others)}  "
            f"topic={a['topic'][:34]:34s} ({a['old_value']}->{a['new_value']})")
    log(f"\n  RAG hit (right file+section): {hit}/{len(edit_anchors)}")
    log(f"  value applied correctly      : {correct}/{len(edit_anchors)}")
    log(f"  runs with ANY extra card     : {spurious_runs}/{len(edit_anchors)}")
    log(f"  hit by format: " + ", ".join(f"{k}={per_fmt_hit[k]}/{per_fmt[k]}" for k in sorted(per_fmt)))

    # ── REMOVE-named anchors ──────────────────────────────────────────────────
    log("\n" + "-"*70)
    log(f"REMOVE-named test — {len(rem_anchors)} anchors")
    log("-"*70)
    rem_hit = 0
    for a in rem_anchors:
        ps = await pipe(a["transcript"])
        want_file = base(a["file"]); want_head = a["section_heading"]
        got = [p for p in ps if p.edit_type == "delete_section"
               and base(p.source_chunk.source_path) == want_file
               and want_head.lower() in p.source_chunk.section_heading.lower()]
        wrong_del = [p for p in ps if p.edit_type == "delete_section" and p not in got]
        if got: rem_hit += 1
        log(f"  [{a['fmt']:4s}] {'HIT ' if got else 'MISS'} wrong_deletes={len(wrong_del)}  «{want_head}»")
    log(f"\n  removal hit: {rem_hit}/{len(rem_anchors)}")

    # ── Corpus-wide RENAME ────────────────────────────────────────────────────
    log("\n" + "-"*70)
    rt = MAN["rename_test"]
    log(f"RENAME test — '{rt['brand']}' -> '{rt['new_value']}' across {rt['doc_count']} brand docs")
    log("-"*70)
    ps = await pipe(rt["transcript"])
    renamed_files = {base(p.source_chunk.source_path) for p in ps if rt["new_value"].split()[0].lower() in (p.after_content or "").lower()}
    brand_files = {base(d["file"]) for d in MAN["documents"] if d["has_brand"]}
    nonbrand_renamed = renamed_files - brand_files
    log(f"  rename cards: {len(ps)} | distinct brand docs renamed: {len(renamed_files & brand_files)}/{len(brand_files)}")
    log(f"  non-brand docs wrongly renamed: {len(nonbrand_renamed)}")

    # ── Per-format apply-to-disk (temp copies) ────────────────────────────────
    log("\n" + "-"*70)
    log("APPLY-to-disk per format (temp copies)")
    log("-"*70)
    applier = SafeApply(audit_dir=tempfile.mkdtemp(prefix="scale_audit_"))
    for fmt, (src, head, after) in sorted(apply_samples.items()):
        tmp = Path(tempfile.mkdtemp(prefix=f"apply_{fmt}_")) / base(src)
        shutil.copy2(src, tmp)
        try:
            applier.apply(str(tmp), head, after, "scale", "x", edit_type="replace")
            # re-read to confirm
            from rag.chunker import chunk_document
            chunks = chunk_document(str(tmp))
            sec = next((c for c in chunks if c.section_heading == head), None)
            ok = sec is not None and (num(after) or "") in sec.content
            log(f"  [{fmt:4s}] {'WROTE-OK' if ok else 'WROTE-?? '}  {base(src)}  «{head}»")
        except Exception as e:
            log(f"  [{fmt:4s}] ERROR {e}")
        finally:
            shutil.rmtree(tmp.parent, ignore_errors=True)

    log("\n" + "="*70)
    log(f"SUMMARY: RAG {hit}/{len(edit_anchors)} hit, {correct}/{len(edit_anchors)} value-correct | "
        f"removals {rem_hit}/{len(rem_anchors)} | "
        f"rename {len(renamed_files & brand_files)}/{len(brand_files)} ({len(nonbrand_renamed)} wrong) | "
        f"apply-formats {len(apply_samples)}/4")
    log("="*70)
    PROG.close()

import asyncio
asyncio.run(main())
