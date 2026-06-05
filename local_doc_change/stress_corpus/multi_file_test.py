"""Huge-transcript, many-file stress test.

Instead of one change per transcript (the needle tests), build ONE long meeting
transcript containing MANY changes spread across MANY of the 100 documents, run
the pipeline ONCE, and check that it produces the right cards for each change
(recall) without spurious edits to unrelated docs (precision).

Two sizes are run: a "BIG" transcript (~15 changes) and a "HUGE" transcript
(all anchors + the corpus-wide rename). Apply is never exercised here, so the
committed corpus is not mutated.
"""
from __future__ import annotations

import json, os, random, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / "../confluence_logic/.env")

import asyncio, openai
from rag.indexer import build_index
from pipeline.run import run_pipeline, PipelineConfig

HERE = Path("stress_corpus")
DOCS = str(HERE / "docs")
MAN = json.loads((HERE / "manifest.json").read_text(encoding="utf-8"))

def base(p): return os.path.basename(p).replace("\\", "/")
def num(s):
    import re
    m = re.search(r"[\d][\d,\.]*", s or ""); return m.group(0) if m else None

FILLER = [
    "Okay, moving on to the next item.", "Any questions before we continue? No? Good.",
    "Let's keep this quick, we have a lot to cover today.", "Right, so as discussed last week,",
    "Someone please take a note on that.", "Great, thanks for the update.",
    "We'll circle back on the other stuff later.", "That reminds me, lunch is at noon.",
    "Sorry, can everyone hear me? Okay.", "Let me share my screen for a second... never mind.",
    "This one's important so write it down.", "Procurement flagged this last sprint.",
    "Quick housekeeping item, then the real decisions.", "The board asked us to tighten this up.",
]

def build_transcript(anchors, include_rename, rng):
    parts = ["Alright team, end of quarter review. Lots of document updates coming "
             "out of this meeting, so listen up and we'll send the proposals around after."]
    items = []
    for a in anchors:
        if a["kind"] == "edit":
            items.append(a["transcript"])
        else:
            items.append(a["transcript"])
    if include_rename:
        items.append(MAN["rename_test"]["transcript"])
    rng.shuffle(items)
    for it in items:
        parts.append(rng.choice(FILLER))
        parts.append(it)
    parts.append("Okay that's everything. The system will draft the changes, review "
                 "them in the UI and accept what looks right. Thanks everyone.")
    return " ".join(parts)


def check(ps, anchors, include_rename):
    edits = [a for a in anchors if a["kind"] == "edit"]
    rems = [a for a in anchors if a["kind"] == "remove_named"]
    expected_keys = set()
    got = {"edit": 0, "remove": 0}
    missed = []
    for a in edits:
        wf, wh = base(a["file"]), a["section_heading"]
        expected_keys.add((wf, wh))
        hit = any(base(p.source_chunk.source_path) == wf and p.source_chunk.section_heading == wh
                  and (num(a["new_value"]) or "") in (p.after_content or "")
                  for p in ps)
        if hit: got["edit"] += 1
        else: missed.append(f"edit {a['fmt']} {wf} «{wh[:30]}»")
    for a in rems:
        wf, wh = base(a["file"]), a["section_heading"]
        expected_keys.add((wf, wh))
        hit = any(p.edit_type == "delete_section" and base(p.source_chunk.source_path) == wf
                  and wh.lower() in p.source_chunk.section_heading.lower() for p in ps)
        if hit: got["remove"] += 1
        else: missed.append(f"remove {a['fmt']} {wf} «{wh[:30]}»")

    brand_files = {base(d["file"]) for d in MAN["documents"] if d["has_brand"]}
    rename_cards = [p for p in ps if "helios" in (p.after_content or "").lower()]
    rename_docs = {base(p.source_chunk.source_path) for p in rename_cards}
    nonbrand_renamed = rename_docs - brand_files

    # spurious = edit/replace card not matching an expected edit target and not a rename
    spurious = []
    for p in ps:
        if "helios" in (p.after_content or "").lower():
            continue  # rename card
        k = (base(p.source_chunk.source_path), p.source_chunk.section_heading)
        if p.edit_type == "delete_section":
            ok = any(base(a["file"]) == k[0] and a["section_heading"].lower() in k[1].lower()
                     for a in rems)
        else:
            ok = k in expected_keys
        if not ok:
            spurious.append(f"{p.edit_type} {k[0]} «{k[1][:28]}»")
    return edits, rems, got, missed, rename_docs, brand_files, nonbrand_renamed, spurious


async def run_case(name, anchors, include_rename, seed):
    rng = random.Random(seed)
    transcript = build_transcript(anchors, include_rename, rng)
    cfg = PipelineConfig(session_id=name, doc_folder=DOCS, use_embeddings=True,
                         rerank=False, contextual_retrieval=False)
    ps = await run_pipeline(transcript, cfg)
    edits, rems, got, missed, rdocs, bfiles, nonbrand, spurious = check(ps, anchors, include_rename)
    print(f"\n{'='*68}\n{name}")
    print(f"transcript: {len(transcript.split())} words | "
          f"expected: {len(edits)} edits + {len(rems)} removals"
          f"{' + rename' if include_rename else ''}")
    print(f"{'='*68}")
    print(f"  cards produced            : {len(ps)}")
    print(f"  edit changes captured     : {got['edit']}/{len(edits)}")
    print(f"  removals captured         : {got['remove']}/{len(rems)}")
    if include_rename:
        print(f"  rename docs covered       : {len(rdocs & bfiles)}/{len(bfiles)}  "
              f"(non-brand wrongly renamed: {len(nonbrand)})")
    print(f"  spurious/wrong cards      : {len(spurious)}")
    for m in missed[:10]:
        print(f"      MISSED: {m}")
    for s in spurious[:10]:
        print(f"      SPURIOUS: {s}")
    return got, len(edits), len(rems), len(spurious)


async def main():
    client = openai.AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    build_index(DOCS, use_embeddings=True, contextual_retrieval=False, openai_client=client)

    edits = [a for a in MAN["anchors"] if a["kind"] == "edit"]
    rems = [a for a in MAN["anchors"] if a["kind"] == "remove_named"]

    # BIG: 10 edits (spread formats) + 3 removals, no rename
    big = edits[:10] + rems[:3]
    await run_case("BIG multi-file transcript", big, include_rename=False, seed=1)

    # HUGE: all 16 edits + all 4 removals + corpus-wide rename
    huge = edits + rems
    await run_case("HUGE everything transcript", huge, include_rename=True, seed=2)

    print("\nDONE")


if __name__ == "__main__":
    asyncio.run(main())
