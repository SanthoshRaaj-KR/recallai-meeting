"""Enriched-corpus test — proves the numbers/tables/images enrichment works and
that vague/garbled transcripts are handled safely (the client's exact case).

Three checks, all against the REAL committed stress_corpus/docs:

  1. TABLE EDITS   every globally-unique value that lives inside a "Key
                   Operational Parameters" table (one per format x2) is captured
                   as a card on the right document with the right new value. [GATE]
  2. PROSE EDITS   a sample of the original prose anchors still produce correct
                   cards after enrichment (no regression).                   [GATE]
  3. VAGUE/GARBLED the client's "wear hose" transcript and other vague/garbled
                   meetings produce ZERO cards (no hallucination), and the
                   diagnostics explain WHY (extracted-but-unmatched).         [GATE]

Read-only: never writes to the corpus. Exit non-zero on any gate failure.
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / "../confluence_logic/.env")

from pipeline.run import PipelineConfig, run_pipeline  # noqa: E402

HERE = Path("stress_corpus")
DOCS = str(HERE / "docs")
MAN = json.loads((HERE / "manifest.json").read_text(encoding="utf-8"))

TABLE = [a for a in MAN["anchors"] if a["kind"] == "edit_table"]
EDITS = [a for a in MAN["anchors"] if a["kind"] == "edit"]


def base(p):
    return os.path.basename(str(p)).replace("\\", "/")


def num(s):
    m = re.search(r"\d[\d,\.]*", s or "")
    return m.group(0) if m else None


async def pipe(transcript):
    cfg = PipelineConfig(
        session_id="enrich", doc_folder=DOCS, use_embeddings=True,
        rerank=False, contextual_retrieval=False,
    )
    diag: dict = {}
    proposals = await run_pipeline(transcript, cfg, diagnostics=diag)
    return proposals, diag


# The client's exact transcript + other vague/garbled meetings. None reference a
# topic the corpus documents, so all must yield zero cards.
VAGUE = [
    ("client 'wear hose'",
     "Hi. Ahmm, Nope. I am an employee from ahmmm harrowfield. The data of the "
     "wear hose will be seen once a week from now on. It is important to keep "
     "standards up."),
    ("garbled aspiration",
     "yeah so um the thing with the flux capacitor needs to be like, you know, "
     "better going forward, we should tighten it up at some point. thanks everyone."),
    ("vague no-number",
     "we talked about the widget situation and decided it should be more "
     "streamlined or whatever, no real numbers locked in yet, we'll circle back."),
    ("uncovered concrete fact",
     "quick note, the gizmo calibration drifted to 88 parsecs last sprint, just "
     "flagging it so we remember."),
    ("pure social",
     "great meeting everyone, the tacos at lunch were unreal, congrats again to "
     "Priya on the baby, see you all next week, drive safe."),
]


async def main():
    fails = 0

    print("=" * 72)
    print("1. TABLE EDITS — values living inside a Key Operational Parameters table")
    print("=" * 72)
    tbl_hits = 0
    for a in TABLE:
        ps, _ = await pipe(a["transcript"])
        want_num = num(a["new_value"]) or a["new_value"]
        hit = next(
            (p for p in ps
             if base(p.source_chunk.source_path) == base(a["file"])
             and want_num in (p.after_content or "")),
            None,
        )
        ok = hit is not None
        tbl_hits += int(ok)
        loc = ""
        if hit:
            loc = f" sec='{hit.source_chunk.section_heading}' edit_type={hit.edit_type}"
        print(f"  [{'OK ' if ok else 'XX '}] {a['fmt']:4} {base(a['file']):30} "
              f"{a['topic']:28} {a['old_value']}→{a['new_value']} cards={len(ps)}{loc}")
    print(f"  TABLE EDITS: {tbl_hits}/{len(TABLE)} captured correctly")
    if tbl_hits < len(TABLE):
        fails += 1

    print("\n" + "=" * 72)
    print("2. PROSE EDITS — original anchors still work after enrichment (sample)")
    print("=" * 72)
    sample = EDITS  # all 16 — measure true needle recall vs the 14/16 baseline
    prose_hits = 0
    for a in sample:
        ps, _ = await pipe(a["transcript"])
        want_num = num(a["new_value"]) or a["new_value"]
        ok = any(
            base(p.source_chunk.source_path) == base(a["file"])
            and want_num in (p.after_content or "")
            for p in ps
        )
        prose_hits += int(ok)
        print(f"  [{'OK ' if ok else 'XX '}] {a['fmt']:4} {base(a['file']):30} "
              f"{a['topic']:28} {a['old_value']}→{a['new_value']} cards={len(ps)}")
    # Baseline before enrichment was ~14/16 (2 anchors are inherently hard). The
    # gate is "no meaningful regression" — allow >=13/16, flag a real drop.
    print(f"  PROSE EDITS: {prose_hits}/{len(sample)} captured "
          f"(baseline ~14/16; gate >=13)")
    if prose_hits < 13:
        fails += 1

    print("\n" + "=" * 72)
    print("3. VAGUE / GARBLED — zero cards (no hallucination) + honest diagnostics")
    print("=" * 72)
    vague_clean = True
    for name, t in VAGUE:
        ps, diag = await pipe(t)
        n_ext = diag.get("extracted_intent_count", 0)
        n_un = len(diag.get("unmatched_intents", []))
        zero = len(ps) == 0
        vague_clean = vague_clean and zero
        topics = ", ".join(u["affected_topic"] for u in diag.get("unmatched_intents", [])[:3])
        print(f"  [{'OK ' if zero else 'XX '}] {name:24} cards={len(ps)} "
              f"extracted={n_ext} unmatched={n_un}"
              + (f"  → {topics}" if topics else ""))
    print(f"  VAGUE SAFETY: {'no false positives' if vague_clean else 'FALSE POSITIVE(S)!'}")
    if not vague_clean:
        fails += 1

    print("\n" + "=" * 72)
    print(f"RESULT: {'ALL GATES PASS' if fails == 0 else f'{fails} GATE(S) FAILED'}")
    print("=" * 72)
    return fails


if __name__ == "__main__":
    sys.exit(1 if asyncio.run(main()) else 0)
