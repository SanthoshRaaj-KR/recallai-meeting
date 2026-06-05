"""Cross-cutting change test: ONE change that must propagate across MANY files,
buried in a big, noisy meeting transcript full of unimportant chatter.

Two cross-cutting changes are exercised inside one long noisy transcript:
  - a value edit ("change the mandatory security re-attestation window from 27
    days to 90 days in every document") that is planted identically in 18 docs
  - the corpus-wide rename (Vantcorex Robotics -> Helios Automata, 14 docs)

Measures: how many of the affected files get a correct card, and whether the
heavy noise produces spurious cards.
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

# A pile of unimportant, non-document chatter to bury the real instruction in.
NOISE = [
    "Morning everyone, grab a coffee, we'll start in a sec.",
    "Did everyone see the offsite photos? Hilarious.",
    "The parking garage is closed Thursday for cleaning, use the lot across the street.",
    "Reminder the all-hands is moved to 3pm, not 2.",
    "Q3 numbers looked solid, shout out to the sales team.",
    "Someone left a blue water bottle in the kitchen, claim it.",
    "We're trialing a new standing-desk vendor, let facilities know if you want one.",
    "The cafeteria is doing taco Tuesday again, big news I know.",
    "Slack was down for ten minutes this morning, it's back.",
    "Let's keep the meeting to thirty minutes today, people have lunch plans.",
    "Oh and the holiday calendar got published, check the wiki.",
    "The new hire cohort starts Monday, say hi when you see them.",
    "Travel approvals are a bit backed up, be patient with finance.",
    "Can someone fix the projector in room 4B, it flickers.",
    "We hit a million sessions last week, pretty cool milestone.",
    "Don't forget to submit your expense reports before month end.",
    "The gym membership perk is renewing, no action needed.",
    "Weather's supposed to be nice this weekend, finally.",
    "I think we're recording this, so wave hi to the recording.",
    "Okay focus everyone, just a couple real items today.",
]


def build_noisy_transcript(rng, include_rename=True):
    parts = ["Alright, quarterly housekeeping plus one compliance decision. Mostly "
             "chatter, but there's a real change in here, so listen for it."]
    # sprinkle lots of noise
    pool = list(NOISE)
    rng.shuffle(pool)
    items = pool[:14]
    # insert the cross-cutting edit somewhere in the middle
    insert_at = rng.randint(4, 9)
    items.insert(insert_at, MAN["cross_cutting"]["transcript"])
    if include_rename:
        items.insert(rng.randint(insert_at + 1, len(items)), MAN["rename_test"]["transcript"])
    parts.extend(items)
    parts.append("That's it. The non-document stuff doesn't need tracking, just the "
                 "compliance change. Thanks all, see you at lunch.")
    return " ".join(parts)


async def main():
    client = openai.AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    build_index(DOCS, use_embeddings=True, contextual_retrieval=False, openai_client=client)

    cc = MAN["cross_cutting"]
    cc_files = {base(f) for f in cc["files"]}
    brand_files = {base(d["file"]) for d in MAN["documents"] if d["has_brand"]}

    rng = random.Random(7)
    transcript = build_noisy_transcript(rng, include_rename=True)
    print(f"transcript: {len(transcript.split())} words "
          f"(1 cross-cutting edit across {len(cc_files)} files + rename across {len(brand_files)})")

    cfg = PipelineConfig(session_id="crossfile", doc_folder=DOCS, use_embeddings=True,
                         rerank=False, contextual_retrieval=False)
    ps = await run_pipeline(transcript, cfg)

    # cross-cutting edit: a card on a cc file whose after has the new value (90) and not old (27)
    cc_hit_files = set()
    for p in ps:
        f = base(p.source_chunk.source_path)
        if f in cc_files and "90" in (p.after_content or "") and "27" not in (p.after_content or ""):
            cc_hit_files.add(f)
    # rename
    rn_files = {base(p.source_chunk.source_path) for p in ps if "helios" in (p.after_content or "").lower()}
    # spurious: any card not on a cc file (with 90) and not a rename card
    spurious = []
    for p in ps:
        f = base(p.source_chunk.source_path)
        is_cc = f in cc_files and "90" in (p.after_content or "")
        is_rn = "helios" in (p.after_content or "").lower()
        if not is_cc and not is_rn:
            spurious.append(f"{p.edit_type} {f} «{p.source_chunk.section_heading[:30]}»")

    print(f"\n  total cards produced        : {len(ps)}")
    print(f"  cross-cutting edit coverage : {len(cc_hit_files)}/{len(cc_files)} files")
    print(f"  rename coverage             : {len(rn_files & brand_files)}/{len(brand_files)} files")
    print(f"  spurious / noise-driven     : {len(spurious)}")
    for s in spurious[:12]:
        print(f"      SPURIOUS: {s}")
    missed = cc_files - cc_hit_files
    for m in sorted(missed)[:20]:
        print(f"      CC MISSED: {m}")

asyncio.run(main())
