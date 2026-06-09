"""Fast noise-safety probe — reuses the launch suite's noise builders so I can
iterate on corpus/config changes without the full 26-minute suite. Runs the same
normal+huge pure-chatter meetings and counts false-positive cards.
"""
from __future__ import annotations

import asyncio
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / "../confluence_logic/.env")

# Reuse the exact noise builders + pipe() the launch suite uses.
import importlib.util

_spec = importlib.util.spec_from_file_location(
    "launch_readiness_test", str(Path("stress_corpus/launch_readiness_test.py"))
)
_lrt = importlib.util.module_from_spec(_spec)
# Avoid the module's open() of launch_progress.txt clobbering — it's fine, it writes there.
_spec.loader.exec_module(_lrt)


async def main():
    # Multiple seeds × normal+huge meetings — adversarial-noise coverage stronger
    # than a single seed, to catch the residual LLM variance that slips a trap.
    total_fp = 0
    n_meetings = 0
    for seed in (7, 13, 101, 2024):
        rng = random.Random(seed)
        metas = [("normal", _lrt.huge_noise(rng, 22)),
                 ("normal", _lrt.huge_noise(rng, 34)),
                 ("huge", _lrt.huge_noise(rng, 140)),
                 ("huge", _lrt.huge_noise(rng, 175))]
        for kind, t in metas:
            n_meetings += 1
            ps = await _lrt.pipe(t)
            total_fp += len(ps)
            flag = "OK  " if len(ps) == 0 else "FALSE-POSITIVE"
            print(f"  seed {seed:>4} [{kind:6}]: {_lrt.wc(t):5} words -> {len(ps)} cards   [{flag}]")
            for p in ps:
                print(f"       -> {p.edit_type} {_lrt.base(p.source_chunk.source_path)} "
                      f"«{p.source_chunk.section_heading}» :: topic={p.intent.affected_topic}")
    print(f"\n  {n_meetings} noise meetings -> TOTAL false-positive cards: {total_fp}   "
          f"GATE={'PASS' if total_fp == 0 else 'FAIL'}")
    return total_fp


if __name__ == "__main__":
    sys.exit(1 if asyncio.run(main()) else 0)
