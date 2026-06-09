"""Reproduce + verify the client's bug: a table change addressed to a NAMED
company must land on THAT company's document, not a same-shaped table in another.

Generic "Key Operational Parameters" rows repeat across companies (e.g. several
docs have "Escalation tiers | 3 tiers"), so the value alone is ambiguous — only
the named company disambiguates. For several companies/formats we craft
"For the {Org} {kind}, change {param} from {value} to {new}" and assert the
single proposed card is on the named company's file.
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
from rag.chunker import chunk_document  # noqa: E402

HERE = Path("stress_corpus")
DOCS = str(HERE / "docs")
MAN = json.loads((HERE / "manifest.json").read_text(encoding="utf-8"))
DOC_META = {d["file"]: d for d in MAN["documents"]}


def base(p):
    return os.path.basename(str(p)).replace("\\", "/")


def parse_params(content: str) -> dict[str, str]:
    """Parse the Key Operational Parameters rows (md/txt pipe tables + odt cells)."""
    rows: dict[str, str] = {}
    for line in content.splitlines():
        if "|" in line:
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if len(cells) == 2 and cells[0] and not set(cells[0]) <= set("-: "):
                if cells[0].lower() != "parameter":
                    rows[cells[0]] = cells[1]
    if not rows:  # odt: cells render as separate lines, paired label/value
        lines = [l.strip() for l in content.splitlines() if l.strip()]
        if lines[:2] == ["Parameter", "Value"]:
            lines = lines[2:]
        for i in range(0, len(lines) - 1, 2):
            rows[lines[i]] = lines[i + 1]
    return rows


def bump(value: str) -> str:
    """Make a plausible new value by changing the number, keeping the unit."""
    m = re.search(r"\d[\d,]*", value)
    if not m:
        return value + " (revised)"
    n = int(m.group(0).replace(",", ""))
    return value.replace(m.group(0), str(n + 1000))


async def main():
    # Pick the FIRST generic param row from a handful of docs across formats.
    targets = []  # (file, org, kind, label, value)
    seen_fmt: dict[str, int] = {}
    for d in MAN["documents"]:
        if seen_fmt.get(d["fmt"], 0) >= 2:
            continue
        chunks = chunk_document(str(HERE / d["file"]))
        params = next((c for c in chunks
                       if "key operational parameters" in c.section_heading.lower()), None)
        if not params:
            continue
        rows = parse_params(params.content)
        # a generic (non-distinctive) row: short value, not the doc's unique anchor
        generic = [(k, v) for k, v in rows.items()
                   if re.match(r"^[\d$]", v) and "gigabyte" not in v]
        if not generic:
            continue
        label, value = generic[0]
        targets.append((d["file"], d["org"], d["kind"], label, value))
        seen_fmt[d["fmt"]] = seen_fmt.get(d["fmt"], 0) + 1

    print(f"Testing company-routing on {len(targets)} table edits\n")
    correct = 0
    for file, org, kind, label, value in targets:
        new_value = bump(value)
        transcript = (f"For the {org} {kind.lower()}, in the key operational "
                      f"parameters, change the {label} from {value} to {new_value}.")
        cfg = PipelineConfig(session_id="xco", doc_folder=DOCS, use_embeddings=True,
                             rerank=False, contextual_retrieval=False)
        ps = await run_pipeline(transcript, cfg)
        on_named = [p for p in ps if base(p.source_chunk.source_path) == base(file)]
        on_other = [p for p in ps if base(p.source_chunk.source_path) != base(file)]
        ok = bool(on_named) and not on_other
        correct += int(ok)
        where = ", ".join(sorted({base(p.source_chunk.source_path) for p in ps})) or "(none)"
        print(f"  [{'OK ' if ok else 'XX '}] {org:12} {label:26} {value:>12} "
              f"-> cards={len(ps)} on={where}")
    print(f"\n  COMPANY ROUTING: {correct}/{len(targets)} landed on the named company "
          f"{'PASS' if correct == len(targets) else 'FAIL'}")
    return 0 if correct == len(targets) else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
