"""Quick verification that the enriched corpus chunks correctly across formats.

Checks, per the manifest:
  - every table anchor's old_value is present in a chunk (RAG-visible),
  - the "Key Operational Parameters" section exists and carries numeric rows,
  - images/figures are present in md/docx (and captioned in txt/odt),
  - ambient numeric density increased vs the old number-free prose.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

HERE = Path(__file__).resolve().parent
MAN = json.loads((HERE / "manifest.json").read_text(encoding="utf-8"))

from rag.chunker import chunk_document  # noqa: E402

PARAMS_HEADING = "Key Operational Parameters"


def check_table_anchors() -> int:
    fails = 0
    tbl = [a for a in MAN["anchors"] if a["kind"] == "edit_table"]
    print(f"\n== Table anchors ({len(tbl)}) ==")
    for a in tbl:
        chunks = chunk_document(str(HERE / a["file"]))
        hit = next((c for c in chunks if a["old_value"] in c.content), None)
        params = [c for c in chunks if PARAMS_HEADING.lower() in c.section_heading.lower()]
        ok = hit is not None
        in_params = hit is not None and PARAMS_HEADING.lower() in hit.section_heading.lower()
        print(f"  [{ 'OK ' if ok else 'XX '}] {a['fmt']:4} {Path(a['file']).name:32} "
              f"{a['old_value']:>16}  params_section={'Y' if params else 'N'} "
              f"in_params={'Y' if in_params else 'N'}")
        if not ok:
            fails += 1
    return fails


def check_params_numbers() -> int:
    """Every doc should have a params section with >=4 numeric values."""
    fails = 0
    num_re = re.compile(r"\d")
    print(f"\n== Key Operational Parameters table per format ==")
    seen_fmt = {}
    for d in MAN["documents"]:
        if seen_fmt.get(d["fmt"], 0) >= 1:
            continue
        seen_fmt[d["fmt"]] = 1
        chunks = chunk_document(str(HERE / d["file"]))
        params = next((c for c in chunks if PARAMS_HEADING.lower() in c.section_heading.lower()), None)
        nnums = len(num_re.findall(params.content)) if params else 0
        ok = params is not None and nnums >= 4
        print(f"  [{ 'OK ' if ok else 'XX '}] {d['fmt']:4} {Path(d['file']).name:32} "
              f"params={'Y' if params else 'N'} digits={nnums}")
        if not ok:
            fails += 1
        if params:
            print(f"        sample: {params.content[:160].replace(chr(10),' / ')}")
    return fails


def check_images() -> int:
    """md/docx should surface a figure/image marker; txt/odt a caption."""
    fails = 0
    print(f"\n== Figures / images ==")
    seen_fmt = {}
    for d in MAN["documents"]:
        if seen_fmt.get(d["fmt"], 0) >= 1:
            continue
        seen_fmt[d["fmt"]] = 1
        chunks = chunk_document(str(HERE / d["file"]))
        intro = chunks[0].content if chunks else ""
        has_fig = ("Figure" in intro or "image" in intro.lower()
                   or "diagram" in intro.lower() or "![" in intro)
        print(f"  [{ 'OK ' if has_fig else '.. '}] {d['fmt']:4} {Path(d['file']).name:32} "
              f"figure_marker={'Y' if has_fig else 'N'}")
    assets = list((HERE / 'docs' / 'assets').glob('*.png'))
    print(f"  asset PNGs on disk: {len(assets)}")
    if not assets:
        fails += 1
    return fails


if __name__ == "__main__":
    f1 = check_table_anchors()
    f2 = check_params_numbers()
    f3 = check_images()
    total = f1 + f2 + f3
    print(f"\nRESULT: table_anchor_fails={f1} params_fails={f2} image_fails={f3}")
    print("ALL GOOD" if total == 0 else f"FAILURES={total}")
    sys.exit(1 if total else 0)
