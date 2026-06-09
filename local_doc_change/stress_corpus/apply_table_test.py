"""Verify write-back (SafeApply) is CORRECT for a value living inside a table.

For one table-anchor doc per format: copy to temp, simulate accepting a card that
swaps old_value -> new_value in the Key Operational Parameters section, apply it,
then re-chunk and assert:
  - new_value present, old_value GONE (the real edit happened),
  - the doc still parses (chunks > 0),
  - no duplicated params section.

No LLM calls — deterministic apply-layer check.
"""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.safe_apply import SafeApply  # noqa: E402
from rag.chunker import chunk_document  # noqa: E402

HERE = Path("stress_corpus")
MAN = json.loads((HERE / "manifest.json").read_text(encoding="utf-8"))
TABLE = [a for a in MAN["anchors"] if a["kind"] == "edit_table"]
# one per format
BY_FMT = {}
for a in TABLE:
    BY_FMT.setdefault(a["fmt"], a)
SAMPLE = list(BY_FMT.values())


def main():
    fails = 0
    tmp = Path(tempfile.mkdtemp(prefix="applytbl_"))
    print("Apply-to-table correctness (one doc per format)\n")
    for a in SAMPLE:
        src = HERE / a["file"]
        dst = tmp / Path(a["file"]).name
        shutil.copy2(src, dst)

        chunks = chunk_document(str(dst))
        params = next((c for c in chunks
                       if "key operational parameters" in c.section_heading.lower()), None)
        if params is None:
            print(f"  [XX] {a['fmt']:4} {dst.name}: no params section to edit")
            fails += 1
            continue

        # Build the 'after' content: same section, value swapped.
        after = params.content.replace(a["old_value"], a["new_value"])
        SafeApply(audit_dir=str(tmp / "audit")).apply(
            file_path=str(dst),
            section_heading=params.section_heading,
            new_content=after,
            session_id="applytbl",
            proposal_id="p1",
            edit_type="replace",
        )

        rechunks = chunk_document(str(dst))
        whole = "\n".join(c.content for c in rechunks)
        new_in = a["new_value"] in whole
        old_gone = a["old_value"] not in whole
        parses = len(rechunks) > 0
        n_params = sum(1 for c in rechunks
                       if "key operational parameters" in c.section_heading.lower())
        ok = new_in and old_gone and parses and n_params >= 1
        fails += int(not ok)
        print(f"  [{'OK ' if ok else 'XX '}] {a['fmt']:4} {dst.name:30} "
              f"new_in={new_in} old_gone={old_gone} parses={parses} "
              f"params_sections={n_params}")
        if not ok:
            # show context to understand the failure
            idx = whole.find(a["new_value"])
            if idx == -1:
                idx = whole.lower().find("key operational")
            print(f"        ...{whole[max(0,idx-40):idx+80].strip()[:140]}...")

    # D-style: replace a table section with (table + trailing prose). The prose
    # (a sentinel) must survive AND the table value stays — mirrors launch
    # Scenario D's REPLACE on the now-first "Key Operational Parameters" section.
    print("\nD-style table+prose replace (sentinel must land, no dup)\n")
    SENT = "ZZSENTINELAPPLIEDZZ"
    for a in SAMPLE:
        dst = tmp_d = Path(tempfile.mkdtemp(prefix="dstyle_")) / Path(a["file"]).name
        shutil.copy2(HERE / a["file"], dst)
        chunks = chunk_document(str(dst))
        params = next((c for c in chunks
                       if "key operational parameters" in c.section_heading.lower()), None)
        new_body = params.content.strip() + f"\n\n{SENT}"
        SafeApply(audit_dir=str(dst.parent / "audit")).apply(
            file_path=str(dst), section_heading=params.section_heading,
            new_content=new_body, session_id="d", proposal_id="p1", edit_type="replace",
        )
        rc = chunk_document(str(dst))
        whole = "\n".join(c.content for c in rc)
        sent_in = SENT in whole
        val_in = a["old_value"] in whole  # table value preserved (we didn't change it)
        no_dup = sum(1 for c in rc
                     if "key operational parameters" in c.section_heading.lower()) == 1
        ok = sent_in and val_in and no_dup
        fails += int(not ok)
        print(f"  [{'OK ' if ok else 'XX '}] {a['fmt']:4} {dst.name:30} "
              f"sentinel={sent_in} table_value={val_in} no_dup={no_dup}")
        shutil.rmtree(dst.parent, ignore_errors=True)

    shutil.rmtree(tmp, ignore_errors=True)
    print(f"\nRESULT: {'APPLY OK for all formats' if fails == 0 else f'{fails} format(s) FAILED apply'}")
    return fails


if __name__ == "__main__":
    sys.exit(1 if main() else 0)
