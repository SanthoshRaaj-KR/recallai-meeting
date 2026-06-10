"""Mirror the client's pattern: company named ONCE up front, then a "set/increase
X to Y" change with NO old value (or a wrong one). Assert it lands on the named
company's doc with the new value."""
from __future__ import annotations
import asyncio, json, os, re, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / "../confluence_logic/.env")
from pipeline.run import PipelineConfig, run_pipeline
from rag.chunker import chunk_document

HERE = Path("stress_corpus"); DOCS = str(HERE / "docs")
MAN = json.loads((HERE / "manifest.json").read_text(encoding="utf-8"))


def b(p): return os.path.basename(str(p)).replace("\\", "/")


def parse_params(content):
    rows = {}
    for line in content.splitlines():
        if "|" in line:
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if len(cells) == 2 and cells[0] and not set(cells[0]) <= set("-: ") and cells[0].lower() != "parameter":
                rows[cells[0]] = cells[1]
    if not rows:
        lines = [l.strip() for l in content.splitlines() if l.strip()]
        if lines[:2] == ["Parameter", "Value"]: lines = lines[2:]
        for i in range(0, len(lines) - 1, 2): rows[lines[i]] = lines[i + 1]
    return rows


async def main():
    targets, seen = [], {}
    for d in MAN["documents"]:
        if seen.get(d["fmt"], 0) >= 1: continue
        ch = chunk_document(str(HERE / d["file"]))
        p = next((c for c in ch if "key operational parameters" in c.section_heading.lower()), None)
        if not p: continue
        rows = parse_params(p.content)
        g = [(k, v) for k, v in rows.items() if re.match(r"^\d", v)]
        if not g: continue
        label, value = g[0]
        targets.append((d["file"], d["org"], d["kind"], label, value)); seen[d["fmt"]] = 1

    correct = 0
    for file, org, kind, label, value in targets:
        m = re.search(r"\d[\d,]*", value); n = int(m.group(0).replace(",", "")); unit = value[m.end():].strip()
        newn = n + 500
        # company named ONCE; NO old value; "increase ... to <new>"
        t = (f"Hey, I am an employee from {org}. Quick update for the team. We decided "
             f"the {label} should be increased to {newn} {unit}. Thanks everyone.")
        cfg = PipelineConfig(session_id="sv", doc_folder=DOCS, use_embeddings=True, rerank=False, contextual_retrieval=False)
        ps = await run_pipeline(t, cfg)
        # correct == lands on ANY doc of the named company with the new value
        o = org.split()[0].lower()
        nv = lambda s: str(newn) in (s or "").replace(",", "")
        on_named = [p for p in ps if o in b(p.source_chunk.source_path).lower() and nv(p.after_content)]
        on_other = [p for p in ps if o not in b(p.source_chunk.source_path).lower()]
        ok = bool(on_named) and not on_other
        correct += int(ok)
        where = ", ".join(sorted({b(p.source_chunk.source_path) for p in ps})) or "(none)"
        print(f"  [{'OK ' if ok else 'XX '}] {org:12} {label:26} ->{newn} {unit:8} cards={len(ps)} on={where}")
    print(f"\n  SET-VALUE ROUTING: {correct}/{len(targets)} "
          f"{'PASS' if correct == len(targets) else 'FAIL'}")
    return 0 if correct == len(targets) else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
