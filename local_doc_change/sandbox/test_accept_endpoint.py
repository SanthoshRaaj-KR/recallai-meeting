"""End-to-end test of the ACTUAL HTTP endpoints the Accept button calls.

Drives the real FastAPI app (src/recall_bridge.py) via httpx ASGI transport:
  1. POST /local-doc/pipeline/start      (runs the real pipeline)
  2. GET  /sessions/{sid}/local-doc/changes   (the cards' data source)
  3. POST /sessions/{sid}/local-doc/execute   (== clicking "Accept" on a card)

Runs against a TEMP COPY of the sandbox docs so originals are never touched.
Proves: accepted proposal edits its file + makes a backup; a NON-accepted
proposal's file is left untouched (Reject / no-accept == no write).
"""
import asyncio
import os
import shutil
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).parent
ROOT = HERE.parent
SRC = ROOT / "src"

# Load OPENAI_API_KEY from confluence_logic/.env
ENV_FILE = ROOT.parent / "confluence_logic" / ".env"
if ENV_FILE.exists():
    for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith("OPENAI_API_KEY=") and "OPENAI_API_KEY" not in os.environ:
            os.environ["OPENAI_API_KEY"] = line.split("=", 1)[1].strip().strip('"').strip("'")

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SRC))

import httpx  # noqa: E402
import recall_bridge as rb  # noqa: E402

TRANSCRIPT = (HERE / "transcript.txt").read_text(encoding="utf-8")
SID = "sandbox-accept-001"


def file_text(path: Path) -> str:
    if path.suffix == ".docx":
        import docx
        return " ".join(p.text for p in docx.Document(str(path)).paragraphs)
    return path.read_text(encoding="utf-8")


async def main() -> int:
    # 1. Temp copy of the sandbox docs
    tmp = Path(tempfile.mkdtemp(prefix="accept_e2e_"))
    docs = tmp / "docs"
    shutil.copytree(HERE / "docs", docs)
    print(f"Temp docs: {docs}")

    # 2. Seed the session transcript (what a finished meeting would have produced)
    rec = rb._SessionRecord(SID, bot_id=None, meeting_url=None)
    rec.transcript = TRANSCRIPT  # the start endpoint reads session.transcript
    rb._sessions[SID] = rec

    transport = httpx.ASGITransport(app=rb.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        # 3. Start the pipeline (real endpoint)
        r = await client.post("/local-doc/pipeline/start", json={
            "session_id": SID,
            "doc_folder": str(docs),
            "use_embeddings": True,
            "rerank": False,
            "contextual_retrieval": False,
        })
        print("start ->", r.status_code, r.json())
        job_id = r.json()["job_id"]

        # 4. Wait for completion (poll the job status in-memory while the loop runs)
        for _ in range(120):
            await asyncio.sleep(1)
            if rb._local_doc_jobs.get(job_id, {}).get("status") in ("completed", "error"):
                break
        status = rb._local_doc_jobs.get(job_id, {}).get("status")
        print("pipeline status ->", status)

        # 5. Fetch proposals (the cards' data)
        r = await client.get(f"/sessions/{SID}/local-doc/changes")
        proposals = r.json()
        print(f"\nproposals returned: {len(proposals)}")
        for p in proposals:
            print(f"  - {Path(p['source_chunk']['source_path']).name} / "
                  f"{p['source_chunk']['section_heading']} "
                  f"[{p['intent']['intent_type']}] {p['proposal_id'][:8]}")

        if not proposals:
            print("NO PROPOSALS — cannot test accept")
            return 1

        # 6. Pick ONE proposal to ACCEPT, and a DIFFERENT file to leave alone
        accept = proposals[0]
        accept_file = Path(accept["source_chunk"]["source_path"])
        other = next((p for p in proposals
                      if Path(p["source_chunk"]["source_path"]) != accept_file), None)

        before_accept = file_text(accept_file)
        before_other = file_text(Path(other["source_chunk"]["source_path"])) if other else None

        # 7. POST execute == clicking "Accept" on that one card
        r = await client.post(f"/sessions/{SID}/local-doc/execute",
                              json={"proposal_ids": [accept["proposal_id"]]})
        print("\nexecute ->", r.status_code, r.json())

    # 8. Verify on disk
    after_accept = file_text(accept_file)
    expected_new = accept["after_content"].strip()[:40]
    backups = list(accept_file.parent.glob(f"{accept_file.stem}.backup.*{accept_file.suffix}"))
    audit = list((Path.cwd() / "local_doc_change" / "audit").glob(f"{SID}.json"))

    print("\n=== RESULTS ===")
    changed = after_accept != before_accept
    print(f"[{'OK ' if changed else 'FAIL'}] accepted file CHANGED on disk: {accept_file.name}")
    has_new = accept["after_content"].strip().split('.')[0][:30] in after_accept
    print(f"[{'OK ' if has_new else 'FAIL'}] new content present (~{expected_new!r}...)")
    print(f"[{'OK ' if backups else 'FAIL'}] backup created: {[b.name for b in backups]}")
    print(f"[{'OK ' if audit else 'FAIL'}] audit log written: {[a.name for a in audit]}")

    if other:
        after_other = file_text(Path(other["source_chunk"]["source_path"]))
        untouched = after_other == before_other
        print(f"[{'OK ' if untouched else 'FAIL'}] NON-accepted file untouched: "
              f"{Path(other['source_chunk']['source_path']).name}")

    shutil.rmtree(tmp, ignore_errors=True)
    print("\n(Original sandbox/docs never modified — all edits were on the temp copy.)")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
