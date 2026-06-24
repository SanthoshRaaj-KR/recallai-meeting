"""Run the adapter and show the line-level diff of each proposal so we can verify
the correct table rows changed."""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, "src")
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env.local")
from review_pipeline.confluence_proposal_adapter import run_local_doc_pipeline  # noqa: E402

TRANSCRIPT = [{"speaker": "Priya", "text": (
    "Team, pricing updates for SmartHub. Change the Managed Threat Hunting add-on "
    "from 5 dollars to 9 dollars per device per year. The extended telemetry "
    "retention add-on should now be 1 dollar 50 per device per year instead of 50 "
    "cents. And the Training online add-on goes from 500 dollars to 750 dollars per seat."
)}]


def changed_lines(before: str, after: str):
    b = [l.strip() for l in (before or "").splitlines() if l.strip()]
    a = [l.strip() for l in (after or "").splitlines() if l.strip()]
    removed = [l for l in b if l not in a]
    added = [l for l in a if l not in b]
    return removed, added


async def main() -> int:
    _meeting, proposals = await run_local_doc_pipeline(
        session_id="diff-test", transcript=TRANSCRIPT, memory_context="")
    print(f"\nproposals = {len(proposals)}")
    for i, p in enumerate(proposals, 1):
        removed, added = changed_lines(p.get("before_content"), p.get("after_content"))
        print(f"\n[{i}] page_id={p.get('page_id')} :: {p.get('section_heading')!r} [{p.get('edit_mode')}]")
        for l in removed[:3]:
            print(f"   -  {l[:100]}")
        for l in added[:3]:
            print(f"   +  {l[:100]}")
        if not removed and not added:
            print("   (no line-level change — NO-OP)")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
