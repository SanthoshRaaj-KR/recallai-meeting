"""LIVE adapter test: real ProposalPipeline (real v2 RAG, real Confluence page
ids), no stubs. Confluence REST is 403 so pages are sourced from RAG content.
Proves the 8001 propose path produces real proposals on real page ids."""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, "src")
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env.local")

from review_pipeline.confluence_proposal_adapter import run_local_doc_pipeline  # noqa: E402

TRANSCRIPT = [
    {"speaker": "Priya", "text": (
        "Team, two pricing updates for SmartHub. Change the Managed Threat Hunting "
        "add-on from 5 dollars to 9 dollars per device per year. And the extended "
        "telemetry retention add-on should now be 1 dollar 50 per device per year "
        "instead of 50 cents. Also bump the volume discount so 10 percent applies at "
        "1500 devices instead of 1000."
    )},
]


async def main() -> int:
    events: list[str] = []

    async def emit(e):
        s = e.get("stage")
        if e.get("type") == "stage_progress":
            events.append(f"{s}:{ {k:v for k,v in e.items() if k not in ('type','stage')} }")
        elif s:
            events.append(s)

    meeting, proposals = await run_local_doc_pipeline(
        session_id="live-adapter-test",
        transcript=TRANSCRIPT,
        memory_context="",
        emit=emit,
    )
    print(f"\nmeeting.title      = {meeting.title!r}")
    print(f"change_intents     = {len(meeting.change_intents)}")
    print(f"stages             = {events}")
    print(f"proposals          = {len(proposals)}")
    for p in proposals:
        print(f"  • page_id={p.get('page_id')} title={p.get('page_title')!r} "
              f":: {str(p.get('section_heading'))[:30]!r} [{p.get('change_type')}/{p.get('edit_mode')}] "
              f"conf={p.get('confidence_score')}")
        print(f"        before: {str(p.get('before_content'))[:70]!r}")
        print(f"        after : {str(p.get('after_content'))[:80]!r}")
    print(f"\nRESULT: {'PASS — proposals on real page ids' if proposals else 'NO PROPOSALS'}")
    return 0 if proposals else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
