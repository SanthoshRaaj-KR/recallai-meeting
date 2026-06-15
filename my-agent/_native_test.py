"""End-to-end test of the vendored local-doc pipeline (Pinecone-hybrid) on the
user's exact SmartHub transcript, over the combined corpus. Prints extracted
intents, each proposal with page attribution + line-level diff, and which intents
produced no card (recall)."""
import asyncio
import io
import sys
from pathlib import Path

# Force UTF-8 stdout so arrows/em-dashes in Confluence content don't crash on cp1252.
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, "src")
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env.local")

from review_pipeline.localdoc import PineconeHybridIndex, PipelineConfig, propose  # noqa: E402
from review_pipeline.confluence_proposal_adapter import _page_map, _proposal_from_native  # noqa: E402

TRANSCRIPT_TEXT = (
    "We are from smarthub. We have decided to raise prices for standard and proffesional "
    "devices by 5 sorry i mean 6 dollars per device per year. 10% discount will be available only on "
    "the purchase of 1500 plus devices. For extended telemetay retention we will be charging 1 and a "
    "half dollars every device. Training cost will be reduced to only 250 dollars. The duration of "
    "training has also been extended to 10 hours. We have also decided to add another level of support "
    "called as P 0. This is for emergency cases. Standard response will be within 2 hours. Enterprise "
    "Response will be immediate. As of the ML anaomaly detection pipeline I am glad to announce that "
    "latency is reduced to just 2 miliseconds. Output dimension is 512 features. Threat classification "
    "is done using XG Boost along with Stacking approach. This has increased latency by 5 miliseconds "
    "but has increased accuracy by 20 percent. We are also migrating the graph database server to kuzu "
    "DB. It runs on local and is fast. We will let the clients handle the processing capacity."
)


def diff(before: str, after: str):
    b = [l.strip() for l in (before or "").splitlines() if l.strip()]
    a = [l.strip() for l in (after or "").splitlines() if l.strip()]
    return [l for l in b if l not in a], [l for l in a if l not in b]


async def main() -> int:
    retriever = PineconeHybridIndex()
    cfg = PipelineConfig(session_id="native-test")
    intents, ld_props = await propose(TRANSCRIPT_TEXT, retriever=retriever, config=cfg)

    print(f"\n=== EXTRACTED INTENTS ({len(intents)}) ===")
    for i in intents:
        print(f"  - [{i.intent_type}] {i.affected_topic!r} -> {i.new_value!r}")

    page_map = _page_map()
    proposals = [m for p in ld_props if (m := _proposal_from_native(p, page_map, "native-test"))]
    covered_topics = {str(p.intent.affected_topic).lower() for p in ld_props}

    print(f"\n=== PROPOSALS ({len(proposals)} Confluence cards; {len(ld_props)} raw) ===")
    by_page: dict[str, int] = {}
    for i, p in enumerate(proposals, 1):
        by_page[p.get("page_title")] = by_page.get(p.get("page_title"), 0) + 1
        removed, added = diff(p.get("before_content"), p.get("after_content"))
        print(f"\n[{i}] page_id={p.get('page_id')}  {p.get('page_title')!r} :: "
              f"{p.get('section_heading')!r} [{p.get('edit_mode')}] conf={p.get('confidence_score'):.2f}")
        for l in removed[:2]:
            print(f"    -  {l[:110]}")
        for l in added[:2]:
            print(f"    +  {l[:110]}")
        if not removed and not added:
            print("    (NO-OP)")

    print(f"\n=== PAGES TOUCHED: {by_page}")
    print("=== INTENTS WITH NO CARD (recall gaps) ===")
    for i in intents:
        if str(i.affected_topic).lower() not in covered_topics:
            print(f"  ~ {i.affected_topic!r} -> {i.new_value!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
