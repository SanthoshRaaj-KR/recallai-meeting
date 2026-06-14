"""End-to-end smoke test of the local-doc Confluence adapter.

Stubs ONLY the live Confluence RAG + page fetch (prev-builder infra that needs
credentials), then exercises the real new code path: materialize pages -> run
the unchanged local-doc pipeline as a subprocess -> map proposals back to
Confluence page ids. Run from my-agent/ with its own venv.
"""
import asyncio
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, "src")

from review_pipeline.confluence_proposal_adapter import run_local_doc_pipeline  # noqa: E402
from review_pipeline.models import ChangeIntent, ExtractedMeeting, PageCandidate  # noqa: E402
from review_pipeline.pipeline import ProposalPipeline  # noqa: E402
from review_pipeline.rag import VectorSearchHit  # noqa: E402

LDOC = Path(__file__).resolve().parents[1] / "local_doc_change"
DOCS = LDOC / "stress_corpus" / "docs"

# Map three real SmartHub pages to fake Confluence page ids.
PAGE_FILES = {
    "P-PRICING": "smarthub_pricing_and_plans.md",
    "P-SLA": "smarthub_sla_support_policy.md",
    "P-ML": "smarthub_ml_anomaly_detection_model_architecture.md",
}

TRANSCRIPT = [
    {"speaker": "Priya", "text": (
        "We are from smarthub. Change the Managed Threat Hunting add-on from 5 dollars "
        "to 9 dollars per device per year. Also the post-incident RCA should be published "
        "within 3 business days instead of 5. And for the anomaly detection model, the "
        "warm-up period after device onboarding should be 48 hours instead of 72."
    )},
]


def _sections_from_md(text: str) -> list[dict]:
    """Split a markdown doc into {heading, text} sections on '## ' headings."""
    sections = []
    cur_h, cur_lines = "", []
    for line in text.splitlines():
        m = re.match(r"^##\s+(.*)$", line)
        if m:
            if cur_lines:
                sections.append({"heading": cur_h, "text": "\n".join(cur_lines).strip(), "html": ""})
            cur_h, cur_lines = m.group(1).strip(), []
        elif line.startswith("# "):
            continue
        else:
            cur_lines.append(line)
    if cur_lines:
        sections.append({"heading": cur_h, "text": "\n".join(cur_lines).strip(), "html": ""})
    return [s for s in sections if s["text"]]


def _fake_pages() -> dict[str, PageCandidate]:
    pages = {}
    for pid, fname in PAGE_FILES.items():
        raw = (DOCS / fname).read_text(encoding="utf-8")
        title = raw.splitlines()[0].lstrip("# ").strip() if raw.startswith("# ") else fname
        pages[pid] = PageCandidate(
            page_id=pid, title=title, space_key="SH",
            url=f"https://example.atlassian.net/wiki/{pid}",
            html="", text=raw, version=3, sections=_sections_from_md(raw),
        )
    return pages


async def main() -> int:
    pages = _fake_pages()

    pp = ProposalPipeline()

    async def fake_extract(transcript, transcript_text, *, query=None):
        return ExtractedMeeting(
            title="SmartHub pricing & ops",
            summary="Pricing, SLA and ML model updates.",
            change_intents=[
                ChangeIntent(instruction="update pricing", subject="Managed Threat Hunting", target_hint="Pricing"),
                ChangeIntent(instruction="update sla", subject="RCA", target_hint="SLA"),
                ChangeIntent(instruction="update ml", subject="warm-up", target_hint="Anomaly model"),
            ],
        )

    async def fake_retrieve(intents):
        hits = [VectorSearchHit(page_id=pid, title=p.title, heading="", score=0.9) for pid, p in pages.items()]
        return [(intents[0], hits)]

    async def fake_fetch(page_id, source="search"):
        return pages[page_id]

    pp._extract_meeting = fake_extract           # type: ignore[assignment]
    pp._retrieve_all_sections = fake_retrieve    # type: ignore[assignment]
    pp._fetch_page_for_retrieval = fake_fetch    # type: ignore[assignment]

    events = []
    async def emit(e):
        events.append(e.get("stage") or e.get("type"))

    meeting, proposals = await run_local_doc_pipeline(
        session_id="adapter-smoke",
        transcript=TRANSCRIPT,
        memory_context="",
        emit=emit,
        pipeline=pp,
    )

    print(f"\nmeeting.title = {meeting.title!r}")
    print(f"stages emitted = {events}")
    print(f"proposals = {len(proposals)}")
    ok_pages = set(PAGE_FILES.keys())
    for p in proposals:
        on_page = p.get("page_id") in ok_pages
        print(f"  [{'OK ' if on_page else 'XX '}] page_id={p.get('page_id')} "
              f"title={p.get('page_title')!r} :: {str(p.get('section_heading'))[:28]!r} "
              f"[{p.get('change_type')}/{p.get('edit_mode')}] conf={p.get('confidence_score')}")
        print(f"        after: {str(p.get('after_content'))[:80]!r}")
    bad = [p for p in proposals if p.get("page_id") not in ok_pages]
    print(f"\nRESULT: {len(proposals)} proposal(s), {len(bad)} mis-mapped -> "
          f"{'PASS' if proposals and not bad else 'FAIL'}")
    return 0 if proposals and not bad else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
