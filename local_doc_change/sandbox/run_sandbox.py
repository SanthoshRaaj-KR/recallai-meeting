"""Sandbox eval: run the local-doc pipeline against realistic docs + a long transcript.

Judges the pipeline on a discriminative transcript that contains BOTH real
actionable changes AND red-herrings (things explicitly NOT decided / unchanged).
A good pipeline proposes the real changes and skips the red herrings.
"""
import asyncio
import os
import sys
from pathlib import Path

HERE = Path(__file__).parent
ROOT = HERE.parent  # local_doc_change/

# Load OPENAI_API_KEY from confluence_logic/.env
ENV_FILE = ROOT.parent / "confluence_logic" / ".env"
if ENV_FILE.exists():
    for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith("OPENAI_API_KEY=") and "OPENAI_API_KEY" not in os.environ:
            os.environ["OPENAI_API_KEY"] = line.split("=", 1)[1].strip().strip('"').strip("'")

sys.path.insert(0, str(ROOT))

from pipeline.run import run_pipeline, PipelineConfig  # noqa: E402

DOCS = str(HERE / "docs")
TRANSCRIPT = (HERE / "transcript.txt").read_text(encoding="utf-8")

# Ground truth for scoring
EXPECTED_CHANGES = [
    "on-call rotation weekly -> half-week",
    "code review 1 -> 2 senior approvals",
    "app log retention 90 days -> 1 year",
    "prod access single manager -> dual (manager + security)",
    "PTO 15 -> 20 days",
]
SHOULD_NOT_CHANGE = [
    "15-min on-call ack window (explicitly unchanged)",
    "remote work office days (explicitly NOT decided)",
    "onboarding buddy count (just an idea)",
    "security training 1-month window (explicitly unchanged)",
    "coffee machine / holiday party (not policy)",
]


async def main() -> int:
    print("=" * 72)
    print("SANDBOX PIPELINE EVAL")
    print("=" * 72)
    print(f"Docs   : {DOCS}")
    print(f"Files  : {[p.name for p in (HERE / 'docs').iterdir()]}")
    print(f"Transcript length: {len(TRANSCRIPT)} chars\n")

    config = PipelineConfig(
        session_id="sandbox-001",
        doc_folder=DOCS,
        use_embeddings=True,
        rerank=False,
        contextual_retrieval=False,
        top_k=3,
        relevance_threshold=0.6,
    )

    proposals = await run_pipeline(transcript=TRANSCRIPT, config=config)

    print("=" * 72)
    print(f"PROPOSALS GENERATED: {len(proposals)}")
    print("=" * 72)
    for i, p in enumerate(proposals, 1):
        print(f"\n--- Proposal {i} ---")
        print(f"  File     : {Path(p.source_chunk.source_path).name}")
        print(f"  Section  : {p.source_chunk.section_heading}")
        print(f"  Intent   : {p.intent.intent_type}  |  topic: {p.intent.affected_topic}")
        print(f"  Old->New : {p.intent.old_value!r} -> {p.intent.new_value!r}")
        print(f"  Quality  : {p.quality_score:.2f}  (conf={p.intent.confidence:.2f})")
        print(f"  BEFORE   : {p.before_content.strip()[:160]!r}")
        print(f"  AFTER    : {p.after_content.strip()[:160]!r}")
        print(f"  Verifier : {p.verifier_note}")

    print("\n" + "=" * 72)
    print("EXPECTED (should appear):")
    for c in EXPECTED_CHANGES:
        print(f"   + {c}")
    print("RED HERRINGS (should NOT appear):")
    for c in SHOULD_NOT_CHANGE:
        print(f"   - {c}")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
