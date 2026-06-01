"""Standalone demo: run the local-doc pipeline with a sample meeting transcript.

No real meeting / Recall bot needed — feeds a hardcoded transcript through the
full pipeline (intent extraction -> RAG retrieval -> evaluation -> drafting ->
verification -> proposals) against the test fixture documents.
"""
import asyncio
import os
import sys
from pathlib import Path

# Load OPENAI_API_KEY from confluence_logic/.env (only key needed)
ENV_FILE = Path(__file__).parent.parent / "confluence_logic" / ".env"
if ENV_FILE.exists():
    for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith("OPENAI_API_KEY=") and "OPENAI_API_KEY" not in os.environ:
            os.environ["OPENAI_API_KEY"] = line.split("=", 1)[1].strip().strip('"').strip("'")

sys.path.insert(0, str(Path(__file__).parent))

from pipeline.run import run_pipeline, PipelineConfig, PIPELINE_STAGES  # noqa: E402

TRANSCRIPT = (
    "Okay team, two policy decisions from today. First, the data retention policy: "
    "we've been keeping records for 7 years, but the new regulations require us to "
    "cut that down to 5 years going forward. Please make sure that's reflected. "
    "Second, the access control process — right now employee access to production "
    "systems requires a single manager approval. Effective immediately, it needs two "
    "separate manager approvals instead of one. Let's get the docs updated."
)

FIXTURES = str(Path(__file__).parent / "tests" / "fixtures")


async def main() -> int:
    print("=" * 70)
    print("LOCAL-DOC PIPELINE DEMO")
    print("=" * 70)
    print(f"\nDoc folder : {FIXTURES}")
    print(f"Transcript : {TRANSCRIPT[:80]}...\n")

    q: asyncio.Queue = asyncio.Queue()

    async def drain():
        seen = []
        while True:
            stage = await q.get()
            if stage is None:
                break
            seen.append(stage)
            print(f"  [stage] {stage}")
        return seen

    config = PipelineConfig(
        session_id="demo-session-001",
        doc_folder=FIXTURES,
        use_embeddings=True,
        rerank=False,            # skip heavy cross-encoder model download
        contextual_retrieval=False,  # skip per-chunk GPT context calls for speed
        top_k=3,
        relevance_threshold=0.6,
    )

    print("Running pipeline...\n")
    drain_task = asyncio.create_task(drain())
    proposals = await run_pipeline(transcript=TRANSCRIPT, config=config, progress_queue=q)
    await q.put(None)
    stages_seen = await drain_task

    print(f"\nStages emitted ({len(stages_seen)}): {stages_seen}")
    print(f"Expected stages ({len(PIPELINE_STAGES)}): {PIPELINE_STAGES}")

    print("\n" + "=" * 70)
    print(f"PROPOSALS GENERATED: {len(proposals)}")
    print("=" * 70)
    for i, p in enumerate(proposals, 1):
        print(f"\n--- Proposal {i} ---")
        print(f"  File        : {Path(p.source_chunk.source_path).name}")
        print(f"  Section     : {p.source_chunk.section_heading}")
        print(f"  Intent type : {p.intent.intent_type}")
        print(f"  Topic       : {p.intent.affected_topic}")
        print(f"  Old -> New  : {p.intent.old_value!r} -> {p.intent.new_value!r}")
        print(f"  Quality     : {p.quality_score:.2f} "
              f"(factual={p.factual_consistency:.2f}, "
              f"format={p.formatting_integrity:.2f}, "
              f"fulfill={p.intent_fulfillment:.2f})")
        print(f"  Verifier    : {p.verifier_note}")
        print(f"  BEFORE: {p.before_content[:120]!r}")
        print(f"  AFTER : {p.after_content[:120]!r}")

    if not proposals:
        print("\n(No proposals — check intent extraction / retrieval / threshold.)")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
