"""Minimal, low-cost check: run ONLY the Rahul transcript 3x and report how many
proposals land and which of the 4 changes were captured. No suites, no re-index."""
import asyncio, os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / "../confluence_logic/.env")
from pipeline.run import PipelineConfig, run_pipeline

DOCS = str(Path("stress_corpus") / "docs")
T = ("Hey. I am Rahul an employee from oakmeridian. Ahmmmm Sorry. For postmortems "
     "after discussing with the team we decided that the metrics for the area will "
     "be reported every week. It will also be supervised by 2 support leads. The max "
     "batch size should be increased to a 1000 records sorry i mean 950 records. We "
     "have also decided to introduce a new escalation tier called emergency tier.")


def b(p): return os.path.basename(str(p))


async def main():
    for r in range(1, 4):
        cfg = PipelineConfig(session_id=f"rahul{r}", doc_folder=DOCS,
                             use_embeddings=True, rerank=False, contextual_retrieval=False)
        ps = await run_pipeline(T, cfg)
        offco = [p for p in ps if "oakmeridian" not in b(p.source_chunk.source_path).lower()]
        print(f"\n===== RUN {r}: {len(ps)} card(s)  off-company={len(offco)} =====")
        for p in ps:
            print(f"  [{p.intent.affected_topic[:34]:34}] -> {b(p.source_chunk.source_path):26} :: {p.source_chunk.section_heading[:28]}")


if __name__ == "__main__":
    asyncio.run(main())
