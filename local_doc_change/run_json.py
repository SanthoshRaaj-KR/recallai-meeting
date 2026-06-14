"""Thin JSON entry point for the local-doc-change pipeline.

This is a logic-free wrapper: it loads a request file, calls the UNCHANGED
``pipeline.run.run_pipeline``, and writes the resulting proposals as JSON. It
exists so another process (e.g. my-agent, running in a different virtualenv) can
invoke the pipeline in *this* environment without importing its heavy deps
(docling/faiss). Nothing in pipeline/, agents_local/, or rag/ is modified.

Usage:
    python run_json.py <request.json> <output.json>

request.json:
    {
      "transcript": "<full meeting transcript text>",
      "doc_folder": "<absolute path to a folder of materialized docs>",
      "session_id": "<id>",
      "config": { ...optional PipelineConfig field overrides... }
    }

output.json: a JSON list of LocalDocProposal.model_dump() dicts.
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402

# OPENAI_API_KEY (and optional PINECONE/LDOC_* config) live in .env.local.
load_dotenv(ROOT / ".env.local", override=True)

from pipeline.run import PipelineConfig, run_pipeline  # noqa: E402


async def _main(request_path: str, output_path: str) -> int:
    req = json.loads(Path(request_path).read_text(encoding="utf-8"))
    overrides = dict(req.get("config") or {})
    overrides.setdefault("use_embeddings", True)
    overrides.setdefault("rerank", False)
    overrides.setdefault("contextual_retrieval", False)
    cfg = PipelineConfig(
        session_id=str(req.get("session_id") or "confluence"),
        doc_folder=req["doc_folder"],
        **overrides,
    )
    proposals = await run_pipeline(req.get("transcript") or "", cfg)
    data = [p.model_dump() for p in proposals]
    Path(output_path).write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    print(f"run_json: wrote {len(data)} proposal(s) -> {output_path}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("usage: python run_json.py <request.json> <output.json>", file=sys.stderr)
        raise SystemExit(2)
    raise SystemExit(asyncio.run(_main(sys.argv[1], sys.argv[2])))
