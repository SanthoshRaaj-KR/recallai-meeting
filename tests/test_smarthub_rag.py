"""SmartHub RAG pipeline evaluation.

Phase 1 — INGEST:  Fetch every page from the configured Confluence space and
                   upsert it into the Pinecone index used by the review
                   pipeline.  Pages already fresh in the index are skipped.

Phase 2 — EVALUATE: Run 25 targeted questions spanning all 26 SmartHub pages
                    and report HIT@1, HIT@3, and MRR for both the basic
                    ``search`` path and the full ``search_with_rerank`` path.

Usage:
    # Ingest + evaluate (default)
    conda run -n meetagents python tests/test_smarthub_rag.py

    # Skip re-indexing (pages already indexed from a previous run)
    conda run -n meetagents python tests/test_smarthub_rag.py --no-ingest

    # Evaluate only one question (quick sanity check)
    conda run -n meetagents python tests/test_smarthub_rag.py --no-ingest --q 9
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any

# ── path bootstrap ─────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "my-agent" / "src"))

# Load .env from repo root
_env_path = ROOT / ".env"
if _env_path.exists():
    for _line in _env_path.read_text().splitlines():
        _line = _line.strip()
        if not _line or _line.startswith("#") or "=" not in _line:
            continue
        _k, _, _v = _line.partition("=")
        os.environ.setdefault(_k.strip(), _v.strip())

from review_pipeline.confluence import RestConfluenceClient  # noqa: E402
from review_pipeline.models import PageCandidate  # noqa: E402
from review_pipeline.rag import ConfluenceVectorIndex, VectorSearchHit  # noqa: E402


# ── evaluation questions ───────────────────────────────────────────────────────
# Each entry: (question, substring_that_must_appear_in_matched_page_title)
# The substring is matched case-insensitively against page titles in the top-k.

QUESTIONS: list[tuple[str, str]] = [
    # ── Company & product ───────────────────────────────────────────
    (
        "Who are the founders and investors of SmartHub.ai?",
        "company overview",
    ),
    (
        "What are the five product pillars of the INFER platform?",
        "product overview",
    ),
    (
        "How does the SD-EDGE Gateway handle local policy enforcement when WAN is down?",
        "sd-edge",
    ),
    (
        "How is the Security Posture Score (SPS) calculated and what are its four dimensions?",
        "security",
    ),
    (
        "What is the top-level system architecture of the INFER platform v2.5?",
        "system architecture",
    ),
    # ── API & onboarding ────────────────────────────────────────────
    (
        "What is the API rate limit for Enterprise tier and what is the base URL?",
        "api reference",
    ),
    (
        "What are the three paths for onboarding devices into INFER?",
        "onboarding",
    ),
    (
        "What are the minimum hardware requirements for deploying the SD-EDGE Gateway?",
        "deployment",
    ),
    # ── Release & troubleshooting ───────────────────────────────────
    (
        "What bug was fixed in issue INF-2341 in the v2.5 release?",
        "release notes",
    ),
    (
        "How do you fix BACnet devices not being discovered by the gateway?",
        "troubleshooting",
    ),
    # ── Pricing & SLA ───────────────────────────────────────────────
    (
        "What is the price per device per year for the Professional plan and what features does it include?",
        "pricing",
    ),
    (
        "What was the compliance problem faced by the hospital network customer?",
        "customer success",
    ),
    (
        "What is the P1 critical incident response time for Enterprise customers?",
        "sla",
    ),
    # ── Telemetry & streaming ───────────────────────────────────────
    (
        "What Kafka topic carries device anomaly scores and what is its retention period?",
        "telemetry schema",
    ),
    (
        "What is the Kafka stream processing topology for telemetry enrichment and feature extraction?",
        "kafka",
    ),
    # ── ML & AI ─────────────────────────────────────────────────────
    (
        "What machine learning models does INFER Secure use for anomaly detection and what are their latencies?",
        "anomaly detection",
    ),
    (
        "What is the Gen-AI NLQ answer faithfulness score target and what retrieval models are used?",
        "natural language query",
    ),
    # ── Graph & auth ────────────────────────────────────────────────
    (
        "What Cypher query finds all devices exposed to a critical CVE?",
        "neo4j",
    ),
    (
        "What are the RBAC roles in INFER and what does the security_analyst role permit?",
        "authentication",
    ),
    # ── OTA & firmware ──────────────────────────────────────────────
    (
        "What is the OTA campaign rollback failure rate threshold and what rings does staged rollout use?",
        "ota firmware",
    ),
    # ── Database ────────────────────────────────────────────────────
    (
        "What PostgreSQL partitioning strategy is used for device telemetry and what is the retention?",
        "postgresql",
    ),
    # ── SIEM & encryption ───────────────────────────────────────────
    (
        "What MITRE ATT&CK technique maps to C2 beaconing in INFER's SIEM connector?",
        "siem",
    ),
    (
        "What encryption algorithm and key management approach protects data at rest in PostgreSQL?",
        "encryption",
    ),
    # ── DevOps & runbooks ───────────────────────────────────────────
    (
        "What CI/CD tool does INFER use for progressive delivery and canary deployments?",
        "ci/cd",
    ),
    (
        "What command scales out the Kafka enrichment-streams service when consumer lag is too high?",
        "runbooks",
    ),
]


# ── helpers ────────────────────────────────────────────────────────────────────

def _title_matches(hits: list[VectorSearchHit], needle: str) -> int | None:
    """Return 1-based rank of first hit whose title contains `needle` (case-insensitive), or None."""
    needle_l = needle.lower()
    for rank, hit in enumerate(hits, 1):
        if needle_l in hit.title.lower():
            return rank
    return None


def _reciprocal_rank(rank: int | None) -> float:
    return 1.0 / rank if rank else 0.0


def _ingest(client: RestConfluenceClient, rag: ConfluenceVectorIndex) -> int:
    """Fetch all pages from Confluence and upsert into Pinecone. Returns page count."""
    print("\n═══ PHASE 1: INGESTION ═══")
    # Fetch up to 200 pages from the space — more than enough for 26 SmartHub pages
    raw_pages = client.list_pages(limit=200)
    print(f"  Found {len(raw_pages)} pages in Confluence space.")

    indexed = 0
    for i, meta in enumerate(raw_pages, 1):
        page_id = meta["page_id"]
        title = meta["title"]
        try:
            page = client.fetch_page(page_id)
            rag.upsert_page(page)
            indexed += 1
            print(f"  [{i:3d}/{len(raw_pages)}] ✓ {title[:70]}")
        except Exception as exc:  # noqa: BLE001
            print(f"  [{i:3d}/{len(raw_pages)}] ✗ {title[:60]} — {exc}")
        # Small delay to avoid rate-limiting the Pinecone upsert calls
        if i % 10 == 0:
            time.sleep(0.5)

    print(f"\n  Indexed {indexed}/{len(raw_pages)} pages.")
    # Give Pinecone a moment to make freshly upserted vectors queryable
    print("  Waiting 5 s for Pinecone index to become consistent …")
    time.sleep(5)
    return indexed


def _evaluate(rag: ConfluenceVectorIndex, questions: list[tuple[str, str]]) -> None:
    """Run all evaluation questions and print a results table."""
    print("\n═══ PHASE 2: EVALUATION ═══\n")

    cols = ("Q#", "BASIC hit@1", "BASIC rank", "RERANK hit@1", "RERANK rank", "Question")
    sep = "─" * 115
    print(f"  {'Q#':>3}  {'B@1':>5}  {'B-rank':>7}  {'R@1':>5}  {'R-rank':>7}  {'Expected page fragment':<28}  Question")
    print("  " + sep)

    basic_hits1 = basic_hits3 = 0
    rerank_hits1 = rerank_hits3 = 0
    basic_rr_sum = rerank_rr_sum = 0.0

    results: list[dict[str, Any]] = []

    for idx, (question, expected_title_frag) in enumerate(questions, 1):
        # ── basic single-query search ──
        basic_raw = rag.search(question, top_k=8)
        basic_rank = _title_matches(basic_raw, expected_title_frag)
        basic_rr = _reciprocal_rank(basic_rank)
        basic_rr_sum += basic_rr

        if basic_rank == 1:
            basic_hits1 += 1
        if basic_rank is not None and basic_rank <= 3:
            basic_hits3 += 1

        # ── multi-query search_with_rerank ──
        queries = _build_queries(question)
        rerank_raw = rag.search_with_rerank(queries, top_k_per_query=25, top_n=8)
        rerank_rank = _title_matches(rerank_raw, expected_title_frag)
        rerank_rr = _reciprocal_rank(rerank_rank)
        rerank_rr_sum += rerank_rr

        if rerank_rank == 1:
            rerank_hits1 += 1
        if rerank_rank is not None and rerank_rank <= 3:
            rerank_hits3 += 1

        b1_sym = "✓" if basic_rank == 1 else ("~" if basic_rank and basic_rank <= 3 else "✗")
        r1_sym = "✓" if rerank_rank == 1 else ("~" if rerank_rank and rerank_rank <= 3 else "✗")
        b_rank_str = str(basic_rank) if basic_rank else "—"
        r_rank_str = str(rerank_rank) if rerank_rank else "—"

        frag_display = expected_title_frag[:26]
        q_display = question[:60]
        print(f"  {idx:>3}  {b1_sym:>5}  {b_rank_str:>7}  {r1_sym:>5}  {r_rank_str:>7}  {frag_display:<28}  {q_display}")

        results.append({
            "q": idx,
            "question": question,
            "expected": expected_title_frag,
            "basic_rank": basic_rank,
            "rerank_rank": rerank_rank,
            # Top-3 titles for debugging misses
            "basic_top3": [h.title for h in basic_raw[:3]],
            "rerank_top3": [h.title for h in rerank_raw[:3]],
        })

    n = len(questions)
    print("  " + sep)
    print(f"\n  {'METRIC':<30} {'BASIC':>8}  {'RERANK':>8}")
    print(f"  {'─'*50}")
    print(f"  {'HIT@1':<30} {basic_hits1/n:>8.1%}  {rerank_hits1/n:>8.1%}")
    print(f"  {'HIT@3':<30} {basic_hits3/n:>8.1%}  {rerank_hits3/n:>8.1%}")
    print(f"  {'MRR':<30} {basic_rr_sum/n:>8.3f}  {rerank_rr_sum/n:>8.3f}")
    print(f"  {'questions evaluated':<30} {n:>8}")

    # ── surface misses ──
    misses = [r for r in results if r["rerank_rank"] is None or r["rerank_rank"] > 3]
    if misses:
        print(f"\n  ── MISSES (rerank not in top-3) ──")
        for m in misses:
            print(f"\n  Q{m['q']:02d}: {m['question']}")
            print(f"       Expected title fragment: {m['expected']}")
            print(f"       Rerank top-3: {m['rerank_top3']}")
            print(f"       Basic  top-3: {m['basic_top3']}")
    else:
        print("\n  No misses — all 25 questions found the correct page in top-3 via rerank!")


def _build_queries(question: str) -> list[str]:
    """Generate 3 query variants from a question for multi-query retrieval."""
    # Variant 1: original question as-is
    # Variant 2: strip question words to make it more keyword-like
    keywords = question.rstrip("?").replace("How do you", "").replace("What is", "").replace(
        "What are", ""
    ).replace("How does", "").strip()
    # Variant 3: append "SmartHub INFER" for domain anchoring
    return [question, keywords, f"SmartHub INFER {keywords}"]


# ── main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="SmartHub RAG pipeline evaluation")
    parser.add_argument("--no-ingest", action="store_true", help="Skip Confluence fetch + Pinecone upsert")
    parser.add_argument("--q", type=int, default=0, metavar="N", help="Run only question N (1-based)")
    args = parser.parse_args()

    # ── verify Pinecone is configured ──
    if not os.getenv("PINECONE_API_KEY", "").strip():
        print("ERROR: PINECONE_API_KEY is not set. Check your .env file.")
        sys.exit(1)

    rag = ConfluenceVectorIndex()
    if not rag.enabled:
        print(f"ERROR: Pinecone RAG is disabled: {rag._disabled_reason}")
        sys.exit(1)

    print(f"Pinecone index: {rag.index_name!r}  namespace: {rag.namespace!r}")

    # ── phase 1: ingest ──
    if not args.no_ingest:
        try:
            client = RestConfluenceClient()
        except ValueError as exc:
            print(f"ERROR: Confluence credentials missing — {exc}")
            sys.exit(1)
        _ingest(client, rag)
    else:
        print("(--no-ingest: skipping Confluence fetch)")

    # ── phase 2: evaluate ──
    questions = QUESTIONS
    if args.q:
        if not 1 <= args.q <= len(QUESTIONS):
            print(f"ERROR: --q must be 1–{len(QUESTIONS)}")
            sys.exit(1)
        questions = [QUESTIONS[args.q - 1]]
        print(f"\nRunning single question #{args.q}: {questions[0][0]}")

    _evaluate(rag, questions)


if __name__ == "__main__":
    main()
