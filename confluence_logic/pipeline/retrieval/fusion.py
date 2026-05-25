"""Reciprocal Rank Fusion utility (RETR-V3-01).

Implements the standard RRF formula (k=60) over multiple ranked lists of
document ids.  RRF is rank-based — it needs no score normalization and avoids
the incommensurable-scale mixing that plagued Phase 10's weighted-sum heuristic
(RESEARCH §Anti-Patterns, Pitfall 4).

Reference:
    bigdataboutique.com/blog/reciprocal-rank-fusion
    The formula is the industry default used by OpenSearch, Elasticsearch,
    Azure AI Search, and Weaviate.

Usage::

    from confluence_logic.pipeline.retrieval.fusion import reciprocal_rank_fusion

    dense_ranking  = ["doc-a", "doc-b", "doc-c"]   # best → worst
    lexical_ranking = ["doc-b", "doc-a", "doc-d"]
    fused = reciprocal_rank_fusion([dense_ranking, lexical_ranking], k=60)
    # fused → [("doc-b", 0.032...), ("doc-a", 0.032...), ("doc-c", ...), ("doc-d", ...)]
    top_id, top_score = fused[0]
"""

from __future__ import annotations

from typing import List, Tuple


def reciprocal_rank_fusion(
    rank_lists: List[List[str]], k: int = 60
) -> List[Tuple[str, float]]:
    """Fuse multiple ranked lists of document ids using Reciprocal Rank Fusion.

    For each ranking in ``rank_lists``, each document at zero-based rank ``r``
    contributes ``1.0 / (k + r + 1)`` to its cumulative score.  Documents
    appearing in multiple rankings accumulate contributions from all of them,
    which is why cross-list agreement drives items to the top.

    The result is sorted descending by fused score (highest relevance first).
    The function is deterministic: identical inputs always produce identical
    output (Python's ``sorted`` is stable, and the arithmetic is
    deterministic given a fixed ``k``).

    Args:
        rank_lists: A list of ranked lists.  Each inner list contains document
            ids ordered from most-relevant (index 0) to least-relevant.
            Empty inner lists are silently skipped (no contribution).
        k: The RRF constant.  k=60 is the production default; higher k
            reduces the score gap between a rank-1 and a rank-2 hit.

    Returns:
        A list of ``(doc_id, fused_score)`` tuples sorted by descending
        fused_score.  The list contains every unique doc_id that appeared
        in at least one input ranking.
    """
    scores: dict[str, float] = {}
    for ranking in rank_lists:
        for rank, doc_id in enumerate(ranking):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda kv: -kv[1])
