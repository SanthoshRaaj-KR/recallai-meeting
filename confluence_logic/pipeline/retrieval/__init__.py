"""Retrieval utilities for the Phase 11 hybrid dense+lexical pipeline.

Exports:
  fusion.reciprocal_rank_fusion  — Rank-based RRF over multiple rank lists
  corpus.fetch_section_corpus     — Per-run section corpus fetch+cache from Neo4j
  bm25_index.BM25Index            — In-process Okapi BM25 over the section corpus
"""
