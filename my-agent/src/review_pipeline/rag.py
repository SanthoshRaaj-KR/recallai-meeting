from __future__ import annotations

import hashlib
import html as _html_module
import logging
import math
import os
import re
from dataclasses import dataclass
from typing import Any

from .models import PageCandidate
from .text_utils import extract_sections, html_to_text, normalize_ws

# Contextual chunk enrichment: when enabled, a short LLM-generated context
# summary is prepended to each chunk's embedding text at upsert time.
# Re-index the corpus after enabling. Uses gpt-4o-mini (one call per chunk).
_CONTEXTUAL_ENRICHMENT = os.getenv("MY_AGENT_RAG_CONTEXTUAL_ENRICHMENT", "1").strip().lower() in {
    "1", "true", "yes", "on"
}

logger = logging.getLogger(__name__)

_BM25_K1 = 1.2
_BM25_B = 0.75
_BM25_BLEND = float(os.getenv("MY_AGENT_BM25_BLEND", "0.25"))


def _tokenize_bm25(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _bm25_scores(query: str, texts: list[str]) -> list[float]:
    """Compute BM25 scores for `texts` against `query`. Returns one score per text."""
    q_terms = _tokenize_bm25(query)
    if not q_terms or not texts:
        return [0.0] * len(texts)

    tokenized = [_tokenize_bm25(t) for t in texts]
    lengths = [len(tok) for tok in tokenized]
    avgdl = sum(lengths) / max(len(lengths), 1)
    n = len(texts)

    # document frequency per term
    df: dict[str, int] = {}
    for tok in tokenized:
        for term in set(tok):
            df[term] = df.get(term, 0) + 1

    scores: list[float] = []
    for tok, dl in zip(tokenized, lengths):
        tf: dict[str, int] = {}
        for term in tok:
            tf[term] = tf.get(term, 0) + 1
        score = 0.0
        for term in q_terms:
            if term not in df:
                continue
            idf = math.log((n - df[term] + 0.5) / (df[term] + 0.5) + 1)
            f = tf.get(term, 0)
            denom = f + _BM25_K1 * (1 - _BM25_B + _BM25_B * dl / max(avgdl, 1))
            score += idf * (f * (_BM25_K1 + 1)) / max(denom, 1e-9)
        scores.append(score)
    return scores


def page_needs_reindex(
    live_version: int | None,
    stored_version: int | None,
    is_indexed: bool,
) -> bool:
    """Decide whether a page must be re-embedded during an incremental sync.

    Re-index unless we can POSITIVELY confirm the indexed copy is current:
      - not indexed yet              -> reindex (new page)
      - stored version unknown       -> reindex. This is the important one: the
        Pinecone version fetch may have FAILED for a batch (metadata unreadable),
        or the stored metadata simply lacked a version. Either way we cannot
        verify freshness, so we must NOT silently skip a possibly-stale page.
      - live version > stored        -> reindex (page was edited in Confluence)
      - live version <= stored       -> fresh, skip
      - live version unknown, stored known -> skip (can't detect a change; avoid
        re-embedding every page on every sync when the listing lacks a version)

    Shared by ConfluenceVectorIndex and PineconeHybridIndex so the two indexes
    can never drift apart on this correctness-critical decision.
    """
    if not is_indexed:
        return True
    if stored_version is None:
        return True
    if live_version is None:
        return False
    return live_version > stored_version


@dataclass
class PageChunk:
    id: str
    page_id: str
    title: str
    space_key: str
    heading: str
    section_order: int
    chunk_order: int
    version: int | None
    text: str


@dataclass
class VectorSearchHit:
    page_id: str
    title: str
    space_key: str = ""
    heading: str = ""
    score: float = 0.0
    text: str = ""
    version: int | None = None


class ConfluenceVectorIndex:
    """Page-wise Confluence vector index for hybrid review retrieval.

    Pinecone is used when configured. The class is deliberately optional: local
    development and tests keep working without Pinecone credentials, while
    production can turn on scalable dense retrieval via environment variables.
    """

    def __init__(self) -> None:
        self.backend = (
            os.getenv("MY_AGENT_RAG_VECTOR_DB")
            or os.getenv("MY_AGENT_VECTOR_DB")
            or "pinecone"
        ).strip().lower()
        self.index_name = (
            os.getenv("MY_AGENT_RAG_INDEX")
            or os.getenv("PINECONE_INDEX_NAME")
            or "confluence-review-rag-v2"
        ).strip()
        self.namespace = (
            os.getenv("MY_AGENT_RAG_NAMESPACE")
            or os.getenv("PINECONE_NAMESPACE")
            or "confluence-review"
        ).strip()
        # Larger chunks -> fewer chunks -> fewer per-chunk enrichment LLM calls
        # (input tokens are cheap, the generated summary output is the costly part).
        # 500 words ~= 665 tokens embed input, well under llama-text-embed-v2's limit
        # and the 5000-char metadata cap below.
        self.max_chunk_words = int(os.getenv("MY_AGENT_RAG_CHUNK_WORDS", "500"))
        self.max_metadata_chars = int(os.getenv("MY_AGENT_RAG_METADATA_CHARS", "5000"))
        self.create_index = os.getenv("MY_AGENT_RAG_CREATE_INDEX", "1").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        self.cloud = os.getenv("PINECONE_CLOUD", "aws").strip()
        self.region = os.getenv("PINECONE_REGION", "us-east-1").strip()
        self._pinecone_index: Any | None = None
        self._disabled_reason: str | None = None

    @property
    def enabled(self) -> bool:
        if self.backend in {"", "none", "off", "disabled"}:
            return False
        if self.backend != "pinecone":
            self._disabled_reason = f"Unsupported vector backend: {self.backend}"
            return False
        if not os.getenv("PINECONE_API_KEY", "").strip():
            self._disabled_reason = "PINECONE_API_KEY is unset"
            return False
        return True

    def upsert_page(self, page: PageCandidate) -> None:
        if not self.enabled or not page.page_id:
            return
        content_hash = self._page_content_hash(page)
        if self._indexed_page_is_fresh(page, content_hash):
            logger.debug("Vector RAG page %s is already fresh; skipping rechunk.", page.page_id)
            return
        chunks = chunk_page(page, max_words=self.max_chunk_words)
        if not chunks:
            return
        index = self._index()
        chunk_hashes = [self._chunk_hash(chunk) for chunk in chunks]
        # Incremental re-index: only rebuild chunks whose text changed, so an edit to
        # one section skips the per-chunk enrichment LLM call + re-embed for every
        # unchanged chunk. Chunk :0 is always rebuilt because its content_hash/version
        # is the page-level freshness sentinel checked by _indexed_page_is_fresh.
        existing = self._fetch_existing_chunk_hashes(page.page_id, len(chunks))
        rebuild = [
            i for i, chunk in enumerate(chunks) if i == 0 or existing.get(chunk.id) != chunk_hashes[i]
        ]
        # Contextual enrichment (MY_AGENT_RAG_CONTEXTUAL_ENRICHMENT=1): prepend an
        # LLM-generated context summary to chunk_text before embedding. Reduces
        # retrieval failure by 35-67% for generic-heading sections (Anthropic, 2024).
        records = []
        for i in rebuild:
            chunk = chunks[i]
            if _CONTEXTUAL_ENRICHMENT:
                try:
                    ctx = self._generate_chunk_context(chunk)
                    embed_text = (ctx + "\n" + self._embedding_text(chunk))[: self.max_metadata_chars]
                except Exception as exc:
                    logger.warning("Context generation failed for chunk %s: %s", chunk.id, exc)
                    embed_text = self._embedding_text(chunk)[: self.max_metadata_chars]
            else:
                embed_text = self._embedding_text(chunk)[: self.max_metadata_chars]
            # Integrated index: Pinecone embeds `chunk_text` server-side (llama-text-embed-v2).
            # `text` is stored as display metadata; `chunk_text` is what gets embedded.
            records.append(
                {
                    "_id": chunk.id,
                    "chunk_text": embed_text,
                    "text": chunk.text[: self.max_metadata_chars],
                    "page_id": chunk.page_id,
                    "title": chunk.title,
                    "space_key": chunk.space_key,
                    "heading": chunk.heading,
                    "section_order": chunk.section_order,
                    "chunk_order": chunk.chunk_order,
                    "version": chunk.version or 0,
                    "content_hash": content_hash,
                    "chunk_hash": chunk_hashes[i],
                    "chunk_count": len(chunks),
                }
            )
        for start in range(0, len(records), 96):
            index.upsert_records(namespace=self.namespace, records=records[start : start + 96])
        self._delete_stale_chunks(page.page_id, len(chunks))

    def _indexed_page_is_fresh(self, page: PageCandidate, content_hash: str) -> bool:
        if not page.page_id:
            return False
        try:
            result = self._index().fetch(ids=[f"{page.page_id}:0"], namespace=self.namespace)
            # Standard index: result.vectors; integrated index fetch may return vectors or records.
            if isinstance(result, dict):
                records = result.get("vectors") or result.get("records") or {}
            else:
                records = getattr(result, "vectors", None) or getattr(result, "records", None) or {}
            first = records.get(f"{page.page_id}:0")
            if not first:
                return False
            # Standard index stores fields in .metadata; integrated may use .fields.
            if isinstance(first, dict):
                fields = first.get("metadata") or first.get("fields") or {}
            else:
                fields = getattr(first, "metadata", None) or getattr(first, "fields", None) or {}
            indexed_hash = str(fields.get("content_hash") or "")
            indexed_version = int(fields.get("version") or 0) or None
            if page.version is not None and indexed_version == page.version and indexed_hash == content_hash:
                return True
            if page.version is None and indexed_hash == content_hash:
                return True
        except Exception as exc:
            logger.debug("Vector RAG freshness check failed for %s: %s", page.page_id, exc)
        return False

    def search(self, query: str, top_k: int = 8) -> list[VectorSearchHit]:
        if not self.enabled:
            if self._disabled_reason:
                logger.debug("Vector RAG disabled: %s", self._disabled_reason)
            return []
        clean_query = normalize_ws(query)
        if not clean_query:
            return []
        result = self._index().search(
            namespace=self.namespace,
            query={"top_k": max(1, top_k), "inputs": {"text": clean_query}},
            fields=["page_id", "title", "space_key", "heading", "text", "version"],
        )
        hits: list[VectorSearchHit] = []
        for match in getattr(getattr(result, "result", result), "hits", []):
            fields = match.fields if hasattr(match, "fields") else {}
            page_id = str(fields.get("page_id") or "")
            if not page_id:
                continue
            hits.append(
                VectorSearchHit(
                    page_id=page_id,
                    title=str(fields.get("title") or page_id),
                    space_key=str(fields.get("space_key") or ""),
                    heading=str(fields.get("heading") or ""),
                    score=float(getattr(match, "_score", None) or getattr(match, "score", 0.0) or 0.0),
                    text=str(fields.get("text") or ""),
                    version=int(fields.get("version") or 0) or None,
                )
            )
        return hits

    def search_with_rerank(
        self,
        queries: list[str],
        *,
        top_k_per_query: int = 25,
        top_n: int = 8,
        rerank_model: str = "bge-reranker-v2-m3",
    ) -> list[VectorSearchHit]:
        """Multi-query Pinecone search with RRF fusion and reranking.

        1. Runs each query against the index at top_k_per_query.
        2. Merges all hits via Reciprocal Rank Fusion keyed on (page_id, heading).
        3. Reranks the top-50 merged candidates with the specified reranker.
        4. Returns up to top_n hits ordered by rerank score.
        """
        if not self.enabled or not queries:
            return []

        index = self._index()

        # Step 1: Search each query via Pinecone integrated inference (llama-text-embed-v2).
        clean_queries = [normalize_ws(q) for q in queries[:5] if normalize_ws(q)]

        query_hit_lists: list[list[tuple[int, VectorSearchHit]]] = []
        for clean in clean_queries:
            try:
                logger.info("Querying index with top_k=%d", top_k_per_query)
                result = index.search(
                    namespace=self.namespace,
                    query={"top_k": max(1, top_k_per_query), "inputs": {"text": clean}},
                    fields=["page_id", "title", "space_key", "heading", "text", "version"],
                )
                ranked: list[tuple[int, VectorSearchHit]] = []
                hits_iter = getattr(getattr(result, "result", result), "hits", [])
                for rank, match in enumerate(hits_iter):
                    fields = match.fields if hasattr(match, "fields") else {}
                    page_id = str(fields.get("page_id") or "")
                    if not page_id:
                        continue
                    ranked.append((
                        rank,
                        VectorSearchHit(
                            page_id=page_id,
                            title=str(fields.get("title") or page_id),
                            space_key=str(fields.get("space_key") or ""),
                            heading=str(fields.get("heading") or ""),
                            score=float(getattr(match, "_score", None) or getattr(match, "score", 0.0) or 0.0),
                            text=str(fields.get("text") or ""),
                            version=int(fields.get("version") or 0) or None,
                        ),
                    ))
                query_hit_lists.append(ranked)
            except Exception as exc:
                logger.warning("search_with_rerank query failed: %s", exc)
                query_hit_lists.append([])

        # Step 2: RRF fusion — key on (page_id, heading) so each unique section
        # accumulates score from however many queries returned it.
        _K = 60
        rrf_scores: dict[str, float] = {}
        best_hit: dict[str, VectorSearchHit] = {}
        for ranked_list in query_hit_lists:
            for rank, hit in ranked_list:
                key = f"{hit.page_id}::{hit.heading}"
                rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (_K + rank)
                if key not in best_hit or hit.score > best_hit[key].score:
                    best_hit[key] = hit

        if not best_hit:
            return []

        # Blend BM25 scores into RRF scores so exact-term hits (version numbers,
        # IDs, names) don't get buried by dense-only ranking.
        if _BM25_BLEND > 0 and any(queries):
            bm25_query = " ".join(q for q in queries[:5] if q)
            candidate_texts = [normalize_ws(f"{best_hit[k].title} {best_hit[k].heading} {best_hit[k].text or ''}") for k in rrf_scores]
            raw_bm25 = _bm25_scores(bm25_query, candidate_texts)
            max_bm25 = max(raw_bm25) if raw_bm25 else 0.0
            max_rrf = max(rrf_scores.values()) if rrf_scores else 0.0
            for key, bm25_val in zip(list(rrf_scores.keys()), raw_bm25):
                norm_bm25 = bm25_val / max(max_bm25, 1e-9)
                norm_rrf = rrf_scores[key] / max(max_rrf, 1e-9)
                rrf_scores[key] = (1 - _BM25_BLEND) * norm_rrf + _BM25_BLEND * norm_bm25

        # Sort by blended score, take top-50 for reranking.
        sorted_keys = sorted(rrf_scores, key=lambda k: rrf_scores[k], reverse=True)
        candidates = [best_hit[k] for k in sorted_keys[:50]]

        # Step 3: Rerank merged candidates with Pinecone inference.
        # Always rerank regardless of candidate count — without it, a page with many
        # indexed sections dominates RRF purely by volume, not relevance.  The early
        # exit that skipped reranking for small indices caused the wrong page to win
        # when the index contains only 2–3 pages.
        rerank_query = normalize_ws(" ".join(q for q in queries[:5] if q)) if queries else ""
        if len(candidates) <= 1 or not rerank_query:
            for hit in candidates[:top_n]:
                hit.score = rrf_scores.get(f"{hit.page_id}::{hit.heading}", 0.0)
            return candidates[:top_n]

        try:
            from pinecone import Pinecone

            pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY", "").strip())
            documents = [
                {
                    "text": normalize_ws(
                        f"Title: {hit.title}\nHeading: {hit.heading}\n{hit.text or ''}"
                    )[:2000]
                }
                for hit in candidates
            ]
            rerank_result = pc.inference.rerank(
                model=rerank_model,
                query=rerank_query,
                documents=documents,
                top_n=min(top_n * 2, len(candidates)),  # fetch extra to allow diversity trim below
                rank_fields=["text"],
                return_documents=False,
                parameters={"truncate": "END"},
            )
            reranked: list[VectorSearchHit] = []
            for ranked in rerank_result.data:
                idx = ranked.index
                if idx < len(candidates):
                    hit = candidates[idx]
                    hit.score = float(ranked.score)
                    reranked.append(hit)

            # Per-page diversity cap: no single page takes more than ceil(top_n/2) slots
            # when multiple pages are candidates.  Prevents a large meeting-overview page
            # from monopolising all slots for a topic-specific query.
            unique_pages = {h.page_id for h in reranked}
            if len(unique_pages) > 1:
                per_page_cap = max(1, (top_n + 1) // 2)
                page_counts: dict[str, int] = {}
                final: list[VectorSearchHit] = []
                for hit in reranked:
                    if page_counts.get(hit.page_id, 0) < per_page_cap:
                        final.append(hit)
                        page_counts[hit.page_id] = page_counts.get(hit.page_id, 0) + 1
                    if len(final) >= top_n:
                        break
            else:
                final = reranked[:top_n]

            return final
        except Exception as exc:
            logger.warning("Reranking failed; falling back to RRF order: %s", exc)
            for hit in candidates[:top_n]:
                hit.score = rrf_scores.get(f"{hit.page_id}::{hit.heading}", 0.0)
            return candidates[:top_n]

    def _index(self) -> Any:
        if self._pinecone_index is not None:
            return self._pinecone_index
        try:
            from pinecone import Pinecone
        except ImportError as exc:
            raise RuntimeError("Install the 'pinecone' package to enable vector RAG.") from exc

        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY", "").strip())
        if self.create_index:
            self._ensure_pinecone_index(pc)
        self._pinecone_index = pc.Index(self.index_name)
        return self._pinecone_index

    def _ensure_pinecone_index(self, pc: Any) -> None:
        try:
            existing = pc.list_indexes()
            names = [item["name"] if isinstance(item, dict) else item.name for item in existing]
            if self.index_name in names:
                return
            # Create an integrated index — llama-text-embed-v2 embeds server-side.
            # chunk_text is the field sent to the embedding model (fieldMap.text).
            pc.create_index_for_model(
                name=self.index_name,
                cloud=self.cloud,
                region=self.region,
                embed={
                    "model": "llama-text-embed-v2",
                    "field_map": {"text": "chunk_text"},
                },
            )
        except Exception as exc:
            logger.warning("Could not ensure Pinecone integrated index %s: %s", self.index_name, exc)
            raise

    def _embedding_text(self, chunk: PageChunk) -> str:
        return normalize_ws(
            "\n".join(
                [
                    f"Title: {chunk.title}",
                    f"Heading: {chunk.heading}",
                    f"Space: {chunk.space_key}",
                    chunk.text,
                ]
            )
        )

    def _oai(self) -> Any:
        if getattr(self, "_oai_client", None) is None:
            import openai
            self._oai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY", "").strip())
        return self._oai_client

    def _generate_chunk_context(self, chunk: PageChunk) -> str:
        """Generate a 1–2 sentence context summary for embedding (Anthropic contextual retrieval).

        Prepending this to chunk_text at upsert time reduces retrieval failure by
        35–67% for chunks with generic headings (e.g. 'Overview', 'Performance').
        Gated by MY_AGENT_RAG_CONTEXTUAL_ENRICHMENT=1; only called at upsert time.
        """
        client = self._oai()
        prompt = (
            f"Document title: {chunk.title}\n"
            f"Space: {chunk.space_key}\n"
            f"Section heading: {chunk.heading}\n\n"
            f"Section text (excerpt):\n{chunk.text[:800]}\n\n"
            "In 1-2 sentences, explain what this section covers in the context of the "
            "document. Be specific about the topics, entities, and values it contains. "
            "Do not repeat the heading verbatim."
        )
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=120,
            temperature=0.0,
        )
        return (resp.choices[0].message.content or "").strip()

    def _delete_stale_chunks(self, page_id: str, live_count: int) -> None:
        stale_ids = [f"{page_id}:{idx}" for idx in range(live_count, live_count + 64)]
        try:
            self._index().delete(ids=stale_ids, namespace=self.namespace)
        except Exception as exc:
            logger.debug("Vector stale chunk cleanup failed for %s: %s", page_id, exc)

    def _page_content_hash(self, page: PageCandidate) -> str:
        value = "\n".join(
            [
                page.page_id or "",
                page.title or "",
                page.space_key or "",
                page.html or "",
                page.text or "",
            ]
        )
        return hashlib.sha256(value.encode("utf-8")).hexdigest()

    def _chunk_hash(self, chunk: PageChunk) -> str:
        """Hash of the inputs that drive a chunk's embedding + enrichment (title,
        heading, space, text). Unchanged hash -> the indexed chunk is still valid,
        so it can be reused without re-enriching or re-embedding."""
        value = "\n".join(
            [chunk.title or "", chunk.heading or "", chunk.space_key or "", chunk.text or ""]
        )
        return hashlib.sha256(value.encode("utf-8")).hexdigest()

    def _fetch_existing_chunk_hashes(self, page_id: str, count: int) -> dict[str, str]:
        """Return {chunk_id: chunk_hash} for the currently-indexed chunks of a page.

        Used to decide which chunks actually changed. On any error (or an index that
        predates chunk_hash) it returns {}, so upsert_page falls back to a full rebuild."""
        if not page_id or count <= 0:
            return {}
        ids = [f"{page_id}:{idx}" for idx in range(count)]
        try:
            result = self._index().fetch(ids=ids, namespace=self.namespace)
        except Exception as exc:
            logger.debug("Vector RAG chunk-hash fetch failed for %s: %s", page_id, exc)
            return {}
        if isinstance(result, dict):
            records = result.get("vectors") or result.get("records") or {}
        else:
            records = getattr(result, "vectors", None) or getattr(result, "records", None) or {}
        hashes: dict[str, str] = {}
        for chunk_id, record in records.items():
            if isinstance(record, dict):
                fields = record.get("metadata") or record.get("fields") or {}
            else:
                fields = getattr(record, "metadata", None) or getattr(record, "fields", None) or {}
            chunk_hash = str(fields.get("chunk_hash") or "")
            if chunk_hash:
                hashes[str(chunk_id)] = chunk_hash
        return hashes


    # ── Incremental sync ──────────────────────────────────────────────────────

    def sync_index(
        self,
        page_listings: list[dict[str, Any]],
        fetch_page_fn: Any,
        progress_cb: Any | None = None,
        *,
        force: bool = False,
    ) -> dict[str, Any]:
        """Incrementally re-index only Confluence pages that changed.

        Algorithm (O(pages/100) Pinecone calls + O(changed) embed calls):

        1. Batch-fetch the first chunk ``{page_id}:0`` for every page from
           Pinecone in groups of 100.  The stored metadata holds the version
           number and content hash that were current at index time.
        2. Compare each page's live Confluence ``version.number`` (from the
           listing) against the stored version.  Pages whose version increased
           — or that aren't in Pinecone yet — are marked stale.
        3. Only for stale pages: call ``fetch_page_fn(page_id)`` to pull the
           full HTML, then rechunk + re-embed via the existing ``upsert_page``
           path.  Fresh pages are skipped entirely.

        Args:
            page_listings:  Lightweight dicts from Confluence REST —
                            at minimum ``{"page_id": str, "version": int|None}``.
            fetch_page_fn:  Callable ``(page_id: str) -> PageCandidate`` that
                            fetches the full page HTML.  Must be synchronous.
            progress_cb:    Optional callable ``(done, total, page_title, status)``
                            called after each page is processed. ``status`` is
                            ``"changed"`` or ``"failed"``.

        Returns:
            ``{"checked": int, "changed": int, "skipped": int, "failed": int}``
        """
        if not self.enabled:
            logger.warning("sync_index: vector RAG is disabled — nothing to sync.")
            return {"checked": 0, "changed": 0, "skipped": 0, "failed": 0}

        checked = changed = skipped = failed = 0
        total = len(page_listings)

        if force:
            logger.info("sync_index: force=True — treating all %d pages as stale", total)

        # ── Step 1: batch-fetch stored versions from Pinecone ──────────────
        # We only fetch chunk :0 per page — it carries content_hash and version
        # in its metadata, which is all we need for the freshness check.
        stored_versions: dict[str, int | None] = {}
        stored_hashes: dict[str, str] = {}
        chunk_ids = [f"{p['page_id']}:0" for p in page_listings if p.get("page_id")]
        index = self._index()
        for batch_start in range(0, len(chunk_ids), 100):
            batch = chunk_ids[batch_start : batch_start + 100]
            try:
                result = index.fetch(ids=batch, namespace=self.namespace)
                # Standard index: result.vectors; integrated index fetch may use records.
                if isinstance(result, dict):
                    records = result.get("vectors") or result.get("records") or {}
                else:
                    records = getattr(result, "vectors", None) or getattr(result, "records", None) or {}
                for rec_id, rec in (records or {}).items():
                    page_id = rec_id.rsplit(":", 1)[0]
                    if isinstance(rec, dict):
                        fields = rec.get("metadata") or rec.get("fields") or {}
                    else:
                        fields = getattr(rec, "metadata", None) or getattr(rec, "fields", None) or {}
                    stored_versions[page_id] = int(fields.get("version") or 0) or None
                    stored_hashes[page_id] = str(fields.get("content_hash") or "")
            except Exception as exc:  # noqa: BLE001
                logger.warning("sync_index: Pinecone batch fetch failed: %s", exc)
                # Treat all pages in this batch as stale so they get re-indexed.
                for cid in batch:
                    pid = cid.rsplit(":", 1)[0]
                    stored_versions.setdefault(pid, None)

        # ── Step 2: compute the delta ──────────────────────────────────────
        stale: list[dict[str, Any]] = []
        for listing in page_listings:
            pid = listing.get("page_id")
            if not pid:
                continue
            if force:
                stale.append(listing)
                continue
            live_version: int | None = listing.get("version")
            if page_needs_reindex(
                live_version, stored_versions.get(pid), pid in stored_versions
            ):
                stale.append(listing)
            else:
                skipped += 1

        logger.info(
            "sync_index: %d pages total — %d stale / %d fresh",
            total,
            len(stale),
            skipped,
        )

        # ── Step 2.5: purge chunks for pages deleted from Confluence ───────
        # The live set is everything Confluence returned. Any chunk in Pinecone
        # whose page_id is NOT in this set belongs to a deleted page.
        live_page_ids = {p["page_id"] for p in page_listings if p.get("page_id")}
        deleted = 0
        try:
            orphan_ids: list[str] = []
            for chunk_id in index.list(namespace=self.namespace):
                pid = chunk_id.rsplit(":", 1)[0]
                if pid not in live_page_ids:
                    orphan_ids.append(chunk_id)
            if orphan_ids:
                for batch_start in range(0, len(orphan_ids), 1000):
                    batch = orphan_ids[batch_start : batch_start + 1000]
                    index.delete(ids=batch, namespace=self.namespace)
                deleted = len(orphan_ids)
                logger.info(
                    "sync_index: purged %d orphan chunk(s) for %d deleted page(s)",
                    deleted,
                    len({cid.rsplit(":", 1)[0] for cid in orphan_ids}),
                )
        except Exception as exc:  # noqa: BLE001
            logger.warning("sync_index: orphan purge failed (non-fatal): %s", exc)

        # ── Step 3: re-embed only stale pages ─────────────────────────────
        for listing in stale:
            pid = listing["page_id"]
            title = listing.get("title", pid)
            status = "changed"
            try:
                page = fetch_page_fn(pid)
                self.upsert_page(page)
                changed += 1
                logger.info("sync_index: [%d/%d] re-indexed %r (page_id=%s)", checked + 1, len(stale), title, pid)
            except Exception as exc:  # noqa: BLE001
                failed += 1
                status = "failed"
                logger.warning("sync_index: failed to re-index %r: %s", title, exc)
            checked += 1
            if progress_cb:
                try:
                    progress_cb(checked, len(stale), title, status)
                except Exception:  # noqa: BLE001
                    pass

        return {
            "checked": checked,
            "changed": changed,
            "skipped": skipped,
            "failed": failed,
            "deleted": deleted,
        }


def _strip_cell(cell_html: str) -> str:
    text = re.sub(r"<[^>]+>", " ", cell_html or "")
    return normalize_ws(_html_module.unescape(text))


def _extract_table_rows(section_html: str) -> list[str]:
    """Parse <table> elements and return one string per data row.

    Format: "Header1: value1 | Header2: value2". Column headers come from <th>
    cells; data rows come from <td> cells. This preserves the column-header →
    cell relationship that plain text extraction loses, which is critical for
    numeric-value lookups (SLA response times, sprint capacities, versions).
    """
    rows: list[str] = []
    for table_match in re.finditer(r"(?is)<table\b[^>]*>(.*?)</table>", section_html or ""):
        table_inner = table_match.group(1)
        headers: list[str] = []
        for tr_match in re.finditer(r"(?is)<tr\b[^>]*>(.*?)</tr>", table_inner):
            row_html = tr_match.group(1)
            th_cells = [_strip_cell(m.group(1)) for m in re.finditer(r"(?is)<th\b[^>]*>(.*?)</th>", row_html)]
            td_cells = [_strip_cell(m.group(1)) for m in re.finditer(r"(?is)<td\b[^>]*>(.*?)</td>", row_html)]
            if th_cells:
                headers = [c for c in th_cells if c]
                continue
            if not td_cells:
                continue
            if headers and len(td_cells) == len(headers):
                row_str = " | ".join(f"{h}: {v}" for h, v in zip(headers, td_cells) if v)
            else:
                row_str = " | ".join(c for c in td_cells if c)
            if row_str:
                rows.append(row_str)
    return rows


def chunk_page(page: PageCandidate, *, max_words: int = 500) -> list[PageChunk]:
    """Split one Confluence page into page-owned chunks.

    Chunks never combine multiple Confluence pages. Heading-delimited sections
    are preferred, then oversized sections are split by paragraph/word windows.
    """
    if not page.page_id:
        return []
    sections = page.sections or (extract_sections(page.html) if page.html else [])
    if not sections:
        text = page.text or html_to_text(page.html)
        sections = [{"heading": "", "level": "0", "text": text, "html": page.html}]

    chunks: list[PageChunk] = []
    for section_order, section in enumerate(sections):
        heading = normalize_ws(str(section.get("heading") or "Page intro"))
        section_html = str(section.get("html") or "")
        text = normalize_ws(str(section.get("text") or html_to_text(section_html)))
        if not text:
            continue
        for part in _split_text(text, max_words=max_words):
            chunks.append(
                PageChunk(
                    id=f"{page.page_id}:{len(chunks)}",
                    page_id=page.page_id,
                    title=page.title,
                    space_key=page.space_key,
                    heading=heading,
                    section_order=section_order,
                    chunk_order=len(chunks),
                    version=page.version,
                    text=part,
                )
            )
        # Table-aware chunks: emit one additional chunk per data row so column-header
        # associations are preserved for numeric-value retrieval (SLAs, metrics, versions).
        if "<table" in section_html.lower():
            for row_text in _extract_table_rows(section_html):
                chunks.append(
                    PageChunk(
                        id=f"{page.page_id}:{len(chunks)}",
                        page_id=page.page_id,
                        title=page.title,
                        space_key=page.space_key,
                        heading=heading,
                        section_order=section_order,
                        chunk_order=len(chunks),
                        version=page.version,
                        text=row_text,
                    )
                )
    return chunks


def _split_text(text: str, *, max_words: int, overlap: int = 50) -> list[str]:
    words = text.split()
    if len(words) <= max_words:
        return [text]
    paragraphs = [p.strip() for p in re.split(r"\n\n+", text) if p.strip()]
    if len(paragraphs) <= 1:
        # Hard word-window path: slide by (max_words - overlap) so consecutive
        # chunks share the last `overlap` words of the previous window.
        step = max(1, max_words - overlap)
        return [
            " ".join(words[idx : idx + max_words])
            for idx in range(0, len(words), step)
            if words[idx : idx + max_words]
        ]

    chunks: list[str] = []
    current: list[str] = []
    current_words = 0
    for paragraph in paragraphs:
        count = len(paragraph.split())
        if current and current_words + count > max_words:
            chunks.append("\n\n".join(current))
            # Carry forward trailing paragraphs that fit within the overlap budget
            # so boundary sentences appear in both the outgoing and incoming chunk.
            overlap_parts: list[str] = []
            overlap_count = 0
            for p in reversed(current):
                p_count = len(p.split())
                if overlap_count + p_count <= overlap:
                    overlap_parts.insert(0, p)
                    overlap_count += p_count
                else:
                    break
            current = overlap_parts + [paragraph]
            current_words = overlap_count + count
        else:
            current.append(paragraph)
            current_words += count
    if current:
        chunks.append("\n\n".join(current))
    return chunks
