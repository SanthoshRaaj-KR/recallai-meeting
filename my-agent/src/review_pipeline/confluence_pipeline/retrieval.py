"""Pinecone-native hybrid retrieval for the vendored Confluence pipeline.

Replaces the confluence-branch FAISS+BM25+OpenAI retriever with a fully managed,
server-side hybrid on Pinecone — nothing local, so it scales with the product:

  * dense embeddings:  llama-text-embed-v2        (integrated inference)
  * sparse embeddings: pinecone-sparse-english-v0 (integrated inference; the
                       managed lexical/BM25 equivalent — exact terms like
                       "Kuzu", "P0", "$6")
  * fusion:            Reciprocal Rank Fusion over both result sets
  * rerank:            bge-reranker-v2-m3          (Pinecone inference)

Both indexes are created with ``create_index_for_model`` so Pinecone embeds the
``chunk_text`` field server-side at upsert and query time. Documents are stored
with all fields needed to reconstruct a full ``ChunkRecord`` (the editor needs
the complete section body to make a surgical edit).
"""

from __future__ import annotations

import hashlib
import logging
import os
import time
from typing import Any

from ..text_utils import clean_inline_text, looks_like_storage_html, storage_to_markdown
from .models import ChunkRecord

logger = logging.getLogger(__name__)


def _page_to_markdown(html: str, title: str) -> str:
    """Convert Confluence storage XHTML to clean markdown (no tags / macro noise)."""
    return storage_to_markdown(html or "", title)


# Dense backend: "openai" embeds with OpenAI text-embedding-3-small and stores plain
# vectors in a standard Pinecone index; "pinecone" uses integrated inference
# (llama-text-embed-v2) which has a 5M-tokens/month free-tier cap.
_DENSE_BACKEND = os.getenv("MY_AGENT_LDOC_DENSE_BACKEND", "openai").strip().lower()
_DENSE_MODEL = os.getenv("MY_AGENT_LDOC_DENSE_MODEL", "llama-text-embed-v2")  # pinecone-integrated mode
_OAI_EMBED_MODEL = os.getenv("MY_AGENT_LDOC_OAI_EMBED_MODEL", "text-embedding-3-small")
_OAI_EMBED_DIM = int(os.getenv("MY_AGENT_LDOC_OAI_EMBED_DIM", "1536"))
_SPARSE_MODEL = os.getenv("MY_AGENT_LDOC_SPARSE_MODEL", "pinecone-sparse-english-v0")
_RERANK_MODEL = os.getenv("MY_AGENT_LDOC_RERANK_MODEL", "bge-reranker-v2-m3")
# Hybrid retrieval: dense (OpenAI text-embedding-3-small) + sparse. Sparse uses
# pinecone-sparse-english-v0 (OpenAI has no sparse model) and carries the lexical /
# exact-token matching that dense is weak on — it's what reliably catches exact
# values like "$250", "512", "2 ms". Default ON; set MY_AGENT_LDOC_USE_SPARSE=0 for
# dense-only.
_USE_SPARSE = os.getenv("MY_AGENT_LDOC_USE_SPARSE", "1").strip().lower() in {"1", "true", "yes", "on"}

_CONTENT_CHARS = int(os.getenv("MY_AGENT_LDOC_CONTENT_CHARS", "12000"))
# What gets embedded (kept smaller than stored content to stay under Pinecone's
# embedding tokens-per-minute ceiling; the heading + section head carries the
# topic signal needed for retrieval, while the full content is still stored for
# the editor to make a surgical edit).
_EMBED_CHARS = int(os.getenv("MY_AGENT_LDOC_EMBED_CHARS", "8000"))
_CONTEXTUAL_ENRICHMENT_LDOC = os.getenv("MY_AGENT_LDOC_CONTEXTUAL_ENRICHMENT", "1").strip().lower() in {
    "1", "true", "yes", "on"
}
_RRF_K = 60

# Upsert pacing — Pinecone integrated embedding has a tokens-per-minute ceiling
# (e.g. 250k TPM for llama-text-embed-v2). Small batches + a short pace + 429
# backoff keep a large corpus reindex under the limit.
_UPSERT_BATCH = int(os.getenv("MY_AGENT_LDOC_UPSERT_BATCH", "40"))
_UPSERT_PACE_S = float(os.getenv("MY_AGENT_LDOC_UPSERT_PACE_S", "1.5"))
_UPSERT_MAX_RETRIES = int(os.getenv("MY_AGENT_LDOC_UPSERT_RETRIES", "8"))

_FIELDS = [
    "content", "source_path", "source_format", "section_heading",
    "section_index", "doc_title", "page_id", "space_key",
]


def _embedding_text(chunk: ChunkRecord) -> str:
    head = f"{chunk.doc_title} — {chunk.section_heading}".strip(" —")
    space = f" [{chunk.space_key}]" if chunk.space_key else ""
    prefix = chunk.context_prefix.strip() + "\n" if chunk.context_prefix else ""
    return (prefix + head + space + "\n" + chunk.content).strip()


class PineconeHybridIndex:
    """Dense + sparse Pinecone indexes with RRF fusion and reranking."""

    def __init__(
        self,
        *,
        dense_index: str | None = None,
        sparse_index: str | None = None,
        namespace: str | None = None,
        create: bool | None = None,
    ) -> None:
        self.dense_index = dense_index or os.getenv("MY_AGENT_LDOC_DENSE_INDEX", "confluence-corpus-dense")
        self.sparse_index = sparse_index or os.getenv("MY_AGENT_LDOC_SPARSE_INDEX", "confluence-corpus-sparse")
        self.namespace = namespace or os.getenv("MY_AGENT_LDOC_NAMESPACE", "smarthub")
        self.cloud = os.getenv("PINECONE_CLOUD", "aws").strip()
        self.region = os.getenv("PINECONE_REGION", "us-east-1").strip()
        env_create = os.getenv("MY_AGENT_LDOC_CREATE_INDEX", "1").strip().lower() in {"1", "true", "yes", "on"}
        self.create = env_create if create is None else create
        self._pc: Any | None = None
        self._dense: Any | None = None
        self._sparse: Any | None = None
        # Disabled by config (OpenAI-dense-only) or when the index can't be reached.
        self._sparse_disabled = not _USE_SPARSE

    # ── Pinecone plumbing ────────────────────────────────────────────────────

    def _client(self) -> Any:
        if self._pc is None:
            from pinecone import Pinecone

            self._pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY", "").strip())
        return self._pc

    def _oai(self) -> Any:
        if getattr(self, "_oai_client", None) is None:
            import openai

            self._oai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY", "").strip())
        return self._oai_client

    def _embed_openai(self, texts: list[str]) -> list[list[float]]:
        """Embed texts with OpenAI text-embedding-3-small (batched, with retry)."""
        import time

        out: list[list[float]] = []
        client = self._oai()
        for start in range(0, len(texts), 100):
            batch = [t or " " for t in texts[start:start + 100]]
            delay = 2.0
            for attempt in range(6):
                try:
                    resp = client.embeddings.create(model=_OAI_EMBED_MODEL, input=batch)
                    out.extend([d.embedding for d in resp.data])
                    break
                except Exception as exc:
                    if attempt < 5 and ("429" in str(exc) or "rate" in str(exc).lower()):
                        time.sleep(delay)
                        delay = min(delay * 2, 30.0)
                        continue
                    raise
        return out

    def _ensure(self, name: str, model: str, *, standard_dim: int | None = None) -> None:
        pc = self._client()
        existing = pc.list_indexes()
        names = [it["name"] if isinstance(it, dict) else it.name for it in existing]
        if name in names:
            return
        if standard_dim:
            # Standard (non-integrated) index for externally-computed OpenAI vectors.
            from pinecone import ServerlessSpec

            logger.info("Creating standard Pinecone index %r (dim=%d, OpenAI vectors)", name, standard_dim)
            pc.create_index(
                name=name,
                dimension=standard_dim,
                metric="cosine",
                spec=ServerlessSpec(cloud=self.cloud, region=self.region),
            )
        else:
            logger.info("Creating Pinecone integrated index %r (%s)", name, model)
            pc.create_index_for_model(
                name=name,
                cloud=self.cloud,
                region=self.region,
                embed={"model": model, "field_map": {"text": "chunk_text"}},
            )
        # Wait until the new index is ready before upserting into it.
        import time

        for _ in range(120):
            try:
                if pc.describe_index(name).status.get("ready"):
                    break
            except Exception:
                pass
            time.sleep(1)

    def _index(self, which: str) -> Any:
        pc = self._client()
        if which == "dense":
            if self._dense is None:
                if self.create:
                    if _DENSE_BACKEND == "openai":
                        self._ensure(self.dense_index, _DENSE_MODEL, standard_dim=_OAI_EMBED_DIM)
                    else:
                        self._ensure(self.dense_index, _DENSE_MODEL)
                self._dense = pc.Index(self.dense_index)
            return self._dense
        # Sparse index is optional: if it can't be created (e.g. Pinecone project
        # serverless-index cap reached) or reached, disable it once and fall back
        # to dense + rerank so retrieval keeps working.
        if self._sparse_disabled:
            return None
        if self._sparse is None:
            try:
                if self.create:
                    self._ensure(self.sparse_index, _SPARSE_MODEL)
                self._sparse = pc.Index(self.sparse_index)
            except Exception as exc:
                self._sparse_disabled = True
                logger.warning(
                    "sparse index %r unavailable (%s) — using dense + rerank only.",
                    self.sparse_index, str(exc)[:160],
                )
                return None
        return self._sparse

    # ── Indexing ─────────────────────────────────────────────────────────────

    def upsert_chunks(
        self,
        chunks: list[ChunkRecord],
        indexes: tuple[str, ...] = ("dense", "sparse"),
    ) -> int:
        """Upsert chunks into the named indexes (default both). Returns count."""
        if not chunks:
            return 0
        records = []
        for c in chunks:
            records.append({
                "_id": c.chunk_id,
                "chunk_text": _embedding_text(c)[:_EMBED_CHARS],
                "content": c.content[:_CONTENT_CHARS],
                "source_path": c.source_path,
                "source_format": c.source_format,
                "section_heading": c.section_heading,
                "section_index": c.section_index,
                "doc_title": c.doc_title,
                "page_id": c.source_path,
                "space_key": c.space_key or "",
                "version": c.version or 0,
                "content_hash": c.content_hash or "",
            })
        for which in indexes:
            index = self._index(which)
            if index is None:
                logger.info("skipping %s upsert (index unavailable)", which)
                continue
            if which == "dense" and _DENSE_BACKEND == "openai":
                self._upsert_openai_dense(index, records)
                continue
            for start in range(0, len(records), _UPSERT_BATCH):
                batch = records[start:start + _UPSERT_BATCH]
                self._upsert_batch_with_backoff(index, batch)
        return len(records)

    def _upsert_openai_dense(self, index: Any, records: list[dict]) -> None:
        """Embed records' chunk_text with OpenAI and upsert plain vectors + metadata."""
        for start in range(0, len(records), 100):
            batch = records[start:start + 100]
            vectors = self._embed_openai([r["chunk_text"] for r in batch])
            items = [
                {
                    "id": r["_id"],
                    "values": vec,
                    "metadata": {k: v for k, v in r.items() if k not in ("_id", "chunk_text")},
                }
                for r, vec in zip(batch, vectors)
            ]
            index.upsert(vectors=items, namespace=self.namespace)

    def _upsert_batch_with_backoff(self, index: Any, batch: list[dict]) -> None:
        """Upsert one batch, backing off on Pinecone embedding TPM (429) limits."""
        import time

        delay = 2.0
        for attempt in range(_UPSERT_MAX_RETRIES):
            try:
                index.upsert_records(namespace=self.namespace, records=batch)
                if _UPSERT_PACE_S > 0:
                    time.sleep(_UPSERT_PACE_S)  # stay under tokens-per-minute
                return
            except Exception as exc:
                is_rate = "429" in str(exc) or "RESOURCE_EXHAUSTED" in str(exc) or "rate" in str(exc).lower()
                if attempt < _UPSERT_MAX_RETRIES - 1 and is_rate:
                    logger.warning("upsert 429 (attempt %d); backing off %.1fs", attempt + 1, delay)
                    time.sleep(delay)
                    delay = min(delay * 2, 60.0)
                    continue
                raise

    # ── Retrieval ────────────────────────────────────────────────────────────

    def _search_one(self, which: str, query: str, top_k: int) -> list[tuple[int, dict, float]]:
        index = self._index(which)
        if index is None:
            return []
        if which == "dense" and _DENSE_BACKEND == "openai":
            return self._search_openai_dense(index, query, top_k)
        try:
            # Pinecone 8.x integrated-inference search nests top_k/inputs under a
            # ``query`` dict (top-level top_k= raises TypeError on this SDK).
            result = index.search(
                namespace=self.namespace,
                query={"top_k": max(1, top_k), "inputs": {"text": query}},
                fields=_FIELDS,
            )
        except Exception as exc:
            logger.warning("hybrid %s search failed: %s", which, exc)
            return []
        out: list[tuple[int, dict, float]] = []
        hits = getattr(getattr(result, "result", result), "hits", [])
        for rank, match in enumerate(hits):
            raw_fields = getattr(match, "fields", None) or {}
            # Pinecone 8.x Hit exposes id/score as ``_id`` / ``_score``.
            cid = str(getattr(match, "_id", None) or getattr(match, "id", "") or "")
            if not cid:
                continue
            score = float(getattr(match, "_score", None) or getattr(match, "score", 0.0) or 0.0)
            fields = dict(raw_fields)
            fields["_id"] = cid
            out.append((rank, fields, score))
        return out

    def _search_openai_dense(self, index: Any, query: str, top_k: int) -> list[tuple[int, dict, float]]:
        """Embed the query with OpenAI and query a standard Pinecone index by vector."""
        try:
            vec = self._embed_openai([query])[0]
            result = index.query(
                namespace=self.namespace,
                vector=vec,
                top_k=max(1, top_k),
                include_metadata=True,
                include_values=False,
            )
        except Exception as exc:
            logger.warning("openai dense search failed: %s", exc)
            return []
        matches = result.get("matches", []) if isinstance(result, dict) else getattr(result, "matches", [])
        out: list[tuple[int, dict, float]] = []
        for rank, m in enumerate(matches):
            meta = (m.get("metadata") if isinstance(m, dict) else getattr(m, "metadata", None)) or {}
            cid = str((m.get("id") if isinstance(m, dict) else getattr(m, "id", "")) or "")
            if not cid:
                continue
            score = float((m.get("score") if isinstance(m, dict) else getattr(m, "score", 0.0)) or 0.0)
            fields = dict(meta)
            fields["_id"] = cid
            out.append((rank, fields, score))
        return out

    def _generate_context_prefix(self, chunk: ChunkRecord) -> str:
        """Generate a 1–2 sentence context summary to prepend to the embedding text."""
        client = self._oai()
        prompt = (
            f"Document title: {chunk.doc_title}\n"
            f"Section heading: {chunk.section_heading}\n\n"
            f"Section text (excerpt):\n{chunk.content[:800]}\n\n"
            "In 1-2 sentences, explain what this section covers in the context of the "
            "document. Be specific about topics, entities, and values. "
            "Do not repeat the heading verbatim."
        )
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=120,
            temperature=0.0,
        )
        return (resp.choices[0].message.content or "").strip()

    def sync_index(
        self,
        page_listings: list[dict[str, Any]],
        fetch_page_fn: Any,
        *,
        force: bool = False,
        progress_cb: Any | None = None,
    ) -> dict[str, Any]:
        """Incrementally sync Confluence pages into the dense + sparse indexes.

        When force=True every page is treated as stale (used when an index was
        just created or found to be missing).  Otherwise only pages whose
        Confluence version.number has increased since the last upsert are
        re-embedded; fresh pages are skipped entirely.

        Args:
            page_listings:  Lightweight dicts — at minimum {page_id, version}.
            fetch_page_fn:  Callable(page_id: str) -> PageCandidate.
            force:          Re-embed every page regardless of stored version.
            progress_cb:    Optional callable(done, total, title) for progress.

        Returns:
            {checked, changed, skipped, failed}
        """
        checked = changed = skipped = failed = 0

        stored_versions: dict[str, int | None] = {}
        stored_hashes: dict[str, str] = {}

        if not force:
            # Batch-fetch stored version from chunk :0 of the dense index.
            chunk_ids = [f"{p['page_id']}:0" for p in page_listings if p.get("page_id")]
            dense = self._index("dense")
            for batch_start in range(0, len(chunk_ids), 100):
                batch = chunk_ids[batch_start: batch_start + 100]
                try:
                    result = dense.fetch(ids=batch, namespace=self.namespace)
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
                    logger.warning("sync_index: dense fetch failed for batch: %s", exc)
                    for cid in batch:
                        stored_versions.setdefault(cid.rsplit(":", 1)[0], None)

        stale: list[dict[str, Any]] = []
        for listing in page_listings:
            pid = listing.get("page_id")
            if not pid:
                continue
            if force:
                stale.append(listing)
            else:
                live_version: int | None = listing.get("version")
                pinecone_version = stored_versions.get(pid)
                is_new = pid not in stored_versions
                is_changed = (
                    live_version is not None
                    and pinecone_version is not None
                    and live_version > pinecone_version
                )
                if is_new or is_changed:
                    stale.append(listing)
                else:
                    skipped += 1

        logger.info(
            "sync_index (hybrid): %d pages total — %d stale / %d fresh",
            len(page_listings), len(stale), skipped,
        )

        for listing in stale:
            pid = listing["page_id"]
            title = listing.get("title", pid)
            try:
                page = fetch_page_fn(pid)
                content_hash = hashlib.sha256(
                    f"{pid}\n{getattr(page, 'title', '')}\n{getattr(page, 'html', '')}".encode()
                ).hexdigest()
                markdown = _page_to_markdown(
                    getattr(page, "html", "") or "",
                    getattr(page, "title", "") or str(pid),
                )
                from .chunker import chunk_markdown_text
                chunks = chunk_markdown_text(markdown, source_path=pid, source_format="confluence")
                space_key = getattr(page, "space_key", "") or ""
                for chunk in chunks:
                    chunk.version = getattr(page, "version", None)
                    chunk.content_hash = content_hash
                    chunk.space_key = space_key
                    if _CONTEXTUAL_ENRICHMENT_LDOC and not chunk.context_prefix:
                        try:
                            chunk.context_prefix = self._generate_context_prefix(chunk)
                        except Exception as _ctx_exc:  # noqa: BLE001
                            logger.warning("Context prefix failed for %s/%s: %s", pid, chunk.section_heading, _ctx_exc)
                self.upsert_chunks(chunks)
                changed += 1
                logger.debug("sync_index (hybrid): re-indexed %r (page_id=%s)", title, pid)
            except Exception as exc:  # noqa: BLE001
                failed += 1
                logger.warning("sync_index (hybrid): failed to re-index %r: %s", title, exc)
            checked += 1
            if progress_cb:
                try:
                    progress_cb(checked, len(stale), title)
                except Exception:  # noqa: BLE001
                    pass

        return {"checked": checked, "changed": changed, "skipped": skipped, "failed": failed}

    def _to_chunk(self, fields: dict) -> ChunkRecord | None:
        cid = str(fields.get("_id") or "")
        if not cid:
            return None
        try:
            section_index = int(float(fields.get("section_index") or 0))
        except (TypeError, ValueError):
            section_index = 0
        content = str(fields.get("content") or "")
        heading = str(fields.get("section_heading") or "")
        # Safety net for indexes written before content cleaning: if a stored chunk
        # still carries Confluence/XHTML tags, clean it on the way out so the editor
        # and the review card never see raw markup (no reindex required).
        if looks_like_storage_html(content):
            content = storage_to_markdown(content, str(fields.get("doc_title") or ""))
        if looks_like_storage_html(heading) or "]]>" in heading:
            heading = clean_inline_text(heading)
        return ChunkRecord(
            chunk_id=cid,
            source_path=str(fields.get("source_path") or ""),
            source_format=str(fields.get("source_format") or "md"),
            section_heading=heading,
            section_index=section_index,
            content=content,
            doc_title=str(fields.get("doc_title") or ""),
        )

    def query(self, query_text: str, top_k: int = 12, *, rerank: bool = True) -> list[ChunkRecord]:
        """Hybrid retrieve: dense + sparse, RRF-fused, reranked. Returns ChunkRecords."""
        q = (query_text or "").strip()
        if not q:
            return []
        per_index = max(top_k, 12)
        dense_hits = self._search_one("dense", q, per_index)
        sparse_hits = self._search_one("sparse", q, per_index)

        # RRF fusion keyed on chunk_id.
        rrf: dict[str, float] = {}
        fields_by_id: dict[str, dict] = {}
        for hit_list in (dense_hits, sparse_hits):
            for rank, fields, _score in hit_list:
                cid = str(fields.get("_id") or "")
                if not cid:
                    continue
                rrf[cid] = rrf.get(cid, 0.0) + 1.0 / (_RRF_K + rank)
                fields_by_id.setdefault(cid, fields)
        if not rrf:
            return []

        ranked_ids = sorted(rrf, key=lambda c: rrf[c], reverse=True)
        candidates = [self._to_chunk(fields_by_id[c]) for c in ranked_ids[:50]]
        candidates = [c for c in candidates if c is not None]

        if not rerank or len(candidates) <= 1:
            return candidates[:top_k]

        # Rerank fused candidates with Pinecone inference (server-side).
        try:
            pc = self._client()
            documents = [
                {"text": f"{c.doc_title} — {c.section_heading}\n{c.content}"[:2000]}
                for c in candidates
            ]
            res = pc.inference.rerank(
                model=_RERANK_MODEL,
                query=q,
                documents=documents,
                top_n=min(top_k, len(candidates)),
                rank_fields=["text"],
                return_documents=False,
                parameters={"truncate": "END"},
            )
            reranked: list[ChunkRecord] = []
            for r in res.data:
                if r.index < len(candidates):
                    reranked.append(candidates[r.index])
            return reranked[:top_k]
        except Exception as exc:
            logger.warning("hybrid rerank failed (%s); using RRF order", exc)
            return candidates[:top_k]
