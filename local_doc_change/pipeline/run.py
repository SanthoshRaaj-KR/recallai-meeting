"""Pipeline orchestrator for the local document change pipeline.

Orchestrates all 8 stages:
  1. transcript_source  — validate transcript
  2. intent_extraction  — extract document change intents via IntentExtractionAgent
  3. rag_indexing       — build or load the BM25/FAISS document index
  4. rag_retrieval      — retrieve candidate sections per intent via HybridRetriever
  5. evaluation         — score sections with EvaluationAgent; filter by threshold
  6. drafting           — draft before/after edits via LocalDocEditorAgent
  7. verification       — verify edit quality via VerifierAgent
  8. ready_for_review   — assemble and return LocalDocProposal list

Each extracted intent is routed by kind (see agents_local.structural):
  - "edit"    → retrieve → evaluate → draft → verify (localized change). An edit
                intent may now produce MULTIPLE proposals when several sections
                qualify (e.g. one change that affects two documents).
  - "rename"  → literal corpus-wide value replacement: one proposal per section
                that mentions the old value.
  - "removal" → whole-section deletion: resolve which ordered sections the
                instruction targets, one delete proposal per section.

Progress is reported via an asyncio.Queue (one stage name per message).
None sentinel closes the stream when the pipeline finishes or errors.
"""

from __future__ import annotations

import asyncio
import datetime
import logging
import os
import re
import uuid
from typing import Optional

import openai
from pydantic import BaseModel

from models import LocalDocProposal
from models.rag import ChunkRecord

logger = logging.getLogger(__name__)

# ── Stage names (must match test_pipeline.py exactly) ────────────────────────

PIPELINE_STAGES: list[str] = [
    "transcript_source",
    "intent_extraction",
    "rag_indexing",
    "rag_retrieval",
    "evaluation",
    "drafting",
    "verification",
    "ready_for_review",
]


# ── PipelineConfig ────────────────────────────────────────────────────────────


class PipelineConfig(BaseModel):
    """Configuration for a single run_pipeline() invocation."""

    session_id: str
    doc_folder: str
    use_embeddings: bool = True
    rerank: bool = True
    contextual_retrieval: bool = True
    top_k: int = 3
    # Minimum (intent, section) relevance for a section to become a card. The
    # primary no-hallucination defense is the intent extractor, which now rejects
    # no-change / deferred / hypothetical / transient-metric chatter before it ever
    # reaches retrieval; this threshold is the second gate. 0.70 is the bottom of
    # the evaluator's "likely the right place" band (0.7-0.89): real edits score
    # 0.85-1.0 and genuinely-relevant prose sections (a postmortem/reporting
    # paragraph that states the value) score ~0.70, which we WANT to admit. The
    # extractor — not this threshold — is what keeps no-change/hypothetical
    # chatter out, so 0.70 maximises recall without re-opening false positives.
    relevance_threshold: float = 0.70
    # How many candidate sections to pull per edit intent before evaluation.
    # Wider than top_k so the right section survives to the eval stage even on a
    # large corpus where many sections are lexically similar. Real corpora are
    # dense with numbers/tables, so a too-narrow net drops the correct section
    # just outside the window; 20 keeps recall robust while the 0.7 eval
    # threshold and max_targets cap still prevent false positives. (Cost is more
    # eval calls per intent, which the throttled runtime absorbs.)
    retrieval_top_k: int = 20
    # Cap on how many sections a single edit intent may change. Default 1: an
    # edit ("change X from A to B") almost always targets one specific section,
    # and selecting only the single best-matching section avoids drafting onto a
    # different section that merely shares the same number or theme. Recall comes
    # from extracting MORE intents and from the rename/removal branches — not
    # from fanning one edit across sections. Raise this only for corpora where a
    # single rule is intentionally duplicated across several sections.
    max_targets_per_intent: int = 1
    # Cross-cutting edits ("change X to Y in every document") must propagate to
    # every section that documents the same thing, so they use a wider retrieval
    # net and a much higher target cap than a normal single-section edit.
    cross_cutting_top_k: int = 40
    cross_cutting_max_targets: int = 30
    # Drafts below this intent-fulfilment score are dropped as failed (no-op /
    # unfulfilled) rather than shown as a misleading accept-able card.
    min_fulfillment: float = 0.4
    skip_contextual: bool = False   # test flag: skip GPT-4o-mini context enrichment
    openai_api_key: Optional[str] = None  # falls back to OPENAI_API_KEY env var


# ── Internal helpers ──────────────────────────────────────────────────────────


async def _emit(q: Optional[asyncio.Queue], stage: str) -> None:
    """Emit a stage name to the progress queue (no-op if queue is None)."""
    if q is not None:
        await q.put(stage)


def _utcnow() -> str:
    return datetime.datetime.utcnow().isoformat() + "Z"


def _get_openai_client(config: PipelineConfig) -> openai.AsyncOpenAI:
    """Return an AsyncOpenAI client using config.openai_api_key or env var."""
    return openai.AsyncOpenAI(
        api_key=config.openai_api_key or os.getenv("OPENAI_API_KEY")
    )


def _ordered_sections(chunks: list[ChunkRecord], source_path: str) -> list[ChunkRecord]:
    """All chunks for one document, in document order, one per heading."""
    by_heading: dict[str, ChunkRecord] = {}
    for c in chunks:
        if c.source_path != source_path:
            continue
        # Keep the first chunk per heading (oversized sections split into .0/.1).
        if c.section_heading not in by_heading:
            by_heading[c.section_heading] = c
    return sorted(by_heading.values(), key=lambda c: c.section_index)


def _make_proposal(
    config: PipelineConfig,
    intent,
    chunk: ChunkRecord,
    before_content: str,
    after_content: str,
    edit_type: str,
    *,
    factual: float,
    formatting: float,
    fulfillment: float,
    quality: float,
    note: str,
) -> LocalDocProposal:
    return LocalDocProposal(
        proposal_id=str(uuid.uuid4()),
        session_id=config.session_id,
        intent=intent,
        source_chunk=chunk,
        before_content=before_content,
        after_content=after_content,
        edit_type=edit_type,
        confidence=quality,
        factual_consistency=factual,
        formatting_integrity=formatting,
        intent_fulfillment=fulfillment,
        quality_score=quality,
        verifier_note=note,
        status="pending",
        created_at=_utcnow(),
    )


# ── Pipeline orchestrator ─────────────────────────────────────────────────────


def _record_diagnostics(
    diagnostics: Optional[dict],
    intents: list,
    proposals: list,
    classify_kind=None,
) -> None:
    """Fill the optional diagnostics dict so callers can explain a result.

    The pipeline is RAG-first: a change only becomes a card when a document
    section actually covers it. When intents are extracted but none match a
    section (e.g. a topic the documents do not cover), the proposal list is
    empty even though the meeting was understood. Surfacing the extracted-but-
    unmatched intents turns a confusing silent zero into an honest explanation —
    without ever fabricating a card for an undocumented topic.
    """
    if diagnostics is None:
        return
    matched_ids = {id(p.intent) for p in proposals}
    unmatched = [i for i in intents if id(i) not in matched_ids]
    diagnostics["extracted_intent_count"] = len(intents)
    diagnostics["proposal_count"] = len(proposals)
    diagnostics["unmatched_intents"] = [
        {
            "affected_topic": i.affected_topic,
            "old_value": i.old_value,
            "new_value": i.new_value,
            "kind": classify_kind(i) if classify_kind else None,
        }
        for i in unmatched
    ]


# Generic category / structure words that are NOT a document's identity — so a
# transcript that merely says "security" or "policy" can't capture a whole group.
# This is domain-agnostic: a real company/page name is a distinctive proper noun
# that survives this filter; generic category words do not.
_DOC_ID_STOPWORDS = {
    "policy", "policies", "handbook", "guide", "guides", "sop", "manual", "manuals",
    "procedure", "procedures", "standard", "standards", "overview", "doc", "document",
    "documents", "the", "and", "for", "of", "an", "section", "part", "team", "teams",
    "security", "finance", "financial", "human", "resources", "engineering", "eng",
    "product", "products", "support", "data", "infrastructure", "operations", "ops",
    "people", "governance", "privacy", "information", "company", "corp", "inc",
    "ltd", "llc", "group", "department", "div", "division",
}


def _doc_identifier_tokens(text: str) -> set:
    """Distinctive identifier tokens from a title/filename (drops generic words)."""
    toks = set(re.findall(r"[a-z][a-z0-9]{2,}", (text or "").lower()))
    return toks - _DOC_ID_STOPWORDS


# Generic words that carry no field identity, so they don't count toward a
# label match (e.g. a "Value" column header or the word "parameter").
_LABEL_STOPWORDS = {
    "the", "a", "an", "of", "for", "to", "and", "or", "per", "by", "in", "on",
    "is", "are", "be", "value", "parameter", "parameters", "standard", "target",
    "current", "new", "max", "min",  # 'max'/'min' alone are too generic to anchor
}


def _sig_tokens(text: str) -> list:
    """Significant (length>=3, non-stopword) lowercase tokens of a phrase."""
    return [
        t for t in re.findall(r"[a-z0-9]+", (text or "").lower())
        if len(t) >= 3 and t not in _LABEL_STOPWORDS
    ]


def _field_labels(content: str) -> list:
    """Field labels declared in a section: table-row left cells and the
    'Label: value' / 'Label - value' prefix of a line. These are the
    structured places a specific parameter is stated."""
    labels: list = []
    for line in (content or "").splitlines():
        s = line.strip()
        if not s:
            continue
        if "|" in s:  # markdown / rendered table row
            cells = [c.strip() for c in s.strip("|").split("|")]
            if len(cells) >= 2 and cells[0] and not set(cells[0]) <= set("-: "):
                labels.append(cells[0])
        else:
            m = re.match(r"^([A-Za-z][\w '/&\-]{2,45})\s*[:\-–]\s+\S", s)
            if m:
                labels.append(m.group(1))
    return labels


def _field_label_match(intent, chunk) -> bool:
    """True when the intent's topic names a structured FIELD that this section
    actually declares (a table row or 'Label: value'). This is a deterministic,
    high-confidence "right place" signal that does not depend on the (noisy) LLM
    evaluator — an exact field match like topic "maximum batch size" against a
    row "Maximum batch size | 100 records" must not be missed because the model
    happened to score it 0.5. Domain-agnostic: it reads whatever labels the
    section declares; nothing about any specific corpus is assumed."""
    topic_toks = _sig_tokens(getattr(intent, "affected_topic", ""))
    if len(topic_toks) < 2:
        return False  # too generic to anchor on a label safely
    topic_set = set(topic_toks)
    for label in _field_labels(chunk.content):
        ltoks = set(_sig_tokens(label))
        if not ltoks:
            continue
        # count topic tokens present in the label (substring-aware so
        # "max"/"maximum", "metric"/"metrics" still align)
        hits = sum(
            1 for t in topic_set
            if any(t in lt or lt in t for lt in ltoks)
        )
        # require a strong overlap: at least two topic tokens, and nearly all of
        # them, present in this one label — prevents flooring on a coincidental
        # single-word brush.
        if hits >= 2 and hits >= len(topic_set) - 1:
            return True
    return False


def _detect_target_documents(transcript: str, chunks: list) -> tuple:
    """Find a document/company the speaker named for the WHOLE meeting.

    Domain-agnostic: builds candidate identifier tokens from each document's
    title + filename (minus generic category/structure words), then sees which
    the transcript mentions. When the speaker establishes one document/org once
    ("I'm from Acme", "for the Acme handbook") and then lists several changes
    without re-naming it, every change must still route to that document. Returns
    (token, set_of_source_paths) or (None, None) when no single document is
    clearly named. NOTHING about any specific corpus is hardcoded — it reads the
    titles/filenames of whatever folder is loaded.
    """
    tok2paths: dict = {}
    all_paths = set()
    for c in chunks:
        all_paths.add(c.source_path)
        stem = os.path.splitext(os.path.basename(c.source_path))[0]
        ident = (getattr(c, "doc_title", "") or "") + " " + stem
        for t in _doc_identifier_tokens(ident):
            tok2paths.setdefault(t, set()).add(c.source_path)
    if not tok2paths:
        return None, None
    matched = [(t, paths) for t, paths in tok2paths.items()
               if t in _doc_identifier_tokens(transcript)]
    if not matched:
        return None, None
    # The most distinctive named identifier tags the FEWEST documents (a specific
    # company/page, not a broad category). Require it to point at a minority of
    # the corpus so a stray word can't capture everything.
    matched.sort(key=lambda kv: len(kv[1]))
    token, paths = matched[0]
    if len(paths) > max(1, int(len(all_paths) * 0.5)):
        return None, None
    return token, paths


async def run_pipeline(
    transcript: str,
    config: PipelineConfig,
    progress_queue: Optional[asyncio.Queue] = None,
    diagnostics: Optional[dict] = None,
) -> list[LocalDocProposal]:
    """Run the full 8-stage local document change pipeline.

    Returns proposals ready for human review. Empty list if no qualifying edits
    are found. If *diagnostics* is provided, it is populated with the extracted
    intent count and any extracted-but-unmatched intents, so a caller can explain
    why a result is empty (e.g. the topic is not covered by any document).
    """
    q = progress_queue

    # ── Stage 1: transcript_source ────────────────────────────────────────────
    await _emit(q, "transcript_source")
    if not transcript or not transcript.strip():
        logger.info("run_pipeline: empty transcript — no proposals to generate")
        _record_diagnostics(diagnostics, [], [])
        await _emit(q, "ready_for_review")
        return []

    # ── Stage 2: intent_extraction ────────────────────────────────────────────
    await _emit(q, "intent_extraction")
    from agents_local.intent_extraction import IntentExtractionAgent  # noqa: PLC0415
    from agents_local.structural import classify_kind  # noqa: PLC0415

    intent_agent = IntentExtractionAgent()
    intents = await intent_agent.extract(transcript)
    if not intents:
        logger.info("run_pipeline: no intents extracted from transcript")
        _record_diagnostics(diagnostics, [], [], classify_kind)
        await _emit(q, "ready_for_review")
        return []

    # Route each intent by kind.
    from agents_local.structural import has_explicit_removal_verb  # noqa: PLC0415

    # An edit card must name a concrete new value; drop placeholders/empties so an
    # inquiry the extractor mis-read ("what's our retention set to?") cannot become
    # a card even if it slipped through extraction.
    _PLACEHOLDER_VALUES = {
        "", "tbd", "tba", "unknown", "n/a", "na", "to be determined",
        "to be decided", "?", "none", "null",
    }

    def _has_concrete_value(i) -> bool:
        return (i.new_value or "").strip().lower() not in _PLACEHOLDER_VALUES

    edit_intents = [
        i for i in intents
        if classify_kind(i) == "edit" and _has_concrete_value(i)
    ]
    rename_intents = [i for i in intents if classify_kind(i) == "rename"]
    # Removals delete whole sections and skip the relevance-score gate, so only
    # honor ones where the speaker explicitly said a removal verb (not inferred
    # from casual chatter) — otherwise a stray intent silently deletes a section.
    removal_intents = [
        i for i in intents
        if classify_kind(i) == "removal" and has_explicit_removal_verb(i)
    ]
    logger.info(
        "run_pipeline: %d intents (edit=%d rename=%d removal=%d)",
        len(intents), len(edit_intents), len(rename_intents), len(removal_intents),
    )

    # ── Stage 3: rag_indexing ─────────────────────────────────────────────────
    await _emit(q, "rag_indexing")
    from rag.indexer import build_index  # noqa: PLC0415

    openai_client = _get_openai_client(config) if config.use_embeddings else None
    index = build_index(
        folder_path=config.doc_folder,
        use_embeddings=config.use_embeddings,
        contextual_retrieval=config.contextual_retrieval and not config.skip_contextual,
        openai_client=openai_client,
    )
    all_chunks: list[ChunkRecord] = index.chunks

    # Transcript-level document routing: if the speaker named one document/company
    # for the whole meeting (e.g. "I'm from Acme ... change X ... change Y"), scope
    # every change to that document instead of letting each change match a
    # same-shaped section in a different company's doc.
    target_token, target_paths = _detect_target_documents(transcript, all_chunks)
    if target_token:
        logger.info(
            "run_pipeline: transcript names document %r (%d files) — scoping changes to it",
            target_token, len(target_paths or []),
        )

    # ── Stage 4: rag_retrieval ────────────────────────────────────────────────
    await _emit(q, "rag_retrieval")
    from rag.retriever import HybridRetriever  # noqa: PLC0415

    retriever = HybridRetriever(index, rerank=config.rerank)

    # Edit intents: collect candidate sections (wide net for recall).
    from agents_local.structural import is_cross_cutting  # noqa: PLC0415

    intent_candidates: list[tuple] = []  # (intent, primary_chunks, fallback_chunks, max_targets)
    for intent in edit_intents:
        # Include the verbatim quote so BM25 gets the exact terms the speaker
        # used — critical for needle-in-haystack retrieval on large corpora where
        # paraphrased topics alone are not distinctive enough.
        query_text = " ".join(
            p for p in (
                intent.affected_topic, intent.old_value, intent.new_value,
                " ".join(intent.verbatim_snippets or []),
            ) if p
        ).strip()
        cross_cutting = is_cross_cutting(intent)
        top_k = config.cross_cutting_top_k if cross_cutting else config.retrieval_top_k
        max_targets = (
            config.cross_cutting_max_targets if cross_cutting
            else config.max_targets_per_intent
        )
        # When the meeting named one document/company, search WITHIN that
        # document's sections (restrict_paths) — many documents share an identical
        # section (e.g. "Key Operational Parameters"), so the named company's copy
        # would never survive a global top-k cut otherwise. There is deliberately
        # NO global fallback for a scoped change: a change addressed to a named
        # company must never land on a DIFFERENT company's document. If the named
        # company has no qualifying section, the change is left unmatched (the
        # diagnostics explain it) rather than silently edited elsewhere.
        scope = target_paths and not cross_cutting
        if scope:
            results = retriever.query(
                query_text, top_k=max(top_k, 20), restrict_paths=target_paths
            )
        else:
            results = retriever.query(query_text, top_k=top_k)
        primary = [r.chunk for r in results]
        intent_candidates.append((intent, primary, max_targets))

    # Removal intents: pick the target document, resolve which sections to drop.
    removal_targets: list[tuple] = []  # (intent, chunk) per section to delete
    if removal_intents:
        from agents_local.structural import RemovalResolverAgent  # noqa: PLC0415

        resolver = RemovalResolverAgent()

        # Positional removals ("the last 4 points", "sections 5-8") need the
        # ordered section list and an LLM to resolve the position. Named/topical
        # removals ("no benefits", "drop the travel section") map directly to the
        # strongest-retrieved section, which is far more reliable than asking an
        # LLM to connect loose wording to a specific heading.
        _POSITIONAL_RE = re.compile(
            r"\b(last|first|final|bottom|top|next|preceding|following)\b"
            r"|\d+\s*(to|through|thru|[-–])\s*\d+|sections?\s+\d",
            re.IGNORECASE,
        )

        async def _plan_removal(intent) -> list[tuple]:
            snippet = " ".join(intent.verbatim_snippets or []) or intent.new_value or ""
            instruction = (
                f'Instruction: "{snippet}". '
                f"Topic: {intent.affected_topic}. "
                f"Intended change: {intent.new_value}."
            )
            # Enrich the doc-selection query with the snippet so a removal that
            # only names its document in the surrounding sentence still routes to
            # the right file. If the meeting named one document, search within it
            # first; only fall back to a global search if it has no match.
            rq = f"{intent.affected_topic} {intent.new_value} {snippet}".strip()
            if target_paths:
                results = retriever.query(
                    rq, top_k=max(config.retrieval_top_k, 20), restrict_paths=target_paths
                )
                if not results:
                    results = retriever.query(rq, top_k=config.retrieval_top_k)
            else:
                results = retriever.query(rq, top_k=config.retrieval_top_k)
            if not results:
                return []
            target_path = results[0].chunk.source_path
            sections = _ordered_sections(all_chunks, target_path)
            headings = [c.section_heading for c in sections]
            by_heading = {c.section_heading: c for c in sections}

            text = f"{instruction} {snippet}"
            is_positional = bool(_POSITIONAL_RE.search(text))

            # Positional removal ("the last 4", "sections 5-8"): resolve position
            # against the ordered heading list of the strongest-matching doc.
            if is_positional:
                to_delete = await resolver.resolve(instruction, headings)
                return [(intent, by_heading[h]) for h in to_delete if h in by_heading]

            # Named / topical removal: match the SUBJECT to a heading across all
            # documents (deterministic) so the delete lands on the section
            # actually named, not a semantically-near one.
            from agents_local.structural import best_named_removal_target  # noqa: PLC0415

            target = best_named_removal_target(intent, all_chunks)
            if target is not None:
                return [(intent, target)]

            # Last resort: let the resolver try the strongest doc's heading list.
            to_delete = await resolver.resolve(instruction, headings)
            return [(intent, by_heading[h]) for h in to_delete if h in by_heading]

        removal_plans = await asyncio.gather(
            *[_plan_removal(i) for i in removal_intents]
        )
        removal_targets = [pair for plan in removal_plans for pair in plan]

    # ── Stage 5: evaluation ───────────────────────────────────────────────────
    await _emit(q, "evaluation")
    from agents_local.evaluation import EvaluationAgent  # noqa: PLC0415

    eval_agent = EvaluationAgent()

    # Deterministic floor for an exact structured-field match. The LLM evaluator
    # is noisy on table rows (an exact "Maximum batch size" match scores anywhere
    # from 0.5 to 0.9 run-to-run); when the section literally declares the field
    # the intent names, treat it as a confident match regardless of the model.
    _LABEL_FLOOR = 0.85

    async def _score_pool(intent, chunks: list[ChunkRecord], max_targets: int) -> list[tuple]:
        """Keep EVERY section that scores above threshold (deduped, capped)."""
        if not chunks:
            return []
        scores = await asyncio.gather(*[eval_agent.score(intent, c) for c in chunks])
        scores = [
            max(s, _LABEL_FLOOR) if _field_label_match(intent, c) else s
            for c, s in zip(chunks, scores)
        ]
        ranked = sorted(zip(chunks, scores), key=lambda cs: cs[1], reverse=True)
        kept: list[tuple] = []
        seen: set[tuple[str, str]] = set()
        for chunk, score in ranked:
            if score < config.relevance_threshold:
                continue
            key = (chunk.source_path, chunk.section_heading)
            if key in seen:
                continue
            seen.add(key)
            kept.append((intent, chunk))
            if len(kept) >= max_targets:
                break
        return kept

    eval_results = await asyncio.gather(
        *[_score_pool(i, chunks, mt) for i, chunks, mt in intent_candidates]
    )
    qualified: list[tuple] = [pair for rl in eval_results for pair in rl]

    # ── Stage 6: drafting ─────────────────────────────────────────────────────
    await _emit(q, "drafting")
    from agents_local.editor import LocalDocEditorAgent  # noqa: PLC0415

    editor = LocalDocEditorAgent()
    drafts = await asyncio.gather(
        *[editor.draft(intent, chunk.content) for intent, chunk in qualified]
    ) if qualified else []

    # ── Stage 7: verification ─────────────────────────────────────────────────
    await _emit(q, "verification")
    from agents_local.verifier import VerifierAgent  # noqa: PLC0415

    verifier = VerifierAgent()
    verifications = await asyncio.gather(
        *[
            verifier.verify(
                draft.before_content,
                draft.after_content,
                f"{intent.intent_type}: {intent.new_value}",
            )
            for (intent, chunk), draft in zip(qualified, drafts)
        ]
    ) if qualified else []

    # ── Stage 8: ready_for_review ─────────────────────────────────────────────
    await _emit(q, "ready_for_review")
    proposals: list[LocalDocProposal] = []

    # Edit proposals — suppress no-ops and clearly-unfulfilled drafts.
    for (intent, chunk), draft, verification in zip(qualified, drafts, verifications):
        before = draft.before_content
        after = draft.after_content
        if after.strip() == before.strip():
            logger.info("run_pipeline: dropping no-op edit on %r", chunk.section_heading)
            continue
        if verification.intent_fulfillment < config.min_fulfillment:
            logger.info(
                "run_pipeline: dropping unfulfilled edit on %r (fulfillment=%.2f)",
                chunk.section_heading, verification.intent_fulfillment,
            )
            continue
        proposals.append(
            _make_proposal(
                config, intent, chunk, before, after,
                edit_type=draft.edit_type if draft.edit_type in
                ("replace", "append", "delete_section") else "replace",
                factual=verification.factual_consistency,
                formatting=verification.formatting_integrity,
                fulfillment=verification.intent_fulfillment,
                quality=verification.quality_score,
                note=verification.verifier_note,
            )
        )

    # Rename proposals — deterministic literal replacement across the corpus.
    if rename_intents:
        from agents_local.structural import (  # noqa: PLC0415
            find_rename_targets,
            infer_replacement_target,
            looks_like_name,
        )

        inferred: Optional[str] = None
        rename_seen: set[tuple[str, str]] = set()
        for intent in rename_intents:
            new_value = (intent.new_value or "").strip()
            # Guard: only act on rename targets that actually look like a name —
            # an over-eager extractor can mistake "update everywhere" for one.
            if not looks_like_name(new_value):
                logger.info("run_pipeline: skipping implausible rename target %r", new_value)
                continue
            old_value = (intent.old_value or "").strip()
            # If the transcript gave no old value (or one that isn't in the
            # docs), infer the dominant cross-document brand name.
            if not old_value or not find_rename_targets(old_value, all_chunks):
                if inferred is None:
                    inferred = infer_replacement_target(all_chunks) or ""
                old_value = inferred
            if not old_value:
                continue
            pattern = re.compile(re.escape(old_value), re.IGNORECASE)
            # A rename must reach every section that mentions the old value across
            # the whole corpus — round-robin per document for full coverage.
            for chunk, original in find_rename_targets(old_value, all_chunks):
                seen_key = (chunk.source_path, chunk.section_heading)
                if seen_key in rename_seen:
                    continue
                new_body = pattern.sub(new_value, original)
                if new_body == original:
                    continue
                rename_seen.add(seen_key)
                proposals.append(
                    _make_proposal(
                        config, intent, chunk, original, new_body,
                        edit_type="replace",
                        factual=1.0, formatting=1.0, fulfillment=1.0, quality=0.9,
                        note=(
                            f"Replaces '{old_value}' with '{new_value}' in this "
                            f"section (deterministic rename)."
                        ),
                    )
                )

    # Removal proposals — whole-section deletions.
    seen_removals: set[tuple[str, str]] = set()
    for intent, chunk in removal_targets:
        key = (chunk.source_path, chunk.section_heading)
        if key in seen_removals:
            continue
        seen_removals.add(key)
        proposals.append(
            _make_proposal(
                config, intent, chunk, chunk.content, "",
                edit_type="delete_section",
                factual=1.0, formatting=1.0, fulfillment=1.0, quality=0.9,
                note="Proposes removing this entire section.",
            )
        )

    _record_diagnostics(diagnostics, intents, proposals, classify_kind)
    logger.info(
        "run_pipeline: completed — session=%s proposals=%d unmatched=%d",
        config.session_id,
        len(proposals),
        len(intents) - len({id(p.intent) for p in proposals}),
    )
    return proposals
