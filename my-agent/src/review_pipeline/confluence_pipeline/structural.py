"""Structural change handling: classify intents and resolve renames/removals.

The standard pipeline (retrieve -> evaluate -> draft -> verify) is built for
*localized* edits: change a number, tweak a sentence. Two common meeting
outcomes do not fit that shape and were being dropped or mis-applied:

- A **rename / rebrand** ("we're renaming the company to X") is a literal
  value replacement that should touch *every* section mentioning the old value
  across the whole corpus — not a single best-matching section.
- A **removal** ("remove the last 4 security points", "no benefits anymore")
  deletes one or more whole sections. The editor cannot express that as a
  before/after value swap, so it used to emit a no-op card.

This module classifies each extracted intent into one of three kinds and
provides deterministic helpers for the rename and removal kinds. Classification
is keyword-based on the *extracted* intent text — domain-agnostic, with no
hardcoding of any specific document's wording.
"""

from __future__ import annotations

import logging
import os
import re
from collections import Counter

from agents import Agent, Runner
from pydantic import BaseModel

from .llm_runtime import guarded_run
from .models import ChunkRecord, ConfluenceIntent

logger = logging.getLogger(__name__)


# ── Intent kind classification ────────────────────────────────────────────────

_RENAME_RE = re.compile(
    r"\b(rename|renamed|renaming|rebrand|rebranding|renanme)\b", re.IGNORECASE
)
_REMOVAL_RE = re.compile(
    r"\b(remove|removing|removal|removed|delete|deleted|deleting|deletion|"
    r"drop|dropping|dropped|eliminate|eliminated|strike|discontinue|"
    r"scrap|scrapped|get rid of|take out|no longer)\b",
    re.IGNORECASE,
)
_NAME_RE = re.compile(r"\bname\b", re.IGNORECASE)
_NAME_VERB_RE = re.compile(
    r"\b(to|is now|now|changed|change|changing|switch|switching|becomes?)\b",
    re.IGNORECASE,
)


_CROSS_CUTTING_RE = re.compile(
    r"\b(everywhere|every document|every doc|every file|every policy|"
    r"all documents|all docs|all files|all policies|all of (our|the) (docs|documents|policies)|"
    r"across all|across the board|company[- ]wide|org[- ]wide|globally|throughout|"
    r"wherever (it|they) (appear|appears|is|are) (mentioned|referenced)?|"
    r"in every (document|doc|file|policy))\b",
    re.IGNORECASE,
)


def is_cross_cutting(intent: ConfluenceIntent) -> bool:
    """True if a change is meant to propagate across MANY documents at once.

    A normal edit targets a single section; a cross-cutting edit ("change X to Y
    in every document", "update this across all policies") must reach every
    section that documents the same thing.
    """
    return bool(_CROSS_CUTTING_RE.search(_intent_text(intent)))


def _intent_text(intent: ConfluenceIntent) -> str:
    """Flatten the human-meaningful fields of an intent into one string."""
    parts = [
        intent.affected_topic or "",
        intent.new_value or "",
        intent.old_value or "",
        intent.rationale or "",
        " ".join(intent.verbatim_snippets or []),
    ]
    return " ".join(parts)


def has_explicit_removal_verb(intent: ConfluenceIntent) -> bool:
    """True only if a removal verb appears in the speaker's QUOTED words.

    Removals delete whole sections and skip the relevance-score gate that guards
    edits, so an over-eager extractor that merely *infers* a removal from casual
    chatter (verb only in its rationale/new_value) would silently delete a real
    section. Requiring the verb in verbatim_snippets keeps genuine "remove the X
    section" instructions while rejecting inferred ones — a precision gate for
    the otherwise un-gated removal branch.
    """
    return bool(_REMOVAL_RE.search(" ".join(intent.verbatim_snippets or [])))


# Phrases with which a speaker confirms a value is STAYING as it is. Deliberately
# high-precision: each one is an explicit "not moving" marker, not merely a mention of
# the current state, because a false positive here silently drops a real change.
_REAFFIRM_RE = re.compile(
    r"\b("
    r"stays? (?:at|the same|as (?:is|it is)|put)|"
    r"staying (?:at|the same|as is)|"
    r"remains? (?:at|the same|unchanged|as is)|"
    r"remaining (?:at|the same|unchanged)|"
    r"unchanged|"
    r"(?:are|is|'re|were) not changing|not changing (?:it|that|this|those)|"
    r"(?:no|isn't|is not|won't be|will not be) chang(?:e|es|ing)(?: there| to (?:it|that|this))?|"
    r"keep(?:ing)? (?:it|that|this|them) (?:that way|the same|as is|as it is|where it is)|"
    r"leav(?:e|ing) (?:it|that|this|them) (?:as is|as it is|alone|where it is|the same)|"
    r"stick(?:ing)? with|"
    r"comfortable (?:with )?where (?:it|they|we) (?:is|are|sit|sits)|"
    r"happy with (?:it|that|the way)"
    r")\b",
    re.IGNORECASE,
)


def is_reaffirmation_phrasing(intent: ConfluenceIntent) -> bool:
    """True when the speaker's QUOTED words confirm a value is staying as it is.

    The deterministic half of the reaffirmation guard, and the analogue of
    :func:`has_explicit_removal_verb`: it reads only ``verbatim_snippets``, never the
    extractor's own paraphrase, so a model that mislabels a "we're keeping it" clause
    as a change is still caught. The contrastive form — "they charge X; ours stays at
    Y" — is exactly where the model's label is unreliable, and exactly where these
    markers are present.
    """
    return bool(_REAFFIRM_RE.search(" ".join(intent.verbatim_snippets or [])))


def classify_kind(intent: ConfluenceIntent) -> str:
    """Return "rename", "removal", or "edit" for a single intent.

    Precedence: rename > removal > edit. Rename wins because a "rename the X to
    Y" instruction can also contain change verbs that would otherwise look like
    an edit; removal wins over edit for the same reason.
    """
    text = _intent_text(intent)
    topic = (intent.affected_topic or "").strip().lower()

    # Rename: explicit rename verbs, or a "name" change phrasing.
    if _RENAME_RE.search(text):
        return "rename"
    if _NAME_RE.search(text) and _NAME_VERB_RE.search(text):
        return "rename"
    if topic.endswith("name") or topic.endswith("naming") or "rebrand" in topic:
        return "rename"

    # Removal: explicit removal verbs, or a "no <thing>" framing.
    if _REMOVAL_RE.search(text):
        return "removal"
    if re.match(r"^\s*no\s+\w", intent.new_value or "", re.IGNORECASE):
        return "removal"
    # "no benefits will be there", "no more X", "there will be no X" anywhere.
    if re.search(
        r"\bno\s+(more\s+)?\w+(\s+\w+)?\s+(will|under|anymore|left|offered|"
        r"available|going forward)",
        text,
        re.IGNORECASE,
    ):
        return "removal"
    if re.search(r"\b(will be|there('s| is| will be)) no\s+\w", text, re.IGNORECASE):
        return "removal"

    return "edit"


# ── Rename helpers (literal corpus-wide value replacement) ────────────────────

# Words that frequently start a sentence and look like proper nouns but are not
# brand names — excluded from company-name inference.
_COMMON_LEADING = {
    "the", "this", "that", "these", "those", "every", "all", "each", "any",
    "employees", "employee", "company", "business", "customer", "customers",
    "we", "our", "new", "users", "user", "data", "access", "support",
    "security", "section", "policy", "service", "services", "team", "teams",
}


def _proper_noun_candidates(text: str) -> list[str]:
    """Extract 2-3 word Capitalized sequences that look like proper nouns."""
    # Sequences of capitalized words (allowing internal & like "Foo & Bar").
    seqs = re.findall(r"\b([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+){1,2})\b", text)
    out: list[str] = []
    for s in seqs:
        first = s.split()[0].lower()
        if first in _COMMON_LEADING:
            continue
        out.append(s.strip())
    return out


def infer_replacement_target(chunks: list[ChunkRecord]) -> str | None:
    """Infer the dominant cross-document proper noun (the company/brand name).

    Heuristic: the brand appears in many *different* documents. Count, for each
    candidate proper noun, how many distinct source files it occurs in; pick the
    one present in the most files (tie-break by total occurrences). Requires the
    winner to appear in at least 2 distinct documents so a phrase that is common
    in a single doc is not mistaken for a brand.
    """
    doc_presence: dict[str, set[str]] = {}
    total_count: Counter[str] = Counter()
    for c in chunks:
        text = f"{c.section_heading}\n{c.content}"
        # Normalise ALL-CAPS headings to Title Case so "NIMBUS ROBOTICS" and
        # "Nimbus Robotics" are counted as the same candidate.
        normalised = re.sub(
            r"\b[A-Z]{2,}(?:\s+[A-Z]{2,})+\b",
            lambda m: m.group(0).title(),
            text,
        )
        for cand in _proper_noun_candidates(normalised):
            doc_presence.setdefault(cand, set()).add(c.source_path)
            total_count[cand] += 1

    best: str | None = None
    best_key = (0, 0)
    for cand, docs in doc_presence.items():
        key = (len(docs), total_count[cand])
        if len(docs) >= 2 and key > best_key:
            best_key = key
            best = cand
    return best


# Verbs / filler that signal an INSTRUCTION phrase rather than a real name —
# guards against an over-eager extractor turning "update everywhere" into a
# rename target.
_NOT_A_NAME_LEADING = {
    "update", "updating", "change", "changing", "reflect", "apply", "use",
    "make", "ensure", "keep", "remove", "rename", "rebrand", "call", "set",
    "everything", "everywhere", "all", "the", "our", "a", "an", "new",
}


def looks_like_name(value: str) -> bool:
    """True if *value* plausibly is a proper name (company/brand/team).

    A real name is short and contains a capitalized word; an instruction phrase
    like "update everywhere" or "reflect the new name" is not a rename target.
    """
    value = (value or "").strip()
    if not value:
        return False
    tokens = value.split()
    if len(tokens) > 5:
        return False
    if tokens[0].lower() in _NOT_A_NAME_LEADING:
        return False
    has_capitalized = any(t[:1].isupper() for t in tokens if t[:1].isalpha())
    return has_capitalized


def find_rename_targets(
    old_value: str,
    chunks: list[ChunkRecord],
    max_targets: int = 200,
    per_doc: int = 8,
) -> list[tuple[ChunkRecord, str]]:
    """Find sections whose body contains *old_value* (case-insensitive).

    A rename must reach EVERY document that mentions the old value. A flat cap
    iterated in chunk order exhausts the budget on the first few documents, so
    matches are grouped per document and taken round-robin (one section per doc
    per round, up to *per_doc* rounds). This guarantees coverage across the whole
    corpus before adding extra sections within any single document.

    Returns (chunk, original_content) pairs; the caller substitutes the new value.
    """
    pattern = re.compile(re.escape(old_value), re.IGNORECASE)
    by_doc: dict[str, list[ChunkRecord]] = {}
    seen: set[tuple[str, str]] = set()
    for c in chunks:
        if not pattern.search(c.content):
            continue
        key = (c.source_path, c.section_heading)
        if key in seen:
            continue
        seen.add(key)
        by_doc.setdefault(c.source_path, []).append(c)

    targets: list[tuple[ChunkRecord, str]] = []
    for r in range(per_doc):
        for secs in by_doc.values():
            if r < len(secs):
                targets.append((secs[r], secs[r].content))
                if len(targets) >= max_targets:
                    return targets
    return targets


# ── Removal helper (which whole sections to delete) ───────────────────────────


# Generic structural / filler words that must not drive a named-removal match
# (every doc has "Policy"/"Section" headings; removal verbs are not subjects).
_NAMED_STOPWORDS = {
    "the", "a", "an", "of", "and", "or", "for", "to", "in", "on", "at", "is",
    "are", "be", "will", "under", "this", "that", "new", "company", "our",
    "policy", "policies", "section", "sections", "procedure", "procedures",
    "standard", "standards", "sop", "guide", "handbook", "overview",
    "remove", "removing", "removal", "removed", "delete", "deleting", "drop",
    "dropping", "no", "longer", "anymore", "whole", "entire", "bit", "part",
    "stuff", "thing", "things", "get", "rid", "out", "take", "scrap",
    "eliminate", "discontinue", "everything", "else",
}


def subject_keywords(intent: ConfluenceIntent) -> set[str]:
    """The content words that name WHAT a removal targets (minus filler/verbs)."""
    text = (
        f"{intent.affected_topic} {intent.new_value} "
        f"{' '.join(intent.verbatim_snippets or [])}"
    )
    toks = re.findall(r"[a-zA-Z]{3,}", text.lower())
    return {t for t in toks if t not in _NAMED_STOPWORDS}


def heading_match_score(heading: str, keywords: set[str]) -> int:
    """How many subject keywords appear in a section heading."""
    htoks = set(re.findall(r"[a-zA-Z]{3,}", heading.lower()))
    return len(htoks & keywords)


def best_named_removal_target(
    intent: ConfluenceIntent, chunks: list[ChunkRecord]
) -> ChunkRecord | None:
    """Pick the section a named removal targets by heading-keyword overlap.

    Scans EVERY section across all documents (not just retrieved ones) so the
    deletion lands on the section actually named — e.g. "deprecation policy" ->
    the "Deprecation Policy" heading, never a semantically-near "Deprovisioning"
    section. Returns None when no heading shares a subject keyword.
    """
    keywords = subject_keywords(intent)
    if not keywords:
        return None
    best: ChunkRecord | None = None
    best_score = 0
    seen: set[tuple[str, str]] = set()
    for c in chunks:
        if c.section_index == 0:
            continue  # never delete a document's title/intro on a topical match
        key = (c.source_path, c.section_heading)
        if key in seen:
            continue
        seen.add(key)
        score = heading_match_score(c.section_heading, keywords)
        if score > best_score:
            best_score = score
            best = c
    return best if best_score > 0 else None


class _RemovalPlan(BaseModel):
    headings_to_delete: list[str]
    reasoning: str


REMOVAL_INSTRUCTIONS = """\
You decide which whole sections of a document a removal instruction targets.

You are given the document's section headings IN ORDER (top to bottom) and a
removal instruction from a meeting. Return the exact headings (copied verbatim
from the provided list) that should be deleted.

Rules:
- Only return headings that appear in the provided list; never invent one.
- Positional language refers to the ordered content sections, ignoring any
  leading title/overview/intro heading: "the last 4 points" = the final 4
  content sections; "the first two" = the first 2 content sections.
- Named / topical language selects the section that is ABOUT that subject. Match
  generously on the subject keyword: if the instruction is about a topic
  (e.g. "benefits", "travel", "refunds") and a heading contains or is clearly
  about that subject, select it — even if the wording differs (e.g. "no benefits
  anymore" -> the "Benefits Enrollment" section; "drop travel reimbursement" ->
  the "Travel and Reimbursement" section).
- "No more X" / "X will be removed" / "we're getting rid of X" all mean: delete
  the section about X.
- A range ("sections 5 to 8") selects each section in that range.
- Only return an empty list when no heading is even plausibly about the subject.
"""


class RemovalResolverAgent:
    """Resolves a removal instruction to the exact section headings to delete."""

    def __init__(self, model: str = None):
        self.model = model or os.getenv("LDOC_REMOVAL_MODEL", "gpt-4o-mini")
        self._agent = Agent(
            name="ConfluenceRemovalResolver",
            model=self.model,
            instructions=REMOVAL_INSTRUCTIONS,
            output_type=_RemovalPlan,
        )

    async def resolve(
        self, instruction: str, ordered_headings: list[str]
    ) -> list[str]:
        """Return the subset of ordered_headings the instruction targets."""
        if not ordered_headings:
            return []
        listing = "\n".join(f"{i + 1}. {h}" for i, h in enumerate(ordered_headings))
        prompt = (
            f"Removal instruction:\n{instruction}\n\n"
            f"Document sections in order:\n{listing}"
        )
        try:
            result = await guarded_run(self._agent, prompt)
            plan: _RemovalPlan = result.final_output
            valid = set(ordered_headings)
            return [h for h in plan.headings_to_delete if h in valid]
        except Exception:
            logger.error("RemovalResolverAgent.resolve() failed", exc_info=True)
            return []
