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

from models import ChunkRecord, LocalDocIntent

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


def _intent_text(intent: LocalDocIntent) -> str:
    """Flatten the human-meaningful fields of an intent into one string."""
    parts = [
        intent.affected_topic or "",
        intent.new_value or "",
        intent.old_value or "",
        intent.rationale or "",
        " ".join(intent.verbatim_snippets or []),
    ]
    return " ".join(parts)


def classify_kind(intent: LocalDocIntent) -> str:
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


def find_rename_targets(
    old_value: str,
    chunks: list[ChunkRecord],
    max_targets: int = 12,
) -> list[tuple[ChunkRecord, str]]:
    """Find every section whose body contains *old_value* (case-insensitive).

    Returns a list of (chunk, new_body) pairs where new_body is the section
    content with all occurrences of old_value replaced — preserving the original
    casing pattern is not attempted; the caller's new_value is inserted verbatim.
    """
    pattern = re.compile(re.escape(old_value), re.IGNORECASE)
    targets: list[tuple[ChunkRecord, str]] = []
    seen: set[tuple[str, str]] = set()
    for c in chunks:
        if not pattern.search(c.content):
            continue
        key = (c.source_path, c.section_heading)
        if key in seen:
            continue
        seen.add(key)
        targets.append((c, c.content))  # new_body filled in by caller
        if len(targets) >= max_targets:
            break
    return targets


# ── Removal helper (which whole sections to delete) ───────────────────────────


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
            name="LocalDocRemovalResolver",
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
            result = await Runner.run(self._agent, prompt)
            plan: _RemovalPlan = result.final_output
            valid = set(ordered_headings)
            return [h for h in plan.headings_to_delete if h in valid]
        except Exception:
            logger.error("RemovalResolverAgent.resolve() failed", exc_info=True)
            return []
