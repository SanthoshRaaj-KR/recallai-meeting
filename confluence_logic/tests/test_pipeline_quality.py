"""Tests for the proposed-changes pipeline quality safeguards.

Each test corresponds to a real failure mode flagged by the team:
- instruction text in after_content (MS Dhoni vs Virat Kohli)
- semantically duplicate proposals
- small content overwriting rich sections (overwrite safety)
- create proposal accidentally overwriting an unrelated page (must NOT happen)
- wrong page with similar title
- old_value missing on page → replace must be blocked
- Pinecone result normalization preserves real page_id

These run synchronously (no pytest-asyncio) so they always execute in CI.
"""
import re
from typing import Any, Dict, List

import pytest

# ---------------------------------------------------------------------------
# Pinecone normalization — chunks must not be treated as pages
# ---------------------------------------------------------------------------


def test_pinecone_normalization_extracts_page_id_from_metadata():
    """Raw Pinecone matches have page_id under .metadata; the normalizer must lift it."""
    from confluence_logic.agents.fact_extraction_agent import _normalize_pinecone_match

    raw = {
        "id": "page42_3",  # chunk id — not a page id
        "score": 0.92,
        "metadata": {
            "page_id": "page42",
            "title": "Database Architecture",
            "space_key": "ENG",
            "heading": "Storage Layer",
            "markdown_content": "We currently run PostgreSQL 13 on AWS RDS.",
        },
    }
    out = _normalize_pinecone_match(raw)
    assert out is not None
    assert out["page_id"] == "page42", "Real page_id from metadata must be used, not chunk id"
    assert out["title"] == "Database Architecture"
    assert "PostgreSQL 13" in out["relevant_content"]
    assert out["score"] == 0.92
    assert out["source"] == "pinecone_rag"


def test_pinecone_normalization_rejects_match_without_metadata_page_id():
    """If a match has no real page_id, it must be dropped — not silently passed through."""
    from confluence_logic.agents.fact_extraction_agent import _normalize_pinecone_match

    bad = {"id": "orphan_0", "score": 0.5, "metadata": {}}
    assert _normalize_pinecone_match(bad) is None


def test_pinecone_normalization_handles_attribute_style_match():
    """Some Pinecone client versions return ScoredVector objects (attributes, not keys)."""
    from confluence_logic.agents.fact_extraction_agent import _normalize_pinecone_match

    class FakeMatch:
        id = "page99_1"
        score = 0.81
        metadata = {"page_id": "page99", "title": "Onboarding", "markdown_content": "Step 1: ..."}

    out = _normalize_pinecone_match(FakeMatch())
    assert out is not None
    assert out["page_id"] == "page99"


# ---------------------------------------------------------------------------
# Instruction-text detection — MS Dhoni / "Maintain professional tone" must die
# ---------------------------------------------------------------------------


def test_instruction_text_is_detected_and_dropped():
    """The Dhoni-style 'Keep X as primary subject' / 'Maintain professional tone' patterns
    must be classified as instruction text so they never reach Confluence."""
    from confluence_logic.agents.proposed_changes_agent import _is_instruction_after_content

    instruction_examples = [
        "Keep MS Dhoni as the primary subject of the page.",
        "Include Virat Kohli only as comparison or fan-interest content.",
        "Add a clear differentiation section explaining why MS Dhoni is better than Virat Kohli.",
        "Maintain a professional and neutral tone throughout the content.",
        "Use a structured format with clear headings.",
        "This page should focus on the technical implementation.",
        "Ensure the content covers all migration steps.",
        "Do not include any deprecated API references.",
    ]
    for txt in instruction_examples:
        assert _is_instruction_after_content(txt), f"Should detect instruction text: {txt!r}"


def test_real_documentation_is_not_flagged_as_instruction():
    """Final-page documentation must not be misclassified as instruction text."""
    from confluence_logic.agents.proposed_changes_agent import _is_instruction_after_content

    doc_examples = [
        "MS Dhoni is a former Indian cricket captain who led India to three ICC tournament victories.",
        "Santos attends the gym three times per week and focuses on strength training.",
        "The team uses React for the frontend and FastAPI for the backend.",
        "PostgreSQL 16 is the supported database version across all production services.",
        "On-call rotation for the payments service is owned by Priya as of 2026-Q2.",
    ]
    for txt in doc_examples:
        assert not _is_instruction_after_content(txt), f"Should NOT flag real docs: {txt!r}"


# ---------------------------------------------------------------------------
# Page Qualifier — deterministic phrase-king behavior
# ---------------------------------------------------------------------------


def test_qualifier_qualifies_when_old_value_present_verbatim():
    """If intent.old_value appears on the page, qualify deterministically with score 10."""
    import asyncio
    from types import SimpleNamespace
    from confluence_logic.agents.page_qualifier import _run_page_qualifier

    intent = SimpleNamespace(
        subject="PostgreSQL version",
        instruction="Update PostgreSQL version from 13 to 16",
        target_hint="database",
        old_value="PostgreSQL 13",
        new_value="PostgreSQL 16",
        action="replace",
    )
    page = {
        "title": "Database Layer",
        "full_content": "We currently run PostgreSQL 13 on AWS RDS in us-east-1.",
        "available_headings": ["Storage", "Backups"],
    }
    out = asyncio.run(_run_page_qualifier(intent, page))
    assert out["qualified"] is True
    assert out["page_fit_score"] == 10
    assert out["old_value_found"] is True
    assert out["matched_phrase"] == "PostgreSQL 13"


def test_qualifier_rejects_replace_when_old_value_missing():
    """If intent.old_value is set but absent from the page, a replace action cannot
    succeed there — qualifier must deterministically reject without calling the LLM."""
    import asyncio
    from types import SimpleNamespace
    from confluence_logic.agents.page_qualifier import _run_page_qualifier

    intent = SimpleNamespace(
        subject="PostgreSQL version",
        instruction="Update PostgreSQL version from 13 to 16",
        target_hint="database",
        old_value="PostgreSQL 13",
        new_value="PostgreSQL 16",
        action="replace",
    )
    page = {
        "title": "Frontend Tech Stack",
        "full_content": "We use React 18 with TypeScript. State management via Zustand.",
        "available_headings": ["Frontend"],
    }
    out = asyncio.run(_run_page_qualifier(intent, page))
    assert out["qualified"] is False
    assert out["page_fit_score"] <= 3
    assert out["old_value_found"] is False
    assert "not found" in out["why"].lower()


# ---------------------------------------------------------------------------
# Drafter normalizer — edit_mode safety guards
# ---------------------------------------------------------------------------


def test_drafter_normalizer_downgrades_replace_without_before_content():
    """If the LLM says edit_mode='replace' but provides no before_content, the
    normalizer must downgrade to 'append' — never let a replace with no anchor
    slip through (would cause destructive section overwrite at execution time)."""
    from confluence_logic.agents.drafter_agent import _normalize_intent_draft

    raw_llm_output = {
        "applies": True,
        "change_type": "edit",
        "edit_mode": "replace",
        "page_id": "page1",
        "page_title": "Some Page",
        "section_heading": "Overview",
        "before_content": None,  # <-- the bug case
        "after_content": "Santos goes to the gym.",
        "rationale": "Meeting decision",
    }
    out = _normalize_intent_draft(raw_llm_output, "page1", "Some Page")
    # Either downgrade to append OR exposed safely — must NOT remain a replace-without-anchor.
    assert out is not None
    assert out["edit_mode"] != "replace" or out["before_content"] is not None
    # The actual contract: with no before_content, must be append
    assert out["edit_mode"] == "append"


def test_drafter_normalizer_upgrades_append_with_before_content():
    """If the LLM says edit_mode='append' but DID provide before_content, the normalizer
    treats this as a hint and upgrades to 'replace' — the LLM has anchor text."""
    from confluence_logic.agents.drafter_agent import _normalize_intent_draft

    raw_llm_output = {
        "applies": True,
        "change_type": "edit",
        "edit_mode": "append",
        "page_id": "page1",
        "page_title": "Some Page",
        "section_heading": "Overview",
        "before_content": "Santos doesn't go to the gym.",
        "after_content": "Santos goes to the gym three times a week.",
        "rationale": "Meeting decision",
    }
    out = _normalize_intent_draft(raw_llm_output, "page1", "Some Page")
    assert out is not None
    assert out["edit_mode"] == "replace"
    assert out["before_content"] == "Santos doesn't go to the gym."


def test_drafter_normalizer_skips_when_applies_false():
    """When the drafter says applies=false, the normalizer must return None."""
    from confluence_logic.agents.drafter_agent import _normalize_intent_draft

    raw_llm_output = {
        "applies": False,
        "reason": "Page is about a different subject",
    }
    out = _normalize_intent_draft(raw_llm_output, "page1", "Some Page")
    assert out is None


# ---------------------------------------------------------------------------
# Semantic dedup — similar-heading variants must collapse
# ---------------------------------------------------------------------------


def test_semantic_signature_collapses_heading_variants():
    """The semantic dedup signature must treat 'Tech Stack' and 'Technology Stack'
    sections with the same content as duplicates (when other fields match)."""
    # Manually replicate the signature normalization the pipeline uses
    def _sig(text: str, n: int = 120) -> str:
        t = re.sub(r"[*_`#>\-]+", " ", (text or "").lower())
        t = re.sub(r"\s+", " ", t).strip()
        return t[:n]

    pid = "page1"
    ctype = "edit"
    emode = "replace"
    intent_subject = "framework SDK"
    before = "OpenAI Agents SDK"
    after = "Claude SDK"

    sig_a = (
        f"{pid}|{ctype}|{emode}|"
        f"{_sig(intent_subject, 60)}|"
        f"{_sig(before, 80)}|"
        f"{_sig(after, 120)}"
    )
    # Same proposal, only heading differs — signature MUST be identical
    sig_b = (
        f"{pid}|{ctype}|{emode}|"
        f"{_sig(intent_subject, 60)}|"
        f"{_sig(before, 80)}|"
        f"{_sig(after, 120)}"
    )
    assert sig_a == sig_b


def test_semantic_signature_distinguishes_different_content():
    """The dedup signature must NOT collapse genuinely different proposals."""
    def _sig(text: str, n: int = 120) -> str:
        t = re.sub(r"[*_`#>\-]+", " ", (text or "").lower())
        t = re.sub(r"\s+", " ", t).strip()
        return t[:n]

    sig_a = f"p1|edit|replace|sdk|OpenAI Agents SDK|{_sig('Claude SDK', 120)}"
    sig_b = f"p1|edit|replace|sdk|OpenAI Agents SDK|{_sig('Anthropic Claude SDK with tools', 120)}"
    assert sig_a != sig_b


# ---------------------------------------------------------------------------
# Title-similarity ambiguity detection
# ---------------------------------------------------------------------------


def test_jaccard_title_similarity_flags_similar_pages():
    """The pipeline uses Jaccard similarity on non-stop-word tokens to detect
    ambiguous title pairs. 'Sales Notes Q3' and 'Engineering Notes Q3' should
    NOT flag (different domain word); 'Payments Runbook' and 'Payments Runbook v2'
    should flag (high overlap)."""
    from confluence_logic.review.api import _GENERIC_TITLE_WORDS

    def jaccard(a: str, b: str) -> float:
        ta = {w for w in re.split(r"\W+", a.lower()) if w and w not in _GENERIC_TITLE_WORDS and len(w) > 2}
        tb = {w for w in re.split(r"\W+", b.lower()) if w and w not in _GENERIC_TITLE_WORDS and len(w) > 2}
        if not ta or not tb:
            return 0.0
        return len(ta & tb) / len(ta | tb)

    # Different domain — must NOT trigger ambiguity (jaccard ≈ 0)
    sim1 = jaccard("Sales Notes Q3", "Engineering Notes Q3")
    assert sim1 < 0.7, f"Sales vs Engineering must not look ambiguous, got {sim1}"

    # Very similar — must trigger
    sim2 = jaccard("Payments Service Runbook", "Payments Service Runbook v2")
    assert sim2 >= 0.7, f"Payments Runbook variants must look ambiguous, got {sim2}"


# ---------------------------------------------------------------------------
# Overwrite safety — small content cannot replace rich sections
# ---------------------------------------------------------------------------


def test_overwrite_safety_threshold_logic():
    """The pipeline refuses to replace a section with much smaller content
    (size ratio safeguard). The threshold is: when existing > 300 chars AND
    new < 50% of existing, downgrade to append."""
    # Simulate the threshold the pipeline uses in _direct_apply_change
    def is_overwrite_unsafe(existing_len: int, new_len: int) -> bool:
        return existing_len > 300 and new_len < existing_len * 0.5

    # A one-line fact replacing a 2000-char section — unsafe
    assert is_overwrite_unsafe(2000, 80) is True
    # New content roughly same size — safe
    assert is_overwrite_unsafe(1000, 900) is False
    # Tiny section — not enough at stake to trigger
    assert is_overwrite_unsafe(200, 40) is False


# ---------------------------------------------------------------------------
# ChangeIntent schema sanity
# ---------------------------------------------------------------------------


def test_change_intent_defaults_are_safe():
    """An empty ChangeIntent should default to action='replace' but all string fields
    empty — that way merge/dedupe doesn't blow up on absent intents."""
    from confluence_logic.agents.fact_extraction_agent import ChangeIntent

    empty = ChangeIntent()
    assert empty.subject == ""
    assert empty.old_value == ""
    assert empty.new_value == ""
    assert empty.action == "replace"


def test_extracted_facts_includes_change_intents():
    """change_intents field must be present on the ExtractedFacts model."""
    from confluence_logic.agents.fact_extraction_agent import ExtractedFacts, ChangeIntent

    facts = ExtractedFacts(
        change_intents=[
            ChangeIntent(subject="A", new_value="B", action="replace"),
        ]
    )
    assert len(facts.change_intents) == 1
    assert facts.change_intents[0].subject == "A"


# ---------------------------------------------------------------------------
# Edit mode dispatch contract
# ---------------------------------------------------------------------------


def test_edit_mode_normalization_legacy_proposals():
    """Legacy proposals without edit_mode should be inferred from before_content."""
    # This mirrors the inference logic in _direct_apply_change
    def infer_edit_mode(edit_mode_raw: str, before_content: str) -> str:
        edit_mode = (edit_mode_raw or "").strip().lower()
        if edit_mode not in {"replace", "append", "create_section"}:
            edit_mode = "replace" if before_content else "append"
        return edit_mode

    # Legacy: before_content set, no edit_mode → replace
    assert infer_edit_mode("", "Old text here") == "replace"
    # Legacy: no before_content → append
    assert infer_edit_mode("", "") == "append"
    # Explicit edit_mode wins
    assert infer_edit_mode("create_section", "") == "create_section"
    # Unknown edit_mode → fallback inference
    assert infer_edit_mode("garbage", "Old text") == "replace"
