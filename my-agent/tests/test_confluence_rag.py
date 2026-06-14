"""Tests for ConfluenceLiveRAG — query building and result formatting.

Simulates exactly how agent.py uses ConfluenceLiveRAG on every wake-word turn:

  1. raw utterance is appended to self._transcript  (before RAG)
  2. query  = _extract_query(raw)                   (strip "Hey Jarvis,")
  3. topic_hint = _extract_topic_hint(_last_compacted_memory)
  4. enriched = rag.build_search_query(query, list(self._transcript), topic_hint=topic_hint)
  5. hits = rag.search(enriched)                    (Pinecone — mocked here)
  6. context = rag.format_context(hits)             (injected into LLM)
  7. _last_compacted_memory updated AFTER the search (so topic_hint still comes from previous turn)

Fake page content is drawn from the actual Deepfake Detection Project page and the
Q3 Meeting Overview page so the queries and expected results are realistic.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agent import _extract_query, _extract_topic_hint  # noqa: E402
from confluence_rag import _SCORE_THRESHOLD, ConfluenceLiveRAG  # noqa: E402


@pytest.fixture(autouse=True)
def _enable_pinecone(monkeypatch):
    """ConfluenceLiveRAG.enabled reads PINECONE_API_KEY; set a dummy so search() runs
    against the mocked index. Real Pinecone is never contacted (index is a MagicMock)."""
    monkeypatch.setenv("PINECONE_API_KEY", "test-key")


# ── Fake Pinecone result builder ──────────────────────────────────────────────

def _make_pinecone_result(chunks: list[dict]) -> SimpleNamespace:
    """Build a fake Pinecone search() result with the structure the code reads:
       result.result.hits[i].fields, result.result.hits[i].score
    """
    hits = [
        SimpleNamespace(
            fields={k: v for k, v in chunk.items() if k != "score"},
            score=chunk["score"],
        )
        for chunk in chunks
    ]
    return SimpleNamespace(result=SimpleNamespace(hits=hits))


# ── Realistic fake page chunks ────────────────────────────────────────────────
# These mirror what would actually be indexed from the two Confluence pages.

_DEEPFAKE_CHUNKS = [
    {
        "page_id": "deepfake-1",
        "title": "Deepfake Detection Project",
        "space_key": "PROJ",
        "heading": "Evaluation Metrics",
        "section_order": 7,
        "text": (
            "Minimum Performance Requirements. "
            "Minimum Recall: 0.92 (92%). Decision date: April 6, 2026. Approver: Rahul. "
            "Rationale: The project prioritizes minimizing missed deepfakes in production; "
            "a recall of 0.92 was chosen based on validation experiments and stakeholder "
            "risk tolerance."
        ),
        "score": 0.91,
    },
    {
        "page_id": "deepfake-1",
        "title": "Deepfake Detection Project",
        "space_key": "PROJ",
        "heading": "Frequency Domain Features (DCT-Based Detection)",
        "section_order": 5,
        "text": (
            "Deepfake models introduce frequency inconsistencies via upsampling artifacts. "
            "Steps: Convert RGB to frequency domain using DCT. Extract high-frequency components. "
            "Use magnitude spectrum as model input. Combine spatial and frequency features. "
            "Advantages: Detects GAN upsampling traces. Robust against visual deception."
        ),
        "score": 0.87,
    },
    {
        "page_id": "deepfake-1",
        "title": "Deepfake Detection Project",
        "space_key": "PROJ",
        "heading": "Decisions",
        "section_order": 11,
        "text": (
            "Move forward with Hybrid DCT + Spatial model. "
            "Use 1% false positive rate as threshold tuning benchmark. "
            "Deploy model as REST microservice. Use ResNet backbone for feature extraction. "
            "Schedule model retraining every 2 weeks."
        ),
        "score": 0.82,
    },
    {
        "page_id": "deepfake-1",
        "title": "Deepfake Detection Project",
        "space_key": "PROJ",
        "heading": "Action Items",
        "section_order": 10,
        "text": (
            "Compare Autoencoder vs VAE performance. Add adaptive thresholding logic. "
            "Implemented hybrid loss function — completed (2026-02-15). "
            "Adaptive thresholding logic implementation — completed (2026-02-15)."
        ),
        "score": 0.78,
    },
    {
        "page_id": "deepfake-1",
        "title": "Deepfake Detection Project",
        "space_key": "PROJ",
        "heading": "Model Design",
        "section_order": 6,
        "text": (
            "Binary classification: 0 → Real, 1 → Fake. "
            "Loss Function: Binary Cross Entropy or Focal Loss for class imbalance. "
            "Anomaly detection: train autoencoder only on real images; "
            "high reconstruction error → fake. Better generalisation to unseen fake methods."
        ),
        "score": 0.76,
    },
]

_MEETING_CHUNKS = [
    {
        "page_id": "meeting-1",
        "title": "Q3 Meeting Overview",
        "space_key": "MEET",
        "heading": "Action Items",
        "section_order": 2,
        "text": "Review Q3 roadmap. Assign owners to delivery milestones. Schedule follow-up call.",
        "score": 0.74,
    },
    {
        "page_id": "meeting-1",
        "title": "Q3 Meeting Overview",
        "space_key": "MEET",
        "heading": "Goals",
        "section_order": 1,
        "text": "Complete product roadmap for Q3. Define team responsibilities. Align on timeline.",
        "score": 0.71,
    },
]

# What Pinecone returns for each query type — the intended top hit is always first.
# Per-page diversity cap in search() keeps 1 chunk per page_id, so the first
# deepfake chunk in each list determines which section comes back.
_DEEPFAKE_FIRST = _make_pinecone_result(_DEEPFAKE_CHUNKS[:3])        # Evaluation Metrics wins
_DCT_FIRST      = _make_pinecone_result([_DEEPFAKE_CHUNKS[1]] + _DEEPFAKE_CHUNKS[2:])  # DCT wins
_DECISIONS_FIRST = _make_pinecone_result([_DEEPFAKE_CHUNKS[2]] + _DEEPFAKE_CHUNKS[3:])  # Decisions wins
_ACTION_ITEMS_FIRST = _make_pinecone_result([_DEEPFAKE_CHUNKS[3]] + _DEEPFAKE_CHUNKS[4:])  # Action Items wins
# What Pinecone returns for a meeting-specific query (meeting page wins):
_MEETING_FIRST = _make_pinecone_result(_MEETING_CHUNKS + _DEEPFAKE_CHUNKS[-1:])
# Mixed result — both pages present (after architecture fixes, this is realistic):
_MIXED_RESULT = _make_pinecone_result([_DEEPFAKE_CHUNKS[0], _MEETING_CHUNKS[0], _DEEPFAKE_CHUNKS[1]])


def _rag_with_fake_pinecone(pinecone_result) -> ConfluenceLiveRAG:
    """Return a ConfluenceLiveRAG whose Pinecone index is pre-loaded with a fake result."""
    rag = ConfluenceLiveRAG()
    fake_index = MagicMock()
    fake_index.search.return_value = pinecone_result
    rag._pinecone_index = fake_index
    return rag


# ── Helper: simulate the exact transcript state at the moment of a wake-word ──

def _build_transcript(*lines: str) -> list[str]:
    """Each line is what ended up in self._transcript — including wake-word lines
    and Jarvis: replies exactly as agent.py appends them."""
    return list(lines)


# ── Section 1: query building (no Pinecone call) ──────────────────────────────


def test_specific_question_suppresses_topic_hint():
    """A question ≥ 5 words must NOT include the old compacted-memory topic hint.

    Scenario: Jarvis previously answered a Q3 meeting question.
    _last_compacted_memory is meeting-focused.  User now asks about deepfake recall.
    The topic_hint from the old memory must not contaminate the new query.
    """
    # Simulated _last_compacted_memory after the previous (meeting) turn:
    old_memory = (
        "The team reviewed Q3 action items and product roadmap milestones. "
        "Alice will own the delivery timeline. Follow-up scheduled for next week."
    )
    topic_hint = _extract_topic_hint(old_memory)

    # Transcript at the moment of this wake-word (includes current line per agent.py):
    transcript = _build_transcript(
        "Rahul: Let's move on — Jarvis, let's ask about the deepfake project.",
        "Hey Jarvis, what is the minimum recall requirement for deepfake detection?",
    )
    query = _extract_query("Hey Jarvis, what is the minimum recall requirement for deepfake detection?")
    assert query  # wake word extracted

    rag = ConfluenceLiveRAG()
    enriched = rag.build_search_query(query, transcript, topic_hint=topic_hint)

    # Deepfake-specific terms must dominate:
    assert "recall" in enriched.lower()
    assert "deepfake" in enriched.lower()
    # Old meeting topic must NOT bleed in (question is 9 words → topic_hint suppressed):
    assert "Q3" not in enriched
    assert "roadmap" not in enriched
    assert "alice" not in enriched.lower()


def test_vague_question_uses_topic_hint():
    """A question < 5 words must include the topic_hint for disambiguation.

    Short questions like 'what was decided?' carry no topical signal on their own.
    The compacted memory tells the embedding model which page to target.
    """
    topic_hint = "Deepfake detection recall threshold discussion VAE autoencoder comparison"

    transcript = _build_transcript(
        "Rahul: So should we go with 0.92 or higher?",
        "Hey Jarvis, what was decided?",
    )
    query = _extract_query("Hey Jarvis, what was decided?")
    assert query

    rag = ConfluenceLiveRAG()
    enriched = rag.build_search_query(query, transcript, topic_hint=topic_hint)

    # topic_hint must appear (question is 3 words < 5):
    assert "VAE" in enriched or "autoencoder" in enriched.lower() or "deepfake" in enriched.lower()
    assert "decided" in enriched.lower()


def test_jarvis_replies_stripped_from_query_context():
    """Jarvis's own TTS replies (prefixed 'Jarvis:') must not be included as context.

    Jarvis replies embed Confluence content from the previous turn — including them
    as query context would bias the embedding toward the old page.
    """
    topic_hint = ""
    transcript = _build_transcript(
        "Rahul: Current recall is 0.71, we need more.",
        "Hey Jarvis, what is our recall target?",
        # Jarvis reply gets appended after TTS — it's now in transcript before next query:
        "Jarvis: Based on the project requirements, the minimum recall target is 0.92 "
        "as decided by Rahul on April 6th. The rationale was to balance sensitivity with "
        "acceptable false-positive rates from the meeting overview page.",
        "Alice: Great, now let's talk about the DCT pipeline.",
        "Hey Jarvis, explain the DCT preprocessing steps.",
    )
    query = _extract_query("Hey Jarvis, explain the DCT preprocessing steps.")

    rag = ConfluenceLiveRAG()
    enriched = rag.build_search_query(query, transcript, topic_hint=topic_hint)

    # The Jarvis reply text (which mentions "meeting overview page") must NOT appear:
    assert "meeting overview page" not in enriched.lower()
    # The actual question and relevant human speech should be in:
    assert "DCT" in enriched or "preprocessing" in enriched.lower()


def test_filler_words_removed_from_transcript_context():
    """Filler words (um, uh, basically, you know) are stripped before embedding."""
    transcript = _build_transcript(
        "Rahul: Um so basically uh we need to you know improve the model recall yeah",
        "Hey Jarvis, what model architecture should we use?",
    )
    query = _extract_query("Hey Jarvis, what model architecture should we use?")
    rag = ConfluenceLiveRAG()
    enriched = rag.build_search_query(query, transcript, topic_hint="")

    fillers = ["um ", "uh ", "basically", "you know", " so "]
    for filler in fillers:
        assert filler not in enriched.lower(), f"filler {filler!r} leaked into query"


def test_query_capped_at_max_chars():
    """Enriched query must not exceed _MAX_QUERY_CHARS (350 chars)."""
    from confluence_rag import _MAX_QUERY_CHARS
    long_transcript = [f"Speaker {i}: This is a very long line about deepfake detection models." for i in range(50)]
    rag = ConfluenceLiveRAG()
    enriched = rag.build_search_query(
        "what is the ensemble model architecture for deepfake detection",
        long_transcript,
        topic_hint="DCT frequency domain autoencoder anomaly detection pipeline",
    )
    assert len(enriched) <= _MAX_QUERY_CHARS


# ── Section 2: search() result processing ────────────────────────────────────


def test_deepfake_recall_question_returns_metrics_section():
    """In a deepfake project meeting, asking about recall should return the
    Evaluation Metrics section (score 0.91) as the top hit."""
    transcript = _build_transcript(
        "Rahul: Current recall is 0.71, needs improvement to meet our target.",
        "ML Team: We need to add adaptive thresholding.",
        "Hey Jarvis, what is the minimum recall we need to hit for production?",
    )
    query = _extract_query("Hey Jarvis, what is the minimum recall we need to hit for production?")

    rag = _rag_with_fake_pinecone(_DEEPFAKE_FIRST)
    enriched = rag.build_search_query(query, transcript, topic_hint="")

    hits = rag.search(enriched)

    assert len(hits) >= 1
    assert hits[0]["title"] == "Deepfake Detection Project"
    assert hits[0]["heading"] == "Evaluation Metrics"
    assert "0.92" in hits[0]["text"]
    assert hits[0]["score"] >= _SCORE_THRESHOLD


def test_dct_pipeline_question_returns_frequency_section():
    """Asking about DCT preprocessing should return the Frequency Domain Features section.

    _DCT_FIRST puts the DCT chunk at rank 0 so the per-page diversity cap keeps it.
    """
    transcript = _build_transcript(
        "ML Team: Should we use block-wise DCT or full-image DCT?",
        "Research Lead: Full-image is simpler but block-wise captures local artifacts better.",
        "Hey Jarvis, explain how the DCT preprocessing works in our pipeline.",
    )
    query = _extract_query("Hey Jarvis, explain how the DCT preprocessing works in our pipeline.")

    rag = _rag_with_fake_pinecone(_DCT_FIRST)
    enriched = rag.build_search_query(query, transcript, topic_hint="")

    hits = rag.search(enriched)

    assert len(hits) == 1
    assert hits[0]["heading"] == "Frequency Domain Features (DCT-Based Detection)"
    assert "upsampling" in hits[0]["text"].lower() or "frequency" in hits[0]["text"].lower()


def test_decisions_question_returns_decisions_section():
    """Asking about what was decided should return the Decisions section."""
    transcript = _build_transcript(
        "Rahul: We had a long debate last week about which backbone to use.",
        "Alice: Right, and we also discussed the false positive rate target.",
        "Hey Jarvis, what decisions were made about the deepfake model architecture?",
    )
    query = _extract_query("Hey Jarvis, what decisions were made about the deepfake model architecture?")

    rag = _rag_with_fake_pinecone(_DECISIONS_FIRST)
    enriched = rag.build_search_query(query, transcript, topic_hint="")

    hits = rag.search(enriched)

    assert len(hits) == 1
    assert hits[0]["heading"] == "Decisions"
    assert "ResNet" in hits[0]["text"] or "DCT" in hits[0]["text"]


def test_action_items_question_returns_action_items_section():
    """Asking about action items in a deepfake meeting should return deepfake action items."""
    transcript = _build_transcript(
        "Rahul: We need to track what's still open from last sprint.",
        "Hey Jarvis, what are the open action items for the deepfake project?",
    )
    query = _extract_query("Hey Jarvis, what are the open action items for the deepfake project?")

    rag = _rag_with_fake_pinecone(_ACTION_ITEMS_FIRST)
    enriched = rag.build_search_query(query, transcript, topic_hint="")

    hits = rag.search(enriched)

    assert len(hits) == 1
    assert hits[0]["heading"] == "Action Items"
    assert "VAE" in hits[0]["text"] or "thresholding" in hits[0]["text"].lower()


# ── Section 3: topic-switch (the exact bug scenario) ─────────────────────────


def test_topic_switch_meeting_to_deepfake_returns_correct_page():
    """The exact bug the user reported:

    Turn 1 (meeting): 'Hey Jarvis, list the action items' → meeting page retrieved.
    _last_compacted_memory is now meeting-focused.

    Turn 2 (deepfake): 'Hey Jarvis, what is the recall target for deepfake detection'
    With the old code: topic_hint contaminated the query → meeting page again.
    With the fix: topic_hint suppressed (question ≥ 5 words) → deepfake page wins.
    """
    # State after Turn 1: compacted memory is meeting-focused
    last_compacted_memory_after_meeting_turn = (
        "The team reviewed Q3 project action items. Alice owns the roadmap. "
        "Bob will schedule the follow-up. Timeline set for end of quarter."
    )
    topic_hint = _extract_topic_hint(last_compacted_memory_after_meeting_turn)

    # Transcript at Turn 2 (includes all previous lines + current wake-word line):
    transcript_at_turn_2 = _build_transcript(
        # From Turn 1:
        "Hey Jarvis, list the action items for the Q3 meeting.",
        "Jarvis: The Q3 meeting action items are: review roadmap, assign owners, schedule follow-up.",
        # Meeting continues:
        "Alice: Good. Now let's shift to the deepfake project status.",
        "Rahul: Our current recall is 0.71 and we need to reach 0.92.",
        # Turn 2 wake-word (appended before RAG call per agent.py):
        "Hey Jarvis, what is the recall target for deepfake detection?",
    )
    query = _extract_query("Hey Jarvis, what is the recall target for deepfake detection?")

    rag = _rag_with_fake_pinecone(_DEEPFAKE_FIRST)
    enriched = rag.build_search_query(query, transcript_at_turn_2, topic_hint=topic_hint)

    # Verify topic_hint was suppressed (question is 7 words ≥ 5):
    assert "roadmap" not in enriched.lower(), "Meeting topic_hint leaked into deepfake query"
    assert "alice" not in enriched.lower(), "Meeting topic_hint leaked into deepfake query"
    assert "recall" in enriched.lower(), "Deepfake question must dominate"

    hits = rag.search(enriched)

    # Must return the deepfake page, NOT the meeting page:
    assert hits[0]["title"] == "Deepfake Detection Project"
    assert hits[0]["page_id"] == "deepfake-1"


def test_model_architecture_question_mid_meeting():
    """Detailed question mid-meeting about model design — verify both query and results."""
    topic_hint = "deepfake detection recall threshold DCT autoencoder"  # from previous deepfake turn

    transcript = _build_transcript(
        "Rahul: The autoencoder approach generalises better to unseen fakes.",
        "Research Lead: Yes but we might miss some injection-based fakes.",
        "ML Team: Should we use classification or anomaly detection?",
        "Hey Jarvis, what are the pros and cons of using an autoencoder versus classification for deepfake detection?",
    )
    query = _extract_query(
        "Hey Jarvis, what are the pros and cons of using an autoencoder versus classification for deepfake detection?"
    )

    rag = _rag_with_fake_pinecone(_DEEPFAKE_FIRST)
    enriched = rag.build_search_query(query, transcript, topic_hint=topic_hint)

    # Question is long (14 words) → topic_hint suppressed even though it's deepfake-relevant:
    assert "autoencoder" in enriched.lower() or "classification" in enriched.lower()

    hits = rag.search(enriched)
    assert all(h["page_id"] == "deepfake-1" for h in hits)


# ── Section 4: per-page diversity and format_context ─────────────────────────


def test_per_page_diversity_limits_one_chunk_per_page():
    """search() must return at most 1 chunk per page_id even when Pinecone returns
    multiple high-scoring chunks from the same page."""
    # Pinecone returns 3 deepfake chunks + 1 meeting chunk
    rag = _rag_with_fake_pinecone(_MIXED_RESULT)
    hits = rag.search("deepfake recall detection threshold")

    page_ids = [h["page_id"] for h in hits]
    assert len(page_ids) == len(set(page_ids)), (
        f"Duplicate page_id in results — diversity cap not working: {page_ids}"
    )


def test_score_threshold_filters_low_confidence_hits():
    """Hits with score below JARVIS_CONFLUENCE_RAG_SCORE_THRESHOLD must be dropped."""
    low_score_result = _make_pinecone_result([
        {**_DEEPFAKE_CHUNKS[0], "score": _SCORE_THRESHOLD - 0.01},
        {**_DEEPFAKE_CHUNKS[1], "score": _SCORE_THRESHOLD + 0.01},
    ])
    rag = _rag_with_fake_pinecone(low_score_result)
    hits = rag.search("deepfake recall")

    assert len(hits) == 1
    assert hits[0]["score"] >= _SCORE_THRESHOLD


def test_format_context_labels_sections_clearly():
    """format_context renders 'Title › Heading [SPACE]' breadcrumbs.

    format_context deduplicates by page_id (1 chunk per page), so we pass one
    deepfake chunk and one meeting chunk to verify both labels appear.
    """
    rag = ConfluenceLiveRAG()
    # One chunk from each page — both must appear with correct labels.
    context = rag.format_context([_DEEPFAKE_CHUNKS[0], _MEETING_CHUNKS[0]])

    assert "Deepfake Detection Project › Evaluation Metrics [PROJ]" in context
    assert "Q3 Meeting Overview › Action Items [MEET]" in context
    assert "0.92" in context
    assert "roadmap" in context.lower() or "milestone" in context.lower()


def test_format_context_one_chunk_per_page():
    """format_context deduplicates by page_id — only the highest-scoring chunk per
    page is included (hits are score-sorted, first occurrence wins)."""
    rag = ConfluenceLiveRAG()
    # Three deepfake chunks — all same page_id
    context = rag.format_context(_DEEPFAKE_CHUNKS[:3])

    # Only the first (highest-score) chunk should appear
    assert context.count("Deepfake Detection Project") == 1


def test_format_context_empty_hits_returns_empty_string():
    rag = ConfluenceLiveRAG()
    assert rag.format_context([]) == ""


# ── Section 5: query builder edge cases ──────────────────────────────────────


def test_back_channel_lines_under_4_words_ignored():
    """Very short lines like 'Sure.', 'Exactly.', 'Right.' must not appear in the query."""
    transcript = _build_transcript(
        "Sure.",
        "Exactly right.",
        "Yes.",
        "Rahul: The recall target is 0.92 based on our stakeholder requirements.",
        "Hey Jarvis, what recall did we set as the minimum threshold?",
    )
    query = _extract_query("Hey Jarvis, what recall did we set as the minimum threshold?")
    rag = ConfluenceLiveRAG()
    enriched = rag.build_search_query(query, transcript, topic_hint="")

    assert "Sure" not in enriched
    assert "Exactly" not in enriched
    assert "recall" in enriched.lower()
    assert "0.92" in enriched or "stakeholder" in enriched.lower()


def test_empty_transcript_falls_back_to_question_only():
    """With no transcript context, the enriched query equals the question alone."""
    rag = ConfluenceLiveRAG()
    enriched = rag.build_search_query(
        "what is the ensemble model architecture",
        [],
        topic_hint="",
    )
    assert "ensemble model architecture" in enriched.lower()
