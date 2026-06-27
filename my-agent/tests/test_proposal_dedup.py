"""Same-section multi-edit survival in the Confluence proposal pipeline.

The editor reproduces the WHOLE section as before_content, so two edits to
different rows of one table/section shared an identical before_content and
collapsed into a single card in _dedupe_same_row — silently dropping all but one.
_narrow_to_changed_lines reduces each edit to just its changed line(s) so sibling
edits get distinct anchors and all survive, while the narrowed text remains a
valid anchor for the storage write-back (apply_section_edit).
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from review_pipeline.confluence_pipeline.models import (
    ChunkRecord,
    ConfluenceIntent,
    ConfluenceProposal,
)
from review_pipeline.confluence_pipeline.pipeline import (
    _dedupe_same_row,
    _narrow_to_changed_lines,
)
from review_pipeline.text_utils import apply_section_edit

SLA_TABLE_BEFORE = (
    "| Severity | Standard Response |\n"
    "| --- | --- |\n"
    "| P1 — Critical | 4 hours |\n"
    "| P2 — High | 8 business hours |"
)


def test_narrow_keeps_only_the_changed_row():
    after = SLA_TABLE_BEFORE.replace("4 hours", "2 hours")
    nb, na = _narrow_to_changed_lines(SLA_TABLE_BEFORE, after)
    assert nb == "| P1 — Critical | 4 hours |"
    assert na == "| P1 — Critical | 2 hours |"
    # The unchanged P2 row is gone from both sides.
    assert "P2" not in nb and "P2" not in na


def test_narrow_single_line_section_falls_back_to_full():
    before = "Secrets rotate automatically every 90 days."
    after = "Secrets rotate automatically every 60 days."
    nb, na = _narrow_to_changed_lines(before, after)
    assert nb == before and na == after


def _proposal(page_id, before, after, quality=0.9):
    chunk = ChunkRecord(
        chunk_id=f"{page_id}:0",
        source_path=page_id,
        source_format="confluence",
        section_heading="Support Tiers",
        section_index=1,
        content=before,
        doc_title="SLA",
    )
    intent = ConfluenceIntent(
        intent_type="policy_update",
        affected_topic="response time",
        new_value="x",
        verbatim_snippets=[],
        confidence=0.9,
        rationale="r",
    )
    p = ConfluenceProposal.create("s", intent, chunk)
    p.before_content = before
    p.after_content = after
    p.edit_type = "replace"
    p.quality_score = quality
    p.confidence = quality
    return p


def test_dedupe_keeps_distinct_rows_but_collapses_duplicates():
    # Two edits to DIFFERENT rows of the same page survive (narrowed to their rows).
    p1 = _proposal(
        "49250305", "| P1 — Critical | 4 hours |", "| P1 — Critical | 2 hours |"
    )
    p2 = _proposal(
        "49250305",
        "| P2 — High | 8 business hours |",
        "| P2 — High | 6 business hours |",
    )
    kept = _dedupe_same_row([p1, p2])
    assert len(kept) == 2, "distinct rows must both survive"

    # Two edits to the SAME row collapse to the highest quality one.
    p3 = _proposal(
        "49250305",
        "| P1 — Critical | 4 hours |",
        "| P1 — Critical | 2 hours |",
        quality=0.7,
    )
    p4 = _proposal(
        "49250305",
        "| P1 — Critical | 4 hours |",
        "| P1 — Critical | 2 hours |",
        quality=0.95,
    )
    kept2 = _dedupe_same_row([p3, p4])
    assert len(kept2) == 1
    assert kept2[0].quality_score == 0.95


def test_dedupe_keeps_two_cells_of_the_same_row():
    # Standard and Professional prices live in ONE header row (different cells).
    # They share a before line but change different words, so both must survive.
    row_before = "| Feature | Standard *$8/device/year* | Professional *$15/device/year* |"
    std = _proposal(
        "49217537", row_before,
        "| Feature | Standard *$10/device/year* | Professional *$15/device/year* |",
    )
    pro = _proposal(
        "49217537", row_before,
        "| Feature | Standard *$8/device/year* | Professional *$18/device/year* |",
    )
    kept = _dedupe_same_row([std, pro])
    assert len(kept) == 2, "different cells of the same row must both survive"


def test_two_same_section_narrowed_edits_both_write_back_to_storage():
    # End-to-end with the real tag roundtrip: two narrowed sibling edits applied to
    # the live storage XHTML both land, tags intact, with no collateral damage.
    storage = (
        "<h2>Support Tiers</h2><table><tbody>"
        "<tr><th>Severity</th><th>Standard Response</th></tr>"
        "<tr><td>P1 — Critical</td><td>4 hours</td></tr>"
        "<tr><td>P2 — High</td><td>8 business hours</td></tr>"
        "</tbody></table>"
    )
    after1 = SLA_TABLE_BEFORE.replace("4 hours", "2 hours")
    after2 = SLA_TABLE_BEFORE.replace("8 business hours", "6 business hours")
    nb1, na1 = _narrow_to_changed_lines(SLA_TABLE_BEFORE, after1)
    nb2, na2 = _narrow_to_changed_lines(SLA_TABLE_BEFORE, after2)

    storage, ok1 = apply_section_edit(storage, nb1, na1, "Support Tiers")
    storage, ok2 = apply_section_edit(storage, nb2, na2, "Support Tiers")
    assert ok1 and ok2
    assert "<td>2 hours</td>" in storage
    assert "<td>6 business hours</td>" in storage
    assert "<td>4 hours</td>" not in storage
    assert "&lt;" not in storage  # no escaped tags leaked
