"""Tests for Confluence storage-XHTML → clean-markdown conversion (text_utils).

The RAG/content layer must hold clean text/markdown only — no XHTML tags and no
Confluence macro chrome (``ac:`` / ``ri:`` / CDATA). Real storage tags are
reintroduced only at write-back time. These tests pin that contract and the
dependency-free fallback used when ``markdownify`` is unavailable.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from review_pipeline.text_utils import (  # noqa: E402
    apply_section_edit,
    looks_like_storage_html,
    storage_to_markdown,
)

SAMPLE = """\
<h1>SmartHub Pricing</h1>
<ac:structured-macro ac:name="info"><ac:rich-text-body><p>Effective Q3.</p></ac:rich-text-body></ac:structured-macro>
<h2>Device Pricing</h2>
<table><tbody>
<tr><th>Tier</th><th>Price</th></tr>
<tr><td>Standard</td><td>$50 per device per year</td></tr>
<tr><td>Professional</td><td>$80 per device per year</td></tr>
</tbody></table>
<ac:structured-macro ac:name="toc"><ac:parameter ac:name="maxLevel">3</ac:parameter></ac:structured-macro>
<h2>Onboarding</h2>
<ac:task-list><ac:task><ac:task-id>1</ac:task-id><ac:task-status>incomplete</ac:task-status>\
<ac:task-body>Send <ac:link><ri:page ri:content-title="Welcome Guide"/></ac:link> to client</ac:task-body></ac:task></ac:task-list>
<p>See <ac:link><ri:attachment ri:filename="setup.pdf"/><ac:link-body>setup doc</ac:link-body></ac:link>.</p>
<ac:structured-macro ac:name="code"><ac:parameter ac:name="language">python</ac:parameter>\
<ac:plain-text-body><![CDATA[latency = 2  # ms]]></ac:plain-text-body></ac:structured-macro>
"""

_TAG_RE = re.compile(r"<[^>]+>")


def _assert_clean(md: str) -> None:
    assert not _TAG_RE.search(md), f"raw tags survived: {md!r}"
    assert "ac:" not in md
    assert "ri:" not in md
    assert "CDATA" not in md


def test_storage_to_markdown_is_tag_free_and_keeps_prose():
    md = storage_to_markdown(SAMPLE, "SmartHub Pricing")
    _assert_clean(md)
    # Headings preserved so the section chunker can still split.
    assert "# SmartHub Pricing" in md
    assert "## Device Pricing" in md
    assert "## Onboarding" in md
    # Real values survive.
    assert "$50 per device per year" in md
    assert "$80 per device per year" in md
    assert "Effective Q3" in md
    assert "latency = 2" in md  # code body (CDATA) kept


def test_macro_and_reference_noise_removed():
    md = storage_to_markdown(SAMPLE, "SmartHub Pricing")
    assert "maxLevel" not in md  # toc macro + its config param dropped
    assert "setup.pdf" not in md  # ri:attachment filename dropped
    assert "language" not in md  # code macro language param dropped
    # …but the human-readable link text is kept.
    assert "Welcome Guide" in md  # ri:content-title surfaced
    assert "setup doc" in md  # ac:link-body surfaced


def test_dependency_free_fallback_when_markdownify_missing():
    real = sys.modules.get("markdownify")
    sys.modules["markdownify"] = None  # force `from markdownify import …` to ImportError
    try:
        md = storage_to_markdown(SAMPLE, "SmartHub Pricing")
    finally:
        if real is not None:
            sys.modules["markdownify"] = real
        else:
            sys.modules.pop("markdownify", None)
    _assert_clean(md)
    assert "## Device Pricing" in md  # headings preserved by the fallback
    assert "$50 per device per year" in md  # one-line table row preserved


def test_looks_like_storage_html():
    assert looks_like_storage_html("<p>hello</p>")
    assert looks_like_storage_html('<ac:task-body>do it</ac:task-body>')
    assert not looks_like_storage_html("clean **markdown** with | a | b | values")
    assert not looks_like_storage_html("just prose, 6 dollars per device")


# ── apply_section_edit: whole-section edits onto multi-block storage XHTML ──────

MULTI_BLOCK = (
    "<h2>Support Tiers</h2>"
    "<table><tbody>"
    "<tr><th>Tier</th><th>Standard response</th></tr>"
    "<tr><td>P1</td><td>within 2 hours</td></tr>"
    "<tr><td>P2</td><td>within 8 hours</td></tr>"
    "</tbody></table>"
    "<h2>Other</h2>"
    "<p>We had 2 incidents last quarter.</p>"
)


def test_apply_section_edit_changes_one_table_cell_keeps_tags():
    before = "| Tier | Standard response |\n| P1 | within 2 hours |\n| P2 | within 8 hours |"
    after = "| Tier | Standard response |\n| P1 | within 1 hour |\n| P2 | within 8 hours |"
    new_html, ok = apply_section_edit(MULTI_BLOCK, before, after, "Support Tiers")
    assert ok
    assert "<td>within 1 hour</td>" in new_html  # changed cell, tags intact
    assert "<td>within 8 hours</td>" in new_html  # sibling row untouched
    assert "<p>We had 2 incidents last quarter.</p>" in new_html  # other section + bare "2" safe
    assert "&lt;" not in new_html


def test_apply_section_edit_scopes_to_named_section():
    # The bare value "2" exists in another section; context anchor + scoping protect it.
    before = "| P1 | within 2 hours |"
    after = "| P1 | within 3 hours |"
    new_html, ok = apply_section_edit(MULTI_BLOCK, before, after, "Support Tiers")
    assert ok
    assert "within 3 hours" in new_html
    assert "We had 2 incidents" in new_html  # untouched


def test_apply_section_edit_phrase_across_blocks():
    html_doc = (
        "<h1>Graph &mdash; Neo4j Device Graph</h1>"
        "<p>INFER uses Neo4j AuraDB as a store.</p>"
        "<p>Neo4j is fast.</p>"
    )
    before = "INFER uses Neo4j AuraDB as a store.\nNeo4j is fast."
    after = "INFER uses Kuzu DB as a store.\nNeo4j is fast."
    new_html, ok = apply_section_edit(html_doc, before, after)
    assert ok
    assert "<p>INFER uses Kuzu DB as a store.</p>" in new_html
    assert "<p>Neo4j is fast.</p>" in new_html  # unrelated mention untouched


def test_apply_section_edit_miss_is_noop():
    new_html, ok = apply_section_edit(MULTI_BLOCK, "value not present here", "something else", "Support Tiers")
    assert not ok
    assert new_html == MULTI_BLOCK


# Real Confluence storage: a header cell splits its label and value with <br/> and
# wraps the value in <em>. The markdown-derived anchor is "Standard *$8/device/year*",
# which exists nowhere literally in the storage — the edit must still anchor on the
# bare, emphasis-stripped value ($8/device/year) inside <em>…</em>.
EMPHASIS_TABLE = (
    "<h2>Plan Comparison</h2><table><tbody>"
    "<tr>"
    "<th>Feature</th>"
    "<th>Starter<br /><em>Free up to 50 devices</em></th>"
    "<th>Standard<br /><em>$8/device/year</em></th>"
    "<th>Professional<br /><em>$15/device/year</em></th>"
    "</tr>"
    "</tbody></table>"
)


def test_apply_section_edit_anchors_bare_value_inside_em_tags():
    before = "| Feature | Starter *Free up to 50 devices* | Standard *$8/device/year* | Professional *$15/device/year* |"
    after = "| Feature | Starter *Free up to 50 devices* | Standard *$10/device/year* | Professional *$15/device/year* |"
    new_html, ok = apply_section_edit(EMPHASIS_TABLE, before, after, "Plan Comparison")
    assert ok, "must anchor on the emphasis-stripped bare value"
    assert "<em>$10/device/year</em>" in new_html  # value changed, tags intact
    assert "<em>$15/device/year</em>" in new_html  # sibling cell untouched
    assert "$8/device/year" not in new_html
    assert "*" not in new_html.split("Plan Comparison", 1)[-1]  # no markdown leaked into storage
    assert "&lt;" not in new_html
