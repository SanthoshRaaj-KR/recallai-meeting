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
