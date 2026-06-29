from __future__ import annotations

import difflib
import html
import re
from typing import Any


def normalize_ws(value: str) -> str:
    return re.sub(r"\s+", " ", (value or "").strip())


def normalize_for_match(value: str) -> str:
    text = html.unescape(value or "").lower()
    text = re.sub(r"\b(\d+)(st|nd|rd|th)\b", r"\1", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[^\w\s/%.-]", " ", text)
    return normalize_ws(text)


def html_to_text(value: str) -> str:
    task_lines = [
        f"[task: {task['status']}] {task['body']}"
        for task in extract_tasks(value)
    ]
    text = _strip_tags(value)
    if task_lines:
        return "\n".join([text, *task_lines] if text else task_lines)
    return text


def _prefer_recall(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return Recall-diarized entries when present, else all entries.

    Keeps the post-meeting pipeline on real speaker names from AssemblyAI
    while staying backwards-compatible with sessions recorded before
    Recall transcription was enabled.
    """
    recall = [e for e in entries if e.get("source") == "recall"]
    return recall if recall else entries


def format_transcript(transcript: list[dict[str, Any]], max_chars: int = 60000) -> str:
    entries = _prefer_recall(transcript)
    lines: list[str] = []
    for entry in entries:
        speaker = entry.get("participant") or entry.get("speaker") or "Speaker"
        text = normalize_ws(str(entry.get("text") or entry.get("content") or ""))
        if text:
            lines.append(f"{speaker}: {text}")
    out = "\n".join(lines)
    if len(out) <= max_chars:
        return out
    head = out[: max_chars // 4]
    tail = out[-(max_chars - len(head)) :]
    return f"{head}\n[... middle of transcript omitted ...]\n{tail}"


def _format_relative_time(seconds: float) -> str:
    """Convert elapsed seconds into a human-readable meeting-relative label."""
    total = max(0, int(seconds))
    minutes, secs = divmod(total, 60)
    if minutes == 0:
        return f"{secs}s"
    return f"{minutes}m {secs:02d}s"


def transcript_highlights(transcript: list[dict[str, Any]], limit: int = 500) -> list[dict[str, str]]:
    # Determine meeting start from the earliest timestamp in the full transcript.
    start_ts: float | None = None
    for entry in transcript:
        ts = entry.get("timestamp")
        try:
            ts_f = float(ts)
            if ts_f > 0 and (start_ts is None or ts_f < start_ts):
                start_ts = ts_f
        except (TypeError, ValueError):
            pass

    entries = _prefer_recall(transcript)
    highlights = []
    for idx, entry in enumerate(entries[-limit:]):
        ts = entry.get("timestamp")
        try:
            ts_f = float(ts)
            if ts_f > 0 and start_ts is not None:
                time_label = _format_relative_time(ts_f - start_ts)
            else:
                time_label = "--"
        except (TypeError, ValueError):
            time_label = "--"

        highlights.append(
            {
                "time": time_label,
                "speaker": str(entry.get("participant") or entry.get("speaker") or "Speaker"),
                "text": normalize_ws(str(entry.get("text") or "")),
            }
        )
    return highlights


_HEADING_RE = re.compile(r"(?is)<h([1-6])\b[^>]*>(.*?)</h\1>")
_TASK_RE = re.compile(r"(?is)<ac:task\b[^>]*>.*?</ac:task>")
_TASK_STATUS_RE = re.compile(r"(?is)<ac:task-status\b[^>]*>(.*?)</ac:task-status>")
_TASK_BODY_RE = re.compile(r"(?is)<ac:task-body\b[^>]*>(.*?)</ac:task-body>")
_TASK_ID_RE = re.compile(r"(?is)<ac:task-id\b[^>]*>(.*?)</ac:task-id>")


def _strip_tags(value: str) -> str:
    text = re.sub(r"(?i)<\s*br\s*/?\s*>", "\n", value or "")
    text = re.sub(r"(?i)</\s*(p|div|li|h[1-6]|tr|ac:task-body)\s*>", "\n", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = html.unescape(text)
    lines = [normalize_ws(line) for line in text.splitlines()]
    return "\n".join(line for line in lines if line)


def extract_tasks(storage_html: str) -> list[dict[str, str]]:
    """Extract Confluence task-list items with status and rendered body text."""
    tasks: list[dict[str, str]] = []
    for match in _TASK_RE.finditer(storage_html or ""):
        raw = match.group(0)
        status_match = _TASK_STATUS_RE.search(raw)
        body_match = _TASK_BODY_RE.search(raw)
        id_match = _TASK_ID_RE.search(raw)
        body_html = body_match.group(1) if body_match else ""
        body = _strip_tags(body_html)
        status = normalize_ws(_strip_tags(status_match.group(1) if status_match else "")) or "unknown"
        if not body:
            continue
        tasks.append(
            {
                "task_id": normalize_ws(_strip_tags(id_match.group(1) if id_match else "")),
                "status": status.lower(),
                "body": body,
                "html": raw,
            }
        )
    return tasks


def task_status_label(body: str, status: str) -> str:
    marker = "[x]" if normalize_ws(status).lower() == "complete" else "[ ]"
    return f"{marker} {normalize_ws(body)}"


def task_body_from_label(label: str) -> str:
    return normalize_ws(re.sub(r"^\s*\[(?:x|X| )\]\s*", "", label or ""))


def replace_task_status_in_storage(storage_html: str, task_body: str, desired_status: str) -> tuple[str, bool]:
    """Change a Confluence task status while preserving task body/metadata."""
    value = storage_html or ""
    desired = "complete" if normalize_ws(desired_status).lower() == "complete" else "incomplete"
    body_norm = normalize_for_match(task_body)
    if not body_norm:
        return value, False

    best: tuple[int, re.Match[str] | None] = (0, None)
    for match in _TASK_RE.finditer(value):
        raw = match.group(0)
        body_match = _TASK_BODY_RE.search(raw)
        rendered_body = _strip_tags(body_match.group(1) if body_match else raw)
        rendered_norm = normalize_for_match(rendered_body)
        score = 0
        if body_norm == rendered_norm:
            score = 100
        elif body_norm in rendered_norm or rendered_norm in body_norm:
            score = min(len(body_norm), len(rendered_norm))
        if score > best[0]:
            best = (score, match)

    match = best[1]
    if match is None:
        return value, False

    raw = match.group(0)
    if not _TASK_STATUS_RE.search(raw):
        return value, False
    updated_task = _TASK_STATUS_RE.sub(
        f"<ac:task-status>{html.escape(desired)}</ac:task-status>",
        raw,
        count=1,
    )
    return f"{value[:match.start()]}{updated_task}{value[match.end():]}", True


def extract_sections(storage_html: str) -> list[dict[str, str]]:
    """Return heading-delimited sections from Confluence storage HTML.

    Each section includes the heading title, heading level, raw section HTML,
    and plain text. Content before the first heading is represented as
    ``heading=None`` so it can still be patched.
    """
    value = storage_html or ""
    matches = list(_HEADING_RE.finditer(value))
    sections: list[dict[str, str]] = []

    if not matches:
        return [{"heading": "", "level": "0", "html": value, "text": html_to_text(value)}]

    if matches[0].start() > 0:
        intro = value[: matches[0].start()]
        if normalize_ws(html_to_text(intro)):
            sections.append({"heading": "", "level": "0", "html": intro, "text": html_to_text(intro)})

    for idx, match in enumerate(matches):
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(value)
        section_html = value[match.start() : end]
        heading = html_to_text(match.group(2))
        sections.append(
            {
                "heading": heading,
                "level": match.group(1),
                "html": section_html,
                "text": html_to_text(section_html),
            }
        )
    return sections


def best_section_heading(storage_html: str, *needles: str) -> str | None:
    """Find the section most likely to contain one of the supplied text needles."""
    sections = extract_sections(storage_html)
    best: tuple[int, str | None] = (0, None)
    for section in sections:
        section_blob = normalize_for_match(f"{section.get('heading', '')}\n{section.get('text', '')}\n{section.get('html', '')}")
        score = 0
        for idx, needle in enumerate(needles):
            norm = normalize_for_match(needle)
            if not norm:
                continue
            if norm in section_blob:
                score += 10 - min(idx, 5)
        heading = section.get("heading") or None
        if score > best[0]:
            best = (score, heading)
    return best[1]


def insert_html_in_section(storage_html: str, section_heading: str | None, addition_html: str) -> str:
    """Insert storage HTML at the end of the chosen section.

    If the section cannot be found, append to the page. This preserves all
    unrelated storage markup.
    """
    value = storage_html or ""
    addition = addition_html or ""
    if not value:
        return addition
    if not section_heading:
        return f"{value}\n{addition}"

    matches = list(_HEADING_RE.finditer(value))
    for idx, match in enumerate(matches):
        heading = html_to_text(match.group(2))
        if normalize_for_match(heading) != normalize_for_match(section_heading):
            continue
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(value)
        return f"{value[:end]}\n{addition}\n{value[end:]}"
    return f"{value}\n{addition}"


def append_new_section(storage_html: str, section_heading: str, section_html: str) -> str:
    heading = html.escape(section_heading or "Update")
    return f"{storage_html or ''}\n<h2>{heading}</h2>\n{section_html or ''}"


def replace_text_in_storage(storage_html: str, before: str, after: str) -> tuple[str, bool]:
    """Replace text in Confluence storage HTML.

    First tries direct storage replacement. If Confluence split or wrapped the
    text, replaces the smallest common block whose rendered text contains the
    anchor. This is deliberately conservative.
    """
    value = storage_html or ""
    if not before:
        return value, False
    escaped_after = html.escape(after or "")
    if before in value:
        return value.replace(before, escaped_after, 1), True

    before_norm = normalize_for_match(before)
    block_re = re.compile(r"(?is)<(p|li|td|th|h[1-6])\b([^>]*)>(.*?)</\1>")
    for match in block_re.finditer(value):
        block_text = html_to_text(match.group(0))
        if before_norm and before_norm in normalize_for_match(block_text):
            tag = match.group(1)
            attrs = match.group(2) or ""
            replacement = f"<{tag}{attrs}>{escaped_after}</{tag}>"
            return f"{value[:match.start()]}{replacement}{value[match.end():]}", True
    return value, False


# ── Section-scoped phrase-diff apply ──────────────────────────────────────────
# replace_text_in_storage matches a single <p>/<td>/<li>/<h*> block, so it can't
# apply a whole-section before/after (what the confluence_pipeline editor emits)
# when the section spans several blocks (e.g. a table). apply_section_edit derives
# the minimal changed phrases from before→after (word-level diff, with surrounding
# context words as a disambiguating anchor) and replaces just those phrases inside
# the located section's storage XHTML — so the wrapping tags are preserved and only
# the changed values move. It only ever applies a replacement whose old text is
# found verbatim, so a miss is a no-op (never corruption).

_WORD_RE = re.compile(r"\S+")
_MD_EDGE_TOKENS = {"#", "##", "###", "####", "#####", "######", "|", "-", "*", "**", "`", "```", ">", "+", "—", "–"}


def _trim_md_edges(tokens: list[str]) -> list[str]:
    lo, hi = 0, len(tokens)
    while lo < hi and tokens[lo] in _MD_EDGE_TOKENS:
        lo += 1
    while hi > lo and tokens[hi - 1] in _MD_EDGE_TOKENS:
        hi -= 1
    return tokens[lo:hi]


def _is_structural(token: str) -> bool:
    """A markdown structural token (table pipe, heading/list marker) that has no
    counterpart in storage text — context must not cross it."""
    return "|" in token or token in _MD_EDGE_TOKENS


def _expand_within_cell(tokens: list[str], lo: int, hi: int, ctx: int) -> list[str]:
    """Pad [lo:hi] with up to `ctx` neighbouring *word* tokens, stopping at any
    structural token so the anchor stays inside one table cell / line segment."""
    left = lo
    taken = 0
    while left > 0 and taken < ctx and not _is_structural(tokens[left - 1]):
        left -= 1
        taken += 1
    right = hi
    taken = 0
    while right < len(tokens) and taken < ctx and not _is_structural(tokens[right]):
        right += 1
        taken += 1
    return tokens[left:right]


def _contextual_replacements(before: str, after: str, ctx: int = 2) -> list[tuple[str, str]]:
    """Minimal (old_phrase -> new_phrase) edits, each padded with `ctx` unchanged
    words on either side (without crossing a table cell / markdown boundary) so the
    anchor is specific enough to replace safely against storage text."""
    b = _WORD_RE.findall(before or "")
    a = _WORD_RE.findall(after or "")
    sm = difflib.SequenceMatcher(a=b, b=a, autojunk=False)
    pairs: list[tuple[str, str]] = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        old = " ".join(_trim_md_edges(_expand_within_cell(b, i1, i2, ctx))).strip()
        new = " ".join(_trim_md_edges(_expand_within_cell(a, j1, j2, ctx))).strip()
        if old and old != new:
            pairs.append((old, new))
    return pairs


def _locate_section_span(storage_html: str, section_heading: str) -> tuple[int, int] | None:
    """Return (start, end) of the section under `section_heading` in storage HTML."""
    target = normalize_for_match(section_heading)
    if not target:
        return None
    matches = list(_HEADING_RE.finditer(storage_html))
    for idx, match in enumerate(matches):
        if normalize_for_match(html_to_text(match.group(2))) != target:
            continue
        level = int(match.group(1))
        end = len(storage_html)
        for nxt in matches[idx + 1:]:
            if int(nxt.group(1)) <= level:
                end = nxt.start()
                break
        return (match.start(), end)
    return None


def apply_section_edit(
    storage_html: str,
    before: str,
    after: str,
    section_heading: str | None = None,
) -> tuple[str, bool]:
    """Apply a whole-section before→after edit to storage XHTML, preserving tags.

    Scopes to the named section when it can be found (so an ambiguous value isn't
    changed elsewhere on the page), then replaces the minimal changed phrases —
    each anchored with neighbouring unchanged words — wherever they appear verbatim
    in that section. Returns (new_html, replaced). A miss leaves the HTML untouched.
    """
    value = storage_html or ""
    if not before or before == after:
        return value, False

    start, end = 0, len(value)
    if section_heading:
        span = _locate_section_span(value, section_heading)
        if span:
            start, end = span
    region = value[start:end]

    applied = False
    for old, new in _contextual_replacements(before, after):
        new_text = html.escape(new, quote=False)  # the inserted value is page text
        # Storage may hold the old text raw or entity-escaped (&amp;/&lt;); try both.
        for candidate in (old, html.escape(old, quote=False)):
            if candidate and candidate != new_text and candidate in region:
                region = region.replace(candidate, new_text, 1)
                applied = True
                break
    if not applied:
        return value, False
    return f"{value[:start]}{region}{value[end:]}", True


# ── Confluence storage XHTML → clean markdown ─────────────────────────────────
# The RAG/content layer must hold clean text/markdown only — no XHTML tags and no
# Confluence macro chrome (``ac:`` / ``ri:`` / CDATA). Real storage tags are
# reintroduced only at write-back time (see replace_text_in_storage /
# insert_html_in_section / append_new_section / pipeline._storage_html), so the
# original Confluence document still renders correctly after an accepted edit.

_COMMENT_RE = re.compile(r"(?is)<!--.*?-->")
_CDATA_RE = re.compile(r"(?is)<!\[CDATA\[(.*?)\]\]>")
# Whole macros that carry no readable prose — drop the element and its body.
_NOISE_MACRO_RE = re.compile(
    r"(?is)<ac:structured-macro\b[^>]*\bac:name=\"(?:toc|children|pagetree|"
    r"livesearch|recently-updated|recently-updated-dashboard|contributors|"
    r"include|excerpt-include|gallery|profile-picture|widget|anchor|view-file|"
    r"attachments)\"[^>]*>.*?</ac:structured-macro>"
)
# Config parameters inside macros (layout, language, …) — drop, keep macro body.
_AC_PARAM_RE = re.compile(r"(?is)<ac:parameter\b[^>]*>.*?</ac:parameter>")
_AC_MACRO_ID_RE = re.compile(r"(?is)<ac:macro-id\b[^>]*>.*?</ac:macro-id>")
_AC_EMOTICON_RE = re.compile(r"(?is)<ac:emoticon\b[^>]*/?>")
# Resource identifiers (attachment filenames, user keys, page refs) — drop.
_RI_PAIR_RE = re.compile(r"(?is)<ri:[\w:-]+\b[^>]*>.*?</ri:[\w:-]+>")
_RI_SELF_RE = re.compile(r"(?is)<ri:[\w:-]+\b[^>]*/?>")
# Links: keep the human-readable anchor text, drop the ri:/href machinery.
_AC_LINK_RE = re.compile(r"(?is)<ac:link\b[^>]*>(.*?)</ac:link>")
_AC_LINK_BODY_RE = re.compile(
    r"(?is)<ac:(?:link-body|plain-text-link-body)\b[^>]*>(.*?)</ac:(?:link-body|plain-text-link-body)>"
)
_RI_TITLE_RE = re.compile(r'(?is)ri:(?:content-title|filename|value)="([^"]*)"')


def _ac_link_to_text(value: str) -> str:
    def _repl(match: "re.Match[str]") -> str:
        inner = match.group(1)
        body = _AC_LINK_BODY_RE.search(inner)
        if body and normalize_ws(_strip_tags(body.group(1))):
            return f" {_strip_tags(body.group(1))} "
        title = _RI_TITLE_RE.search(inner)
        if title and title.group(1).strip():
            return f" {title.group(1)} "
        return " "

    return _AC_LINK_RE.sub(_repl, value)


def _strip_confluence_macros(value: str) -> str:
    """Remove Confluence-specific macro/reference noise, keeping readable prose."""
    text = value or ""
    text = _COMMENT_RE.sub(" ", text)
    text = _NOISE_MACRO_RE.sub(" ", text)
    text = _ac_link_to_text(text)  # before stripping ri: (links wrap ri: refs)
    text = _AC_PARAM_RE.sub(" ", text)
    text = _AC_MACRO_ID_RE.sub(" ", text)
    text = _AC_EMOTICON_RE.sub(" ", text)
    text = _RI_PAIR_RE.sub(" ", text)
    text = _RI_SELF_RE.sub(" ", text)
    # Keep code/plain-text bodies (CDATA) as literal text for downstream conversion.
    # Escape only & < > (not quotes) so embedded "<" can't reopen tag parsing while
    # code stays readable.
    text = _CDATA_RE.sub(lambda m: html.escape(m.group(1), quote=False), text)
    # Drop any stray CDATA markers left by malformed/legacy storage.
    text = text.replace("<![CDATA[", " ").replace("]]>", " ")
    return text


def clean_inline_text(value: str) -> str:
    """Clean a short inline string (e.g. a section heading) to tag-free plain text."""
    return normalize_ws(_strip_tags(_strip_confluence_macros(value or "")))


def _html_to_markdown_fallback(value: str) -> str:
    """Dependency-free HTML→markdown used when ``markdownify`` is unavailable.

    Preserves ATX headings, bullet lists and one-line table rows so the section
    chunker can still split a page and the editor can make per-row edits. Never
    emits a raw tag.
    """
    text = value or ""
    for lvl in range(1, 7):
        text = re.sub(
            rf"(?is)<h{lvl}\b[^>]*>(.*?)</h{lvl}>",
            lambda m, _l=lvl: "\n" + "#" * _l + " " + normalize_ws(_strip_tags(m.group(1))) + "\n",
            text,
        )
    text = re.sub(
        r"(?is)<li\b[^>]*>(.*?)</li>",
        lambda m: "\n- " + normalize_ws(_strip_tags(m.group(1))),
        text,
    )

    def _row(match: "re.Match[str]") -> str:
        cells = re.findall(r"(?is)<t[dh]\b[^>]*>(.*?)</t[dh]>", match.group(1))
        if not cells:
            return "\n"
        return "\n| " + " | ".join(normalize_ws(_strip_tags(c)) for c in cells) + " |"

    text = re.sub(r"(?is)<tr\b[^>]*>(.*?)</tr>", _row, text)
    text = re.sub(r"(?is)</(p|div|table|ul|ol)\s*>", "\n\n", text)
    text = re.sub(r"(?i)<\s*br\s*/?\s*>", "\n", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = html.unescape(text)
    lines = [normalize_ws(ln) for ln in text.splitlines()]
    out: list[str] = []
    for ln in lines:
        if ln or (out and out[-1] != ""):
            out.append(ln)
    return re.sub(r"\n{3,}", "\n\n", "\n".join(out)).strip()


def looks_like_storage_html(value: str) -> bool:
    """True if the text still carries XHTML/Confluence-storage tags worth cleaning."""
    return bool(
        re.search(r"</?(?:ac:|ri:|p|div|span|h[1-6]|table|tr|td|th|ul|ol|li|br)\b", value or "")
    )


def storage_to_markdown(storage_html: str, title: str = "") -> str:
    """Convert Confluence storage XHTML to clean markdown (no tags, no macro noise).

    Pipeline: strip ``ac:``/``ri:``/CDATA macro noise → convert standard HTML to
    markdown (``markdownify`` when importable, else a dependency-free fallback that
    keeps ``#`` headings for the chunker) → guarantee a level-1 ``# {title}``. The
    output is what gets embedded, retrieved, evaluated, edited and shown on review
    cards; it is deliberately tag-free.
    """
    cleaned = _strip_confluence_macros(storage_html or "")
    try:
        from markdownify import markdownify as _md

        markdown = _md(
            cleaned,
            heading_style="ATX",
            strip=["span"],
            escape_asterisks=False,
            escape_underscores=False,
            escape_misc=False,
        ).strip()
    except ImportError:
        markdown = _html_to_markdown_fallback(cleaned)
    # Belt-and-suspenders: never let a raw tag survive into the indexed text.
    if re.search(r"<[^>]+>", markdown):
        markdown = _strip_tags(markdown)
    markdown = re.sub(r"\n{3,}", "\n\n", markdown).strip()
    title = normalize_ws(title)
    if title and not markdown.lstrip().startswith("# "):
        markdown = f"# {title}\n\n{markdown}"
    return markdown