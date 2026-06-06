from __future__ import annotations

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


def format_transcript(transcript: list[dict[str, Any]], max_chars: int = 60000) -> str:
    lines: list[str] = []
    for entry in transcript:
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

    highlights = []
    for idx, entry in enumerate(transcript[-limit:]):
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