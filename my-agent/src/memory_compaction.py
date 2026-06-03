from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

_SALIENT_RE = re.compile(
    r"\b("
    r"action|agreed|approve|approved|block|blocked|blocker|cancel|cancelled|"
    r"change|changed|complete|completed|confluence|create|created|date|deadline|"
    r"decide|decided|decision|delete|deleted|defer|deferred|document|due|issue|"
    r"launch|migrate|move|moved|must|need|owner|plan|priority|problem|release|"
    r"rename|renamed|risk|ship|shipped|should|status|task|todo|update|updated|"
    r"version|will"
    r")\b",
    re.IGNORECASE,
)
_SPACE_RE = re.compile(r"\s+")


def normalize_memory_text(value: str) -> str:
    """Normalize whitespace for memory snippets without changing wording."""
    return _SPACE_RE.sub(" ", (value or "").strip())


def format_memory_entry(entry: dict[str, Any] | str) -> str:
    """Render transcript entries the same way the pipeline expects context."""
    if isinstance(entry, str):
        return normalize_memory_text(entry)
    speaker = entry.get("participant") or entry.get("speaker") or "Speaker"
    text = normalize_memory_text(str(entry.get("text") or entry.get("content") or ""))
    if not text:
        return ""
    return f"{speaker}: {text}"


@dataclass
class _MemoryPoint:
    text: str
    order: int
    score: int


@dataclass
class TranscriptCompactor:
    """Keep a compact memory of transcript lines that have left a recent window.

    The live Jarvis prompt still receives the existing sliding window. This class
    only adds a compressed, bounded memory of older turns so those turns do not
    vanish completely once they fall outside that window.
    """

    window_size: int
    max_memory_chars: int = 8000
    max_points: int = 80
    chunk_size: int = 24
    _recent: list[str] = field(default_factory=list)
    _staged: list[str] = field(default_factory=list)
    _points: list[_MemoryPoint] = field(default_factory=list)
    _seen_keys: set[str] = field(default_factory=set)
    _next_order: int = 0
    compacted_utterances: int = 0

    def observe_utterance(self, text: str) -> None:
        line = normalize_memory_text(text)
        if not line:
            return
        self._recent.append(line)
        overflow = len(self._recent) - max(1, self.window_size)
        if overflow > 0:
            self._staged.extend(self._recent[:overflow])
            del self._recent[:overflow]
            self.compacted_utterances += overflow
        if len(self._staged) >= max(1, self.chunk_size):
            self.force_compact()

    def observe_entry(self, entry: dict[str, Any] | str) -> None:
        self.observe_utterance(format_memory_entry(entry))

    def force_compact(self) -> None:
        if not self._staged:
            return
        for text in self._summarize_chunk(self._staged):
            key = self._dedupe_key(text)
            if not key or key in self._seen_keys:
                continue
            self._seen_keys.add(key)
            self._points.append(
                _MemoryPoint(
                    text=text,
                    order=self._next_order,
                    score=self._score(text),
                )
            )
            self._next_order += 1
        self._staged.clear()
        self._trim()

    def memory_text(self) -> str:
        self.force_compact()
        if not self._points:
            return ""
        lines = [
            "[Compacted meeting memory]",
            (
                "Earlier transcript compressed from "
                f"{self.compacted_utterances} utterance(s). Treat this as prior "
                "meeting context; prefer the recent transcript if details conflict."
            ),
            "Key retained context:",
        ]
        lines.extend(f"- {point.text}" for point in sorted(self._points, key=lambda p: p.order))
        return "\n".join(lines)

    def _summarize_chunk(self, lines: list[str]) -> list[str]:
        candidates = [line for line in lines if _SALIENT_RE.search(line)]
        if not candidates:
            candidates = [*lines[:2], *lines[-2:]]
        out: list[str] = []
        for line in candidates:
            short = line[:280].rstrip()
            if short:
                out.append(short)
            if len(out) >= 10:
                break
        return out

    def _trim(self) -> None:
        if not self._points:
            return
        if len(self._points) > self.max_points:
            keep = sorted(
                self._points,
                key=lambda p: (p.score, p.order),
                reverse=True,
            )[: self.max_points]
            self._points = sorted(keep, key=lambda p: p.order)
            self._seen_keys = {self._dedupe_key(p.text) for p in self._points}
        while self._memory_chars() > self.max_memory_chars and len(self._points) > 1:
            weakest = min(self._points, key=lambda p: (p.score, p.order))
            self._points.remove(weakest)
            self._seen_keys = {self._dedupe_key(p.text) for p in self._points}

    def _memory_chars(self) -> int:
        return sum(len(point.text) + 3 for point in self._points)

    def _score(self, text: str) -> int:
        return len(_SALIENT_RE.findall(text)) + min(3, len(text) // 90)

    def _dedupe_key(self, text: str) -> str:
        return re.sub(r"[^\w\s]", "", normalize_memory_text(text).lower())[:220]


def add_memory_context(transcript_text: str, memory_context: str | None) -> str:
    memory = (memory_context or "").strip()
    if not memory:
        return transcript_text
    if not transcript_text:
        return memory
    return f"{memory}\n\n[Meeting transcript (recent/raw)]\n{transcript_text}"
