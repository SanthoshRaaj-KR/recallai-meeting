from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any, Callable

from openai import OpenAI

logger = logging.getLogger(__name__)

_SPACE_RE = re.compile(r"\s+")
CompactFn = Callable[[str, list[str], int], str]


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
class TranscriptCompactor:
    """Keep an LLM-compacted memory of transcript lines outside the recent window.

    The live Jarvis prompt still receives the existing sliding window. This class
    rewrites older turns into a short memory block so those turns do not vanish
    completely once they fall outside that window.
    """

    window_size: int
    max_memory_chars: int = 8000
    chunk_size: int = 24
    model: str = field(
        default_factory=lambda: (
            os.getenv("MY_AGENT_MEMORY_MODEL")
            or os.getenv("JARVIS_MEMORY_MODEL")
            or os.getenv("MY_AGENT_REVIEW_MODEL")
            or os.getenv("JARVIS_REVIEW_MODEL")
            or "gpt-4o-mini"
        ).strip()
    )
    compact_fn: CompactFn | None = None
    _recent: list[str] = field(default_factory=list)
    _staged: list[str] = field(default_factory=list)
    _memory: str = ""
    _openai: OpenAI | None = None
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
        staged = [line for line in self._staged if line.strip()]
        if not staged:
            self._staged.clear()
            return
        self._memory = self._compact_block(self._memory, staged)
        self._staged.clear()
        self._memory = self._trim_text(self._memory)

    def memory_text(self) -> str:
        self.force_compact()
        if not self._memory:
            return ""
        lines = [
            "[Compacted meeting memory]",
            (
                "Earlier transcript LLM-compacted from "
                f"{self.compacted_utterances} utterance(s). Treat this as prior "
                "meeting context; prefer the recent transcript if details conflict."
            ),
            "Key retained context:",
            self._memory,
        ]
        return "\n".join(lines)

    def _compact_block(self, previous_memory: str, transcript_block: list[str]) -> str:
        try:
            if self.compact_fn:
                compacted = self.compact_fn(previous_memory, transcript_block, self.max_memory_chars)
            else:
                compacted = self._compact_block_with_openai(previous_memory, transcript_block)
            return self._clean_llm_memory(compacted)
        except Exception as exc:
            logger.warning("LLM transcript compaction failed; preserving trimmed raw memory: %s", exc)
            return self._fallback_memory(previous_memory, transcript_block)

    def _compact_block_with_openai(self, previous_memory: str, transcript_block: list[str]) -> str:
        if self._openai is None:
            self._openai = OpenAI()
        prompt = (
            "Compact meeting transcript memory. You will receive any older compacted "
            "memory plus the next chronological transcript block. Produce a new compacted "
            "memory that is a continuation of the older memory, as short as possible.\n\n"
            "Keep only important durable facts: decisions, action items, owners, due dates, "
            "status changes, risks, blockers, explicit documentation/Confluence changes, "
            "page names, old values, new values, and final agreed state. Drop small talk, "
            "repetition, uncertainty that was later resolved, and transcript mechanics.\n\n"
            "If the new block has no durable information and older memory exists, return "
            "the older memory cleaned up. If nothing important remains, return an empty string. "
            "Use terse bullets or short fragments. Do not invent facts."
        )
        payload = {
            "older_compacted_memory": previous_memory or "",
            "new_transcript_block": "\n".join(transcript_block),
            "max_characters": self.max_memory_chars,
        }
        opts: dict[str, Any] = {"model": self.model}
        if self.model.startswith(("gpt-5", "o1", "o3", "o4")):
            opts["max_completion_tokens"] = 900
        else:
            opts["max_tokens"] = 900
            opts["temperature"] = 0.0
        response = self._openai.chat.completions.create(
            **opts,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
        )
        return response.choices[0].message.content or ""

    def _clean_llm_memory(self, value: str) -> str:
        text = (value or "").strip()
        text = re.sub(r"(?i)^\s*\[?compacted meeting memory\]?\s*", "", text).strip()
        text = re.sub(r"(?i)^key retained context:\s*", "", text).strip()
        return self._trim_text(text)

    def _fallback_memory(self, previous_memory: str, transcript_block: list[str]) -> str:
        parts = [previous_memory.strip(), "\n".join(transcript_block).strip()]
        return self._trim_text("\n".join(part for part in parts if part))

    def _trim_text(self, value: str) -> str:
        text = (value or "").strip()
        if len(text) <= self.max_memory_chars:
            return text
        return text[-self.max_memory_chars :].lstrip()


def add_memory_context(transcript_text: str, memory_context: str | None) -> str:
    memory = (memory_context or "").strip()
    if not memory:
        return transcript_text
    if not transcript_text:
        return memory
    return f"{memory}\n\n[Meeting transcript (recent/raw)]\n{transcript_text}"
