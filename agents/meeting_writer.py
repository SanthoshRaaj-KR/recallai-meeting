"""
MeetingWriterAgent: pure I/O class that maintains per-meeting .md files and a
JSON meeting index. No LLM calls — this is a structured file writer only.

Responsibilities:
- Write the meeting header block to a new .md file when a meeting starts.
- Append a timestamped batch section each time a batch is flushed.
- Upsert a MeetingIndexEntry into the JSON index after each flush.
- Provide read access to the index and individual .md files.

Design constraints:
- All file writes protected by asyncio.Lock (Phase 1 mandate for shared state).
- All file I/O uses aiofiles to avoid blocking the asyncio event loop.
- Append-only after the header — existing content is never rewritten (D-06).
- Index format: JSON array of MeetingIndexEntry dicts at meetings/meeting_index.json.
- .md file path convention: {base_dir}/{channel_id}/{date}_{meeting_id}.md
"""

import asyncio
import json
import logging
from pathlib import Path

import aiofiles

from storage.models import MeetingIndexEntry, MeetingMetadataOutput

logger = logging.getLogger(__name__)


class MeetingWriterAgent:
    """Writes structured .md files and maintains a JSON meeting index.

    This is a plain Python class, NOT an openai-agents Agent() instance.
    All public methods are async to integrate cleanly with the asyncio event loop.
    All file writes are protected by a single asyncio.Lock to prevent concurrent
    write corruption.
    """

    def __init__(self, base_dir: str = "./meetings"):
        self._base_dir = Path(base_dir)
        self._lock = asyncio.Lock()
        self._index_path = Path(base_dir) / "meeting_index.json"
        self._index_cache: list[MeetingIndexEntry] | None = None

    def _md_path(self, channel_id: str, date_str: str, meeting_id: str) -> Path:
        """Return the canonical .md file path for a meeting.

        Format: {base_dir}/{channel_id}/{date_str}_{meeting_id}.md
        """
        return self._base_dir / channel_id / f"{date_str}_{meeting_id}.md"

    async def write_meeting_header(self, entry: MeetingIndexEntry) -> None:
        """Create the .md file and write the meeting header block.

        Opens in write mode ("w") — only called once per meeting to establish
        the header. Subsequent writes use append_batch which opens in "a" mode.

        Args:
            entry: MeetingIndexEntry with meeting metadata for the header.
        """
        async with self._lock:
            md_path = self._md_path(entry.channel_id, entry.date, entry.meeting_id)
            md_path.parent.mkdir(parents=True, exist_ok=True)
            participants_str = ", ".join(entry.participants) if entry.participants else "TBD"
            header = (
                f"# Meeting: {entry.title}\n"
                f"\n"
                f"**Date:** {entry.date}\n"
                f"**Channel:** {entry.channel_name}\n"
                f"**Participants:** {participants_str}\n"
                f"**Meeting ID:** {entry.meeting_id}\n"
                f"\n"
                f"---\n"
                f"\n"
            )
            async with aiofiles.open(md_path, "w") as f:
                await f.write(header)
            logger.debug("Wrote meeting header: %s", md_path)

    async def append_transcript_lines(
        self,
        entry: MeetingIndexEntry,
        lines: list[str],
    ) -> None:
        """Append raw transcript lines to the meeting's .md file.

        Opens in append mode — existing content is never rewritten.
        Lines should be in "Speaker: utterance" format.

        Args:
            entry: MeetingIndexEntry identifying which .md file to append to.
            lines: List of "Speaker: utterance" strings to append.
        """
        if not lines:
            return
        async with self._lock:
            md_path = self._md_path(entry.channel_id, entry.date, entry.meeting_id)
            block = "\n".join(lines) + "\n"
            async with aiofiles.open(md_path, "a") as f:
                await f.write(block)
            logger.debug("Appended %d transcript lines to %s", len(lines), md_path)

    async def finalize_meeting(
        self,
        entry: MeetingIndexEntry,
        metadata: MeetingMetadataOutput,
    ) -> None:
        """Append the structured Meeting Summary section at end-of-meeting.

        Called once on disconnect after MeetingMetadataAgent has run. Appends
        a clean summary block containing goals, key decisions, and conclusions.

        Args:
            entry: MeetingIndexEntry identifying which .md file to finalize.
            metadata: MeetingMetadataOutput from MeetingMetadataAgent.
        """
        async with self._lock:
            md_path = self._md_path(entry.channel_id, entry.date, entry.meeting_id)

            goals_block = "\n".join(f"- {g}" for g in metadata.goals) or "- None recorded"
            decisions_block = "\n".join(f"- {d}" for d in metadata.key_decisions) or "- None recorded"
            conclusions_block = "\n".join(f"- {c}" for c in metadata.conclusions) or "- None recorded"

            section = (
                f"\n---\n\n"
                f"## Meeting Summary\n\n"
                f"**Goals:**\n{goals_block}\n\n"
                f"**Key Decisions:**\n{decisions_block}\n\n"
                f"**Conclusions:**\n{conclusions_block}\n"
            )
            async with aiofiles.open(md_path, "a") as f:
                await f.write(section)
            logger.debug("Finalized meeting summary for %s", entry.meeting_id)

    async def upsert_index(self, entry: MeetingIndexEntry) -> None:
        """Create or update the meeting's entry in the JSON index.

        Reads the current index (or uses in-memory cache), replaces an existing
        entry with the same meeting_id, or appends a new entry. Writes the
        updated list back and refreshes the cache atomically.

        Args:
            entry: MeetingIndexEntry to insert or update.
        """
        async with self._lock:
            entries: list[dict] = []
            if self._index_path.exists():
                async with aiofiles.open(self._index_path, "r") as f:
                    raw = await f.read()
                entries = json.loads(raw) if raw.strip() else []

            # Replace existing entry with same meeting_id, or append
            entry_dict = json.loads(entry.model_dump_json())
            updated = False
            for i, item in enumerate(entries):
                if item.get("meeting_id") == entry.meeting_id:
                    entries[i] = entry_dict
                    updated = True
                    break
            if not updated:
                entries.append(entry_dict)

            async with aiofiles.open(self._index_path, "w") as f:
                await f.write(json.dumps(entries, indent=2))

            # Refresh in-memory cache so the next read_index() is instant
            self._index_cache = [MeetingIndexEntry.model_validate(item) for item in entries]
            logger.debug("Upserted index entry: %s", entry.meeting_id)

    async def read_index(self) -> list[MeetingIndexEntry]:
        """Read and return all entries from the JSON meeting index.

        Returns the in-memory cache when available (populated by upsert_index
        or a prior read). Falls back to disk on the first call or after a cache
        miss. Returns an empty list if the index file does not exist yet.

        Returns:
            List of MeetingIndexEntry objects, one per meeting.
        """
        # Fast path: return cached list without touching disk
        if self._index_cache is not None:
            return self._index_cache
        async with self._lock:
            # Re-check after acquiring lock — another coroutine may have
            # populated the cache while we waited.
            if self._index_cache is not None:
                return self._index_cache
            if not self._index_path.exists():
                self._index_cache = []
                return self._index_cache
            async with aiofiles.open(self._index_path, "r") as f:
                raw = await f.read()
            entries = json.loads(raw) if raw.strip() else []
            self._index_cache = [MeetingIndexEntry.model_validate(item) for item in entries]
            return self._index_cache

    async def read_md(self, md_path: str) -> str:
        """Read and return the full content of a meeting .md file.

        No lock needed — the .md file is append-only after the header, making
        concurrent reads safe (D-06).

        Args:
            md_path: Relative or absolute path to the .md file.

        Returns:
            Full string content of the .md file.

        Raises:
            FileNotFoundError: If the .md file does not exist.
        """
        path = Path(md_path)
        if not path.exists():
            raise FileNotFoundError(f"Meeting .md file not found: {md_path}")
        async with aiofiles.open(path, "r") as f:
            return await f.read()
