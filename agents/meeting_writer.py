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

from storage.models import BatchSummaryOutput, MeetingIndexEntry

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

    async def append_batch(
        self,
        entry: MeetingIndexEntry,
        batch_num: int,
        batch_summary: BatchSummaryOutput,
        timestamp_str: str,
    ) -> None:
        """Append a batch summary section to the meeting's .md file.

        Always opens in append mode ("a") — existing content is never rewritten.
        This method is the only place batch content is written; the file is safe
        to read concurrently while being written (D-06).

        Args:
            entry: MeetingIndexEntry identifying which .md file to append to.
            batch_num: Sequential batch number (1-based).
            batch_summary: BatchSummaryOutput from RollingSummarizerAgent.
            timestamp_str: Human-readable time formatted as "HH:MM AM/PM" by caller.
        """
        async with self._lock:
            md_path = self._md_path(entry.channel_id, entry.date, entry.meeting_id)
            key_points_block = "\n".join(f"- {p}" for p in batch_summary.key_points)
            speakers_str = ", ".join(batch_summary.speakers)
            section = (
                f"## {timestamp_str} — Batch {batch_num}\n"
                f"\n"
                f"{batch_summary.summary_text}\n"
                f"\n"
                f"**Key Points:**\n"
                f"{key_points_block}\n"
                f"\n"
                f"**Speakers:** {speakers_str}\n"
                f"\n"
                f"---\n"
                f"\n"
            )
            async with aiofiles.open(md_path, "a") as f:
                await f.write(section)
            logger.debug("Appended batch %d to %s", batch_num, md_path)

    async def upsert_index(self, entry: MeetingIndexEntry) -> None:
        """Create or update the meeting's entry in the JSON index.

        Reads the current index, replaces an existing entry with the same
        meeting_id, or appends a new entry. Writes the updated list back.

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
            updated = False
            for i, item in enumerate(entries):
                if item.get("meeting_id") == entry.meeting_id:
                    entries[i] = json.loads(entry.model_dump_json())
                    updated = True
                    break
            if not updated:
                entries.append(json.loads(entry.model_dump_json()))

            async with aiofiles.open(self._index_path, "w") as f:
                await f.write(json.dumps(entries, indent=2))
            logger.debug("Upserted index entry: %s", entry.meeting_id)

    async def read_index(self) -> list[MeetingIndexEntry]:
        """Read and return all entries from the JSON meeting index.

        Returns an empty list if the index file does not exist yet.

        Returns:
            List of MeetingIndexEntry objects, one per meeting.
        """
        async with self._lock:
            if not self._index_path.exists():
                return []
            async with aiofiles.open(self._index_path, "r") as f:
                raw = await f.read()
            entries = json.loads(raw) if raw.strip() else []
            return [MeetingIndexEntry.model_validate(item) for item in entries]

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
