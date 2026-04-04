"""
Async JSON file storage for meeting records.

MetadataStore provides read/write/list operations for MeetingRecord objects,
persisting each meeting as a JSON file at:
    {base_dir}/{channel_id}/{meeting_id}.json

Uses aiofiles for all file I/O to avoid blocking the asyncio event loop
(per project constraint: blocking I/O in async handlers stalls the event loop).
Uses pathlib.Path for all path operations.
"""

import logging
from pathlib import Path

import aiofiles

from storage.models import MeetingRecord

logger = logging.getLogger(__name__)


class MetadataStore:
    """
    Async JSON file store for MeetingRecord objects.

    Files are organized as:
        {base_dir}/{channel_id}/{meeting_id}.json

    All I/O is non-blocking (uses aiofiles). Directory creation is synchronous
    (pathlib mkdir) which is acceptable since it is a rare, fast operation.
    """

    def __init__(self, base_dir: str = "./meetings"):
        self._base_dir = Path(base_dir)

    async def write(self, record: MeetingRecord) -> Path:
        """Write meeting record as JSON file. Returns the file path.

        Creates intermediate directories if they don't exist.
        Overwrites an existing file for the same meeting_id.

        Args:
            record: The MeetingRecord to persist.

        Returns:
            Path to the written JSON file.
        """
        dir_path = self._base_dir / record.channel_id
        dir_path.mkdir(parents=True, exist_ok=True)
        file_path = dir_path / f"{record.meeting_id}.json"
        async with aiofiles.open(file_path, "w") as f:
            await f.write(record.model_dump_json(indent=2))
        logger.debug("✅ Wrote meeting record: %s", file_path)
        return file_path

    async def read(self, channel_id: str, meeting_id: str) -> MeetingRecord:
        """Read meeting record from JSON file.

        Args:
            channel_id: The Slack channel ID.
            meeting_id: The meeting UUID.

        Returns:
            The deserialized MeetingRecord.

        Raises:
            FileNotFoundError: If no JSON file exists for the given channel/meeting.
        """
        file_path = self._base_dir / channel_id / f"{meeting_id}.json"
        if not file_path.exists():
            raise FileNotFoundError(
                f"Meeting record not found: {file_path}"
            )
        async with aiofiles.open(file_path, "r") as f:
            data = await f.read()
        logger.debug("📖 Read meeting record: %s", file_path)
        return MeetingRecord.model_validate_json(data)

    async def list_by_channel(self, channel_id: str) -> list[str]:
        """List all meeting_ids for a channel.

        Returns empty list if the channel directory doesn't exist.

        Args:
            channel_id: The Slack channel ID.

        Returns:
            List of meeting_id strings (stems of .json filenames).
        """
        dir_path = self._base_dir / channel_id
        if not dir_path.exists():
            return []
        return [p.stem for p in dir_path.glob("*.json")]
