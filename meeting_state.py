"""
Thread-safe meeting state protected by asyncio.Lock.

This module provides MeetingState, a class that encapsulates all
meeting-related state fields (bot_id, transcript_log, is_active,
jarvis_listening) behind an asyncio.Lock to prevent race conditions
when multiple async tasks access or mutate state concurrently.
"""

import asyncio
from typing import Optional


class MeetingState:
    """Thread-safe meeting state protected by asyncio.Lock."""

    def __init__(self):
        self._lock = asyncio.Lock()
        self._bot_id: Optional[str] = None
        self._transcript_log: list[dict] = []
        self._is_active: bool = False
        self._jarvis_listening: bool = False

    async def add_transcript(self, participant: str, text: str, timestamp: float) -> None:
        """Append a transcript entry under lock."""
        async with self._lock:
            self._transcript_log.append({
                "participant": participant,
                "text": text,
                "timestamp": timestamp,
            })

    async def get_transcript(self) -> str:
        """Return accumulated transcript as readable string, or placeholder if empty."""
        async with self._lock:
            if not self._transcript_log:
                return "[No transcript yet]"
            lines = [f"{e['participant']}: {e['text']}" for e in self._transcript_log]
            return "\n".join(lines)

    async def get_transcript_count(self) -> int:
        """Return number of transcript entries."""
        async with self._lock:
            return len(self._transcript_log)

    async def get_bot_id(self) -> Optional[str]:
        """Return current bot_id."""
        async with self._lock:
            return self._bot_id

    async def set_bot_id(self, bot_id: Optional[str]) -> None:
        """Set the bot_id."""
        async with self._lock:
            self._bot_id = bot_id

    async def set_active(self, active: bool) -> None:
        """Set the is_active flag."""
        async with self._lock:
            self._is_active = active

    async def is_active(self) -> bool:
        """Return whether meeting is currently active."""
        async with self._lock:
            return self._is_active

    async def set_listening(self, listening: bool) -> None:
        """Set the jarvis_listening flag."""
        async with self._lock:
            self._jarvis_listening = listening

    async def is_listening(self) -> bool:
        """Return whether Jarvis is in listening mode (wake word heard, awaiting query)."""
        async with self._lock:
            return self._jarvis_listening

    async def get_health_snapshot(self) -> dict:
        """Return a dict snapshot suitable for the /health endpoint."""
        async with self._lock:
            return {
                "bot_id": self._bot_id,
                "active": self._is_active,
                "transcript_lines": len(self._transcript_log),
            }
