"""Recall.ai diarized transcript fetch + normalization (SPK-V3-01).

Fetches the post-meeting diarized transcript from the Recall.ai bot-transcript
endpoint and normalizes each utterance into a flat dict with a real participant
name.  When diarization is enabled, participants are identified by name (e.g.
"JohnDoe"); when only a speaker index is available, the entry is labelled
"Speaker N" — still far better than collapsing everyone to "Meeting:".

BOUNDARY:
  This module does NOT import from ``confluence_logic.jarvis_agentic`` or
  ``confluence_logic.agent_worker`` — those are the locked live voice-path
  modules (Phase-7 / SPK-V3-01 boundary rule).  RECALL_BASE_URL and
  RECALL_API_KEY are re-derived here via ``os.getenv`` using the same
  derivation formula as ``jarvis_agentic.py`` lines 65-67.

KILLSWITCH:
  ``RECALL_TRANSCRIPT_ENABLED`` — set ``JARVIS_RECALL_TRANSCRIPT_ENABLED=1``
  (or ``true`` / ``True``) to enable.  Default ``"0"`` preserves the Phase-7
  cost posture (Recall async-transcription charges are opt-in).

GRACEFUL DEGRADATION:
  Any failure (missing bot_id, HTTP error, empty/pending transcript, parse
  error) returns ``None`` and logs a warning; it never raises.  The Stage 0
  caller (``stages/transcript_source.py``) falls back to the LiveKit
  transcript_log on ``None``.

Threat notes (T-11-22 / T-11-23):
  T-11-22  DoS/availability: bounded timeout (20 s) + broad except → None.
  T-11-23  Cost: killswitch default OFF; env-var opt-in documented in
           ``jarvis_agentic.py`` user_setup block.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, Dict, List, Optional

import requests

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Killswitch — default OFF to preserve the Phase-7 cost posture (T-11-23)
# ---------------------------------------------------------------------------

RECALL_TRANSCRIPT_ENABLED: bool = os.getenv(
    "JARVIS_RECALL_TRANSCRIPT_ENABLED", "0"
) in {"1", "true", "True"}

# ---------------------------------------------------------------------------
# Recall.ai endpoint derivation — mirrors jarvis_agentic.py lines 65-67
# without importing the locked voice module.
# ---------------------------------------------------------------------------

_RECALL_API_KEY: Optional[str] = os.getenv("RECALL_API_KEY")
_RECALL_API_REGION: str = os.getenv("RECALL_API_REGION", "ap-northeast-1")
RECALL_BASE_URL: str = f"https://{_RECALL_API_REGION}.recall.ai/api/v1"

# HTTP timeout in seconds for the transcript GET request (T-11-22).
_FETCH_TIMEOUT_SECS: int = 20


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _make_headers() -> Dict[str, str]:
    """Build Recall.ai auth headers — mirrors create_bot in jarvis_agentic.py."""
    return {
        "Authorization": f"Token {_RECALL_API_KEY or ''}",
        "Content-Type": "application/json",
    }


def _normalize_utterances(raw_transcript: Any) -> Optional[List[Dict[str, Any]]]:
    """Normalize a Recall.ai transcript response into flat dicts.

    Recall.ai transcript endpoint returns a list of objects.  Each object has:
      ``speaker_id``   — an integer index (always present)
      ``speaker_name`` — a display name string (present when diarization
                         identified the participant by name)
      ``words``        — list of word objects each with ``text`` and
                         ``start_time`` / ``end_time`` floats

    Normalizes each utterance to::

        {
            "participant": "<name or 'Speaker N'>",
            "text": "<joined words>",
            "timestamp": <start_time of first word>,
            "source": "recall",
        }

    Returns ``None`` on any structural issue or if the list is empty/None.
    """
    if not raw_transcript:
        return None

    if not isinstance(raw_transcript, list):
        logger.warning("recall_transcript: unexpected payload type %s", type(raw_transcript))
        return None

    entries: List[Dict[str, Any]] = []
    for item in raw_transcript:
        if not isinstance(item, dict):
            continue

        # Resolve participant name — prefer the display name, fall back to index.
        name: str = ""
        if item.get("speaker_name"):
            name = str(item["speaker_name"]).strip()
        if not name:
            speaker_id = item.get("speaker_id")
            if speaker_id is not None:
                name = f"Speaker {speaker_id}"
            else:
                name = "Speaker 0"

        # Join word texts.
        words = item.get("words") or []
        if not isinstance(words, list):
            words = []
        text = " ".join(
            str(w.get("text", "")) for w in words if isinstance(w, dict) and w.get("text")
        ).strip()
        if not text:
            continue

        # Timestamp from first word's start_time.
        timestamp: float = 0.0
        if words and isinstance(words[0], dict):
            timestamp = float(words[0].get("start_time") or 0.0)

        entries.append(
            {
                "participant": name,
                "text": text,
                "timestamp": timestamp,
                "source": "recall",
            }
        )

    if not entries:
        return None

    # Sort by timestamp (ascending) — ensures chronological order.
    entries.sort(key=lambda e: e["timestamp"])
    return entries


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def fetch_recall_transcript(
    bot_id: str,
) -> Optional[List[Dict[str, Any]]]:
    """Fetch and normalize the diarized transcript for a Recall.ai bot.

    Makes a synchronous HTTP GET off the event loop via ``asyncio.to_thread``
    with a bounded timeout (T-11-22).

    Args:
        bot_id: The Recall.ai bot id whose transcript to fetch.

    Returns:
        A list of normalized utterance dicts with keys ``participant``,
        ``text``, ``timestamp``, ``source="recall"``, ordered by timestamp.
        Returns ``None`` on any failure (missing bot_id, HTTP error, empty or
        pending transcript, parse error) — never raises.
    """
    if not bot_id:
        logger.warning("recall_transcript: bot_id is empty — skipping fetch")
        return None

    url = f"{RECALL_BASE_URL}/bot/{bot_id}/transcript/"

    def _get() -> requests.Response:
        return requests.get(
            url,
            headers=_make_headers(),
            timeout=_FETCH_TIMEOUT_SECS,
        )

    try:
        response: requests.Response = await asyncio.to_thread(_get)
    except Exception as exc:
        logger.warning("recall_transcript: HTTP request failed for bot %s: %s", bot_id, exc)
        return None

    try:
        if response.status_code != 200:
            logger.warning(
                "recall_transcript: non-200 status %d for bot %s",
                response.status_code,
                bot_id,
            )
            return None

        data = response.json()
    except Exception as exc:
        logger.warning(
            "recall_transcript: failed to parse JSON for bot %s: %s", bot_id, exc
        )
        return None

    try:
        return _normalize_utterances(data)
    except Exception as exc:
        logger.warning(
            "recall_transcript: normalization error for bot %s: %s", bot_id, exc
        )
        return None
