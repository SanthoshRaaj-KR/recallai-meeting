"""
Audio cache for pre-generated acknowledgement MP3 files.
Loads all .mp3 files from assets/audio/ into memory on first access,
then serves random selections without any TTS API call.
"""
import logging
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

AUDIO_DIR = Path(__file__).resolve().parent / "assets" / "audio"

# In-memory cache: filename_stem -> bytes
_cache: Dict[str, bytes] = {}
_cache_loaded: bool = False


def load_audio_cache(directory: Optional[Path] = None) -> int:
    """
    Load all .mp3 files from the audio assets directory into memory.
    Returns the number of files loaded.
    Call once at startup or lazily on first access.
    """
    global _cache, _cache_loaded
    audio_dir = directory or AUDIO_DIR
    _cache.clear()

    if not audio_dir.is_dir():
        logger.warning("Audio cache directory does not exist: %s", audio_dir)
        _cache_loaded = True
        return 0

    for mp3_file in audio_dir.glob("*.mp3"):
        try:
            _cache[mp3_file.stem] = mp3_file.read_bytes()
            logger.debug("Loaded cached audio: %s (%d bytes)", mp3_file.stem, len(_cache[mp3_file.stem]))
        except Exception as e:
            logger.error("Failed to load audio file %s: %s", mp3_file.name, e)

    _cache_loaded = True
    logger.info("Audio cache loaded: %d files from %s", len(_cache), audio_dir)
    return len(_cache)


def get_random_ack_audio() -> Optional[Tuple[str, bytes]]:
    """
    Return a random (filename_stem, audio_bytes) tuple from the cache.
    Returns None if the cache is empty or not loaded.
    Lazily loads the cache on first call.
    """
    global _cache_loaded
    if not _cache_loaded:
        load_audio_cache()

    if not _cache:
        return None

    key = random.choice(list(_cache.keys()))
    return (key, _cache[key])


def get_random_filler_audio() -> Optional[Tuple[str, bytes]]:
    """
    Return a random pre-generated gap filler audio clip (gap_filler_*.mp3).
    These are longer phrases (8-14 words) designed to bridge processing time
    while the LLM generates the actual answer.

    Returns None if no gap filler files are cached — callers should fall back
    to the LLM-generated contextual filler in that case.
    Lazily loads the cache on first call.
    """
    global _cache_loaded
    if not _cache_loaded:
        load_audio_cache()

    filler_keys = [k for k in _cache if k.startswith("gap_filler_")]
    if not filler_keys:
        return None

    key = random.choice(filler_keys)
    return (key, _cache[key])


def get_wake_ack_audio() -> Optional[Tuple[str, bytes]]:
    """
    Return ("yes", audio_bytes) if the wake-ack clip is cached, else None.
    Used by _handle_bare_wake to play a pre-generated "Yes?" without a live TTS call.
    Lazily loads the cache on first call.
    """
    global _cache_loaded
    if not _cache_loaded:
        load_audio_cache()

    data = _cache.get("yes")
    if data is None:
        return None
    return ("yes", data)


def get_busy_ack_audio() -> Optional[Tuple[str, bytes]]:
    """
    Return ("busy", audio_bytes) if the busy-ack clip is cached, else None.
    Used by _handle_bare_wake to play a pre-generated "I'm already on it" without a
    live TTS call.
    Lazily loads the cache on first call.
    """
    global _cache_loaded
    if not _cache_loaded:
        load_audio_cache()

    data = _cache.get("busy")
    if data is None:
        return None
    return ("busy", data)


def get_cache_size() -> int:
    """Return the number of cached audio files."""
    if not _cache_loaded:
        load_audio_cache()
    return len(_cache)
