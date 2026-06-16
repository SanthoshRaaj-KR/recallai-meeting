"""
Generate src/warmup_prompt.wav — the pre-determined silent backend prompt.

The WAV file contains the phrase "Hello Jarvis, please introduce yourself to the
meeting."  Its TEXT CONTENT drives the LLM intro at session start; the audio file
itself is kept in sync as documentation and for future audio-injection use cases.

Run once (or whenever the prompt text changes):
    conda run -n meetagents uv run python src/generate_warmup_prompt.py
"""

import asyncio
from pathlib import Path

import edge_tts

_PROMPT_TEXT = "Hello Jarvis, please introduce yourself to the meeting."
_OUTPUT_PATH = Path(__file__).parent / "warmup_prompt.wav"


async def _generate() -> None:
    communicate = edge_tts.Communicate(_PROMPT_TEXT, voice="en-US-JennyNeural")
    await communicate.save(str(_OUTPUT_PATH))
    print(f"Saved: {_OUTPUT_PATH}  ({_OUTPUT_PATH.stat().st_size} bytes)")


if __name__ == "__main__":
    asyncio.run(_generate())
