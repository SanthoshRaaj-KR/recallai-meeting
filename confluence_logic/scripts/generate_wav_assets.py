"""
Pre-generate cached acknowledgement audio files.

Usage:
    python -m confluence_logic.scripts.generate_wav_assets
"""
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

AUDIO_DIR = Path(__file__).resolve().parent.parent / "assets" / "audio"

# Acknowledgement phrases to cache — these replace _INSTANT_ACKS TTS calls
ACK_PHRASES = [
    ("on_it", "On it."),
    ("sure_thing", "Sure thing."),
    ("give_me_a_sec", "Give me a sec."),
    ("got_it", "Got it."),
    ("one_moment", "One moment."),
    ("right_away", "Right away."),
    ("working_on_it", "Working on it."),
    ("let_me_check", "Let me check."),
    ("give_me_a_moment", "Give me a moment."),
    ("let_me_handle_that", "Sure, let me handle that."),
]


def generate_all():
    client = OpenAI()
    model = os.getenv("JARVIS_TTS_MODEL", "gpt-4o-mini-tts").strip()
    voice = os.getenv("JARVIS_TTS_VOICE", "echo").strip()
    speed = float(os.getenv("JARVIS_TTS_SPEED", "1.0"))

    AUDIO_DIR.mkdir(parents=True, exist_ok=True)

    for filename, phrase in ACK_PHRASES:
        out_path = AUDIO_DIR / f"{filename}.mp3"
        if out_path.exists():
            print(f"  SKIP {out_path.name} (already exists)")
            continue
        print(f"  Generating {out_path.name}: '{phrase}'")
        try:
            response = client.audio.speech.create(
                model=model,
                voice=voice,
                input=phrase,
                response_format="mp3",
                speed=speed,
            )
            audio_bytes = response.read() if hasattr(response, "read") else response.content
            out_path.write_bytes(audio_bytes)
            print(f"  OK {out_path.name} ({len(audio_bytes)} bytes)")
        except Exception as e:
            print(f"  FAIL {out_path.name}: {e}", file=sys.stderr)

    print(f"\nDone. Audio files in: {AUDIO_DIR}")


if __name__ == "__main__":
    generate_all()
