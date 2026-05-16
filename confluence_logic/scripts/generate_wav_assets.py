"""
Pre-generate cached acknowledgement and gap-filler audio files.

Two asset types:
  ACK_PHRASES      — ultra-short (1-4 word) micro-acks used on wake detection
  GAP_FILLER_PHRASES — full sentences that bridge the gap while the LLM is thinking

Usage:
    python -m confluence_logic.scripts.generate_wav_assets
    python -m confluence_logic.scripts.generate_wav_assets --acks-only
    python -m confluence_logic.scripts.generate_wav_assets --fillers-only
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
    # Wake-word acknowledgements (used by _handle_bare_wake via get_wake_ack_audio /
    # get_busy_ack_audio — must keep these exact keys).
    ("yes", "Yes?"),
    ("busy", "I'm already on it. Give me a moment."),
    # General micro-acks
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

# Gap filler phrases — spoken while the LLM generates its answer.
# Designed to be formal, context-agnostic, varied, and naturally paced
# (8–14 words each — long enough to cover ~2–3 seconds of processing time).
GAP_FILLER_PHRASES = [
    # Retrieval / lookup signals
    ("gap_filler_01", "Sure, let me retrieve that information for you now."),
    ("gap_filler_02", "Of course — pulling that together right away."),
    ("gap_filler_03", "Certainly, let me look into that for you."),
    ("gap_filler_04", "Let me bring up the relevant details — one moment."),
    ("gap_filler_05", "Understood — accessing that right away."),
    # Processing / thinking signals
    ("gap_filler_06", "Give me just a moment to work through this."),
    ("gap_filler_07", "Allow me a brief moment — I am on it."),
    ("gap_filler_08", "Bear with me for just a second while I process that."),
    ("gap_filler_09", "One moment — let me work through the details."),
    ("gap_filler_10", "Let me gather what you need — this will be brief."),
    # Review / scan signals
    ("gap_filler_11", "Let me review the discussion and come right back."),
    ("gap_filler_12", "I will scan through the conversation — just a moment."),
    ("gap_filler_13", "Let me go through the relevant context for you."),
    ("gap_filler_14", "Reviewing that now — I will be with you shortly."),
    # Formal acknowledgment + action
    ("gap_filler_15", "Noted — let me take care of that for you now."),
    ("gap_filler_16", "Right, let me get to that immediately."),
    ("gap_filler_17", "I will have that ready for you in just a moment."),
    ("gap_filler_18", "Sure — let me circle back on this right away."),
    # Check / verify signals
    ("gap_filler_19", "Let me verify that and report back shortly."),
    ("gap_filler_20", "I am checking on that now — please hold a moment."),
]


def _generate_batch(client, model: str, voice: str, speed: float, phrases: list, label: str):
    for filename, phrase in phrases:
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


def generate_all(acks: bool = True, fillers: bool = True):
    client = OpenAI()
    model = os.getenv("JARVIS_TTS_MODEL", "gpt-4o-mini-tts").strip()
    voice = os.getenv("JARVIS_TTS_VOICE", "echo").strip()
    speed = float(os.getenv("JARVIS_TTS_SPEED", "1.0"))

    AUDIO_DIR.mkdir(parents=True, exist_ok=True)

    if acks:
        print(f"\n--- Micro-acknowledgements ({len(ACK_PHRASES)} phrases) ---")
        _generate_batch(client, model, voice, speed, ACK_PHRASES, "ack")

    if fillers:
        print(f"\n--- Gap fillers ({len(GAP_FILLER_PHRASES)} phrases) ---")
        _generate_batch(client, model, voice, speed, GAP_FILLER_PHRASES, "gap_filler")

    print(f"\nDone. Audio files in: {AUDIO_DIR}")


if __name__ == "__main__":
    acks_only = "--acks-only" in sys.argv
    fillers_only = "--fillers-only" in sys.argv
    generate_all(
        acks=not fillers_only,
        fillers=not acks_only,
    )
