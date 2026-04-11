"""
Meeting transcript responder for Jarvis voice assistant.
Provides LLM-powered summarization and opinion generation from meeting transcript logs.
"""
import asyncio
import logging
import os
from typing import Optional, List, Dict, Any

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

_openai_client: Optional[OpenAI] = None


def _get_client() -> OpenAI:
    global _openai_client
    if _openai_client is None:
        _openai_client = OpenAI()
    return _openai_client


MEETING_RESPONDER_MODEL = os.getenv("JARVIS_GENERAL_MODEL", "gpt-4o-mini").strip()
JARVIS_SUMMARY_MAX_TOKENS = int(os.getenv("JARVIS_SUMMARY_MAX_TOKENS", "400"))
JARVIS_SUMMARY_BRIEF_MAX_TOKENS = int(os.getenv("JARVIS_SUMMARY_BRIEF_MAX_TOKENS", "150"))
JARVIS_OPINION_MAX_TOKENS = int(os.getenv("JARVIS_OPINION_MAX_TOKENS", "200"))

_EMPTY_TRANSCRIPT_FALLBACK = "I haven't heard anything in the meeting yet."


def _format_transcript(transcript_log: List[Dict[str, Any]]) -> str:
    """Format transcript_log entries as '[Participant]: [text]' lines in chronological order."""
    return "\n".join(
        f"{entry['participant']}: {entry['text']}"
        for entry in transcript_log
    )


async def summarize_meeting(transcript_log: List[Dict[str, Any]], detail_level: str = "detailed") -> str:
    """
    Generate a spoken-language summary of everything in the meeting transcript.

    Args:
        transcript_log: List of dicts with 'participant', 'text', 'timestamp' keys.
        detail_level: "brief" for a short bullet-style summary, "detailed" (default) for full narrative.

    Returns:
        A TTS-friendly summary paragraph, or fallback string if transcript is empty.
    """
    if not transcript_log:
        return _EMPTY_TRANSCRIPT_FALLBACK

    transcript_text = _format_transcript(transcript_log)
    # Truncate from the start if excessively long (keep the most recent content)
    max_chars = 8000
    if len(transcript_text) > max_chars:
        transcript_text = transcript_text[-max_chars:]

    if detail_level == "brief":
        system_prompt = (
            "You are Jarvis, an AI assistant attending a live meeting. "
            "The user wants a brief summary. Give a short, punchy bullet-point style summary of the 3-5 most important points discussed. "
            "No more than 5 bullet points. Speak naturally — say 'Here are the main points:' then list them conversationally."
        )
        max_tokens = JARVIS_SUMMARY_BRIEF_MAX_TOKENS
    else:
        system_prompt = (
            "You are Jarvis, an AI assistant attending a live meeting. "
            "The user has asked you to summarize what was discussed. "
            "Write a clear, spoken-language summary of the key topics, decisions, and points raised. "
            "Speak as if delivering the summary aloud — no markdown, no bullet points, no formatting. "
            "Use natural sentences. Cover all significant content without omitting important details. "
            "Do not editorialize; just report what was said."
        )
        max_tokens = JARVIS_SUMMARY_MAX_TOKENS

    user_prompt = f"Here is the meeting transcript so far:\n\n{transcript_text}\n\nPlease summarize what was discussed."

    try:
        response = await asyncio.to_thread(
            lambda: _get_client().chat.completions.create(
                model=MEETING_RESPONDER_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=max_tokens,
                temperature=0.5,
            )
        )
        answer = (response.choices[0].message.content or "").strip()
        if not answer:
            return "I wasn't able to put together a summary right now."
        logger.info("Meeting summary generated (%d chars)", len(answer))
        return answer
    except Exception as e:
        logger.error("Meeting summarizer failed: %s", e)
        return "Sorry, I couldn't generate a summary right now."


async def generate_opinion(transcript_log: List[Dict[str, Any]], query: str = "") -> str:
    """
    Generate a confident first-person opinion grounded in the meeting transcript.

    Args:
        transcript_log: List of dicts with 'participant', 'text', 'timestamp' keys.
        query: The user's specific question or topic they want an opinion on.

    Returns:
        A TTS-friendly opinion string starting with a grounding phrase, or fallback if empty.
    """
    if not transcript_log:
        return _EMPTY_TRANSCRIPT_FALLBACK

    transcript_text = _format_transcript(transcript_log)
    max_chars = 8000
    if len(transcript_text) > max_chars:
        transcript_text = transcript_text[-max_chars:]

    system_prompt = (
        "You are Jarvis, an AI assistant attending a live meeting. "
        "The user wants your opinion or recommendation based on what was discussed. "
        "Deliver a confident, direct, first-person opinion. Pick a side — do not hedge with 'it depends' alone. "
        "Always open with a short grounding phrase such as 'Based on what I heard,' or 'From the discussion so far,' or 'Given what the team discussed,' — then immediately give your opinion. "
        "Reference specific points or trade-offs you heard to justify your view. "
        "Speak naturally as if talking aloud in the meeting. "
        "No markdown, no bullet points, no formatting. 2 to 4 sentences maximum."
    )

    user_content = f"Meeting transcript:\n\n{transcript_text}"
    if query:
        user_content += f"\n\nThe user asked: {query}"
    user_content += "\n\nWhat is your opinion or recommendation?"

    try:
        response = await asyncio.to_thread(
            lambda: _get_client().chat.completions.create(
                model=MEETING_RESPONDER_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
                max_tokens=JARVIS_OPINION_MAX_TOKENS,
                temperature=0.7,
            )
        )
        answer = (response.choices[0].message.content or "").strip()
        if not answer:
            return "I'm not sure what to recommend based on what I heard."
        logger.info("Meeting opinion generated (%d chars)", len(answer))
        return answer
    except Exception as e:
        logger.error("Meeting opinion generator failed: %s", e)
        return "Sorry, I couldn't form an opinion right now."
