"""
Meeting transcript responder for Jarvis voice assistant.
Provides LLM-powered summarization and opinion generation from meeting transcript logs.
"""
import asyncio
import difflib
import logging
import os
import re
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
JARVIS_SUMMARY_BRIEF_MAX_TOKENS = int(os.getenv("JARVIS_SUMMARY_BRIEF_MAX_TOKENS", "280"))
JARVIS_OPINION_MAX_TOKENS = int(os.getenv("JARVIS_OPINION_MAX_TOKENS", "350"))
JARVIS_ACTION_ITEMS_MAX_TOKENS = int(os.getenv("JARVIS_ACTION_ITEMS_MAX_TOKENS", "300"))
JARVIS_SPEAKER_QUERY_MAX_TOKENS = int(os.getenv("JARVIS_SPEAKER_QUERY_MAX_TOKENS", "250"))

_EMPTY_TRANSCRIPT_FALLBACK = "I haven't heard anything in the meeting yet."


def _format_transcript(transcript_log: List[Dict[str, Any]]) -> str:
    """Format transcript_log entries as '[Participant]: [text]' lines in chronological order."""
    return "\n".join(
        f"{entry['participant']}: {entry['text']}"
        for entry in transcript_log
    )


async def summarize_meeting(transcript_log: List[Dict[str, Any]], detail_level: str = "brief") -> str:
    """
    Generate a spoken-language summary of everything in the meeting transcript.

    Args:
        transcript_log: List of dicts with 'participant', 'text', 'timestamp' keys.
        detail_level: "brief" (default) for a spoken-list summary, "detailed" for full narrative.

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
            "The user wants a brief summary. Give a short spoken-style summary of the 3-5 most important points discussed. "
            "Say 'Here are the main points:' then present each point using 'First,' 'Second,' 'Third,' etc. "
            "No markdown, no bullet points, no dashes, no formatting — speak in natural sentences only."
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
        "Do not use phrases like 'might', 'could potentially', or 'it depends' without committing to a clear recommendation. "
        "Always open with a short grounding phrase such as 'Based on what I heard,' or 'From the discussion so far,' or 'Given what the team discussed,' — then immediately give your opinion. "
        "Reference specific points, numbers, costs, or percentages from the transcript to justify your view — quote exact figures when they are relevant. "
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


async def extract_action_items(transcript_log: List[Dict[str, Any]]) -> str:
    """
    Extract action items, decisions, and commitments from the meeting transcript.
    Returns a spoken-language list of action items suitable for TTS.
    """
    if not transcript_log:
        return "I haven't heard anything in the meeting yet, so there are no action items."

    transcript_text = _format_transcript(transcript_log)
    max_chars = 8000
    if len(transcript_text) > max_chars:
        transcript_text = transcript_text[-max_chars:]

    system_prompt = (
        "You are Jarvis, an AI assistant in a live meeting. "
        "Extract all action items, decisions, and commitments from this transcript. "
        "For each action item, note WHO is responsible (if mentioned) and WHAT they need to do. "
        "Present them as a spoken list — say 'Here are the action items:' then list them naturally. "
        "If no clear action items exist, say so honestly. "
        "Speak naturally — no markdown, no bullet points, no formatting. "
        "Number them verbally (e.g., 'First, ...', 'Second, ...', 'Third, ...')."
    )

    try:
        response = await asyncio.to_thread(
            lambda: _get_client().chat.completions.create(
                model=MEETING_RESPONDER_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Meeting transcript:\n{transcript_text}"},
                ],
                max_tokens=JARVIS_ACTION_ITEMS_MAX_TOKENS,
                temperature=0.3,
            )
        )
        answer = (response.choices[0].message.content or "").strip()
        return answer or "I couldn't identify any clear action items from the discussion so far."
    except Exception as e:
        logger.error("Action items extraction failed: %s", e)
        return "Sorry, I had trouble extracting action items right now."


async def summarize_speaker(transcript_log: List[Dict[str, Any]], speaker_name: str) -> str:
    """
    Summarize what a specific participant said in the meeting.
    Uses fuzzy matching on speaker_name against transcript participant names.
    """
    if not transcript_log:
        return "I haven't heard anything in the meeting yet."

    # Fuzzy match speaker name against participants.
    # Strategy: compare against full name AND each individual name token (handles
    # first-name-only queries like "Anjali" matching "Anjali Singh").
    speaker_lower = speaker_name.lower().strip()
    participants = set(entry["participant"] for entry in transcript_log)
    matched_participant = None
    best_ratio = 0.0
    for p in participants:
        p_lower = p.lower()
        # Full-name similarity
        ratio = difflib.SequenceMatcher(None, speaker_lower, p_lower).ratio()
        # Also check against each individual token (first name, last name)
        for token in p_lower.split():
            token_ratio = difflib.SequenceMatcher(None, speaker_lower, token).ratio()
            if token_ratio > ratio:
                ratio = token_ratio
        if ratio >= 0.70 and ratio > best_ratio:
            best_ratio = ratio
            matched_participant = p
    # Substring containment fallback for exact first-name queries (e.g. "Anjali" in "Anjali Singh")
    if not matched_participant:
        for p in participants:
            if speaker_lower in p.lower() or p.lower() in speaker_lower:
                matched_participant = p
                break

    if not matched_participant:
        available = ", ".join(sorted(participants))
        return f"I don't see anyone named {speaker_name} in the transcript. The participants I've heard are: {available}."

    # Filter transcript to only this speaker's entries
    speaker_entries = [e for e in transcript_log if e["participant"] == matched_participant]
    if not speaker_entries:
        return f"{matched_participant} hasn't said anything yet."

    speaker_text = "\n".join(e["text"] for e in speaker_entries)
    max_chars = 4000
    if len(speaker_text) > max_chars:
        speaker_text = speaker_text[-max_chars:]

    system_prompt = (
        f"You are Jarvis, an AI assistant in a live meeting. "
        f"Summarize what {matched_participant} has said and contributed to the discussion. "
        "Cover their key points, opinions, and any decisions or commitments they made. "
        "Speak naturally — no markdown, no bullet points. Keep it concise (3-5 sentences max)."
    )

    try:
        response = await asyncio.to_thread(
            lambda: _get_client().chat.completions.create(
                model=MEETING_RESPONDER_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"{matched_participant}'s contributions:\n{speaker_text}"},
                ],
                max_tokens=JARVIS_SPEAKER_QUERY_MAX_TOKENS,
                temperature=0.4,
            )
        )
        answer = (response.choices[0].message.content or "").strip()
        return answer or f"I couldn't summarize {matched_participant}'s contributions right now."
    except Exception as e:
        logger.error("Speaker summarization failed: %s", e)
        return f"Sorry, I had trouble summarizing what {matched_participant} said."


# ---------------------------------------------------------------------------
# Streaming variants — yield complete sentences as the LLM generates them
# ---------------------------------------------------------------------------

_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?]) {1,2}")


async def _stream_sentences(messages: list, max_tokens: int, temperature: float):
    """Async generator: stream an OpenAI chat completion and yield complete sentences.

    Runs the blocking OpenAI streaming iterator in a thread pool and pushes
    complete sentences (split on sentence-ending punctuation) into an asyncio
    Queue so callers can process them concurrently with other work.
    """
    sentence_queue: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_running_loop()

    def _worker():
        buffer = ""
        try:
            stream = _get_client().chat.completions.create(
                model=MEETING_RESPONDER_MODEL,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                stream=True,
            )
            for chunk in stream:
                token = chunk.choices[0].delta.content or ""
                buffer += token
                # Extract all complete sentences from the buffer
                while True:
                    m = _SENTENCE_BOUNDARY.search(buffer)
                    if not m:
                        break
                    sentence = buffer[: m.start() + 1].strip()
                    buffer = buffer[m.end() :]
                    if sentence:
                        asyncio.run_coroutine_threadsafe(sentence_queue.put(sentence), loop)
        except Exception as e:
            logger.error("LLM streaming failed: %s", e)
        finally:
            if buffer.strip():
                asyncio.run_coroutine_threadsafe(sentence_queue.put(buffer.strip()), loop)
            asyncio.run_coroutine_threadsafe(sentence_queue.put(None), loop)  # sentinel

    worker_task = asyncio.create_task(asyncio.to_thread(_worker))
    try:
        while True:
            sentence = await sentence_queue.get()
            if sentence is None:
                break
            yield sentence
    finally:
        try:
            await worker_task
        except asyncio.CancelledError:
            pass


async def summarize_meeting_streaming(transcript_log: List[Dict[str, Any]], detail_level: str = "brief"):
    """Streaming version of summarize_meeting — yields sentences as the LLM generates them."""
    if not transcript_log:
        yield _EMPTY_TRANSCRIPT_FALLBACK
        return

    transcript_text = _format_transcript(transcript_log)
    if len(transcript_text) > 8000:
        transcript_text = transcript_text[-8000:]

    if detail_level == "brief":
        system_prompt = (
            "You are Jarvis, an AI assistant attending a live meeting. "
            "The user wants a brief summary. Give a short spoken-style summary of the 3-5 most important points discussed. "
            "Say 'Here are the main points:' then present each point using 'First,' 'Second,' 'Third,' etc. "
            "No markdown, no bullet points, no dashes, no formatting — speak in natural sentences only."
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

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Here is the meeting transcript so far:\n\n{transcript_text}\n\nPlease summarize what was discussed."},
    ]
    async for sentence in _stream_sentences(messages, max_tokens, 0.5):
        yield sentence


async def generate_opinion_streaming(transcript_log: List[Dict[str, Any]], query: str = ""):
    """Streaming version of generate_opinion — yields sentences as the LLM generates them."""
    if not transcript_log:
        yield _EMPTY_TRANSCRIPT_FALLBACK
        return

    transcript_text = _format_transcript(transcript_log)
    if len(transcript_text) > 8000:
        transcript_text = transcript_text[-8000:]

    system_prompt = (
        "You are Jarvis, an AI assistant attending a live meeting. "
        "The user wants your opinion or recommendation based on what was discussed. "
        "Deliver a confident, direct, first-person opinion. Pick a side — do not hedge with 'it depends' alone. "
        "Do not use phrases like 'might', 'could potentially', or 'it depends' without committing to a clear recommendation. "
        "Always open with a short grounding phrase such as 'Based on what I heard,' or 'From the discussion so far,' or 'Given what the team discussed,' — then immediately give your opinion. "
        "Reference specific points, numbers, costs, or percentages from the transcript to justify your view — quote exact figures when they are relevant. "
        "Speak naturally as if talking aloud in the meeting. "
        "No markdown, no bullet points, no formatting. 2 to 4 sentences maximum."
    )
    user_content = f"Meeting transcript:\n\n{transcript_text}"
    if query:
        user_content += f"\n\nThe user asked: {query}"
    user_content += "\n\nWhat is your opinion or recommendation?"

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]
    async for sentence in _stream_sentences(messages, JARVIS_OPINION_MAX_TOKENS, 0.7):
        yield sentence


async def extract_action_items_streaming(transcript_log: List[Dict[str, Any]]):
    """Streaming version of extract_action_items — yields sentences as the LLM generates them."""
    if not transcript_log:
        yield "I haven't heard anything in the meeting yet, so there are no action items."
        return

    transcript_text = _format_transcript(transcript_log)
    if len(transcript_text) > 8000:
        transcript_text = transcript_text[-8000:]

    system_prompt = (
        "You are Jarvis, an AI assistant in a live meeting. "
        "Extract all action items, decisions, and commitments from this transcript. "
        "For each action item, note WHO is responsible (if mentioned) and WHAT they need to do. "
        "Present them as a spoken list — say 'Here are the action items:' then list them naturally. "
        "If no clear action items exist, say so honestly. "
        "Speak naturally — no markdown, no bullet points, no formatting. "
        "Number them verbally (e.g., 'First, ...', 'Second, ...', 'Third, ...')."
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Meeting transcript:\n{transcript_text}"},
    ]
    async for sentence in _stream_sentences(messages, JARVIS_ACTION_ITEMS_MAX_TOKENS, 0.3):
        yield sentence


async def summarize_speaker_streaming(transcript_log: List[Dict[str, Any]], speaker_name: str):
    """Streaming version of summarize_speaker — yields sentences as the LLM generates them.

    Performs the same fuzzy-match speaker lookup as summarize_speaker before streaming.
    Yields a single error sentence if the speaker cannot be matched.
    """
    if not transcript_log:
        yield "I haven't heard anything in the meeting yet."
        return

    speaker_lower = speaker_name.lower().strip()
    participants = set(entry["participant"] for entry in transcript_log)
    matched_participant = None
    best_ratio = 0.0
    for p in participants:
        p_lower = p.lower()
        ratio = difflib.SequenceMatcher(None, speaker_lower, p_lower).ratio()
        for token in p_lower.split():
            token_ratio = difflib.SequenceMatcher(None, speaker_lower, token).ratio()
            if token_ratio > ratio:
                ratio = token_ratio
        if ratio >= 0.70 and ratio > best_ratio:
            best_ratio = ratio
            matched_participant = p
    if not matched_participant:
        for p in participants:
            if speaker_lower in p.lower() or p.lower() in speaker_lower:
                matched_participant = p
                break

    if not matched_participant:
        available = ", ".join(sorted(participants))
        yield f"I don't see anyone named {speaker_name} in the transcript. The participants I've heard are: {available}."
        return

    speaker_entries = [e for e in transcript_log if e["participant"] == matched_participant]
    if not speaker_entries:
        yield f"{matched_participant} hasn't said anything yet."
        return

    speaker_text = "\n".join(e["text"] for e in speaker_entries)
    if len(speaker_text) > 4000:
        speaker_text = speaker_text[-4000:]

    system_prompt = (
        f"You are Jarvis, an AI assistant in a live meeting. "
        f"Summarize what {matched_participant} has said and contributed to the discussion. "
        "Cover their key points, opinions, and any decisions or commitments they made. "
        "Speak naturally — no markdown, no bullet points. Keep it concise (3-5 sentences max)."
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"{matched_participant}'s contributions:\n{speaker_text}"},
    ]
    async for sentence in _stream_sentences(messages, JARVIS_SPEAKER_QUERY_MAX_TOKENS, 0.4):
        yield sentence
