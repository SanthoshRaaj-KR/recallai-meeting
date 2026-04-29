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
JARVIS_SUMMARY_MAX_TOKENS = int(os.getenv("JARVIS_SUMMARY_MAX_TOKENS", "700"))
JARVIS_SUMMARY_BRIEF_MAX_TOKENS = int(os.getenv("JARVIS_SUMMARY_BRIEF_MAX_TOKENS", "400"))
JARVIS_OPINION_MAX_TOKENS = int(os.getenv("JARVIS_OPINION_MAX_TOKENS", "700"))
JARVIS_ACTION_ITEMS_MAX_TOKENS = int(os.getenv("JARVIS_ACTION_ITEMS_MAX_TOKENS", "400"))
JARVIS_SPEAKER_QUERY_MAX_TOKENS = int(os.getenv("JARVIS_SPEAKER_QUERY_MAX_TOKENS", "400"))

_EMPTY_TRANSCRIPT_FALLBACK = "I haven't heard anything in the meeting yet."


def _format_transcript(transcript_log: List[Dict[str, Any]]) -> str:
    """Format transcript_log entries as '[Participant]: [text]' lines in chronological order."""
    return "\n".join(
        f"{entry['participant']}: {entry['text']}"
        for entry in transcript_log
    )


_AMBIGUOUS_OPINION_RE = re.compile(
    r"\b(?:this|that|it|the plan|the approach|the idea|what do you think|your take|your opinion)\b",
    re.IGNORECASE,
)


def _is_ambiguous_opinion_query(query: str) -> bool:
    normalized = (query or "").strip().lower()
    if not normalized:
        return True
    explicit_topic_markers = ("about ", "regarding ", "on ", "for ", "with ")
    has_short_deictic = any(phrase in normalized for phrase in ("this", "that", "it", "the plan", "the idea"))
    if has_short_deictic:
        return True
    return bool(_AMBIGUOUS_OPINION_RE.search(normalized)) and not any(marker in normalized for marker in explicit_topic_markers)


def _build_opinion_context(transcript_log: List[Dict[str, Any]], query: str = "") -> Dict[str, str]:
    recent_entries = transcript_log[-10:]
    prior_entries = transcript_log[:-10]

    current_discussion = _format_transcript(recent_entries)
    earlier_context = _format_transcript(prior_entries)

    if len(current_discussion) > 5000:
        current_discussion = current_discussion[-5000:]
    if len(earlier_context) > 3000:
        earlier_context = earlier_context[:1200] + "\n[... earlier meeting context omitted ...]\n" + earlier_context[-1800:]

    return {
        "current_discussion": current_discussion,
        "earlier_context": earlier_context,
        "is_ambiguous": "yes" if _is_ambiguous_opinion_query(query) else "no",
    }


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
    # Preserve both the meeting opening and recent content when truncating.
    # Keeping only the tail loses early topics (intros, first agenda items).
    max_chars = 8000
    if len(transcript_text) > max_chars:
        head = transcript_text[:2000]
        tail = transcript_text[-(max_chars - 2000):]
        transcript_text = head + "\n[... earlier content omitted for brevity ...]\n" + tail

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

    opinion_context = _build_opinion_context(transcript_log, query)

    system_prompt = (
        "You are Jarvis, an AI assistant attending a live meeting. "
        "The user wants your opinion, recommendation, critique, or strategy. "
        "Use the meeting transcript as context, but do not be limited to it. "
        "The 'current discussion' section is the highest-priority context. "
        "For ambiguous questions like 'what do you think about this?', answer only about the current discussion unless the user explicitly names an earlier topic. "
        "Do not connect unrelated earlier topics to the current topic just because they appeared earlier in the meeting. "
        "Combine what was discussed with your broader knowledge and independent reasoning. "
        "You may respectfully disagree with the plan discussed, point out missing risks, and propose your own better plan. "
        "Clearly separate meeting facts from your assessment when useful. "
        "Deliver a confident, direct, first-person opinion. Pick a side when the question calls for it, while acknowledging real tradeoffs. "
        "Reference specific points, numbers, costs, or percentages from the transcript when they matter, and add external reasoning when it improves the answer. "
        "Speak naturally as if talking aloud in the meeting. "
        "Stay respectful and practical. No markdown, no bullet points, no formatting. 3 to 6 sentences maximum."
    )

    user_content = (
        f"Current discussion (highest priority):\n\n{opinion_context['current_discussion']}\n\n"
        f"Earlier meeting background (use only if the user explicitly asks about it or it directly clarifies the current discussion):\n\n"
        f"{opinion_context['earlier_context'] or '[none]'}\n\n"
        f"Ambiguous current-topic question: {opinion_context['is_ambiguous']}"
    )
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
        head = transcript_text[:2000]
        tail = transcript_text[-6000:]
        transcript_text = head + "\n[... earlier content omitted for brevity ...]\n" + tail

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

    opinion_context = _build_opinion_context(transcript_log, query)

    query_focus = (
        f" The user's specific question is: '{query}'. Address this question directly — "
        "do not drift to other meeting topics unless they directly support your answer."
        if query else ""
    )
    system_prompt = (
        "You are Jarvis, an AI assistant attending a live meeting. "
        "The user wants your opinion, recommendation, critique, or strategy. "
        "Use the meeting transcript as context, but do not be limited to it. "
        "The 'current discussion' section is the highest-priority context. "
        "For ambiguous questions like 'what do you think about this?', answer only about the current discussion unless the user explicitly names an earlier topic. "
        "Do not connect unrelated earlier topics to the current topic just because they appeared earlier in the meeting. "
        "Combine what was discussed with your broader knowledge and independent reasoning. "
        "You may respectfully disagree with a decision or consensus from the meeting, explain why, and propose your own plan. "
        "Do not pretend a decision was not made; acknowledge it first, then validate, critique, or improve it. "
        "Deliver a confident, direct, first-person opinion. Pick a side when the question calls for it, while acknowledging real tradeoffs. "
        "Reference specific points, numbers, costs, or percentages from the transcript when they matter, and add external reasoning when it improves the answer. "
        "Speak naturally as if talking aloud in the meeting. "
        "Stay respectful and practical. No markdown, no bullet points, no formatting. 3 to 6 sentences maximum."
        + query_focus
    )
    # Put the specific question first in the user content so the LLM treats it as the primary task
    if query:
        user_content = (
            f"Question: {query}\n\n"
            f"Current discussion (highest priority):\n\n{opinion_context['current_discussion']}\n\n"
            f"Earlier meeting background (use only if the user explicitly asks about it or it directly clarifies the current discussion):\n\n"
            f"{opinion_context['earlier_context'] or '[none]'}\n\n"
            f"Ambiguous current-topic question: {opinion_context['is_ambiguous']}\n\n"
            "Answer the question above with a direct opinion. If the question is ambiguous, treat it as referring to the current discussion only. "
            "Use your own reasoning and broader knowledge where helpful."
        )
    else:
        user_content = (
            f"Current discussion (highest priority):\n\n{opinion_context['current_discussion']}\n\n"
            f"Earlier meeting background:\n\n{opinion_context['earlier_context'] or '[none]'}\n\n"
            "What is your opinion or recommendation about the current discussion?"
        )

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


async def summarize_speaker_streaming(transcript_log: List[Dict[str, Any]], speaker_name: str, topic: str = ""):
    """Streaming version of summarize_speaker — yields sentences as the LLM generates them.

    Performs the same fuzzy-match speaker lookup as summarize_speaker before streaming.
    Yields a single error sentence if the speaker cannot be matched.

    Args:
        topic: Optional topic qualifier (e.g. "the database migration"). When provided,
               the LLM is instructed to focus only on what the speaker said about that topic.
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

    topic_focus = (
        f" Focus specifically on what {matched_participant} said about {topic}. "
        "If they did not say anything relevant to that topic, say so clearly."
        if topic else ""
    )
    system_prompt = (
        f"You are Jarvis, an AI assistant in a live meeting. "
        f"Summarize what {matched_participant} has said and contributed to the discussion."
        + topic_focus
        + " Cover their key points, opinions, and any decisions or commitments they made. "
        "Speak naturally — no markdown, no bullet points. Keep it concise (3-5 sentences max)."
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"{matched_participant}'s contributions:\n{speaker_text}"},
    ]
    async for sentence in _stream_sentences(messages, JARVIS_SPEAKER_QUERY_MAX_TOKENS, 0.4):
        yield sentence
