"""
Intent classifier for Jarvis voice assistant.

Classifies user text into one of six intents using an LLM.
All queries go directly to the LLM — no regex fast-path — so that
STT noise (filler words, split words, mangled names, spelled-out acronyms)
does not confuse heuristic matching.
"""
import asyncio
import logging
import os
from typing import Optional

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

logger = logging.getLogger(__name__)

_openai_client: Optional[OpenAI] = None


def _get_client() -> OpenAI:
    global _openai_client
    if _openai_client is None:
        _openai_client = OpenAI()
    return _openai_client


async def classify_intent(text: str) -> str:
    """
    Classify user text into one of six intents:
    'confluence', 'general', 'meeting_summary', 'meeting_opinion', 'action_items', 'speaker_query'.
    Returns one of the six intent strings.
    All queries go through the LLM — no regex pre-filter.
    """
    system_prompt = (
        "You are a classifier. The user is speaking to a voice assistant called Jarvis in a meeting. "
        "Jarvis can edit, create, delete, and list Confluence wiki pages, answer general questions, "
        "summarize the meeting transcript, give opinions on what was discussed, extract action items, "
        "and summarize what specific participants said. "
        "Classify the user's message into exactly one category:\n"
        "- 'confluence' if the user wants to create, edit, update, delete, rename, list, or otherwise act on Confluence pages or content. "
        "IMPORTANT: also use 'confluence' when the user wants to summarize or do anything with a specific Confluence document, page, or section, "
        "OR when they want to save/create a page with a meeting summary (e.g., 'summarize the meeting and put it in a new page'). "
        "Any action that involves a Confluence page or document is 'confluence', even if it includes summarization.\n"
        "- 'general' if the user is asking a general question, making conversation, or asking something unrelated to Confluence or the current meeting\n"
        "- 'meeting_summary' ONLY if the user wants to hear a spoken summary/recap of the current meeting transcript with NO Confluence action involved "
        "(e.g., 'catch me up', 'what did I miss', 'summarize the meeting' — with no mention of pages or documents)\n"
        "- 'meeting_opinion' if the user wants Jarvis's opinion, recommendation, or take on what was discussed (e.g., 'what do you think', 'which option is better', 'how should we proceed')\n"
        "- 'action_items' if the user wants to know the action items, tasks, commitments, or next steps from the meeting\n"
        "- 'speaker_query' if the user wants to know what a specific person said, contributed, or mentioned in the meeting\n\n"
        "Respond with ONLY one of these six words: 'confluence', 'general', 'meeting_summary', 'meeting_opinion', 'action_items', 'speaker_query'. Nothing else."
    )

    try:
        response = await asyncio.to_thread(
            lambda: _get_client().chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": text},
                ],
                max_tokens=5,
                temperature=0.0,
            )
        )
        result = (response.choices[0].message.content or "").strip().lower()
        if result in ("confluence", "general", "meeting_summary", "meeting_opinion", "action_items", "speaker_query"):
            logger.info("Classifier LLM: %s -> %s", text[:60], result)
            return result
        logger.warning("Classifier LLM returned unexpected: %s, defaulting to confluence", result)
        return "confluence"
    except Exception as e:
        logger.error("Classifier LLM call failed: %s, defaulting to confluence", e)
        return "confluence"
