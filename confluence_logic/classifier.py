"""
Intent classifier for Jarvis voice assistant.

Classifies user transcript into 'confluence' (edit/create/delete intent)
or 'general' (question/conversation).

Fast-path heuristic handles obvious cases; LLM fallback handles ambiguous ones.
"""
import asyncio
import logging
import os
import re
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


_CONFLUENCE_VERBS = frozenset({
    "create", "edit", "update", "delete", "remove", "add", "rename",
    "list", "show", "make", "write", "rewrite", "change", "replace",
})

_CONFLUENCE_NOUNS = (
    "page", "pages", "section", "heading", "title", "document",
    "confluence", "workspace", "content",
)

_SUMMARY_TRIGGERS = frozenset({
    "summarize", "summary", "recap", "recapping", "missed", "miss",
})
_SUMMARY_PHRASES = (
    "catch me up", "what was said", "what did i miss", "what have i missed",
    "summarize the meeting", "give me a summary", "what happened so far",
)

_OPINION_TRIGGERS = frozenset({
    "think", "opinion", "recommend", "recommendation", "suggestion", "suggest", "prefer", "choose",
})
_OPINION_PHRASES = (
    "what do you think", "what's your take", "what is your take",
    "how should we proceed", "which option", "what would you do",
    "what do you recommend", "your thoughts", "your opinion",
)


def _fast_classify(text: str) -> Optional[str]:
    """Return intent string if heuristic is confident, else None for LLM fallback."""
    normalized = text.strip().lower()
    words = normalized.split()
    if not words:
        return None

    # A Confluence-specific noun in the query means the action targets a page/document,
    # not just the spoken meeting transcript.
    has_confluence_target = any(noun in normalized for noun in _CONFLUENCE_NOUNS)
    has_summary_trigger = (
        any(phrase in normalized for phrase in _SUMMARY_PHRASES)
        or any(word in _SUMMARY_TRIGGERS for word in words)
    )

    # If a summary trigger AND a Confluence target both appear, the user wants Jarvis to
    # summarize or act on a Confluence document (or create a page with the summary).
    # Route to confluence so the editor pipeline handles it end-to-end.
    if has_summary_trigger and has_confluence_target:
        return "confluence"

    # Pure meeting-summary heuristics — only when there is no Confluence target.
    if not has_confluence_target and has_summary_trigger:
        return "meeting_summary"

    # Meeting opinion heuristics (per D-03)
    if any(phrase in normalized for phrase in _OPINION_PHRASES):
        return "meeting_opinion"
    if words[0] in ("what", "how", "which") and any(word in _OPINION_TRIGGERS for word in words):
        return "meeting_opinion"

    # If first word is a confluence verb AND mentions a confluence noun, it's confluence
    if words[0] in _CONFLUENCE_VERBS:
        if any(noun in normalized for noun in _CONFLUENCE_NOUNS):
            return "confluence"
        # First word is action verb but no confluence noun — still likely confluence
        # (e.g., "delete the roadmap section")
        return "confluence"
    # Obvious question patterns are general
    if words[0] in ("what", "why", "how", "when", "where", "who", "is", "are", "can", "could", "would", "do", "does", "did", "tell", "explain"):
        if not any(noun in normalized for noun in _CONFLUENCE_NOUNS):
            return "general"
    return None


async def classify_intent(text: str) -> str:
    """
    Classify user text into one of four intents: 'confluence', 'general', 'meeting_summary', 'meeting_opinion'.
    Returns: 'confluence', 'general', 'meeting_summary', or 'meeting_opinion'
    """
    # Try fast heuristic first
    fast = _fast_classify(text)
    if fast is not None:
        logger.info("Classifier fast-path: %s -> %s", text[:60], fast)
        return fast

    # LLM fallback for ambiguous cases
    system_prompt = (
        "You are a classifier. The user is speaking to a voice assistant called Jarvis in a meeting. "
        "Jarvis can edit, create, delete, and list Confluence wiki pages, answer general questions, "
        "summarize the meeting transcript, and give opinions on what was discussed. "
        "Classify the user's message into exactly one category:\n"
        "- 'confluence' if the user wants to create, edit, update, delete, rename, list, or otherwise act on Confluence pages or content. "
        "IMPORTANT: also use 'confluence' when the user wants to summarize or do anything with a specific Confluence document, page, or section, "
        "OR when they want to save/create a page with a meeting summary (e.g., 'summarize the meeting and put it in a new page'). "
        "Any action that involves a Confluence page or document is 'confluence', even if it includes summarization.\n"
        "- 'general' if the user is asking a general question, making conversation, or asking something unrelated to Confluence or the current meeting\n"
        "- 'meeting_summary' ONLY if the user wants to hear a spoken summary/recap of the current meeting transcript with NO Confluence action involved "
        "(e.g., 'catch me up', 'what did I miss', 'summarize the meeting' — with no mention of pages or documents)\n"
        "- 'meeting_opinion' if the user wants Jarvis's opinion, recommendation, or take on what was discussed (e.g., 'what do you think', 'which option is better', 'how should we proceed')\n\n"
        "Respond with ONLY one of these four words: 'confluence', 'general', 'meeting_summary', 'meeting_opinion'. Nothing else."
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
        if result in ("confluence", "general", "meeting_summary", "meeting_opinion"):
            logger.info("Classifier LLM: %s -> %s", text[:60], result)
            return result
        logger.warning("Classifier LLM returned unexpected: %s, defaulting to confluence", result)
        return "confluence"
    except Exception as e:
        logger.error("Classifier LLM call failed: %s, defaulting to confluence", e)
        return "confluence"
