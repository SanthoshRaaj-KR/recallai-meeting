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


def _fast_classify(text: str) -> Optional[str]:
    """Return 'confluence' or 'general' if heuristic is confident, else None for LLM fallback."""
    normalized = text.strip().lower()
    words = normalized.split()
    if not words:
        return None
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
    Classify user text as 'confluence' (edit/create/delete intent) or 'general' (question/conversation).
    Returns: 'confluence' or 'general'
    """
    # Try fast heuristic first
    fast = _fast_classify(text)
    if fast is not None:
        logger.info("Classifier fast-path: %s -> %s", text[:60], fast)
        return fast

    # LLM fallback for ambiguous cases
    system_prompt = (
        "You are a classifier. The user is speaking to a voice assistant called Jarvis in a meeting. "
        "Jarvis can edit, create, delete, and list Confluence wiki pages. "
        "Classify the user's message into exactly one category:\n"
        "- 'confluence' if the user wants to create, edit, update, delete, rename, list, or otherwise modify Confluence pages or content\n"
        "- 'general' if the user is asking a general question, making conversation, or asking something unrelated to Confluence page operations\n\n"
        "Respond with ONLY the word 'confluence' or 'general'. Nothing else."
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
        if result in ("confluence", "general"):
            logger.info("Classifier LLM: %s -> %s", text[:60], result)
            return result
        logger.warning("Classifier LLM returned unexpected: %s, defaulting to confluence", result)
        return "confluence"
    except Exception as e:
        logger.error("Classifier LLM call failed: %s, defaulting to confluence", e)
        return "confluence"
