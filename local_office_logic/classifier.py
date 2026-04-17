"""
Intent classifier for the local office Jarvis assistant.

Routes office-file actions to the local artifact pipeline and keeps general
conversation / meeting-summary behavior aligned with the meeting assistant.
"""

import asyncio
import logging
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


_OFFICE_VERBS = frozenset(
    {"create", "edit", "update", "delete", "remove", "add", "rename", "list", "show", "make", "write", "replace"}
)
_OFFICE_NOUNS = (
    "file",
    "document",
    "doc",
    "word",
    "writer",
    "sheet",
    "spreadsheet",
    "excel",
    "calc",
    "workbook",
    "worksheet",
    "local",
)

_SUMMARY_TRIGGERS = frozenset({"summarize", "summary", "recap", "missed", "miss"})
_SUMMARY_PHRASES = (
    "catch me up",
    "what was said",
    "what did i miss",
    "what have i missed",
    "summarize the meeting",
    "give me a summary",
    "what happened so far",
)

_OPINION_TRIGGERS = frozenset({"think", "opinion", "recommend", "recommendation", "suggestion", "suggest", "prefer", "choose"})
_OPINION_PHRASES = (
    "what do you think",
    "what's your take",
    "what is your take",
    "how should we proceed",
    "which option",
    "what would you do",
    "what do you recommend",
    "your thoughts",
    "your opinion",
)


def _fast_classify(text: str) -> Optional[str]:
    normalized = text.strip().lower()
    words = normalized.split()
    if not words:
        return None

    has_office_target = any(noun in normalized for noun in _OFFICE_NOUNS)
    has_summary_trigger = any(phrase in normalized for phrase in _SUMMARY_PHRASES) or any(word in _SUMMARY_TRIGGERS for word in words)

    if has_summary_trigger and has_office_target:
        return "office"
    if not has_office_target and has_summary_trigger:
        return "meeting_summary"
    if any(phrase in normalized for phrase in _OPINION_PHRASES):
        return "meeting_opinion"
    if words[0] in ("what", "how", "which") and any(word in _OPINION_TRIGGERS for word in words):
        return "meeting_opinion"
    if words[0] in _OFFICE_VERBS:
        return "office"
    if has_office_target and any(word in _OFFICE_VERBS for word in words):
        return "office"
    if words[0] in ("what", "why", "how", "when", "where", "who", "is", "are", "can", "could", "would", "do", "does", "did", "tell", "explain"):
        if not has_office_target:
            return "general"
    return None


async def classify_intent(text: str) -> str:
    fast = _fast_classify(text)
    if fast is not None:
        logger.info("Local office classifier fast-path: %s -> %s", text[:60], fast)
        return fast

    system_prompt = (
        "You are a classifier for Jarvis, a voice assistant in a meeting. "
        "Jarvis can edit, create, delete, rename, and list sandboxed local office files, "
        "answer general questions, summarize the meeting transcript, and give opinions on what was discussed. "
        "Classify the user's message into exactly one category:\n"
        "- 'office' for actions on local documents, spreadsheets, workbooks, sheets, files, or office content\n"
        "- 'general' for general conversation or unrelated questions\n"
        "- 'meeting_summary' only when the user wants a spoken summary of the current meeting with no office-file action involved\n"
        "- 'meeting_opinion' when the user wants Jarvis's opinion or recommendation on what was discussed\n\n"
        "Respond with only one of these words: office, general, meeting_summary, meeting_opinion."
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
        if result in {"office", "general", "meeting_summary", "meeting_opinion"}:
            logger.info("Local office classifier LLM: %s -> %s", text[:60], result)
            return result
        logger.warning("Unexpected local office classifier result: %s; defaulting to office", result)
        return "office"
    except Exception as exc:
        logger.error("Local office classifier LLM call failed: %s; defaulting to office", exc)
        return "office"
