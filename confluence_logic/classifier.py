"""Intent classifier for Jarvis voice assistant."""
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

_CONTEXT_REFERENTS = (
    "this",
    "that",
    "it",
    "they",
    "them",
    "these",
    "those",
    "the issue",
    "the problem",
    "the bug",
    "the blocker",
    "the risk",
    "the plan",
    "the approach",
    "the option",
    "the decision",
    "the proposal",
    "the recommendation",
)
_CONTEXT_DEPENDENT_PATTERNS = (
    r"\bwhat(?:'s| is)?\s+(?:the\s+)?(?:fix|solution|answer|best\s+move|next\s+step|plan)\b",
    r"\bhow\s+(?:do|should|can|could)\s+we\s+(?:fix|solve|handle|address|approach|proceed|unblock|mitigate|deal\s+with)\b",
    r"\bwhat\s+(?:do|should|can|could)\s+we\s+(?:do|try|change|fix|choose|pick)\b",
    r"\bshould\s+we\s+(?:do|use|choose|pick|ship|delay|fix|change|go\s+with)\b",
    r"\bwhich\s+(?:one|option|approach|path|plan)\s+(?:is|should|would)\b",
    r"\b(?:is|was)\s+(?:that|this|it)\s+(?:good|bad|right|wrong|better|safe|risky)\b",
    r"\bwhat\s+(?:about|if)\s+(?:this|that|it|them|those)\b",
)
_STANDALONE_GENERAL_PATTERNS = (
    r"\bweather\b",
    r"\bstock\s+price\b",
    r"\bshare\s+price\b",
    r"\bexchange\s+rate\b",
    r"\bcurrent\s+(?:price|time|date|score)\b",
    r"\bwhat\s+is\s+(?:a|an)?\s*[a-z0-9][\w\s-]{1,80}\??$",
    r"\bwhat\s+does\s+[a-z0-9][\w\s-]{1,80}\s+mean\??$",
    r"\bexplain\s+[a-z0-9][\w\s-]{1,80}\??$",
)


def _get_client() -> OpenAI:
    global _openai_client
    if _openai_client is None:
        _openai_client = OpenAI()
    return _openai_client


def _normalize_for_classifier(text: str) -> str:
    return " ".join((text or "").strip().lower().replace("’", "'").split())


def _fast_classify_context_dependency(text: str) -> Optional[str]:
    normalized = _normalize_for_classifier(text)
    if not normalized:
        return None

    if any(re.search(pattern, normalized) for pattern in _CONTEXT_DEPENDENT_PATTERNS):
        return "meeting_opinion"

    question_starter = normalized.startswith(
        ("what ", "what's ", "how ", "why ", "which ", "should ", "can we ", "could we ", "do we ", "is ")
    )
    if question_starter and any(re.search(rf"\b{re.escape(ref)}\b", normalized) for ref in _CONTEXT_REFERENTS):
        if not any(re.search(pattern, normalized) for pattern in _STANDALONE_GENERAL_PATTERNS):
            return "meeting_opinion"

    if any(re.search(pattern, normalized) for pattern in _STANDALONE_GENERAL_PATTERNS):
        return "general"

    return None


async def classify_intent(text: str) -> str:
    """
    Classify user text into one of six intents:
    'confluence', 'general', 'meeting_summary', 'meeting_opinion', 'action_items', 'speaker_query'.
    Returns one of the six intent strings.
    All queries go through the LLM — no regex pre-filter.
    """
    fast = _fast_classify_context_dependency(text)
    if fast is not None:
        logger.info("Classifier context fast-path: %s -> %s", text[:60], fast)
        return fast

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
        "- 'general' if the user is asking a standalone general question, making conversation, or asking something unrelated to Confluence or the current meeting. "
        "Only use 'general' when the question makes sense by itself without the meeting transcript, such as weather, stock prices, definitions, facts, or general explanations.\n"
        "- 'meeting_summary' ONLY if the user wants to hear a spoken summary/recap of the current meeting transcript with NO Confluence action involved "
        "(e.g., 'catch me up', 'what did I miss', 'summarize the meeting' — with no mention of pages or documents)\n"
        "- 'meeting_opinion' if the user wants Jarvis's opinion, recommendation, or take on what was discussed (e.g., 'what do you think', 'which option is better', 'how should we proceed'). "
        "Also use 'meeting_opinion' for context-dependent questions that do not make sense alone and need the meeting transcript, such as 'what is the fix?', 'how do we solve the problem?', "
        "'should we do that?', 'is that a good idea?', or 'how do we fix it?'. These are NOT general questions.\n"
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
