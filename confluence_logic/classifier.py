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
        "Jarvis is connected to the team's Confluence wiki and can answer questions about it. "
        "Classify the user's message into exactly one category:\n\n"
        "- 'confluence' for TWO types of queries:\n"
        "  (A) WRITE: create, edit, update, delete, rename, list, or otherwise act on Confluence pages.\n"
        "  (B) READ/LOOKUP: asking about information that would be tracked in the team's Confluence wiki — "
        "project timelines, certification plans, feature roadmaps, audit schedules, process documentation, "
        "team ownership, sprint plans, launch dates, or any workspace-specific fact. "
        "Examples of READ confluence queries: 'when is our SOC2 audit?', 'what does the roadmap say about X?', "
        "'who owns the security page?', 'when is the launch?', 'what is the plan for Y?', "
        "'when is SOC2 coming?', 'what are the compliance requirements?', 'what is the status of Z?'. "
        "If the question is about something that could plausibly be documented in a work wiki, use 'confluence'.\n\n"
        "- 'general' ONLY for pure world-knowledge questions with no workspace relevance: "
        "weather, stock prices, currency rates, generic definitions ('what is OAuth?'), or casual conversation. "
        "Do NOT use 'general' for questions about certifications, audits, launches, plans, or timelines — those belong in 'confluence'.\n\n"
        "- 'meeting_summary' ONLY if the user wants a spoken recap of the current meeting transcript "
        "(e.g., 'catch me up', 'what did I miss', 'summarize the meeting') with no Confluence action.\n\n"
        "- 'meeting_opinion' if the user wants Jarvis's opinion on what was discussed, or asks a context-dependent question "
        "that only makes sense with the meeting transcript ('what do you think', 'which option is better', "
        "'what is the fix?', 'should we do that?', 'how do we solve it?').\n\n"
        "- 'action_items' if the user asks for action items, tasks, or next steps from the meeting.\n\n"
        "- 'speaker_query' if the user asks what a specific person said in the meeting.\n\n"
        "Respond with ONLY one word: 'confluence', 'general', 'meeting_summary', 'meeting_opinion', 'action_items', or 'speaker_query'."
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
