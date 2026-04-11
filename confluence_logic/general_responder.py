"""
General question responder for non-Confluence queries.
Uses the LLM to generate a conversational answer, then speaks it via TTS.
"""
import asyncio
import logging
import os
import re
import requests as _requests
from typing import Optional
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


GENERAL_RESPONDER_MODEL = os.getenv("JARVIS_GENERAL_MODEL", "gpt-4o-mini").strip()

# Keywords that signal the answer might be stale from training data
_FRESHNESS_PATTERNS = re.compile(
    r"\b(latest|current|available|list all|which models?|what models?|"
    r"new model|released|recent|right now|today|price|cost|how much|"
    r"version|api key|tier|quota|limit)\b",
    re.IGNORECASE,
)


def _needs_web_search(question: str) -> bool:
    """Return True if the question is likely to need live/current data."""
    return bool(_FRESHNESS_PATTERNS.search(question))


def _quick_web_search(query: str) -> str:
    """
    Fetch a quick answer snippet from DuckDuckGo Instant Answer API.
    Returns a short context string, or empty string if nothing useful found.
    Free, no API key required, returns in <1s.
    """
    try:
        resp = _requests.get(
            "https://api.duckduckgo.com/",
            params={"q": query, "format": "json", "no_html": "1", "skip_disambig": "1"},
            timeout=3,
        )
        if resp.status_code != 200:
            return ""
        data = resp.json()
        # Prefer AbstractText (Wikipedia-style summary), then Answer (instant answer)
        snippet = data.get("AbstractText") or data.get("Answer") or ""
        if snippet and len(snippet) > 20:
            return snippet[:600]  # cap at 600 chars
    except Exception as e:
        logger.debug("Web search failed (non-fatal): %s", e)
    return ""


def _history_to_messages(conversation_history: str) -> list:
    """Convert 'User: ...\nAssistant: ...' history string into proper OpenAI message objects.

    Building real alternating user/assistant turns (rather than stuffing history into a
    single user message) lets the LLM treat prior exchanges as actual conversation context,
    so follow-up questions correctly reference earlier answers.
    """
    messages = []
    if not conversation_history or conversation_history.strip() == "[none]":
        return messages

    current_role: Optional[str] = None
    current_lines: list = []

    for line in conversation_history.splitlines():
        if line.startswith("User: "):
            if current_role and current_lines:
                messages.append({"role": current_role, "content": "\n".join(current_lines).strip()})
            current_role = "user"
            current_lines = [line[len("User: "):]]
        elif line.startswith("Assistant: "):
            if current_role and current_lines:
                messages.append({"role": current_role, "content": "\n".join(current_lines).strip()})
            current_role = "assistant"
            current_lines = [line[len("Assistant: "):]]
        else:
            if current_role:
                current_lines.append(line)

    if current_role and current_lines:
        messages.append({"role": current_role, "content": "\n".join(current_lines).strip()})

    return messages


async def answer_general_question(
    question: str,
    conversation_history: str = "",
) -> str:
    """
    Generate a conversational answer to a general (non-Confluence) question.

    Args:
        question: The user's question text.
        conversation_history: Recent conversation context (formatted as "User: ...\nAssistant: ...").

    Returns:
        A natural language answer string suitable for TTS playback.
    """
    system_prompt = (
        "You are Jarvis, a helpful and friendly AI assistant in a live meeting. "
        "Answer naturally and conversationally, as if speaking out loud in a meeting. "
        "Keep your answer concise — 1 to 3 sentences maximum. "
        "Do not use markdown, bullet points, or formatting. "
        "Do not mention Confluence or page editing unless the user asks about it. "
        "Speak in a warm, professional tone."
    )

    messages = [{"role": "system", "content": system_prompt}]

    # Inject prior turns as real chat history so the LLM can follow the thread.
    messages.extend(_history_to_messages(conversation_history))

    # Selective web search for questions that need current data
    web_context = ""
    if _needs_web_search(question):
        logger.info("Web search triggered for: %s", question[:60])
        web_context = await asyncio.to_thread(_quick_web_search, question)
        if web_context:
            logger.info("Web search returned %d chars", len(web_context))

    user_content = question
    if web_context:
        user_content = (
            f"{question}\n\n"
            f"[Live web search result — use this for accuracy]:\n{web_context}"
        )

    messages.append({"role": "user", "content": user_content})

    try:
        response = await asyncio.to_thread(
            lambda: _get_client().chat.completions.create(
                model=GENERAL_RESPONDER_MODEL,
                messages=messages,
                max_tokens=150,
                temperature=0.7,
            )
        )
        answer = (response.choices[0].message.content or "").strip()
        if not answer:
            return "I'm not sure how to answer that."
        logger.info("General responder: %s -> %s", question[:50], answer[:80])
        return answer
    except Exception as e:
        logger.error("General responder failed: %s", e)
        return "Sorry, I couldn't process that question right now."
