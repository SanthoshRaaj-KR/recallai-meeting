"""
General question responder for non-Confluence queries.
Uses the LLM to generate a conversational answer, then speaks it via TTS.
"""
import asyncio
import logging
import os
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
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "").strip()

_WEB_SEARCH_ROUTER_PROMPT = (
    "You are a routing classifier. Answer only 'yes' or 'no'.\n"
    "Does this question require real-time or current-day data to answer accurately?\n"
    "Answer 'yes' for: weather, sports scores, news headlines, stock prices, "
    "current events, today's date/time, live data, anything that changes daily.\n"
    "Answer 'no' for: factual/historical questions, explanations, opinions, "
    "meeting transcript questions.\n"
    "Question: {question}"
)


async def _needs_web_search(question: str) -> bool:
    """Return True if the question likely needs live/current data (per D-07, D-08)."""
    try:
        response = await asyncio.to_thread(
            lambda: _get_client().chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": _WEB_SEARCH_ROUTER_PROMPT.format(question=question)}],
                max_tokens=5,
                temperature=0.0,
            )
        )
        return (response.choices[0].message.content or "").strip().lower().startswith("yes")
    except Exception as e:
        logger.debug("Web search router failed (non-fatal): %s", e)
        return False


def _quick_web_search(query: str) -> str:
    """
    Fetch search results from Tavily API.
    Returns a context string with top result snippets, or empty string on failure.
    Requires TAVILY_API_KEY env var.
    """
    if not TAVILY_API_KEY:
        logger.debug("tavily: API key not set, skipping web search")
        return ""
    try:
        resp = _requests.post(
            "https://api.tavily.com/search",
            json={
                "api_key": TAVILY_API_KEY,
                "query": query,
                "search_depth": "basic",
                "max_results": 3,
                "include_answer": True,
            },
            timeout=5,
        )
        if resp.status_code != 200:
            logger.debug("tavily: search failed with status %d", resp.status_code)
            return ""
        data = resp.json()
        # Prefer the AI-generated answer if available
        answer = data.get("answer", "")
        if answer and len(answer) > 20:
            return answer[:800]
        # Fallback to concatenated result snippets
        results = data.get("results", [])
        snippets = [r.get("content", "") for r in results[:3] if r.get("content")]
        combined = " ".join(snippets)
        return combined[:800] if combined else ""
    except Exception as e:
        logger.debug("tavily: web search failed (non-fatal): %s", e)
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
    graph_context: str = "",
    speech_rewrite_enabled: bool = False,
    multiturn_reference: bool = False,
    force_web_search: bool = False,
) -> str:
    """
    Generate a conversational answer to a general (non-Confluence) question.

    Args:
        question: The user's question text.
        conversation_history: Recent conversation context (formatted as "User: ...\nAssistant: ...").
        speech_rewrite_enabled: When True, relax the conciseness constraint since _rewrite_for_speech
            will handle final condensing downstream (avoids double-condensing).
        multiturn_reference: When True, append multi-turn referencing instructions.

    Returns:
        A natural language answer string suitable for TTS playback.
    """
    conciseness_instruction = (
        "Keep it reasonably concise. "
        if speech_rewrite_enabled
        else "Keep your answer concise — 1 to 3 sentences maximum. "
    )
    system_prompt = (
        "You are Jarvis, a helpful and friendly AI assistant in a live meeting. "
        "Answer naturally and conversationally, as if speaking out loud in a meeting. "
        + conciseness_instruction
        + "Do not use markdown, bullet points, or formatting. "
        "Do not mention Confluence or page editing unless the user asks about it. "
        "Speak in a warm, professional tone."
    )

    if multiturn_reference:
        system_prompt += (
            "\n\nYou are in a multi-turn conversation. When your answer relates to something discussed earlier, "
            "naturally reference it (e.g., 'As I mentioned...', 'Building on what we discussed...', "
            "'Going back to your earlier question...'). Only do this when genuinely relevant — do not force it."
        )

    if graph_context:
        system_prompt += (
            "\n\nMeeting context (from the current conversation):\n"
            + graph_context
            + "\n\nIf the question relates to something in the meeting context above, "
            "explicitly tie your answer to what was discussed — for example, say "
            "'which is exactly what the team is working on' or 'as was mentioned earlier in this meeting'. "
            "Do not answer generically if the meeting context is directly relevant."
        )

    messages = [{"role": "system", "content": system_prompt}]

    # Inject prior turns as real chat history so the LLM can follow the thread.
    messages.extend(_history_to_messages(conversation_history))

    # Selective web search for questions that need current data
    web_context = ""
    needs_live_data = force_web_search or await _needs_web_search(question)
    if needs_live_data:
        logger.info("Web search triggered for: %s", question[:60])
        web_context = await asyncio.to_thread(_quick_web_search, question)
        if web_context:
            logger.info("Web search returned %d chars", len(web_context))

    user_content = question
    if web_context:
        user_content = (
            f"{question}\n\n"
            f"[Live web search result — use this if it clearly matches the question; "
            f"if the results seem to be about a different entity or topic, note that openly]:\n{web_context}"
        )
    elif needs_live_data:
        # Web search was warranted but returned nothing (e.g. no API key).
        # Inject a hedge to prevent the LLM from hallucinating stale/invented live data.
        user_content = (
            f"{question}\n\n"
            f"[Note: a live web search was attempted but no results are available. "
            f"Do NOT invent or guess current scores, prices, or live data. "
            f"If you cannot answer without live data, say so clearly and suggest where the user can check.]"
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
