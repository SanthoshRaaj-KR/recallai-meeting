"""
General question responder for non-Confluence queries.
Uses the LLM to generate a conversational answer, then speaks it via TTS.
"""
import asyncio
import logging
import os
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
        "The user has asked you a general question (not related to editing Confluence pages). "
        "Answer naturally and conversationally, as if speaking out loud in a meeting. "
        "Keep your answer concise — 1 to 3 sentences maximum. "
        "Do not use markdown, bullet points, or formatting. "
        "Do not mention Confluence or page editing unless the user asks about it. "
        "Speak in a warm, professional tone."
    )

    messages = [{"role": "system", "content": system_prompt}]

    if conversation_history and conversation_history.strip() != "[none]":
        messages.append({
            "role": "user",
            "content": f"Recent conversation context:\n{conversation_history}\n\nNow answer this question:",
        })

    messages.append({"role": "user", "content": question})

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
