"""Async batch contextualizer for ChunkRecord objects.

Adds short 1-2 sentence context prefixes to each chunk via an OpenAI chat
completion call.  Designed for contextual retrieval (Anthropic 2024 pattern):
prepending context improves BM25 and dense retrieval quality by anchoring
each chunk to its document-level topic.

Security — T-12-04: Document content is passed only in the *user* message.
The *system* prompt is a fixed string with no f-string interpolation from
user/document data, preventing prompt injection via section content.
"""

from __future__ import annotations

import asyncio
import logging

from models.rag import ChunkRecord

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are a document context summarizer. "
    "Your task is to write a 1-2 sentence description that situates a document "
    "section within its broader document context."
)

_SEMAPHORE_SIZE = 10


async def add_context_prefixes(
    chunks: list[ChunkRecord],
    openai_client,
    model: str = "gpt-4o-mini",
) -> list[ChunkRecord]:
    """Batch-contextualize a list of ChunkRecord objects.

    For each chunk, calls the OpenAI chat completions API to generate a short
    1-2 sentence context description.  Requests are issued concurrently with a
    semaphore cap of 10.  Any API failure sets ``context_prefix = ""`` and
    logs a warning (graceful fallback).

    Parameters
    ----------
    chunks:
        Input list of ChunkRecord objects (unmodified copies are returned).
    openai_client:
        An initialised ``openai.AsyncOpenAI`` (or ``openai.OpenAI``) client.
    model:
        Chat model to use for context generation.

    Returns
    -------
    list[ChunkRecord]
        New list of ChunkRecord objects with ``context_prefix`` populated.
    """
    semaphore = asyncio.Semaphore(_SEMAPHORE_SIZE)
    tasks = [
        _contextualize_one(chunk, openai_client, model, semaphore) for chunk in chunks
    ]
    return await asyncio.gather(*tasks)


async def _contextualize_one(
    chunk: ChunkRecord,
    openai_client,
    model: str,
    semaphore: asyncio.Semaphore,
) -> ChunkRecord:
    """Contextualize a single ChunkRecord; return updated copy."""
    async with semaphore:
        try:
            user_message = (
                f"Document: {chunk.source_path}\n"
                f"Section: {chunk.section_heading}\n\n"
                f"Content:\n{chunk.content[:1000]}\n\n"
                "Write a 1-2 sentence context description that situates this section "
                "within the document. Start with 'This section...'"
            )
            # Support both sync and async clients
            if hasattr(openai_client, "chat") and hasattr(
                openai_client.chat, "completions"
            ):
                completions = openai_client.chat.completions
                if asyncio.iscoroutinefunction(completions.create):
                    response = await completions.create(
                        model=model,
                        messages=[
                            {"role": "system", "content": _SYSTEM_PROMPT},
                            {"role": "user", "content": user_message},
                        ],
                        max_tokens=80,
                        temperature=0.0,
                    )
                else:
                    # Sync client wrapped in thread
                    response = await asyncio.to_thread(
                        completions.create,
                        model=model,
                        messages=[
                            {"role": "system", "content": _SYSTEM_PROMPT},
                            {"role": "user", "content": user_message},
                        ],
                        max_tokens=80,
                        temperature=0.0,
                    )
                prefix = response.choices[0].message.content.strip()
            else:
                prefix = ""
        except Exception as exc:
            logger.warning(
                "Context prefix generation failed for chunk %s: %s",
                chunk.chunk_id,
                exc,
            )
            prefix = ""

    # Return updated copy (ChunkRecord has frozen=False so direct mutation is OK,
    # but returning a model_copy is cleaner for immutability conventions)
    updated = chunk.model_copy()
    updated.context_prefix = prefix
    return updated
