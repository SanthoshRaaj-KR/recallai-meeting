"""Shared LLM call runtime: global concurrency throttle + transient-error retry.

The pipeline fans out many concurrent gpt-4o-mini calls (one evaluation per
candidate section across every intent, plus per-segment extraction, drafting and
verification). On a large transcript this bursts past the org's tokens-per-minute
(TPM) limit; the Agents SDK then raises a 429 ``RateLimitError`` (or a transient
5xx), which callers turn into a dropped section (score 0.0) — i.e. a missing
proposal and a noisy traceback.

For this pipeline latency is acceptable but errors are not. So every agent call
goes through :func:`guarded_run`:

  * a per-event-loop semaphore caps how many calls are in flight at once,
    smoothing the token rate so the TPM ceiling is rarely hit;
  * transient failures (429 rate limit, 5xx, timeouts, connection resets) are
    retried with exponential backoff + jitter, so a momentary limit never
    surfaces as an error.

Tunable via env: ``LDOC_LLM_CONCURRENCY`` (default 4), ``LDOC_LLM_MAX_RETRIES``
(default 8), ``LDOC_LLM_RETRY_BASE`` (s, default 1.0), ``LDOC_LLM_RETRY_MAX``
(s, default 30.0).
"""

from __future__ import annotations

import asyncio
import logging
import os
import random
from typing import Any

from agents import Runner

logger = logging.getLogger(__name__)

_MAX_CONCURRENCY = max(1, int(os.getenv("LDOC_LLM_CONCURRENCY", "4")))
_MAX_RETRIES = max(1, int(os.getenv("LDOC_LLM_MAX_RETRIES", "8")))
_BASE_DELAY = max(0.05, float(os.getenv("LDOC_LLM_RETRY_BASE", "1.0")))
_MAX_DELAY = max(_BASE_DELAY, float(os.getenv("LDOC_LLM_RETRY_MAX", "30.0")))

# asyncio primitives are bound to the loop they are first used in; the pipeline
# may run under different loops (uvicorn vs. asyncio.run in tests), so the
# semaphore is recreated whenever the running loop changes.
_sem: asyncio.Semaphore | None = None
_sem_loop: asyncio.AbstractEventLoop | None = None


def _semaphore() -> asyncio.Semaphore:
    global _sem, _sem_loop
    loop = asyncio.get_running_loop()
    if _sem is None or _sem_loop is not loop:
        _sem = asyncio.Semaphore(_MAX_CONCURRENCY)
        _sem_loop = loop
    return _sem


_RETRYABLE_NAMES = {
    "RateLimitError", "APITimeoutError", "APIConnectionError",
    "InternalServerError", "APIError", "APIStatusError",
    "ServiceUnavailableError", "Timeout", "ConnectionError",
}
_RETRYABLE_STATUS = {408, 409, 429, 500, 502, 503, 504}
_RETRYABLE_SUBSTR = (
    "rate limit", "rate_limit", "429", "timeout", "timed out",
    "500", "502", "503", "504", "overloaded", "temporarily unavailable",
    "service unavailable", "connection error", "connection reset",
    "upstream connect error", "disconnect/reset",
)


def _is_retryable(exc: BaseException) -> bool:
    """True for transient OpenAI/network errors worth retrying."""
    status = getattr(exc, "status_code", None)
    if status in _RETRYABLE_STATUS:
        return True
    if type(exc).__name__ in _RETRYABLE_NAMES:
        # Broad SDK error classes: if a non-transient status is attached, skip.
        if status is not None and status not in _RETRYABLE_STATUS:
            return False
        return True
    msg = str(exc).lower()
    return any(s in msg for s in _RETRYABLE_SUBSTR)


async def guarded_run(agent: Any, prompt: Any) -> Any:
    """Run an Agents-SDK agent with global throttling + transient-error retry.

    Returns the ``Runner.run`` result. Non-transient errors are re-raised
    immediately; transient ones are retried up to ``LDOC_LLM_MAX_RETRIES`` times
    before the last exception is re-raised (callers keep their own fallback).
    """
    delay = _BASE_DELAY
    last_exc: BaseException | None = None
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            async with _semaphore():
                return await Runner.run(agent, prompt)
        except Exception as exc:  # noqa: BLE001 — classified by _is_retryable
            last_exc = exc
            if attempt >= _MAX_RETRIES or not _is_retryable(exc):
                raise
            sleep_s = min(delay, _MAX_DELAY) + random.uniform(0.0, 0.5)
            logger.warning(
                "LLM transient failure (%s); throttled retry %d/%d in %.1fs",
                type(exc).__name__, attempt, _MAX_RETRIES, sleep_s,
            )
            await asyncio.sleep(sleep_s)
            delay = min(delay * 2, _MAX_DELAY)
    assert last_exc is not None
    raise last_exc
