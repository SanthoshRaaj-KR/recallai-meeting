"""Shared pytest config for my-agent tests.

Splits a fast, deterministic *unit* gate from *live* tests that need external
services (Pinecone / Cerebras / OpenAI) and seeded data. Live tests are skipped
by default; run them with RUN_LIVE_TESTS=1.

    uv run pytest                      # unit gate (fast, no creds)
    RUN_LIVE_TESTS=1 uv run pytest      # everything, incl. live integration/eval
"""

import os

import pytest


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "live: requires live external services (Pinecone/LLM) and seeded data; "
        "skipped unless RUN_LIVE_TESTS=1",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if os.getenv("RUN_LIVE_TESTS") == "1":
        return
    skip_live = pytest.mark.skip(reason="live test — set RUN_LIVE_TESTS=1 to run")
    for item in items:
        if "live" in item.keywords:
            item.add_marker(skip_live)
