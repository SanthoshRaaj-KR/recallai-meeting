"""Standalone ML model pre-download for the Docker build.

Imports ONLY the model plugins (Silero VAD + the turn detector) and triggers
their downloads through the LiveKit plugin registry — deliberately WITHOUT
importing the agent app (agent.py) or any business source.

Why: the Dockerfile runs this BEFORE `COPY . .`, so this build layer depends only
on this file plus the installed dependencies. A change to any other application
source file therefore does NOT invalidate it, and the models are never
re-downloaded on an app-code rebuild — the same caching guarantee `uv sync` gets.

Downloads the exact same files as `agent.py download-files` (which iterates the
same plugin registry after importing the full app).
"""

from livekit.agents import Plugin
from livekit.plugins import silero, turn_detector  # noqa: F401 — import registers the plugins


def main() -> None:
    for plugin in Plugin.registered_plugins:
        plugin.download_files()


if __name__ == "__main__":
    main()
