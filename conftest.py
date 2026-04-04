"""
pytest conftest — project-level test configuration.

Problem: The openai-agents SDK registers its package as "agents". This project
also uses an "agents/" directory for project-specific agent implementations
(e.g. agents/summarizer.py). Without intervention, only one of these can win
the "agents" namespace.

Solution: Import the SDK's "agents" package first (it takes priority since it
has __init__.py), then extend its __path__ to include our local agents/ directory.
This allows "from agents.summarizer import SummarizerAgent" to work alongside
"from agents import Agent, Runner" (both resolve correctly).
"""

import os

# Extend the SDK's agents package path to include our local agents/ directory.
# This must happen before any test module imports agents.summarizer.
import agents as _sdk_agents

_LOCAL_AGENTS_DIR = os.path.join(os.path.dirname(__file__), "agents")
if _LOCAL_AGENTS_DIR not in _sdk_agents.__path__:
    _sdk_agents.__path__.append(_LOCAL_AGENTS_DIR)
