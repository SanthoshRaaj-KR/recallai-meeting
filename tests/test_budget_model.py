"""Regression tests enforcing gpt-4o-mini across all LLM-calling agents.

These tests prevent accidental model upgrades to expensive gpt-4o.
They inspect constructor defaults and verify override propagation without
making any real API calls.
"""

import os
import pytest
from unittest.mock import MagicMock, patch


class TestBudgetModelDefaults:
    def test_summarizer_default_model(self):
        from agents.summarizer import SummarizerAgent
        agent = SummarizerAgent()
        assert agent._model == "gpt-4o-mini"

    def test_answer_agent_default_model(self):
        from agents.answer_agent import AnswerAgent
        agent = AnswerAgent()
        assert agent._model == "gpt-4o-mini"

    def test_orchestrator_default_model(self):
        from agents.orchestrator import OrchestratorAgent
        with patch("agents.orchestrator.AsyncOpenAI"):
            orch = OrchestratorAgent(
                retriever=MagicMock(),
                date_resolver=MagicMock(),
                answer_agent=MagicMock(),
            )
        assert orch._model == "gpt-4o-mini"

    def test_jarvis_openai_model_env_default(self):
        # Verify the default in code is gpt-4o-mini; if env var is set,
        # that value should also be gpt-4o-mini (or the test env is overriding).
        import jarvis
        default_in_code = "gpt-4o-mini"
        assert jarvis.OPENAI_MODEL in (default_in_code, os.getenv("OPENAI_MODEL", default_in_code))

    def test_summarizer_model_override(self):
        from agents.summarizer import SummarizerAgent
        agent = SummarizerAgent(model="gpt-3.5-turbo")
        assert agent._model == "gpt-3.5-turbo"

    def test_answer_agent_model_override(self):
        from agents.answer_agent import AnswerAgent
        agent = AnswerAgent(model="gpt-3.5-turbo")
        assert agent._model == "gpt-3.5-turbo"

    def test_orchestrator_model_override(self):
        from agents.orchestrator import OrchestratorAgent
        with patch("agents.orchestrator.AsyncOpenAI"):
            orch = OrchestratorAgent(
                retriever=MagicMock(),
                date_resolver=MagicMock(),
                answer_agent=MagicMock(),
                model="gpt-3.5-turbo",
            )
        assert orch._model == "gpt-3.5-turbo"
