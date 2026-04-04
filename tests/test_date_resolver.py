"""
TDD RED phase — tests for DateResolutionAgent.

All tests in this file will FAIL before agents/date_resolver.py is created.
Run: pytest tests/test_date_resolver.py
"""

import datetime
import pytest
from unittest.mock import patch

from agents.date_resolver import DateResolutionAgent, DateResolutionResult

# Fixed datetime used across tests — noon UTC on 2026-03-30
FIXED_DT = datetime.datetime(2026, 3, 30, 12, 0, 0, tzinfo=datetime.timezone.utc)

# Expected start/end of day for FIXED_DT
EXPECTED_START_TS = int(
    datetime.datetime(2026, 3, 30, 0, 0, 0, tzinfo=datetime.timezone.utc).timestamp()
)
EXPECTED_END_TS = int(
    datetime.datetime(2026, 3, 30, 23, 59, 59, tzinfo=datetime.timezone.utc).timestamp()
)


class TestDateResolutionResult:
    def test_result_has_start_ts_end_ts_expression_is_relative(self):
        result = DateResolutionResult(
            start_ts=100,
            end_ts=200,
            expression="last Monday",
            is_relative=True,
        )
        assert result.start_ts == 100
        assert result.end_ts == 200
        assert result.expression == "last Monday"
        assert result.is_relative is True

    def test_start_ts_and_end_ts_are_ints(self):
        result = DateResolutionResult(
            start_ts=100,
            end_ts=200,
            expression="last Monday",
            is_relative=True,
        )
        assert isinstance(result.start_ts, int)
        assert isinstance(result.end_ts, int)


class TestDateResolutionAgentInit:
    def test_init_creates_instance(self):
        agent = DateResolutionAgent()
        assert agent is not None


class TestDateResolutionAgentResolve:
    def test_resolve_calls_dateparser_with_settings(self):
        with patch("agents.date_resolver.dateparser.parse", return_value=FIXED_DT) as mock_parse:
            agent = DateResolutionAgent()
            agent.resolve("last Monday")
            mock_parse.assert_called_once_with(
                "last Monday",
                settings={
                    "RETURN_AS_TIMEZONE_AWARE": True,
                    "TIMEZONE": "UTC",
                    "PREFER_DAY_OF_MONTH": "first",
                    "PREFER_DATES_FROM": "past",
                },
            )

    def test_resolve_returns_date_resolution_result(self):
        with patch("agents.date_resolver.dateparser.parse", return_value=FIXED_DT):
            agent = DateResolutionAgent()
            result = agent.resolve("last Monday")
            assert isinstance(result, DateResolutionResult)

    def test_resolve_start_ts_is_start_of_day(self):
        with patch("agents.date_resolver.dateparser.parse", return_value=FIXED_DT):
            agent = DateResolutionAgent()
            result = agent.resolve("last Monday")
            assert result.start_ts == EXPECTED_START_TS

    def test_resolve_end_ts_is_end_of_day(self):
        with patch("agents.date_resolver.dateparser.parse", return_value=FIXED_DT):
            agent = DateResolutionAgent()
            result = agent.resolve("last Monday")
            assert result.end_ts == EXPECTED_END_TS

    def test_resolve_expression_stored_in_result(self):
        with patch("agents.date_resolver.dateparser.parse", return_value=FIXED_DT):
            agent = DateResolutionAgent()
            result = agent.resolve("last Monday")
            assert result.expression == "last Monday"

    def test_resolve_is_relative_true_for_relative_expressions(self):
        with patch("agents.date_resolver.dateparser.parse", return_value=FIXED_DT):
            agent = DateResolutionAgent()
            result = agent.resolve("last Monday")
            assert result.is_relative is True

    def test_resolve_is_relative_false_for_absolute_expressions(self):
        with patch("agents.date_resolver.dateparser.parse", return_value=FIXED_DT):
            agent = DateResolutionAgent()
            result = agent.resolve("2026-03-30")
            assert result.is_relative is False

    def test_resolve_raises_value_error_when_dateparser_returns_none(self):
        with patch("agents.date_resolver.dateparser.parse", return_value=None):
            agent = DateResolutionAgent()
            with pytest.raises(ValueError, match="Cannot parse date expression"):
                agent.resolve("not a date XYZZY")
