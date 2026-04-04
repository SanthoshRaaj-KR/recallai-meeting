"""
DateResolutionAgent — natural language date expression parser.

Converts NL date expressions like "last Monday" or "two weeks ago" into exact
UTC day ranges (start_ts, end_ts as Unix epoch integers). Uses dateparser for
parsing with locale-aware, UTC-anchored settings.

Design notes:
- DateResolutionAgent is a plain Python class (NOT an openai-agents Agent() instance).
  Phase 5 (OrchestratorAgent) calls it directly before retrieval.
- dateparser.parse() is called synchronously — it performs no network I/O.
- All timestamps are Unix epoch integers (not datetime objects) to match
  Pinecone metadata schema established in Phase 2.
"""

import datetime

import dateparser
from pydantic import BaseModel

_SETTINGS = {
    "RETURN_AS_TIMEZONE_AWARE": True,
    "TIMEZONE": "UTC",
    "PREFER_DAY_OF_MONTH": "first",
    "PREFER_DATES_FROM": "past",
}


class DateResolutionResult(BaseModel):
    """Result of resolving a natural language date expression to a UTC day range.

    Fields:
        start_ts: Unix epoch integer for 00:00:00 UTC of the resolved day.
        end_ts:   Unix epoch integer for 23:59:59 UTC of the resolved day.
        expression: The original NL expression passed to resolve().
        is_relative: True if the expression uses relative terms (e.g. "last Monday"),
            False if it is an absolute date string (e.g. "2026-03-30").
    """

    start_ts: int
    end_ts: int
    expression: str
    is_relative: bool


class DateResolutionAgent:
    """Parses natural language date expressions to UTC day ranges.

    Uses dateparser.parse() with settings that ensure UTC timezone-aware
    output, preferring past dates for relative expressions.

    Usage:
        agent = DateResolutionAgent()
        result = agent.resolve("last Monday")
        print(result.start_ts, result.end_ts)
    """

    # Keywords that indicate a relative (not absolute) date expression.
    # Detection: any token in the lowercased, whitespace-split expression
    # matching one of these keywords → is_relative=True.
    _RELATIVE_KEYWORDS = frozenset([
        "last", "this", "next", "ago", "yesterday",
        "today", "tomorrow", "week", "month", "year",
        "monday", "tuesday", "wednesday", "thursday",
        "friday", "saturday", "sunday",
    ])

    def resolve(self, expression: str) -> DateResolutionResult:
        """Parse a natural language date expression to a UTC day range.

        Uses dateparser.parse() with settings:
            RETURN_AS_TIMEZONE_AWARE: True
            TIMEZONE: "UTC"
            PREFER_DAY_OF_MONTH: "first"
            PREFER_DATES_FROM: "past"

        For any expression that parses to a datetime, produces:
            start_ts = int of that day's 00:00:00 UTC
            end_ts   = int of that day's 23:59:59 UTC

        Args:
            expression: NL date string e.g. "last Monday", "two weeks ago", "2026-03-30"

        Returns:
            DateResolutionResult with start_ts, end_ts, expression, and is_relative.

        Raises:
            ValueError: If dateparser.parse() returns None (unparseable expression).
        """
        dt = dateparser.parse(expression, settings=_SETTINGS)
        if dt is None:
            raise ValueError(f"Cannot parse date expression: {expression!r}")

        # Snap to day boundaries in UTC
        start = dt.replace(hour=0, minute=0, second=0, microsecond=0)
        end = dt.replace(hour=23, minute=59, second=59, microsecond=0)
        start_ts = int(start.timestamp())
        end_ts = int(end.timestamp())

        # Relative detection: split on whitespace and check for relative keywords
        tokens = expression.lower().split()
        is_relative = any(kw in tokens for kw in self._RELATIVE_KEYWORDS)

        return DateResolutionResult(
            start_ts=start_ts,
            end_ts=end_ts,
            expression=expression,
            is_relative=is_relative,
        )
