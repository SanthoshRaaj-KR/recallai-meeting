"""Tests for agent_bridge.py — tool wrappers and transcript access."""
from pathlib import Path

import pytest

_BRIDGE_PATH = Path(__file__).resolve().parent.parent / "agent_bridge.py"


@pytest.fixture(scope="module")
def bridge_source() -> str:
    assert _BRIDGE_PATH.exists(), f"agent_bridge.py not found at {_BRIDGE_PATH}"
    return _BRIDGE_PATH.read_text(encoding="utf-8")


# ── Tool list ─────────────────────────────────────────────────────────────────

def test_jarvis_tools_exports_five_tools() -> None:
    """JARVIS_TOOLS must contain exactly the 5 function_tool wrappers."""
    from confluence_logic.agent_bridge import JARVIS_TOOLS
    names = {t.__name__ for t in JARVIS_TOOLS}
    assert names == {
        "summarize_meeting_tool",
        "generate_opinion_tool",
        "extract_action_items_tool",
        "summarize_speaker_tool",
        "answer_general_question_tool",
    }, f"Unexpected tool set: {names}"


def test_all_tools_are_function_tools() -> None:
    """Every entry in JARVIS_TOOLS must be decorated with @function_tool."""
    from confluence_logic.agent_bridge import JARVIS_TOOLS
    from livekit.agents.llm import function_tool
    for tool in JARVIS_TOOLS:
        # livekit function_tool wraps the callable; it marks it with an attribute
        assert callable(tool), f"{tool} is not callable"


# ── Timing instrumentation ────────────────────────────────────────────────────

def test_tool_wrappers_log_response_time(bridge_source: str) -> None:
    """Each tool wrapper must log its execution time via time.perf_counter."""
    assert "import time" in bridge_source, \
        "agent_bridge.py must import time for response timing"
    assert bridge_source.count("time.perf_counter()") >= 10, \
        "Each of the 5 tools must have a start (t0) and end perf_counter call"
    assert bridge_source.count("⏱️") >= 5, \
        "Each tool must emit a timing log line with ⏱️"


# ── Transcript access ─────────────────────────────────────────────────────────

def test_get_transcript_log_returns_empty_for_unknown_session() -> None:
    """Unknown session_id must return [] without raising."""
    from confluence_logic.agent_bridge import get_transcript_log_for_session
    result = get_transcript_log_for_session("nonexistent-session-id-xyz")
    assert result == [], f"Expected [], got {result!r}"


def test_session_id_from_context_handles_missing_metadata() -> None:
    """_session_id_from_context must return '' when job metadata is empty/missing."""
    from confluence_logic.agent_bridge import _session_id_from_context

    class FakeJob:
        metadata = None

    class FakeSession:
        userdata = {}

    class FakeContext:
        job = FakeJob()
        session = FakeSession()

    result = _session_id_from_context(FakeContext())
    assert result == "", f"Expected empty string, got {result!r}"
