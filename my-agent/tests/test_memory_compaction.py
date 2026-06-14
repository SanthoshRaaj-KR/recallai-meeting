import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from memory_compaction import TranscriptCompactor  # noqa: E402
from review_pipeline.models import ExtractedMeeting  # noqa: E402
from review_pipeline.pipeline import ProposalPipeline  # noqa: E402


def test_compactor_uses_llm_summary_for_lines_that_leave_recent_window():
    calls = []

    def fake_compact(previous_memory, transcript_block, _max_chars):
        calls.append((previous_memory, list(transcript_block)))
        return "\n".join(
            [
                previous_memory,
                "- Launch date moves to Friday.",
                "- Update the Confluence launch plan.",
            ]
        ).strip()

    compactor = TranscriptCompactor(
        window_size=2,
        chunk_size=2,
        max_memory_chars=2000,
        compact_fn=fake_compact,
    )

    compactor.observe_utterance("Asha: We decided the launch date moves to Friday.")
    compactor.observe_utterance("Ben: The weather came up briefly.")
    compactor.observe_utterance("Asha: Action item is to update the Confluence launch plan.")
    compactor.observe_utterance("Ben: The release owner will be Maya.")
    compactor.observe_utterance("Asha: Recent chatter stays verbatim.")
    compactor.observe_utterance("Ben: Latest note stays verbatim too.")

    memory = compactor.memory_text()
    memory_lower = memory.lower()

    assert "[Compacted meeting memory]" in memory
    assert "launch date moves to friday" in memory_lower
    assert "confluence launch plan" in memory_lower
    assert compactor.compacted_utterances == 4
    assert len(calls) == 2


def test_compactor_feeds_previous_compacted_memory_into_next_llm_block():
    calls = []

    def fake_compact(previous_memory, transcript_block, _max_chars):
        calls.append((previous_memory, list(transcript_block)))
        if previous_memory:
            return f"{previous_memory}\n- Owner changed to Maya."
        return "- Launch date moves to Friday."

    compactor = TranscriptCompactor(
        window_size=1,
        chunk_size=1,
        max_memory_chars=2000,
        compact_fn=fake_compact,
    )

    compactor.observe_utterance("Asha: Launch date moves to Friday.")
    compactor.observe_utterance("Asha: Release owner changed to Maya.")
    compactor.observe_utterance("Ben: Latest line stays recent.")

    memory = compactor.memory_text()

    assert len(calls) == 2
    assert calls[0][0] == ""
    assert "Launch date moves to Friday" in calls[1][0]
    assert calls[1][1] == ["Asha: Release owner changed to Maya."]
    assert "Owner changed to Maya" in memory


def test_pipeline_includes_optional_compacted_memory_context(monkeypatch):
    pipeline = ProposalPipeline()
    captured = {}

    async def fake_extract(_transcript, transcript_text, query=None):
        captured["transcript_text"] = transcript_text
        return ExtractedMeeting(
            title="Memory Test",
            summary="No changes needed.",
            change_intents=[],
        )

    monkeypatch.setattr(pipeline, "_extract_meeting", fake_extract)
    # Stubbed extraction returns no change_intents, so run() returns early
    # (before RAG/drafting/verification) — those stages need no stubbing here.

    meeting, proposals = asyncio.run(
        pipeline.run(
            session_id="s1",
            transcript=[{"participant": "Asha", "text": "Recent update only."}],
            memory_context="[Compacted meeting memory]\nOld decision: ship on Friday.",
        )
    )

    assert meeting.title == "Memory Test"
    assert proposals == []
    assert "Old decision: ship on Friday." in captured["transcript_text"]
    assert "[Meeting transcript (recent/raw)]" in captured["transcript_text"]
    assert "Asha: Recent update only." in captured["transcript_text"]
