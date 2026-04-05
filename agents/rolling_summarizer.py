"""
MeetingMetadataAgent: processes the full meeting transcript at end-of-meeting
and returns a MeetingMetadataOutput with overview, goals, key_decisions, and
conclusions.

This agent is invoked ONCE per meeting on disconnect — not per-batch. During
the meeting, transcript lines are appended directly to the .md file without any
LLM processing. The metadata section (goals, decisions, conclusions) is generated
here and appended as a structured "Meeting Summary" block at the end of the .md
file, and stored in the JSON index for the History Manager's selection LLM.

Design notes:
- Follows the same Agent(output_type=...) pattern as SummarizerAgent.
- Uses gpt-4o-mini per Phase 6 mandate (budget model switch).
- Does NOT catch exceptions — callers (jarvis.py disconnect handler) handle failures.

Backwards compatibility:
- RollingSummarizerAgent is re-exported as an alias so any existing test imports
  do not break during the migration window.
"""

from agents import Agent, Runner  # openai-agents SDK (namespace extended via conftest.py)
from storage.models import MeetingMetadataOutput

_INSTRUCTIONS = """\
You are a meeting analyst. Given a full meeting transcript (format: "Speaker: utterance"),
extract concise, structured metadata for the meeting.

Return:
- overview: 1-2 sentence description of what the meeting was about and who attended.
- goals: Up to 5 short phrases describing what the meeting aimed to achieve.
- key_decisions: Up to 10 short phrases, each a clear decision that was reached.
  Only include decisions explicitly agreed upon — not proposals or suggestions.
- conclusions: Up to 5 short phrases summarising how the meeting concluded or
  what the overall outcome was.

Stay factual. Do not infer beyond what is said. Be concise — each item should be
a short phrase, not a full sentence.
"""


class MeetingMetadataAgent:
    """Generates structured metadata from the full meeting transcript at end-of-meeting.

    Called once on meeting disconnect. Takes the entire accumulated transcript and
    returns goals, key decisions, and conclusions for the .md summary section and
    the JSON meeting index.
    """

    def __init__(self, model: str = "gpt-4o-mini"):
        self._model = model
        self._agent = Agent(
            name="meeting-metadata-agent",
            instructions=_INSTRUCTIONS,
            model=model,
            output_type=MeetingMetadataOutput,
        )

    async def run(self, full_transcript: str) -> MeetingMetadataOutput:
        """Generate meeting metadata from the full transcript.

        Args:
            full_transcript: Complete meeting transcript as a single string,
                one "Speaker: utterance" line per line.

        Returns:
            MeetingMetadataOutput with overview, goals, key_decisions, conclusions.

        Raises:
            Any exception raised by the SDK Runner is propagated to the caller.
        """
        result = await Runner.run(self._agent, input=full_transcript)
        return result.final_output


# Backwards-compat alias — keeps existing imports working during migration
RollingSummarizerAgent = MeetingMetadataAgent
