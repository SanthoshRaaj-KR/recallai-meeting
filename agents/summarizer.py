"""
SummarizerAgent: processes a raw meeting transcript and returns a MeetingRecord.

Uses the OpenAI Agents SDK with structured output (output_type=SummaryOutput) so
the LLM response is validated as a typed Pydantic object — not parsed from free text.

Design notes:
- SummaryOutput captures the LLM's structured response.
- Post-processing converts SummaryOutput → MeetingRecord with status tagging.
- If any ActionItem has a blank owner, a ValueError is raised immediately.
- status is forced to "partial" when len(transcript) < 500.
"""

import time
from typing import Optional

from pydantic import BaseModel

from agents import Agent, Runner  # openai-agents SDK (namespace extended via conftest.py)
from storage.models import ActionItem, MeetingRecord

_INSTRUCTIONS = """\
You are a meeting summarizer. Given a meeting transcript where each line is
"{speaker}: {utterance}", extract the following as structured JSON:

- summary_text: A 2-4 sentence prose summary of what was discussed and decided.
- topics_covered: A list of distinct topics discussed (short phrases, e.g. "API contract", "changelog update").
- decisions: A list of decisions the group reached (complete sentences).
- action_items: A list of tasks where each item has:
    - owner: The FULL NAME of the participant who committed to the task.
      Use the last speaker who explicitly accepted or volunteered for it.
      If ownership is genuinely ambiguous, use the last speaker who mentioned it.
      NEVER leave owner blank. NEVER use "TBD", "Unknown", or "Someone".
    - task: A clear description of what must be done.
    - due: Optional due date string if mentioned, otherwise omit.
- participants: The list of unique speaker names who appeared in the transcript.

Accuracy rules:
- Only include decisions that were explicitly agreed upon, not just proposed.
- Only include action items where someone explicitly accepted ownership.
- Extract participant names exactly as they appear in the transcript.
"""


class SummaryOutput(BaseModel):
    """Structured output schema for the LLM response.

    This is the intermediate schema — the agent returns a SummaryOutput, which
    is then converted to a MeetingRecord with additional post-processing fields.
    """

    summary_text: str
    topics_covered: list[str]
    decisions: list[str]
    action_items: list[ActionItem]  # Uses storage.models.ActionItem
    participants: list[str]


class SummarizerAgent:
    """Processes a meeting transcript and returns a structured MeetingRecord.

    Uses the OpenAI Agents SDK with output_type=SummaryOutput so the SDK
    enforces structured JSON output — no regex parsing of free text.
    """

    def __init__(self, model: str = "gpt-4o-mini"):
        self._model = model
        self._agent = Agent(
            name="meeting-summarizer",
            instructions=_INSTRUCTIONS,
            model=self._model,
            output_type=SummaryOutput,
        )

    async def run(
        self,
        transcript: str,
        meeting_meta: dict,
    ) -> MeetingRecord:
        """Run the summarizer agent on the given transcript.

        Args:
            transcript: Raw meeting transcript string. Each line should be
                "{participant_name}: {utterance}".
            meeting_meta: Dict with meeting metadata fields:
                meeting_id, channel_id, channel_name, start_ts,
                end_ts (optional), duration_seconds (optional),
                participants (optional list).

        Returns:
            MeetingRecord with all structured fields populated.

        Raises:
            ValueError: If any ActionItem has a blank owner field.
            Any exception raised by the SDK Runner is propagated to the caller.
        """
        prompt = f"Meeting transcript:\n\n{transcript}"

        result = await Runner.run(self._agent, input=prompt)
        output: SummaryOutput = result.final_output

        # Validate action item attribution — blank owners are never allowed
        for item in output.action_items:
            if not item.owner or not item.owner.strip():
                raise ValueError(
                    f"ActionItem owner must not be blank: {item.task!r}"
                )

        # Post-processing: compute derived fields
        raw_chars = len(transcript)
        status = "partial" if raw_chars < 500 else "complete"
        now_ts = int(time.time())

        return MeetingRecord(
            meeting_id=meeting_meta.get("meeting_id", ""),
            channel_id=meeting_meta.get("channel_id", ""),
            channel_name=meeting_meta.get("channel_name", ""),
            start_ts=meeting_meta.get("start_ts", 0),
            end_ts=meeting_meta.get("end_ts"),
            duration_seconds=meeting_meta.get("duration_seconds"),
            participants=output.participants,
            summary_text=output.summary_text,
            topics_covered=output.topics_covered,
            action_items=output.action_items,
            decisions=output.decisions,
            status=status,
            raw_transcript_chars=raw_chars,
            summarized_at=now_ts,
        )
