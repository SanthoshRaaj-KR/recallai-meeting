"""
RollingSummarizerAgent: processes a single batch of meeting transcript lines and
returns a BatchSummaryOutput with summary_text, key_points, and speakers.

This agent is invoked each time the sentence buffer is flushed (either by sentence
count threshold or the time ceiling). It does NOT accumulate context across batches —
each call summarizes only the lines passed to it.

Design notes:
- Follows the same Agent(output_type=...) pattern as SummarizerAgent in summarizer.py.
- Does NOT catch exceptions — callers (jarvis.py flush handler) handle failures.
- Uses gpt-4o-mini per Phase 6 mandate (budget model switch).
"""

from agents import Agent, Runner  # openai-agents SDK (namespace extended via conftest.py)
from storage.models import BatchSummaryOutput

_INSTRUCTIONS = """\
You are a real-time meeting summarizer. Given a batch of meeting transcript lines
(format: "Speaker: utterance"), produce a concise summary of this batch only.

Return:
- summary_text: 2-4 sentence prose. Be specific — name topics and outcomes discussed.
- key_points: Up to 5 short bullet-point phrases capturing the most important points.
- speakers: List of unique speaker names who appear in the transcript lines.

Stay factual. Do not infer beyond what is said. Do not reference other batches.
"""


class RollingSummarizerAgent:
    """Summarizes a single flushed batch of transcript lines into structured output.

    Each call is stateless — the agent does not retain context between batches.
    The batch_transcript string is the complete input for one summarization call.
    """

    def __init__(self, model: str = "gpt-4o-mini"):
        self._model = model
        self._agent = Agent(
            name="rolling-batch-summarizer",
            instructions=_INSTRUCTIONS,
            model=model,
            output_type=BatchSummaryOutput,
        )

    async def run(self, batch_transcript: str) -> BatchSummaryOutput:
        """Summarize one batch of transcript lines.

        Args:
            batch_transcript: String of transcript lines, one per line, in the
                format "Speaker: utterance\\nSpeaker2: utterance\\n..."

        Returns:
            BatchSummaryOutput with summary_text, key_points, and speakers.

        Raises:
            Any exception raised by the SDK Runner is propagated to the caller.
        """
        result = await Runner.run(self._agent, input=batch_transcript)
        return result.final_output
