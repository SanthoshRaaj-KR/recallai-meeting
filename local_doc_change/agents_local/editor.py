"""LocalDocEditorAgent — drafts document edits given a change intent and section content.

Uses the OpenAI Agents SDK (agents.Agent + agents.Runner) with structured output
to return an EditorDraft with before_content and after_content strings.
"""

from __future__ import annotations

import logging
import os

from agents import Agent, Runner
from pydantic import BaseModel

from models import LocalDocIntent

logger = logging.getLogger(__name__)

EDITOR_INSTRUCTIONS = """\
You are a precise, minimal-diff document editor. Given a change intent and the current section content, you produce an edited version that changes ONLY what the intent requires and leaves everything else byte-for-byte identical.

Rules:
1. before_content: reproduce the current section content EXACTLY as given (character-accurate copy). Do not trim, reflow, or reformat it.
2. after_content: before_content with the SMALLEST possible edit that satisfies the intent. Touch only the specific words, numbers, or sentences the change affects. Every other sentence must remain word-for-word unchanged.
3. edit_type: "replace" when you change existing text; "append" when you only add new text at the end; "delete_section" when the entire section should be removed.
4. Make ONLY the change the intent describes. Do NOT invent, compute, summarise, annotate, or add any value the intent did not state. For example, if the change is "15 per year" -> "3 per month", do NOT also add a derived total like "totaling 36 days" — the human did not ask for it.
5. Do NOT restate the old value alongside the new one, and do NOT add parentheticals, clarifications, or editor notes. The result must read as clean final prose.
6. Removals: if the intent is to remove specific items, sentences, or clauses, delete exactly those and keep the surrounding text intact and grammatical. Do not rewrite what remains.
7. Preserve the document's writing style, tone, punctuation, and formatting conventions (markdown markers, list style, spacing).
8. If the section already matches the intent's new_value, return after_content identical to before_content (no note, no change).
9. Applicability gate: only apply the change if THIS section actually contains the specific value, clause, or statement the intent targets. If the exact thing being changed is not present here — even when the section is on a related topic with a similar-looking value — return after_content IDENTICAL to before_content. Never approximate the change onto different wording or a different number. A section that merely shares a theme with the change, but does not contain the specific rule/value being changed, must be left untouched.
"""


class EditorDraft(BaseModel):
    """The result of a document edit operation."""

    before_content: str
    after_content: str
    edit_type: str  # "replace" | "append" | "delete_section"


class LocalDocEditorAgent:
    """Drafts before/after content for a document section given a change intent."""

    def __init__(self, model: str = None):
        self.model = model or os.getenv("LDOC_EDITOR_MODEL", "gpt-4o-mini")
        self._agent = Agent(
            name="LocalDocEditor",
            model=self.model,
            instructions=EDITOR_INSTRUCTIONS,
            output_type=EditorDraft,
        )

    async def draft(self, intent: LocalDocIntent, section_content: str) -> EditorDraft:
        """Produce a before/after draft for the given section and intent.

        On exception, returns an identity draft (before == after) and logs the error.
        """
        prompt = (
            f"Change intent:\n"
            f"  type: {intent.intent_type}\n"
            f"  topic: {intent.affected_topic}\n"
            f"  old_value: {intent.old_value}\n"
            f"  new_value: {intent.new_value}\n"
            f"  rationale: {intent.rationale}\n\n"
            f"Current section content:\n{section_content}"
        )
        try:
            result = await Runner.run(self._agent, prompt)
            return result.final_output
        except Exception:
            logger.error(
                "LocalDocEditorAgent.draft() failed for topic=%s",
                intent.affected_topic,
                exc_info=True,
            )
            return EditorDraft(
                before_content=section_content,
                after_content=section_content,
                edit_type="replace",
            )
