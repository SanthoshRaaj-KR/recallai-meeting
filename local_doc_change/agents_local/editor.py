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
You are a precise document editor. Given a change intent and the current section content, you produce an edited version.

Rules:
1. before_content: reproduce the current section content EXACTLY as given (character-accurate copy).
2. after_content: the section content AFTER applying the change described by the intent.
3. edit_type: "replace" if you are changing existing text; "append" if you are adding new content at the end without removing anything; "delete_section" only if the entire section should be removed.
4. Make ONLY the change described by the intent. Do not invent or add unrelated content.
5. Preserve the document's writing style, tone, and formatting conventions.
6. If the section is already consistent with the intent's new_value, still return the section unchanged with a note in after_content.
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
