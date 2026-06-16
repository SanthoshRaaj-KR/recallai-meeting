"""ConfluenceEditorAgent — drafts document edits given a change intent and section content.

Uses the OpenAI Agents SDK (agents.Agent + agents.Runner) with structured output
to return an EditorDraft with before_content and after_content strings.
"""

from __future__ import annotations

import logging
import os

from agents import Agent, Runner
from pydantic import BaseModel

from .llm_runtime import guarded_run
from .models import ConfluenceIntent

logger = logging.getLogger(__name__)

EDITOR_INSTRUCTIONS = """\
You are a precise, minimal-diff document editor. Given a change intent and the current section content, you produce an edited version that changes ONLY what the intent requires and leaves everything else byte-for-byte identical.

Rules:
1. before_content: reproduce the current section content EXACTLY as given (character-accurate copy). Do not trim, reflow, or reformat it.
2. after_content: before_content with the SMALLEST possible edit that satisfies the intent. Touch only the specific words, numbers, or sentences the change affects. Every other sentence must remain word-for-word unchanged.
3. edit_type: almost always "replace" — use it for any modification of existing text (changing a number, a word, a clause). Use "append" ONLY when the intent adds entirely new content and nothing existing is being modified. Use "delete_section" only when the whole section is removed. Never "append" a value change.
4. Do NOT invent, summarise, annotate, or add any value the intent did not ask for. For example, if the change is "15 per year" -> "3 per month", just write the new figure — do NOT also add a derived total like "totaling 36 days", and do NOT add an editor note.
5. Relative changes: when the intent describes a change relative to the current value (e.g. "increased by 3 days", "current number + 3", "double it"), READ the current value in this section and WRITE the resulting value as clean prose. Example: the section says "ten paid sick days" and the intent is "+3 days" -> after_content says "thirteen paid sick days". Replace the number in place; do not append a note.
6. Do NOT restate the old value alongside the new one, and do NOT add parentheticals or clarifications. The result must read as clean final prose.
7. Removals: if the intent is to remove specific items, sentences, or clauses, delete exactly those and keep the surrounding text intact and grammatical. Do not rewrite what remains.
8. Preserve the document's writing style, tone, punctuation, and formatting conventions (markdown markers, list style, spacing).
9. If the section already fully matches the intent's new_value, return after_content identical to before_content.
10. Wrong-section guard: this section was already selected as the best match, so normally you should apply the change. The ONLY time you return after_content identical to before_content is when this section is clearly about a DIFFERENT rule or subject and merely happens to share a number or keyword with the intent (e.g. an intent about a support response SLA must not edit a security incident-reporting window that coincidentally also says "24 hours"). Do not approximate a change onto an unrelated rule.
11. old_value is the speaker's recollection and may be WRONG or ABSENT. When the topic clearly identifies a specific field, parameter, labeled value, or table row in THIS section (e.g. topic "maximum batch size" and the section has a row "Maximum batch size | 250 records", or "set the response time to 2 hours" and the section states a response time), SET that field to new_value even if the section's current value differs from old_value. Identify the field by its TOPIC/label, not by matching the old number. ("increase X to 950", "set X to 2 hours", "change X to weekly" all mean: put new_value on the field named X.) This still obeys rule 10 — if no field in this section is actually about the topic, do not force a change.
12. ADDING a new item to a category (intent like "introduce/add a new escalation tier called emergency", "add a new approval level", "create a new region"): this ADDS one item — it does NOT overwrite the field with the item's name. If the matching field is a COUNT (e.g. "Escalation tiers | 5 tiers", "Approval levels: 3"), increment the count by one to reflect the addition ("5 tiers" -> "6 tiers"); NEVER replace the count with the new item's name ("5 tiers" -> "emergency tier" is WRONG). If the matching field is a LIST of items (bullets, comma-separated, or a row that enumerates the items), append the new item to that list, preserving the existing entries. If the section neither counts nor lists the category, leave it unchanged (after_content == before_content) rather than corrupting an unrelated value.
"""


class EditorDraft(BaseModel):
    """The result of a document edit operation."""

    before_content: str
    after_content: str
    edit_type: str  # "replace" | "append" | "delete_section"


class ConfluenceEditorAgent:
    """Drafts before/after content for a document section given a change intent."""

    def __init__(self, model: str = None):
        self.model = model or os.getenv("LDOC_EDITOR_MODEL", "gpt-5.4-mini")
        self._agent = Agent(
            name="ConfluenceEditor",
            model=self.model,
            instructions=EDITOR_INSTRUCTIONS,
            output_type=EditorDraft,
        )

    async def draft(self, intent: ConfluenceIntent, section_content: str) -> EditorDraft:
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
            result = await guarded_run(self._agent, prompt)
            return result.final_output
        except Exception:
            logger.error(
                "ConfluenceEditorAgent.draft() failed for topic=%s",
                intent.affected_topic,
                exc_info=True,
            )
            return EditorDraft(
                before_content=section_content,
                after_content=section_content,
                edit_type="replace",
            )
