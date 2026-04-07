import logging
from typing import List, Tuple
from agents import Agent, Runner
from .tools import search_workspace_knowledge, fetch_live_page, preview_edit, preview_delete, commit_delete, commit_document_edit, update_page_title, reset_tool_state, get_tool_state
from .reframer_agent import ReframerAgent

logger = logging.getLogger(__name__)

class EditorAgent:
    def __init__(self, model: str = "gpt-5-mini"):
        self.model = model
        self.reframer = ReframerAgent(model=model)
        self.history: List[Tuple[str, str]] = []
        self.max_history_turns = 6
        from .tools import search_workspace_knowledge, fetch_live_page, preview_edit, preview_delete, commit_delete, commit_document_edit, update_page_title, create_confluence_page
        self.agent = Agent(
            name="Jarvis Editor",
            model=model,
            instructions=(
                "You are Jarvis, an agentic document editor bot operating in a live meeting. "
                "CRITICAL EDITING RULES:\n"
                "- You will receive recent conversation context. Use it to resolve follow-ups like 'yes', 'that page', 'do it', or 'the one we just created'.\n"
                "- A resolver agent will provide you with 'Resolver context'. Treat it as strong guidance for which page and heading to target.\n"
                "- SEARCH: use 'search_workspace_knowledge' to discover live Confluence page candidates first. Prefer exact or near-exact title matches before editing.\n"
                "- CREATE: Use 'create_confluence_page(space_key, title, body_text, sections, parent_page_id)' only when the user explicitly asks to create a new page. Never create a new page as a fallback for an edit request.\n"
                "- RENAME: If the user asks to change the title of an existing page, use 'update_page_title(page_id, expected_version, new_title)'. Do not create a replacement page.\n"
                "- FETCH: use 'fetch_live_page(page_id, heading_string)' to retrieve true components. Use 'Root' only for top-of-page intro edits. Use 'FULL_PAGE' when the user wants to replace or delete the entire page body.\n"
                "- PREVIEW: use 'preview_edit(page_id, heading_string, old_block_html, new_block_html)' to diff-test. Pass 'Root' for top intro edits and 'FULL_PAGE' for full-page rewrites.\n"
                "- COMMIT: use 'commit_document_edit(page_id, expected_version, heading_string, old, new)'.\n"
                "- DELETE: For deletions, prefer the dedicated delete tools instead of empty-string edits. Use 'preview_delete(page_id, heading_string, target_html_or_text, delete_entire_section)' and then 'commit_delete(page_id, expected_version, heading_string, target_html_or_text, delete_entire_section)'.\n"
                "- If the user says to remove or delete an entire section, pass the real heading name with delete_entire_section=true.\n"
                "- If the user wants to delete only one block, paragraph, bullet group, or visible text chunk, pass that target text to the delete tools and leave delete_entire_section=false.\n"
                "- If the user says to remove all content, replace the entire page, rewrite the whole page, or start fresh, you must use heading_string='FULL_PAGE'. Do not use 'Root' for that.\n"
                "- If the user wants to replace visible text that is not a heading, use heading_string='FULL_PAGE' or the relevant real heading, and pass the visible text itself as old_block_html. The edit tools can resolve a unique visible-text block.\n"
                "- Use a heading name only when the target is actually a page heading from available_headings. Do not treat arbitrary text like '14' as a heading unless you saw it in available_headings.\n"
                "- Never claim success unless the most recent create/commit tool returned success=true. If a tool fails or conflicts, explicitly say the edit did not complete.\n"
                "Your final text response to the user must be a highly concise, friendly voice confirmation (1-2 sentences max)."
            ),
            tools=[search_workspace_knowledge, fetch_live_page, preview_edit, preview_delete, commit_delete, commit_document_edit, update_page_title, create_confluence_page]
        )

    def clear_memory(self) -> None:
        self.history.clear()

    def _remember(self, user_query: str, assistant_reply: str) -> None:
        self.history.append((user_query.strip(), assistant_reply.strip()))
        if len(self.history) > self.max_history_turns:
            self.history = self.history[-self.max_history_turns:]

    def _format_recent_history(self) -> str:
        if not self.history:
            return "[none]"

        lines: List[str] = []
        for user_text, assistant_text in self.history[-self.max_history_turns:]:
            lines.append(f"User: {user_text}")
            lines.append(f"Assistant: {assistant_text}")
        return "\n".join(lines)

    def _needs_reframing(self, query: str) -> bool:
        normalized = query.strip().lower()
        if not normalized:
            return False

        short_followups = {
            "yes", "yeah", "yep", "do it", "go ahead", "that one",
            "this one", "the page", "that page", "same page", "continue",
            "use that", "edit it", "update it",
        }
        if normalized in short_followups:
            return True

        if len(normalized.split()) <= 4:
            return True

        referential_phrases = [
            " it ",
            " that ",
            " this ",
            " same ",
            "the one we just",
            "the page",
            "created",
            "write about anything",
            "something about",
        ]
        padded = f" {normalized} "
        return any(phrase in padded for phrase in referential_phrases)

    async def handle_query(self, query: str) -> str:
        try:
            normalized_query = query.strip()
            if normalized_query.lower() in {"reset memory", "clear memory", "/reset"}:
                self.clear_memory()
                return "Memory cleared."

            recent_history = self._format_recent_history()
            if self._needs_reframing(normalized_query):
                reframed = await self.reframer.handle_query(normalized_query, conversation_history=recent_history)
                enriched_query = (
                    f"Recent conversation history:\n{recent_history}\n\n"
                    f"Original user request:\n{normalized_query}\n\n"
                    f"Resolver context:\n{reframed}"
                )
            else:
                enriched_query = (
                    f"Recent conversation history:\n{recent_history}\n\n"
                    f"Original user request:\n{normalized_query}"
                )
            reset_tool_state()
            result = await Runner.run(self.agent, enriched_query)
            if hasattr(result, 'final_output'):
                answer = result.final_output.strip()
            else:
                answer = str(result)

            tool_state = get_tool_state()
            if tool_state.get("last_action") in {"commit", "create"} and tool_state.get("success") is False:
                answer = f"The requested change did not complete. {tool_state.get('message', '')}".strip()

            self._remember(normalized_query, answer)
            return answer
        except Exception as e:
            logger.error(f"Agent flow failed: {e}")
            return "I encountered an issue running the agent workflow."
