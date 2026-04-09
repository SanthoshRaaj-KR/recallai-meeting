import logging
from typing import Callable, List, Optional, Tuple
from agents import Agent, Runner
from ..core.schemas import MasterVoiceDecision, ResolverDecision
from .tools import search_workspace_knowledge, fetch_live_page, preview_edit, preview_delete, commit_delete, commit_document_edit, update_page_title, reset_tool_state, get_tool_state, list_workspace_pages, format_page_titles_for_user, set_mutation_observer
from .reframer_agent import ReframerAgent

logger = logging.getLogger(__name__)

class EditorAgent:
    def __init__(self, model: str = "gpt-5-mini"):
        self.model = model
        self.reframer = ReframerAgent(model=model)
        self.history: List[Tuple[str, str]] = []
        self.max_history_turns = 6
        from .tools import search_workspace_knowledge, fetch_live_page, preview_edit, preview_delete, commit_delete, commit_document_edit, update_page_title, create_confluence_page
        self.edit_agent = Agent(
            name="Jarvis Page Editor",
            model=model,
            instructions=(
                "You edit existing Confluence pages by adding, updating, rewriting, or renaming content. "
                "CRITICAL EDITING RULES:\n"
                "- Do NOT blindly trust page IDs from conversation history. ALWAYS verify the current page existence using 'search_workspace_knowledge' or 'list_workspace_pages' first.\n"
                "- You will receive recent conversation context. Use it to resolve follow-ups like 'yes', 'that page', 'do it', or 'the one we just created'.\n"
                "- Resolver context or prepared request context is strong guidance for which page and heading to target.\n"
                "- SEARCH: use 'search_workspace_knowledge' to discover live Confluence page candidates first. Prefer exact or near-exact title matches before editing.\n"
                "- If a page name is approximately given, search and pick the closest match. Do NOT ask for confirmation.\n"
                "- RENAME: If the user asks to change the title of an existing page, use 'update_page_title(page_id, expected_version, new_title)'. Do not create a replacement page.\n"
                "- FETCH: use 'fetch_live_page(page_id, heading_string)' to retrieve true components. Use 'Root' only for top-of-page intro edits. Use 'FULL_PAGE' when the user wants to replace or delete the entire page body.\n"
                "- PREVIEW: use 'preview_edit(page_id, heading_string, old_block_html, new_block_html)' to diff-test. Pass 'Root' for top intro edits and 'FULL_PAGE' for full-page rewrites.\n"
                "- COMMIT: use 'commit_document_edit(page_id, expected_version, heading_string, old, new)'.\n"
                "- If the user says to remove all content, replace the entire page, rewrite the whole page, or start fresh, you must use heading_string='FULL_PAGE'. Do not use 'Root' for that.\n"
                "- If the user wants to replace visible text that is not a heading, use heading_string='FULL_PAGE' or the relevant real heading, and pass the visible text itself as old_block_html. The edit tools can resolve a unique visible-text block.\n"
                "- Use a heading name only when the target is actually a page heading from available_headings. Do not treat arbitrary text like '14' as a heading unless you saw it in available_headings.\n"
                "- TABLES: When creating or editing tables, ALWAYS use Confluence storage format classes: <table class=\"confluenceTable\"><tbody><tr><th class=\"confluenceTh\">...</th></tr><tr><td class=\"confluenceTd\">...</td></tr></tbody></table>. Plain markdown/HTML tables will not render properly.\n"
                "- ALMOST NEVER return 'NEEDS_CLARIFICATION'. You are a worker agent; do NOT ask questions directly. If search returns zero results, return 'ERROR: Could not find target page.' Prefer best-guess execution over failing.\n"
                "- NEVER claim success unless the most recent create/commit tool returned success=true. If a tool fails or conflicts completely, you MUST return 'ERROR: The update failed. <reason>' instead of failing silently.\n"
                "If the edit succeeds, return a short internal completion note without extra conversational padding."
            ),
            tools=[search_workspace_knowledge, fetch_live_page, preview_edit, commit_document_edit, update_page_title, list_workspace_pages],
        )
        self.delete_agent = Agent(
            name="Jarvis Page Deleter",
            model=model,
            instructions=(
                "You delete content from existing Confluence pages. "
                "Use 'search_workspace_knowledge' or 'list_workspace_pages' to find the right page, 'fetch_live_page' to inspect it, "
                "then 'preview_delete' and 'commit_delete' for the actual removal. "
                "Do NOT blindly trust page IDs from history. ALWAYS use 'search_workspace_knowledge' or 'list_workspace_pages' to verify the target first. "
                "If a page name is approximately given, search and pick the closest match. Do NOT ask for confirmation. "
                "If the user wants an entire section removed, set delete_entire_section=true with the real heading. "
                "If they want one paragraph, block, bullet group, or visible text removed, leave delete_entire_section=false and pass the exact visible target text. "
                "Never create a page. ALMOST NEVER ask questions. If search returns zero results AND you cannot guess the target, return 'ERROR: Could not find target page.' Prefer best-guess execution. "
                "If deletion fails completely, return 'ERROR: The deletion failed. <reason>'"
            ),
            tools=[search_workspace_knowledge, fetch_live_page, preview_delete, commit_delete, list_workspace_pages],
        )
        self.create_agent = Agent(
            name="Jarvis Page Creator",
            model=model,
            instructions=(
                "You create brand-new Confluence pages. "
                "Only create a page when the user explicitly asks for a new page, document, note, or draft. "
                "Use 'search_workspace_knowledge' or 'list_workspace_pages' first to check if a similar page already exists. If it does, do NOT create a duplicate. "
                "Use 'create_confluence_page(space_key, title, body_text, sections, parent_page_id)' for creation. "
                "If the user gives a clear topic but no explicit title, derive a concise sensible title from that topic. Never ask for a title. "
                "If the request sounds like an edit to an existing page, do not create anything. "
                "TABLES: When providing body_text or section HTML that includes tables, ALWAYS use Confluence storage format classes: <table class=\"confluenceTable\"><tbody><tr><th class=\"confluenceTh\">...</th></tr><tr><td class=\"confluenceTd\">...</td></tr></tbody></table>. Plain HTML tables will not render properly. "
                "NEVER ask questions. Always infer the best course of action. "
                "If page creation fails completely, return 'ERROR: Page creation failed. <reason>'"
            ),
            tools=[create_confluence_page, search_workspace_knowledge, list_workspace_pages],
        )
        self.list_agent = Agent(
            name="Jarvis Page Lister",
            model=model,
            instructions=(
                "You list available Confluence pages. "
                "Always use 'list_workspace_pages(limit=20)'. "
                "Return only clean human-friendly page titles, separated by commas. "
                "Never include page IDs, space keys, snippets, or other internal metadata unless the user explicitly asks for them."
            ),
            tools=[list_workspace_pages],
        )
        self.resolve_tool = self.reframer.agent.as_tool(
            tool_name="resolve_request",
            tool_description="Resolve vague, referential, or ambiguous requests into a concrete page editing intent.",
        )
        self.edit_tool = self.edit_agent.as_tool(
            tool_name="edit_existing_page",
            tool_description="Add, update, rewrite, or rename content on an existing Confluence page.",
        )
        self.delete_tool = self.delete_agent.as_tool(
            tool_name="delete_from_page",
            tool_description="Delete a section or content block from an existing Confluence page.",
        )
        self.create_tool = self.create_agent.as_tool(
            tool_name="create_new_page",
            tool_description="Create a brand-new Confluence page when the user explicitly asks for one.",
        )
        self.list_tool = self.list_agent.as_tool(
            tool_name="list_recent_pages",
            tool_description="List the recent 20 Confluence page titles cleanly for the user.",
        )
        self.agent = Agent(
            name="Jarvis Editor Master",
            model=model,
            instructions=(
                "You are Jarvis, the master editor agent. "
                "You stay in control of the conversation and use specialist agents as tools.\n"
                "RESOLVE FIRST:\n"
                "- ALWAYS use 'resolve_request' or 'list_recent_pages' to identify the exact target page BEFORE delegating to any worker.\n"
                "- When delegating, include the resolved page title, page_id, and heading in the input to the worker tool.\n"
                "- Workers should receive enough info to execute without asking questions.\n\n"
                "ROUTING RULES:\n"
                "- Use 'list_recent_pages' for page-discovery requests like 'what pages are available' or 'list pages'.\n"
                "- Use 'create_new_page' only when the user explicitly asks for a new page.\n"
                "- Use 'delete_from_page' for delete/remove requests against an existing page.\n"
                "- Use 'edit_existing_page' for additions, updates, rewrites, and renames on existing pages.\n"
                "- If resolver context is present in the request, trust it strongly and skip calling 'resolve_request' again.\n"
                "- Prefer calling a single specialist unless the request truly requires multiple distinct steps.\n\n"
                "CLARIFICATION RULES:\n"
                "- ALMOST NEVER ask the user a question. Prefer best-guess execution.\n"
                "- Only ask if search and list tools returned zero useful results AND the request is genuinely impossible to guess.\n"
                "- If a worker returns an 'ERROR:' message, evaluate the failure. If it's a hard failure that needs user input, YOU must formulate the clarification question (e.g., 'The update failed. Would you like me to try again?'). Workers do NOT ask questions, YOU do.\n"
                "- Treat clarification context in the prompt as the user's answer. Do not repeat questions.\n"
                "- Never expose internal metadata or raw error strings in the final user-facing reply.\n"
                "Your final reply must be a concise internal completion note."
            ),
            tools=[self.resolve_tool, self.list_tool, self.edit_tool, self.delete_tool, self.create_tool],
        )
        self.voice_master_agent = Agent(
            name="Jarvis Editor Master",
            model=model,
            instructions=(
                "You are Jarvis, the master editor agent for live voice requests. "
                "You operate in MINIMAL SPEECH mode — the user has already heard a short acknowledgement.\n"
                "Your ONLY job: decide if clarification is genuinely needed, and produce an execution_request.\n"
                "RULES:\n"
                "- Set immediate_reply to an empty string always.\n"
                "- Set proceed_reply to an empty string always. Do NOT narrate what you're about to do.\n"
                "- Ask a clarification ONLY when the request is genuinely impossible to execute without more info.\n"
                "- Ask at most ONE concise question. Never ask multiple questions.\n"
                "- Use recent page titles and resolver tools to resolve ambiguity BEFORE asking the user.\n"
                "- If clarification context is present, treat it as the user's answer. Do not repeat the same doubt.\n"
                "- If the request is clear enough, set needs_clarification=false and produce a concise execution_request.\n"
                "- For create-page requests with a clear topic, infer a sensible title rather than asking for one.\n"
                "- For page-listing requests, set intent='list_pages' and execution_request='LIST_PAGES'.\n"
                "- Never include internal IDs or metadata in any field.\n"
                "- When in doubt, prefer executing with best-guess rather than asking."
            ),
            output_type=MasterVoiceDecision,
            tools=[self.resolve_tool, self.list_tool],
        )

    def clear_memory(self) -> None:
        self.history.clear()

    def get_recent_history_text(self) -> str:
        return self._format_recent_history()

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

    def _format_resolver_context(self, decision: ResolverDecision) -> str:
        return (
            f"ACTION: {decision.action}\n"
            f"PAGE_TITLE: {decision.page_title or 'NONE'}\n"
            f"PAGE_ID: {decision.page_id or 'NONE'}\n"
            f"HEADING: {decision.heading or 'UNKNOWN'}\n"
            f"REFRAMED_REQUEST: {decision.reframed_request}\n"
            f"RATIONALE: {decision.rationale}"
        )

    async def _run_editor(
        self,
        memory_query: str,
        enriched_query: str,
        mutation_started_callback: Optional[Callable[[str], None]] = None,
    ) -> str:
        reset_tool_state()
        set_mutation_observer(mutation_started_callback)
        try:
            result = await Runner.run(self.agent, enriched_query)
            if hasattr(result, 'final_output'):
                answer = result.final_output.strip()
            else:
                answer = str(result)

            if answer.startswith("NEEDS_CLARIFICATION:"):
                detail = answer.split(":", 1)[1].strip()
                if detail and not detail.endswith("?"):
                    detail = f"{detail}?"
                answer = detail or "What should I clarify?"

            tool_state = get_tool_state()
            if tool_state.get("last_action") in {"commit", "create"} and tool_state.get("success") is False:
                answer = f"The requested change did not complete. {tool_state.get('message', '')}".strip()

            self._remember(memory_query, answer)
            return answer
        finally:
            set_mutation_observer(None)

    async def handle_query(
        self,
        query: str,
        mutation_started_callback: Optional[Callable[[str], None]] = None,
    ) -> str:
        try:
            normalized_query = query.strip()
            if normalized_query.lower() in {"reset memory", "clear memory", "/reset"}:
                self.clear_memory()
                return "Memory cleared."

            recent_history = self._format_recent_history()
            if self._needs_reframing(normalized_query):
                enriched_query = (
                    f"Recent conversation history:\n{recent_history}\n\n"
                    f"Original user request:\n{normalized_query}\n\n"
                    "Routing hint:\n"
                    "This request is likely vague, referential, or missing a concrete page target. "
                    "Use 'resolve_request' before choosing a specialist unless the correct action is already obvious from the history."
                )
            else:
                enriched_query = (
                    f"Recent conversation history:\n{recent_history}\n\n"
                    f"Original user request:\n{normalized_query}"
                )
            return await self._run_editor(
                normalized_query,
                enriched_query,
                mutation_started_callback=mutation_started_callback,
            )
        except Exception as e:
            logger.error(f"Agent flow failed: {e}")
            return "I encountered an issue running the agent workflow."

    async def handle_voice_query(
        self,
        query: str,
        clarification_context: str = "",
        mutation_started_callback: Optional[Callable[[str], None]] = None,
    ) -> str:
        try:
            normalized_query = query.strip()
            if not normalized_query:
                return "I need a clearer request before I can help."

            recent_history = self._format_recent_history()
            sections = [
                f"Recent conversation history:\n{recent_history}",
                f"Original user request:\n{normalized_query}",
            ]
            if clarification_context.strip():
                sections.extend(
                    [
                        f"Clarification context:\n{clarification_context.strip()}",
                        "Voice routing hint:\n"
                        "Use the clarification context as the user's answer to your earlier questions. "
                        "If it already resolves the ambiguity, do not ask the same question again. "
                        "Ask at most one new concise follow-up only if the request is still genuinely unresolved.",
                    ]
                )

            return await self._run_editor(
                normalized_query,
                "\n\n".join(sections),
                mutation_started_callback=mutation_started_callback,
            )
        except Exception as e:
            logger.error(f"Voice agent flow failed: {e}")
            return "I encountered an issue running the agent workflow."

    async def plan_voice_turn(
        self,
        query: str,
        clarification_context: str = "",
        has_other_pending_work: bool = False,
        request_label: str = "",
        is_queued_followup: bool = False,
    ) -> MasterVoiceDecision:
        try:
            normalized_query = query.strip()
            if not normalized_query:
                return MasterVoiceDecision(
                    immediate_reply="",
                    needs_clarification=True,
                    clarification_question="What would you like me to change?",
                    intent="edit",
                    rationale="Empty voice request.",
                )

            recent_history = self._format_recent_history()
            sections = [
                f"Recent conversation history:\n{recent_history}",
                f"Original user request:\n{normalized_query}",
                f"Other pending work waiting:\n{'yes' if has_other_pending_work else 'no'}",
                f"Current request label:\n{request_label or '[none]'}",
            ]
            if is_queued_followup:
                sections.append(
                    "Queue context:\n"
                    "This request was queued while a previous task was running. "
                    "That previous task has now completed — its result is visible in the conversation history above. "
                    "Use that context to resolve any ambiguities in this request. "
                    "If this request relates to what was just done, leverage that knowledge."
                )
            if clarification_context.strip():
                sections.append(f"Clarification context:\n{clarification_context.strip()}")

            result = await Runner.run(self.voice_master_agent, "\n\n".join(sections))
            if isinstance(result.final_output, MasterVoiceDecision):
                return result.final_output
            return MasterVoiceDecision.model_validate(result.final_output)
        except Exception as e:
            logger.error(f"Voice planning flow failed: {e}")
            return MasterVoiceDecision(
                immediate_reply="",
                needs_clarification=True,
                clarification_question="Which page should I work on?",
                intent="edit",
                rationale="Voice planning fallback.",
            )

    async def handle_prepared_query(
        self,
        prepared_query: str,
        original_query: Optional[str] = None,
        mutation_started_callback: Optional[Callable[[str], None]] = None,
    ) -> str:
        try:
            normalized_prepared = prepared_query.strip()
            if not normalized_prepared:
                return "I need a clearer request before I can help."

            if normalized_prepared.upper() == "LIST_PAGES":
                page_response = list_workspace_pages(limit=20)
                if not page_response.candidates:
                    answer = "I couldn't find any pages right now."
                else:
                    answer = format_page_titles_for_user(page_response.candidates)
                memory_query = (original_query or normalized_prepared).strip()
                self._remember(memory_query, answer)
                return answer

            memory_query = (original_query or normalized_prepared).strip()
            recent_history = self._format_recent_history()
            enriched_query = (
                f"Recent conversation history:\n{recent_history}\n\n"
                f"Prepared request from spoken master:\n{normalized_prepared}"
            )
            return await self._run_editor(
                memory_query,
                enriched_query,
                mutation_started_callback=mutation_started_callback,
            )
        except Exception as e:
            logger.error(f"Agent flow failed: {e}")
            return "I encountered an issue running the agent workflow."
