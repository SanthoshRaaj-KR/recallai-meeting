import logging
from typing import Callable, List, Optional, Tuple

from agents import Agent, Runner

from ..core.schemas import MasterVoiceDecision, ResolverDecision
from .reframer_agent import ReframerAgent
from .tools import (
    commit_artifact_delete,
    commit_artifact_edit,
    create_local_artifact,
    fetch_live_artifact,
    format_artifact_titles_for_user,
    get_tool_state,
    list_sandbox_artifacts,
    preview_artifact_delete,
    preview_artifact_edit,
    rename_local_artifact,
    reset_tool_state,
    search_sandbox_artifacts,
    set_mutation_observer,
)


logger = logging.getLogger(__name__)


class EditorAgent:
    def __init__(self, model: str = "gpt-5-mini"):
        self.model = model
        self.reframer = ReframerAgent(model=model)
        self.history: List[Tuple[str, str]] = []
        self.max_history_turns = 6

        self.edit_agent = Agent(
            name="Jarvis Local Office Editor",
            model=model,
            instructions=(
                "You edit existing sandboxed office artifacts, including documents and spreadsheets. "
                "CRITICAL RULES:\n"
                "- Always verify the current target file using 'search_sandbox_artifacts' or 'list_sandbox_artifacts' first.\n"
                "- If a filename is approximate, search and pick the closest match. Do not ask for confirmation.\n"
                "- Use 'fetch_live_artifact(artifact_id, section_label)' to inspect real available targets.\n"
                "- Use section_label='Root' for top-of-document intro edits and 'FULL_PAGE' for full-body rewrites.\n"
                "- For spreadsheet edits, use the worksheet name as section_label.\n"
                "- Use 'preview_artifact_edit' before 'commit_artifact_edit'.\n"
                "- For rename requests on an existing file, use 'rename_local_artifact'. Do not create a replacement file.\n"
                "- Use visible text targets for targeted replacements when the user references a specific phrase or cell text.\n"
                "- ALMOST NEVER return NEEDS_CLARIFICATION. If search returns zero results, return 'ERROR: Could not find target file.'\n"
                "- Never claim success unless the most recent commit/create tool returned success=true.\n"
                "If the edit succeeds, return a short internal completion note without extra conversational padding."
            ),
            tools=[
                search_sandbox_artifacts,
                fetch_live_artifact,
                preview_artifact_edit,
                commit_artifact_edit,
                rename_local_artifact,
                list_sandbox_artifacts,
            ],
        )
        self.delete_agent = Agent(
            name="Jarvis Local Office Deleter",
            model=model,
            instructions=(
                "You delete content from existing sandboxed office artifacts. "
                "Use 'search_sandbox_artifacts' or 'list_sandbox_artifacts' to find the file, "
                "'fetch_live_artifact' to inspect it, then 'preview_artifact_delete' and 'commit_artifact_delete'. "
                "For document sections, use the real heading. For spreadsheets, use the worksheet name. "
                "If the user wants a whole section removed, set delete_entire_section=true with the real section label. "
                "If search returns zero results and you cannot reasonably infer the target, return 'ERROR: Could not find target file.'"
            ),
            tools=[
                search_sandbox_artifacts,
                fetch_live_artifact,
                preview_artifact_delete,
                commit_artifact_delete,
                list_sandbox_artifacts,
            ],
        )
        self.create_agent = Agent(
            name="Jarvis Local Office Creator",
            model=model,
            instructions=(
                "You create new sandboxed office files. "
                "Only create a file when the user explicitly asks for a new document, workbook, spreadsheet, sheet file, note, or draft. "
                "Use 'search_sandbox_artifacts' or 'list_sandbox_artifacts' first to avoid duplicates. "
                "Use 'create_local_artifact(title, artifact_family, file_format, body_text, sections)' for creation. "
                "If the user gives a clear topic but no explicit title, derive a concise sensible title from that topic. "
                "For spreadsheets, default artifact_family='spreadsheet' when the user asks for Excel/Calc/worksheet style output. "
                "Never ask questions. Always infer the best course of action. "
                "If creation fails completely, return 'ERROR: Artifact creation failed. <reason>'"
            ),
            tools=[create_local_artifact, search_sandbox_artifacts, list_sandbox_artifacts],
        )
        self.list_agent = Agent(
            name="Jarvis Local Office Lister",
            model=model,
            instructions=(
                "You list available sandboxed office artifacts. "
                "Always use 'list_sandbox_artifacts(limit=20)'. "
                "Return only clean human-friendly titles, separated by commas. "
                "Never include internal ids or absolute paths unless the user explicitly asks for them."
            ),
            tools=[list_sandbox_artifacts],
        )

        self.resolve_tool = self.reframer.agent.as_tool(
            tool_name="resolve_request",
            tool_description="Resolve vague local office file requests into a concrete editing intent.",
        )
        self.edit_tool = self.edit_agent.as_tool(
            tool_name="edit_existing_artifact",
            tool_description="Add, update, rewrite, or rename content in an existing sandboxed office file.",
        )
        self.delete_tool = self.delete_agent.as_tool(
            tool_name="delete_from_artifact",
            tool_description="Delete a section or content block from an existing sandboxed office file.",
        )
        self.create_tool = self.create_agent.as_tool(
            tool_name="create_new_artifact",
            tool_description="Create a brand-new sandboxed office file.",
        )
        self.list_tool = self.list_agent.as_tool(
            tool_name="list_recent_artifacts",
            tool_description="List the recent 20 sandboxed office file titles cleanly for the user.",
        )
        self.agent = Agent(
            name="Jarvis Local Office Master",
            model=model,
            instructions=(
                "You are Jarvis, the master editor agent for sandboxed office files. "
                "You stay in control of the conversation and use specialist agents as tools.\n"
                "RESOLVE FIRST:\n"
                "- Always use 'resolve_request' or 'list_recent_artifacts' to identify the exact target file before delegating.\n"
                "- Include the resolved title, artifact_id, and section label in the worker request.\n"
                "- Workers should receive enough information to execute without asking questions.\n\n"
                "ROUTING RULES:\n"
                "- Use 'list_recent_artifacts' for discovery requests like 'what local files are available'.\n"
                "- Use 'create_new_artifact' only when the user explicitly asks for a new file.\n"
                "- Use 'delete_from_artifact' for delete/remove requests.\n"
                "- Use 'edit_existing_artifact' for additions, updates, rewrites, and renames.\n"
                "- If resolver context is present, trust it strongly and skip re-resolving.\n\n"
                "CLARIFICATION RULES:\n"
                "- ALMOST NEVER ask the user a question. Prefer best-guess execution.\n"
                "- Only ask if search/list returned zero useful results and the request is impossible to infer.\n"
                "- If a worker returns an ERROR message that truly needs user input, you formulate the clarification.\n"
                "- Never expose internal ids or raw sandbox paths in the final user-facing reply.\n"
                "Your final reply must be a concise internal completion note."
            ),
            tools=[self.resolve_tool, self.list_tool, self.edit_tool, self.delete_tool, self.create_tool],
        )
        self.voice_master_agent = Agent(
            name="Jarvis Local Office Voice Master",
            model=model,
            instructions=(
                "You are Jarvis, the master local office editor for live voice requests. "
                "The user has already heard a short acknowledgement.\n"
                "Your job is to decide whether clarification is genuinely needed and produce an execution_request.\n"
                "RULES:\n"
                "- Set immediate_reply to an empty string always.\n"
                "- Set proceed_reply to an empty string always.\n"
                "- Ask at most one concise clarification question when the request is truly impossible to execute.\n"
                "- Use recent artifact titles and resolver tools to resolve ambiguity before asking the user.\n"
                "- If clarification context is present, treat it as the user's answer.\n"
                "- For list-file requests, set intent='list_artifacts' and execution_request='LIST_ARTIFACTS'.\n"
                "- For new spreadsheets, prefer artifact_family='spreadsheet'. For new documents, prefer artifact_family='document'.\n"
                "- Never include internal ids or absolute paths in any field.\n"
                "- Prefer executing with best-guess rather than asking when the target is reasonably inferable."
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
            self.history = self.history[-self.max_history_turns :]

    def _format_recent_history(self) -> str:
        if not self.history:
            return "[none]"
        lines: List[str] = []
        for user_text, assistant_text in self.history[-self.max_history_turns :]:
            lines.append(f"User: {user_text}")
            lines.append(f"Assistant: {assistant_text}")
        return "\n".join(lines)

    _CLEAR_ACTION_VERBS = frozenset(
        {"create", "list", "show", "delete", "add", "update", "edit", "rename", "remove", "make", "write"}
    )
    _REFERENTIAL_PHRASES = (
        " it ",
        " that ",
        " this ",
        " same ",
        "the one we just",
        "the file",
        "the document",
        "the sheet",
        "the spreadsheet",
        "created",
    )

    def _needs_reframing(self, query: str) -> bool:
        normalized = query.strip().lower()
        if not normalized:
            return False
        short_followups = {
            "yes",
            "yeah",
            "yep",
            "do it",
            "go ahead",
            "that one",
            "this one",
            "the file",
            "that file",
            "same file",
            "continue",
            "use that",
            "edit it",
            "update it",
        }
        if normalized in short_followups:
            return True
        padded = f" {normalized} "
        has_referential = any(phrase in padded for phrase in self._REFERENTIAL_PHRASES)
        words = normalized.split()
        if words and words[0] in self._CLEAR_ACTION_VERBS and not has_referential:
            return False
        if len(words) <= 4:
            return True
        return has_referential

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
            answer = result.final_output.strip() if hasattr(result, "final_output") else str(result)
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
                    "This request is likely vague, referential, or missing a concrete file target. "
                    "Use 'resolve_request' before choosing a specialist unless the correct action is already obvious."
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
        except Exception as exc:
            logger.error("Local office agent flow failed: %s", exc)
            return "I encountered an issue running the local office agent workflow."

    async def handle_voice_query(
        self,
        query: str,
        clarification_context: str = "",
        meeting_context: str = "",
        mutation_started_callback: Optional[Callable[[str], None]] = None,
    ) -> str:
        try:
            normalized_query = query.strip()
            if not normalized_query:
                return "I need a clearer request before I can help."
            enriched_input = (
                f"[Recent meeting discussion for context]\n{meeting_context}\n\n[User request]\n{normalized_query}"
                if meeting_context
                else normalized_query
            )
            recent_history = self._format_recent_history()
            sections = [
                f"Recent conversation history:\n{recent_history}",
                f"Original user request:\n{enriched_input}",
            ]
            if clarification_context.strip():
                sections.extend(
                    [
                        f"Clarification context:\n{clarification_context.strip()}",
                        "Voice routing hint:\n"
                        "Use the clarification context as the user's answer to your earlier question. "
                        "Ask at most one new concise follow-up only if the request is still genuinely unresolved.",
                    ]
                )
            return await self._run_editor(
                normalized_query,
                "\n\n".join(sections),
                mutation_started_callback=mutation_started_callback,
            )
        except Exception as exc:
            logger.error("Local office voice flow failed: %s", exc)
            return "I encountered an issue running the local office agent workflow."

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
                    "That previous task has now completed and its result is visible in the conversation history above."
                )
            if clarification_context.strip():
                sections.append(f"Clarification context:\n{clarification_context.strip()}")

            result = await Runner.run(self.voice_master_agent, "\n\n".join(sections))
            if isinstance(result.final_output, MasterVoiceDecision):
                return result.final_output
            return MasterVoiceDecision.model_validate(result.final_output)
        except Exception as exc:
            logger.error("Local office voice planning flow failed: %s", exc)
            return MasterVoiceDecision(
                immediate_reply="",
                needs_clarification=True,
                clarification_question="Which local office file should I work on?",
                intent="edit",
                rationale="Voice planning fallback.",
            )

    async def handle_prepared_query(
        self,
        prepared_query: str,
        original_query: Optional[str] = None,
        meeting_context: str = "",
        mutation_started_callback: Optional[Callable[[str], None]] = None,
    ) -> str:
        try:
            normalized_prepared = prepared_query.strip()
            if not normalized_prepared:
                return "I need a clearer request before I can help."

            if normalized_prepared.upper() == "LIST_ARTIFACTS":
                artifact_response = list_sandbox_artifacts(limit=20)
                answer = (
                    "I couldn't find any local office files right now."
                    if not artifact_response.candidates
                    else format_artifact_titles_for_user(artifact_response.candidates)
                )
                memory_query = (original_query or normalized_prepared).strip()
                self._remember(memory_query, answer)
                return answer

            memory_query = (original_query or normalized_prepared).strip()
            enriched_prepared = (
                f"[Recent meeting discussion for context]\n{meeting_context}\n\n[User request]\n{normalized_prepared}"
                if meeting_context
                else normalized_prepared
            )
            recent_history = self._format_recent_history()
            enriched_query = (
                f"Recent conversation history:\n{recent_history}\n\n"
                f"Prepared request from spoken master:\n{enriched_prepared}"
            )
            return await self._run_editor(
                memory_query,
                enriched_query,
                mutation_started_callback=mutation_started_callback,
            )
        except Exception as exc:
            logger.error("Local office prepared-query flow failed: %s", exc)
            return "I encountered an issue running the local office agent workflow."
