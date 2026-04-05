import logging

from agents import Agent, Runner

from .tools import list_workspace_pages, search_workspace_knowledge


logger = logging.getLogger(__name__)


class ReframerAgent:
    def __init__(self, model: str = "gpt-5-mini"):
        self.model = model
        self.agent = Agent(
            name="Jarvis Reframer",
            model=model,
            instructions=(
                "You are a resolver agent for enterprise document-editing requests. "
                "Users are often vague. Your job is to inspect workspace pages and rewrite the request into a concrete editing intent. "
                "Always inspect real workspace pages before asking the user to restate an exact title. "
                "Use 'search_workspace_knowledge' for targeted candidates and 'list_workspace_pages' to inspect recent real page options. "
                "For vague requests, you should prefer calling 'list_workspace_pages(limit=100)' so you can reason over the available pages. "
                "Produce a short structured text block with these exact lines:\n"
                "ACTION: edit|create|clarify\n"
                "PAGE_TITLE: <best page title or NONE>\n"
                "PAGE_ID: <best page id or NONE>\n"
                "HEADING: <best heading or Root or UNKNOWN>\n"
                "REFRAMED_REQUEST: <clear task statement using the selected page>\n"
                "RATIONALE: <one short sentence>\n"
                "Prefer the most likely existing page before deciding to create a new one. "
                "Only choose ACTION: clarify if there is genuinely no plausible page candidate after inspecting the workspace page list."
            ),
            tools=[search_workspace_knowledge, list_workspace_pages],
        )

    async def handle_query(self, query: str, conversation_history: str = "") -> str:
        try:
            reframer_input = (
                f"Recent conversation history:\n{conversation_history or '[none]'}\n\n"
                f"Latest user request:\n{query}"
            )
            result = await Runner.run(self.agent, reframer_input)
            if hasattr(result, "final_output"):
                return result.final_output.strip()
            return str(result)
        except Exception as e:
            logger.error(f"Reframer flow failed: {e}")
            return (
                "ACTION: clarify\n"
                "PAGE_TITLE: NONE\n"
                "PAGE_ID: NONE\n"
                "HEADING: UNKNOWN\n"
                "REFRAMED_REQUEST: Clarify which page should be edited.\n"
                "RATIONALE: Resolver failed, so explicit confirmation is safer."
            )
