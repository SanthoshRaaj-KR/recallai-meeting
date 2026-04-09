import logging

from agents import Agent, Runner

from ..core.schemas import ResolverDecision
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
                "Do NOT blindly trust page IDs from conversation history as pages can be deleted or recreated. Use 'search_workspace_knowledge' for targeted candidates and 'list_workspace_pages' to inspect recent real page options. "
                "For vague requests, you should prefer calling 'list_workspace_pages(limit=100)' so you can reason over the available pages. "
                "Return a structured resolver decision. "
                "Use action values edit, create, or clarify. "
                "Set page_title/page_id/heading when you have a likely target. "
                "Prefer the most likely existing page before deciding to create a new one. "
                "Only choose ACTION: clarify if there is genuinely no plausible page candidate after inspecting the workspace page list."
            ),
            output_type=ResolverDecision,
            tools=[search_workspace_knowledge, list_workspace_pages],
        )

    async def handle_query(self, query: str, conversation_history: str = "") -> ResolverDecision:
        try:
            reframer_input = (
                f"Recent conversation history:\n{conversation_history or '[none]'}\n\n"
                f"Latest user request:\n{query}"
            )
            result = await Runner.run(self.agent, reframer_input)
            if isinstance(result.final_output, ResolverDecision):
                return result.final_output
            return ResolverDecision.model_validate(result.final_output)
        except Exception as e:
            logger.error(f"Reframer flow failed: {e}")
            return ResolverDecision(
                action="clarify",
                page_title=None,
                page_id=None,
                heading="UNKNOWN",
                reframed_request="Clarify which page should be edited.",
                rationale="Resolver failed, so explicit confirmation is safer.",
            )
