import logging

from agents import Agent, Runner

from ..core.schemas import ResolverDecision
from .tools import list_sandbox_artifacts, search_sandbox_artifacts


logger = logging.getLogger(__name__)


class ReframerAgent:
    def __init__(self, model: str = "gpt-5-mini"):
        self.model = model
        self.agent = Agent(
            name="Jarvis Local Office Resolver",
            model=model,
            instructions=(
                "You resolve vague local office editing requests into concrete artifact intents. "
                "Users may refer to files loosely. Inspect the sandbox first before asking questions. "
                "Use 'search_sandbox_artifacts' for targeted candidates and 'list_sandbox_artifacts' for recent options. "
                "Return a structured resolver decision. Use action values edit, create, or clarify. "
                "Set artifact_title/artifact_id/section_label when you have a likely target. "
                "Prefer the most likely existing file before deciding to create a new one. "
                "Only choose clarify if there is genuinely no plausible candidate after inspecting the sandbox."
            ),
            output_type=ResolverDecision,
            tools=[search_sandbox_artifacts, list_sandbox_artifacts],
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
        except Exception as exc:
            logger.error("Local office resolver flow failed: %s", exc)
            return ResolverDecision(
                action="clarify",
                artifact_title=None,
                artifact_id=None,
                section_label="UNKNOWN",
                reframed_request="Clarify which local office file should be edited.",
                rationale="Resolver failed, so explicit confirmation is safer.",
            )
