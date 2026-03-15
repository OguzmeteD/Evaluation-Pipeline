from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Protocol

from pydantic_ai import Agent, RunContext

from src.core.langfuse_mcp import build_langfuse_mcp_server
from src.core.web_search import WebSearchClient
from src.schemas.prompt_coach import PromptApplyTarget, PromptCoachDecision, PromptCoachRequest, PromptCoachResponse


SYSTEM_PROMPT = """
You are Prompt Coach, a prompt engineering assistant for a Langfuse and LLM-as-a-Judge workflow.

Your job:
- review the user's requested prompt change
- decide whether to approve, revise, or reject it
- propose a stronger prompt when useful
- suggest judge/evaluator considerations for LLM-as-a-Judge
- use web search when external context would improve the answer

Rules:
- be concise, concrete, and implementation-oriented
- prefer revising weak prompt ideas instead of rejecting unless clearly harmful
- if a prompt can be improved, provide a ready-to-use recommended prompt
- if current task/judge prompt exists, reason against that context
- only cite web sources you actually used
- when Langfuse MCP tools are available, inspect existing prompts before recommending a replacement or change
- prefer using Langfuse MCP prompt tools to compare prompt versions before suggesting apply/publish actions
"""


@dataclass
class PromptCoachDeps:
    search_client: WebSearchClient
    


class PromptCoachGateway(Protocol):
    def run(self, request: PromptCoachRequest) -> PromptCoachResponse: ...


class PydanticAIPromptCoachGateway:
    def __init__(self, *, model_name: str | None = None, search_client: WebSearchClient | None = None) -> None:
        self.model_name = model_name or os.getenv("PROMPT_COACH_MODEL") or os.getenv("EXPERIMENT_JUDGE_MODEL")
        self.search_client = search_client or WebSearchClient()
        self.mcp_server = build_langfuse_mcp_server()

    def run(self, request: PromptCoachRequest) -> PromptCoachResponse:
        if not self.model_name:
            raise ValueError("PROMPT_COACH_MODEL veya EXPERIMENT_JUDGE_MODEL ayarlanmis olmali.")

        toolsets = [self.mcp_server] if self.mcp_server is not None else []
        agent = Agent(
            self.model_name,
            output_type=PromptCoachResponse,
            deps_type=PromptCoachDeps,
            system_prompt=SYSTEM_PROMPT,
            retries=2,
            toolsets=toolsets,
        )

        @agent.tool
        def search_web(ctx: RunContext[PromptCoachDeps], query: str, max_results: int = 5) -> list[dict[str, str | None]]:
            results = ctx.deps.search_client.search(query, max_results=max_results)
            return [result.model_dump() for result in results]

        prompt = _build_prompt(request)
        return agent.run_sync(prompt, deps=PromptCoachDeps(search_client=self.search_client)).output


class PromptCoachService:
    def __init__(self, gateway: PromptCoachGateway | None = None) -> None:
        self.gateway = gateway or PydanticAIPromptCoachGateway()

    def coach(self, request: PromptCoachRequest) -> PromptCoachResponse:
        return self.gateway.run(request)


def _build_prompt(request: PromptCoachRequest) -> str:
    return (
        f"Active page: {request.active_page or 'unknown'}\n\n"
        f"User request:\n{request.user_request}\n\n"
        f"Current task prompt:\n{request.current_task_prompt or 'n/a'}\n\n"
        f"Current task prompt name:\n{request.current_task_prompt_name or 'n/a'}\n\n"
        f"Current task prompt label:\n{request.current_task_prompt_label or 'n/a'}\n\n"
        f"Current judge prompt:\n{request.current_judge_prompt or 'n/a'}\n\n"
        f"Current judge prompt name:\n{request.current_judge_prompt_name or 'n/a'}\n\n"
        f"Current judge prompt label:\n{request.current_judge_prompt_label or 'n/a'}\n\n"
        f"Current page context:\n{request.current_system_context or 'n/a'}\n\n"
        "Return a decision, summary, reasons, improved prompt if needed, judge guidance, "
        "apply target hint, suggested evaluators, and any web sources used."
    )


_DEFAULT_PROMPT_COACH: PromptCoachService | None = None


def _get_service() -> PromptCoachService:
    global _DEFAULT_PROMPT_COACH
    if _DEFAULT_PROMPT_COACH is None:
        _DEFAULT_PROMPT_COACH = PromptCoachService()
    return _DEFAULT_PROMPT_COACH


def get_prompt_coach_response(request: PromptCoachRequest) -> PromptCoachResponse:
    return _get_service().coach(request)


def fallback_prompt_coach_response(message: str) -> PromptCoachResponse:
    return PromptCoachResponse(
        decision=PromptCoachDecision.REVISE,
        summary=message,
        recommended_prompt=None,
        judge_guidance="Current configuration could not produce a coached suggestion.",
        apply_target=PromptApplyTarget.NONE,
        warnings=[message],
    )
