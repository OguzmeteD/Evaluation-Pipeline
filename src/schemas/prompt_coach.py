from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class PromptCoachDecision(str, Enum):
    APPROVE = "approve"
    REVISE = "revise"
    REJECT = "reject"


class PromptApplyTarget(str, Enum):
    TASK = "task"
    JUDGE = "judge"
    BOTH = "both"
    NONE = "none"


class WebSearchResult(BaseModel):
    title: str
    url: str
    snippet: str | None = None


class PromptCoachRequest(BaseModel):
    user_request: str
    active_page: str | None = None
    current_task_prompt: str | None = None
    current_judge_prompt: str | None = None
    current_task_prompt_name: str | None = None
    current_judge_prompt_name: str | None = None
    current_task_prompt_label: str | None = None
    current_judge_prompt_label: str | None = None
    current_system_context: str | None = None


class PromptCoachResponse(BaseModel):
    decision: PromptCoachDecision
    summary: str
    recommended_prompt: str | None = None
    judge_guidance: str | None = None
    apply_target: PromptApplyTarget = PromptApplyTarget.NONE
    reasons: list[str] = Field(default_factory=list)
    suggested_evaluators: list[str] = Field(default_factory=list)
    web_sources: list[WebSearchResult] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
