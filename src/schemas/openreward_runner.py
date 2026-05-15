from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator


class OpenRewardConfig(BaseModel):
    environment_name: str = ""
    variant: str | None = None
    tool_name: str = ""
    task_spec_template: dict[str, Any] | None = None
    tool_input_template: dict[str, Any] | None = None
    tool_input_field_name: str = "input"
    log_rollout: bool = False
    rollout_run_name: str | None = None
    print_rollout_messages: bool = False
    base_url: str | None = None

    @field_validator("environment_name", "tool_name", "tool_input_field_name")
    @classmethod
    def normalize_strings(cls, value: str) -> str:
        return value.strip()


class OpenRewardItemResult(BaseModel):
    dataset_item_id: str | None = None
    session_id: str | None = None
    prompt_blocks: list[dict[str, Any]] = Field(default_factory=list)
    available_tools: list[str] = Field(default_factory=list)
    tool_input: dict[str, Any] = Field(default_factory=dict)
    tool_output: dict[str, Any] | None = None
    output: Any = None
    reward: float | None = None
    finished: bool | None = None
    rollout_id: str | None = None
    rollout_url: str | None = None
    latency_ms: float | None = None
    error: str | None = None
    metadata: dict[str, Any] | None = None


class OpenRewardExecutionRequest(BaseModel):
    config: OpenRewardConfig
    run_name: str
    dataset_name: str
    enable_judging: bool = False


class OpenRewardExecutionResult(BaseModel):
    processed_items: int = 0
    failed_items: int = 0
    item_results: list[OpenRewardItemResult] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    summary: dict[str, Any] = Field(default_factory=dict)
