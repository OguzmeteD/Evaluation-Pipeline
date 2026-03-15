from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


TOOL_EVALUATOR_PRESETS = ["rag", "embedding", "retrieval", "rerank", "tool-call"]


class MetricsTimeGranularity(str, Enum):
    AUTO = "auto"
    MINUTE = "minute"
    HOUR = "hour"
    DAY = "day"
    WEEK = "week"
    MONTH = "month"


class ToolMatchStrategy(str, Enum):
    NAME = "name"
    TAGS = "tags"


class PromptMetricsFilters(BaseModel):
    from_date: datetime | None = None
    to_date: datetime | None = None
    prompt_name: str | None = None
    prompt_versions: list[int] = Field(default_factory=list)
    run_name: str | None = None
    dataset_name: str | None = None
    environment: str | None = None
    model_name: str | None = None
    time_granularity: MetricsTimeGranularity | None = None
    limit: int = 100


class PromptMetricsSummary(BaseModel):
    total_observations: int = 0
    total_cost: float = 0.0
    avg_latency_ms: float | None = None
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_tokens: int = 0


class PromptVersionRow(BaseModel):
    prompt_name: str | None = None
    prompt_version: int | None = None
    observation_count: int = 0
    avg_latency_ms: float | None = None
    total_cost: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    model_name: str | None = None


class PromptTrendRow(BaseModel):
    bucket: str
    prompt_name: str | None = None
    prompt_version: int | None = None
    observation_count: int = 0
    avg_latency_ms: float | None = None
    total_cost: float = 0.0
    total_tokens: int = 0


class PromptRunRow(BaseModel):
    created_at: datetime
    run_name: str
    dataset_name: str
    status: str
    mode: str
    task_prompt_name: str | None = None
    task_prompt_version: int | None = None
    judge_prompt_name: str | None = None
    judge_prompt_version: int | None = None
    task_model: str | None = None
    judge_model: str | None = None
    processed_items: int = 0
    failed_items: int = 0
    matched_total_cost: float | None = None
    matched_avg_latency_ms: float | None = None
    matched_total_tokens: int | None = None
    dataset_run_url: str | None = None


class PromptAnalyticsDataset(BaseModel):
    summary: PromptMetricsSummary
    version_rows: list[PromptVersionRow] = Field(default_factory=list)
    trend_rows: list[PromptTrendRow] = Field(default_factory=list)
    run_rows: list[PromptRunRow] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class ToolJudgeFilters(BaseModel):
    from_date: datetime | None = None
    to_date: datetime | None = None
    tool_names: list[str] = Field(default_factory=list)
    environment: str | None = None
    observation_types: list[str] = Field(default_factory=list)
    evaluator_names: list[str] = Field(default_factory=list)
    limit: int = 100


class ToolMetricsSummary(BaseModel):
    tool_count: int = 0
    observation_count: int = 0
    total_cost: float = 0.0
    avg_latency_ms: float | None = None
    total_tokens: int = 0


class ToolMetricsRow(BaseModel):
    tool_name: str
    matched_by: ToolMatchStrategy
    observation_count: int = 0
    avg_latency_ms: float | None = None
    total_cost: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    top_model: str | None = None
    observation_names: list[str] = Field(default_factory=list)


class ToolEvaluatorRow(BaseModel):
    tool_name: str
    evaluator_name: str
    matched_by: ToolMatchStrategy
    average_score: float | None = None
    count: int = 0
    score_source: str | None = None
    data_type: str | None = None
    categorical_breakdown: dict[str, int] = Field(default_factory=dict)


class ToolJudgeDataset(BaseModel):
    summary: ToolMetricsSummary
    tool_rows: list[ToolMetricsRow] = Field(default_factory=list)
    evaluator_rows: list[ToolEvaluatorRow] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
