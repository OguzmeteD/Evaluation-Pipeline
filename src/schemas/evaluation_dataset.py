from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class PromptSource(str, Enum):
    LANGFUSE_PROMPT = "langfuse_prompt"
    OBSERVATION_INPUT = "observation_input"
    TRACE_INPUT = "trace_input"
    UNAVAILABLE = "unavailable"


class JudgeDatasetFilters(BaseModel):
    from_date: datetime | None = None
    to_date: datetime | None = None
    experiment_id: str | None = None
    trace_ids: list[str] = Field(default_factory=list)
    session_ids: list[str] = Field(default_factory=list)
    judge_names: list[str] = Field(default_factory=list)
    score_names: list[str] = Field(default_factory=list)
    min_score: float | None = None
    limit: int = 100
    cursor: str | None = None


class JudgeScoreRecord(BaseModel):
    score_id: str
    trace_id: str | None = None
    observation_id: str | None = None
    session_id: str | None = None
    experiment_id: str | None = None
    judge_name: str
    score_name: str
    score_value: float | None = None
    score_label: str | None = None
    score_comment: str | None = None
    score_source: str | None = None
    author_user_id: str | None = None
    data_type: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None
    metadata: dict[str, Any] | None = None


class ScoreBreakdown(BaseModel):
    score_name: str
    judge_name: str
    average_score: float | None = None
    count: int = 0


class GenerationRow(BaseModel):
    trace_id: str
    observation_id: str | None = None
    session_id: str | None = None
    experiment_id: str | None = None
    trace_name: str | None = None
    observation_name: str | None = None
    model: str | None = None
    latency_ms: float | None = None
    total_tokens: int | None = None
    total_cost: float | None = None
    system_prompt: str | None = None
    prompt_messages: list[dict[str, Any]] = Field(default_factory=list)
    generation_text: str | None = None
    judge_scores: list[JudgeScoreRecord] = Field(default_factory=list)
    input_payload: Any = None
    output_payload: Any = None
    started_at: datetime | None = None
    ended_at: datetime | None = None
    prompt_source: PromptSource = PromptSource.UNAVAILABLE
    has_prompt: bool = False
    has_generation: bool = False


class TraceSummary(BaseModel):
    trace_id: str
    session_id: str | None = None
    experiment_id: str | None = None
    trace_name: str | None = None
    observation_count: int = 0
    avg_score: float | None = None
    score_breakdown: list[ScoreBreakdown] = Field(default_factory=list)
    trace_scores: list[JudgeScoreRecord] = Field(default_factory=list)
    has_prompt: bool = False
    has_generation: bool = False
    started_at: datetime | None = None
    ended_at: datetime | None = None
    latency_ms: float | None = None
    total_cost: float | None = None
    total_tokens: int | None = None
    observation_ids: list[str] = Field(default_factory=list)


class DatasetCounts(BaseModel):
    traces: int = 0
    rows: int = 0
    observation_scores: int = 0
    trace_scores: int = 0


class DatasetMeta(BaseModel):
    filters: JudgeDatasetFilters
    counts: DatasetCounts
    prompt_coverage: float = 0.0
    generation_coverage: float = 0.0
    average_score: float | None = None
    warnings: list[str] = Field(default_factory=list)
    next_cursor: str | None = None
    metrics: dict[str, Any] = Field(default_factory=dict)


class EvaluationDataset(BaseModel):
    traces: list[TraceSummary] = Field(default_factory=list)
    rows: list[GenerationRow] = Field(default_factory=list)
    meta: DatasetMeta
