from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class DatasetMetricThreshold(BaseModel):
    metric_name: str
    min_score: float
    judge_name: str | None = None


class DatasetBuilderFilters(BaseModel):
    from_date: datetime | None = None
    to_date: datetime | None = None
    experiment_id: str | None = None
    session_ids: list[str] = Field(default_factory=list)
    metric_thresholds: list[DatasetMetricThreshold] = Field(default_factory=list)
    limit: int = 100


class DatasetCandidateScore(BaseModel):
    metric_name: str
    judge_name: str | None = None
    score_value: float
    score_id: str | None = None
    created_at: datetime | None = None
    comment: str | None = None


class DatasetCandidateTrace(BaseModel):
    trace_id: str
    trace_name: str | None = None
    input_payload: Any = None
    output_payload: Any = None
    score_summary: list[DatasetCandidateScore] = Field(default_factory=list)
    matched_metrics: list[str] = Field(default_factory=list)
    avg_score: float | None = None
    session_id: str | None = None
    experiment_id: str | None = None
    metadata: dict[str, Any] | None = None


class DatasetCandidateResult(BaseModel):
    filters: DatasetBuilderFilters
    candidates: list[DatasetCandidateTrace] = Field(default_factory=list)
    total_candidates: int = 0
    warnings: list[str] = Field(default_factory=list)


class DatasetCreationRequest(BaseModel):
    dataset_name: str
    description: str | None = None
    metadata: dict[str, Any] | None = None
    candidates: list[DatasetCandidateTrace] = Field(default_factory=list)


class DatasetCreationResult(BaseModel):
    dataset_id: str | None = None
    dataset_name: str
    created_items: int = 0
    failed_items: int = 0
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    item_ids: list[str] = Field(default_factory=list)
