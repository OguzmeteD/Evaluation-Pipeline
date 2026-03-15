from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


DEFAULT_LITELLM_LOG_TABLE = "litellm_request_logs"


class LiteLLMFieldMapping(BaseModel):
    table_name: str | None = DEFAULT_LITELLM_LOG_TABLE
    id_column: str | None = "request_id"
    created_at_column: str | None = "created_at"
    model_column: str | None = "model_name"
    provider_column: str | None = "provider"
    total_cost_column: str | None = "total_cost"
    input_tokens_column: str | None = "input_tokens"
    output_tokens_column: str | None = "output_tokens"
    total_tokens_column: str | None = "total_tokens"
    latency_ms_column: str | None = "latency_ms"
    status_column: str | None = "status"
    input_column: str | None = "request_input"
    output_column: str | None = "request_output"
    metadata_column: str | None = "metadata"
    langfuse_trace_id_column: str | None = "langfuse_trace_id"
    langfuse_observation_id_column: str | None = "langfuse_observation_id"

    def missing_required_fields(self) -> list[str]:
        missing: list[str] = []
        if not self.table_name:
            missing.append("LITELLM_LOG_TABLE")
        if not self.id_column:
            missing.append("LITELLM_ID_COLUMN")
        if not self.created_at_column:
            missing.append("LITELLM_CREATED_AT_COLUMN")
        if not self.model_column:
            missing.append("LITELLM_MODEL_COLUMN")
        if not self.total_cost_column:
            missing.append("LITELLM_COST_COLUMN")
        return missing


class LiteLLMStoreConfig(BaseModel):
    enabled: bool = False
    dsn_present: bool = False
    timeout_seconds: int = 30
    auto_create_table: bool = True
    schema_mode: str = "code_first"
    table_bootstrapped: bool = False
    mapping: LiteLLMFieldMapping = Field(default_factory=LiteLLMFieldMapping)
    missing_fields: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class LiteLLMCostFilters(BaseModel):
    from_date: datetime | None = None
    to_date: datetime | None = None
    model_names: list[str] = Field(default_factory=list)
    providers: list[str] = Field(default_factory=list)
    statuses: list[str] = Field(default_factory=list)
    min_cost: float | None = None
    max_cost: float | None = None
    min_total_tokens: int | None = None
    max_total_tokens: int | None = None
    min_latency_ms: float | None = None
    max_latency_ms: float | None = None
    require_langfuse_join: bool = False
    limit: int = 100


class LiteLLMCostCandidateRow(BaseModel):
    request_id: str
    created_at: datetime | None = None
    model_name: str | None = None
    provider: str | None = None
    total_cost: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    latency_ms: float | None = None
    status: str | None = None
    request_input: Any = None
    request_output: Any = None
    langfuse_trace_id: str | None = None
    langfuse_observation_id: str | None = None
    metadata: dict[str, Any] | None = None


class LiteLLMCostPreviewSummary(BaseModel):
    total_requests: int = 0
    total_cost: float = 0.0
    total_tokens: int = 0
    avg_latency_ms: float | None = None


class LiteLLMCostDatasetPreview(BaseModel):
    config: LiteLLMStoreConfig
    filters: LiteLLMCostFilters
    summary: LiteLLMCostPreviewSummary
    rows: list[LiteLLMCostCandidateRow] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class LiteLLMCostDatasetRequest(BaseModel):
    dataset_name: str
    description: str | None = None
    metadata: dict[str, Any] | None = None
    filters: LiteLLMCostFilters | None = None
    rows: list[LiteLLMCostCandidateRow] = Field(default_factory=list)


class LiteLLMCostDatasetResult(BaseModel):
    dataset_id: str | None = None
    dataset_name: str
    created_items: int = 0
    failed_items: int = 0
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    item_ids: list[str] = Field(default_factory=list)
