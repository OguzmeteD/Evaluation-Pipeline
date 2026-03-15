from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class LiteLLMIngestionRow(BaseModel):
    request_id: str
    created_at: datetime | None = None
    model_name: str
    provider: str | None = None
    total_cost: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int | None = None
    latency_ms: float | None = None
    status: str | None = None
    request_input: Any = None
    request_output: Any = None
    metadata: dict[str, Any] | list[Any] | str | None = None
    langfuse_trace_id: str | None = None
    langfuse_observation_id: str | None = None


class LiteLLMIngestionRequest(BaseModel):
    rows: list[LiteLLMIngestionRow] = Field(default_factory=list)


class LiteLLMIngestionResult(BaseModel):
    requested_rows: int = 0
    upserted_rows: int = 0
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
