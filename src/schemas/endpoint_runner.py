from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator


class EndpointConfig(BaseModel):
    url: str = ""
    method: str = "POST"
    headers: dict[str, str] = Field(default_factory=dict)
    auth_type: str = "none"
    auth_token_env: str | None = None
    timeout_seconds: int = 30
    retry_count: int = 1

    @field_validator("method")
    @classmethod
    def validate_method(cls, value: str) -> str:
        method = value.upper().strip()
        if method not in {"GET", "POST"}:
            raise ValueError("Endpoint method yalnizca GET veya POST olabilir.")
        return method

    @field_validator("auth_type")
    @classmethod
    def validate_auth_type(cls, value: str) -> str:
        auth_type = value.strip().lower()
        if auth_type not in {"none", "bearer", "api_key_header"}:
            raise ValueError("Auth type none, bearer veya api_key_header olmali.")
        return auth_type


class EndpointPayloadMapping(BaseModel):
    input_path_mode: str = "dataset_input_direct"
    request_template: dict[str, Any] | None = None
    input_field_name: str = "input"
    expected_output_field_name: str | None = None
    metadata_field_name: str | None = None


class EndpointResponseMapping(BaseModel):
    response_type: str = "json"
    output_json_path: str | None = None
    trace_id_json_path: str | None = None
    observation_id_json_path: str | None = None
    metadata_json_path: str | None = None
    tool_trace_json_path: str | None = None

    @field_validator("response_type")
    @classmethod
    def validate_response_type(cls, value: str) -> str:
        response_type = value.strip().lower()
        if response_type not in {"json", "text"}:
            raise ValueError("Response type json veya text olmali.")
        return response_type


class EndpointItemResult(BaseModel):
    dataset_item_id: str | None = None
    request_payload: Any = None
    raw_response: Any = None
    output: Any = None
    trace_id: str | None = None
    observation_id: str | None = None
    status_code: int | None = None
    latency_ms: float | None = None
    error: str | None = None
    response_metadata: dict[str, Any] | None = None


class EndpointExecutionRequest(BaseModel):
    endpoint_config: EndpointConfig
    payload_mapping: EndpointPayloadMapping = Field(default_factory=EndpointPayloadMapping)
    response_mapping: EndpointResponseMapping = Field(default_factory=EndpointResponseMapping)
    enable_judging: bool = False


class EndpointExecutionResult(BaseModel):
    processed_items: int = 0
    failed_items: int = 0
    item_results: list[EndpointItemResult] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    summary: dict[str, Any] = Field(default_factory=dict)
