from __future__ import annotations

import json
import os
import time
from typing import Any, Protocol

import httpx

from src.schemas.endpoint_runner import (
    EndpointConfig,
    EndpointExecutionRequest,
    EndpointExecutionResult,
    EndpointItemResult,
    EndpointPayloadMapping,
    EndpointResponseMapping,
)
from src.schemas.experiment_runner import NormalizedDatasetItem


class EndpointHTTPClient(Protocol):
    def request(
        self,
        method: str,
        url: str,
        *,
        headers: dict[str, str] | None = None,
        json: Any = None,
        timeout: float | None = None,
    ) -> Any: ...


class EndpointRunnerService:
    def __init__(self, http_client: EndpointHTTPClient | None = None) -> None:
        self.http_client = http_client or httpx.Client()

    def run_endpoint_execution(
        self,
        *,
        items: list[NormalizedDatasetItem],
        request: EndpointExecutionRequest,
    ) -> EndpointExecutionResult:
        warnings: list[str] = []
        item_results: list[EndpointItemResult] = []
        failed_items = 0
        for item in items:
            result = self._run_single_item(item=item, request=request)
            item_results.append(result)
            if result.error:
                failed_items += 1
        processed_items = len(item_results)
        successful_latencies = [row.latency_ms for row in item_results if row.error is None and row.latency_ms is not None]
        successful_items = processed_items - failed_items
        summary = {
            "success_rate": (successful_items / processed_items) if processed_items else 0.0,
            "avg_latency_ms": (sum(successful_latencies) / len(successful_latencies)) if successful_latencies else None,
        }
        if failed_items:
            warnings.append(f"{failed_items} endpoint request basarisiz oldu.")
        return EndpointExecutionResult(
            processed_items=processed_items,
            failed_items=failed_items,
            item_results=item_results,
            warnings=warnings,
            errors=[],
            summary=summary,
        )

    def _run_single_item(
        self,
        *,
        item: NormalizedDatasetItem,
        request: EndpointExecutionRequest,
    ) -> EndpointItemResult:
        payload = self._build_payload(item=item, mapping=request.payload_mapping)
        headers = self._build_headers(config=request.endpoint_config)
        retries = max(0, request.endpoint_config.retry_count)
        attempt = 0
        last_error: str | None = None
        while attempt <= retries:
            started = time.perf_counter()
            try:
                response = self.http_client.request(
                    request.endpoint_config.method,
                    request.endpoint_config.url,
                    headers=headers,
                    json=payload,
                    timeout=request.endpoint_config.timeout_seconds,
                )
                latency_ms = (time.perf_counter() - started) * 1000.0
                if 400 <= response.status_code < 500:
                    return EndpointItemResult(
                        dataset_item_id=item.id,
                        request_payload=payload,
                        raw_response=response.text,
                        output=None,
                        status_code=response.status_code,
                        latency_ms=latency_ms,
                        error=f"HTTP {response.status_code}: {response.text}",
                    )
                if response.status_code >= 500:
                    last_error = f"HTTP {response.status_code}: {response.text}"
                    if attempt < retries:
                        attempt += 1
                        continue
                last_error = None
                output, trace_id, observation_id, response_metadata, raw_response = self._normalize_response(
                    response=response,
                    mapping=request.response_mapping,
                )
                return EndpointItemResult(
                    dataset_item_id=item.id,
                    request_payload=payload,
                    raw_response=raw_response,
                    output=output,
                    trace_id=trace_id,
                    observation_id=observation_id,
                    status_code=response.status_code,
                    latency_ms=latency_ms,
                    error=last_error,
                    response_metadata=response_metadata,
                )
            except (httpx.TimeoutException, httpx.NetworkError) as exc:
                last_error = str(exc)
                if attempt < retries:
                    attempt += 1
                    continue
                return EndpointItemResult(
                    dataset_item_id=item.id,
                    request_payload=payload,
                    raw_response=None,
                    output=None,
                    status_code=None,
                    latency_ms=None,
                    error=str(exc),
                )
            except Exception as exc:
                return EndpointItemResult(
                    dataset_item_id=item.id,
                    request_payload=payload,
                    raw_response=None,
                    output=None,
                    status_code=None,
                    latency_ms=None,
                    error=str(exc),
                )
        return EndpointItemResult(
            dataset_item_id=item.id,
            request_payload=payload,
            raw_response=None,
            output=None,
            status_code=None,
            latency_ms=None,
            error=last_error or "Unknown endpoint execution error",
        )

    def _build_headers(self, *, config: EndpointConfig) -> dict[str, str]:
        headers = dict(config.headers)
        token_env = (config.auth_token_env or "").strip()
        token = os.getenv(token_env) if token_env else None
        if config.auth_type == "bearer":
            if not token:
                raise ValueError("Bearer auth icin auth_token_env tanimli olmali ve environment degeri dolu olmali.")
            headers["Authorization"] = f"Bearer {token}"
        elif config.auth_type == "api_key_header":
            if not token:
                raise ValueError("API key header auth icin auth_token_env tanimli olmali ve environment degeri dolu olmali.")
            headers["x-api-key"] = token
        return headers

    def _build_payload(self, *, item: NormalizedDatasetItem, mapping: EndpointPayloadMapping) -> Any:
        context = {
            "input": item.input,
            "expected_output": item.expected_output,
            "metadata": item.metadata or {},
            "dataset_item_id": item.id,
            "source_trace_id": item.source_trace_id,
            "source_observation_id": item.source_observation_id,
        }
        if mapping.request_template:
            return self._inject_placeholders(mapping.request_template, context)
        payload: dict[str, Any] = {
            mapping.input_field_name: item.input,
        }
        if mapping.expected_output_field_name:
            payload[mapping.expected_output_field_name] = item.expected_output
        if mapping.metadata_field_name:
            payload[mapping.metadata_field_name] = item.metadata or {}
        return payload

    def _inject_placeholders(self, value: Any, context: dict[str, Any]) -> Any:
        if isinstance(value, dict):
            return {key: self._inject_placeholders(child, context) for key, child in value.items()}
        if isinstance(value, list):
            return [self._inject_placeholders(child, context) for child in value]
        if isinstance(value, str):
            for key, replacement in context.items():
                token = f"{{{{{key}}}}}"
                if value == token:
                    return replacement
                if token in value:
                    value = value.replace(token, self._replacement_to_string(replacement))
            return value
        return value

    @staticmethod
    def _replacement_to_string(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        try:
            return json.dumps(value, ensure_ascii=True, default=str)
        except TypeError:
            return str(value)

    def _normalize_response(
        self,
        *,
        response: Any,
        mapping: EndpointResponseMapping,
    ) -> tuple[Any, str | None, str | None, dict[str, Any] | None, Any]:
        if mapping.response_type == "text":
            return response.text, None, None, None, response.text
        try:
            raw_json = response.json()
        except Exception as exc:
            raise ValueError(f"JSON response bekleniyordu fakat parse edilemedi: {exc}") from exc
        output = self._extract_path(raw_json, mapping.output_json_path) if mapping.output_json_path else raw_json
        trace_id = self._coerce_optional_str(self._extract_path(raw_json, mapping.trace_id_json_path))
        observation_id = self._coerce_optional_str(self._extract_path(raw_json, mapping.observation_id_json_path))
        metadata_value = self._extract_path(raw_json, mapping.metadata_json_path) if mapping.metadata_json_path else None
        tool_trace = self._extract_path(raw_json, mapping.tool_trace_json_path) if mapping.tool_trace_json_path else None
        response_metadata: dict[str, Any] | None = None
        if isinstance(metadata_value, dict):
            response_metadata = dict(metadata_value)
        elif metadata_value is not None:
            response_metadata = {"raw_metadata": metadata_value}
        if tool_trace is not None:
            response_metadata = response_metadata or {}
            response_metadata["tool_trace"] = tool_trace
        return output, trace_id, observation_id, response_metadata, raw_json

    def _extract_path(self, payload: Any, path: str | None) -> Any:
        if not path:
            return None
        current = payload
        for part in path.split("."):
            if isinstance(current, list):
                try:
                    index = int(part)
                except ValueError as exc:
                    raise ValueError(f"Gecersiz JSON path segment'i: {part}") from exc
                if index >= len(current):
                    raise ValueError(f"JSON path list index disinda: {part}")
                current = current[index]
                continue
            if isinstance(current, dict):
                if part not in current:
                    raise ValueError(f"JSON path bulunamadi: {path}")
                current = current[part]
                continue
            raise ValueError(f"JSON path ilerletilemedi: {path}")
        return current

    @staticmethod
    def _coerce_optional_str(value: Any) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        return text or None
