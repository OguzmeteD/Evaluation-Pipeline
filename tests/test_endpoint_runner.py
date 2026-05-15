from __future__ import annotations

import unittest
from dataclasses import dataclass
from typing import Any

import httpx

from src.core.endpoint_runner import EndpointRunnerService
from src.schemas.endpoint_runner import (
    EndpointConfig,
    EndpointExecutionRequest,
    EndpointPayloadMapping,
    EndpointResponseMapping,
)
from src.schemas.experiment_runner import NormalizedDatasetItem


@dataclass
class FakeResponse:
    status_code: int
    text: str
    json_data: Any = None

    def json(self) -> Any:
        if self.json_data is None:
            raise ValueError("not json")
        return self.json_data


class FakeHTTPClient:
    def __init__(self, responses: list[Any]) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def request(self, method: str, url: str, *, headers=None, json=None, timeout=None):
        self.calls.append(
            {
                "method": method,
                "url": url,
                "headers": headers,
                "json": json,
                "timeout": timeout,
            }
        )
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class EndpointRunnerServiceTest(unittest.TestCase):
    def test_request_template_injects_placeholders(self) -> None:
        service = EndpointRunnerService(http_client=FakeHTTPClient([]))
        item = NormalizedDatasetItem(
            id="item-1",
            input={"question": "hello"},
            expected_output="world",
            metadata={"channel": "chat"},
            source_trace_id="trace-1",
            source_observation_id="obs-1",
        )
        payload = service._build_payload(
            item=item,
            mapping=EndpointPayloadMapping(
                request_template={
                    "payload": "{{input}}",
                    "reference": "{{expected_output}}",
                    "meta": "{{metadata}}",
                    "trace": "{{source_trace_id}}",
                }
            ),
        )
        self.assertEqual(payload["payload"], {"question": "hello"})
        self.assertEqual(payload["reference"], "world")
        self.assertEqual(payload["meta"], {"channel": "chat"})
        self.assertEqual(payload["trace"], "trace-1")

    def test_json_response_extracts_output_and_metadata(self) -> None:
        client = FakeHTTPClient(
            [
                FakeResponse(
                    status_code=200,
                    text='{"result":{"text":"ok"},"trace_id":"trace-1","metadata":{"source":"endpoint"}}',
                    json_data={"result": {"text": "ok"}, "trace_id": "trace-1", "metadata": {"source": "endpoint"}},
                )
            ]
        )
        service = EndpointRunnerService(http_client=client)
        result = service.run_endpoint_execution(
            items=[NormalizedDatasetItem(id="item-1", input="hello")],
            request=EndpointExecutionRequest(
                endpoint_config=EndpointConfig(url="https://example.com/run"),
                response_mapping=EndpointResponseMapping(
                    response_type="json",
                    output_json_path="result.text",
                    trace_id_json_path="trace_id",
                    metadata_json_path="metadata",
                ),
            ),
        )
        self.assertEqual(result.processed_items, 1)
        self.assertEqual(result.item_results[0].output, "ok")
        self.assertEqual(result.item_results[0].trace_id, "trace-1")
        self.assertEqual(result.item_results[0].response_metadata, {"source": "endpoint"})

    def test_text_response_is_returned_as_output(self) -> None:
        client = FakeHTTPClient([FakeResponse(status_code=200, text="plain text")])
        service = EndpointRunnerService(http_client=client)
        result = service.run_endpoint_execution(
            items=[NormalizedDatasetItem(id="item-1", input="hello")],
            request=EndpointExecutionRequest(
                endpoint_config=EndpointConfig(url="https://example.com/run"),
                response_mapping=EndpointResponseMapping(response_type="text"),
            ),
        )
        self.assertEqual(result.item_results[0].output, "plain text")

    def test_http_4xx_marks_item_failed_without_retry(self) -> None:
        client = FakeHTTPClient([FakeResponse(status_code=422, text="bad request")])
        service = EndpointRunnerService(http_client=client)
        result = service.run_endpoint_execution(
            items=[NormalizedDatasetItem(id="item-1", input="hello")],
            request=EndpointExecutionRequest(
                endpoint_config=EndpointConfig(url="https://example.com/run", retry_count=3),
            ),
        )
        self.assertEqual(result.failed_items, 1)
        self.assertEqual(len(client.calls), 1)
        self.assertIn("HTTP 422", result.item_results[0].error or "")

    def test_retry_retries_network_error_then_succeeds(self) -> None:
        client = FakeHTTPClient(
            [
                httpx.ReadTimeout("timed out"),
                FakeResponse(status_code=200, text='{"ok":true}', json_data={"ok": True}),
            ]
        )
        service = EndpointRunnerService(http_client=client)
        result = service.run_endpoint_execution(
            items=[NormalizedDatasetItem(id="item-1", input="hello")],
            request=EndpointExecutionRequest(
                endpoint_config=EndpointConfig(url="https://example.com/run", retry_count=1),
            ),
        )
        self.assertEqual(result.failed_items, 0)
        self.assertEqual(len(client.calls), 2)


if __name__ == "__main__":
    unittest.main()
