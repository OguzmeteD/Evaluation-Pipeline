from __future__ import annotations

import unittest
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

from src.core.litellm_cost_builder import LiteLLMCostBuilderService
from src.core.litellm_ingestion import LiteLLMIngestionWriterService
from src.core.litellm_store import PostgresLiteLLMStore
from src.frontend.pages.litellm_cost_builder import (
    _parse_ingestion_rows,
    _parse_json_object,
    _selected_rows_from_state,
    _send_dataset_to_experiment_studio,
)
from src.schemas.litellm_ingestion import LiteLLMIngestionRequest
from src.schemas.litellm_cost_builder import (
    LiteLLMCostCandidateRow,
    LiteLLMCostDatasetPreview,
    LiteLLMCostDatasetRequest,
    LiteLLMCostFilters,
    LiteLLMFieldMapping,
    LiteLLMCostPreviewSummary,
    LiteLLMStoreConfig,
    DEFAULT_LITELLM_LOG_TABLE,
)


class FakeLiteLLMStore:
    def __init__(self) -> None:
        self.config = LiteLLMStoreConfig(
            enabled=True,
            dsn_present=True,
            timeout_seconds=30,
            auto_create_table=True,
            schema_mode="code_first",
            table_bootstrapped=False,
            mapping=LiteLLMFieldMapping(
                table_name="litellm_logs",
                id_column="request_id",
                created_at_column="created_at",
                model_column="model",
                total_cost_column="cost",
            ),
            missing_fields=[],
            warnings=[],
        )
        self.upserted_rows: list[dict[str, Any]] = []

    def get_config(self):
        return self.config

    def list_requests(self, filters: LiteLLMCostFilters):
        return (
            [
                {
                    "request_id": "req-1",
                    "created_at": datetime(2026, 3, 15, tzinfo=UTC),
                    "model_name": "gpt-4.1",
                    "provider": "openai",
                    "total_cost": 0.12,
                    "input_tokens": 100,
                    "output_tokens": 40,
                    "total_tokens": 140,
                    "latency_ms": 320.0,
                    "status": "success",
                    "request_input": {"question": "hello"},
                    "request_output": {"answer": "world"},
                    "langfuse_trace_id": "trace-1",
                    "langfuse_observation_id": None,
                    "metadata": {"customer": "acme"},
                },
                {
                    "request_id": "req-2",
                    "created_at": datetime(2026, 3, 15, tzinfo=UTC),
                    "model_name": "claude-3-7-sonnet",
                    "provider": "anthropic",
                    "total_cost": 0.2,
                    "input_tokens": 80,
                    "output_tokens": 20,
                    "total_tokens": 100,
                    "latency_ms": 450.0,
                    "status": "success",
                    "request_input": "plain input",
                    "request_output": "plain output",
                    "langfuse_trace_id": None,
                    "langfuse_observation_id": None,
                    "metadata": None,
                },
            ],
            [],
        )

    def ensure_schema(self):
        return []

    def upsert_requests(self, rows: list[dict[str, Any]]) -> int:
        self.upserted_rows.extend(rows)
        return len(rows)


class FakeLangfuseCollector:
    def __init__(self) -> None:
        self.created_dataset_calls: list[dict[str, Any]] = []
        self.created_item_calls: list[dict[str, Any]] = []

    def get_dataset(self, name: str) -> Any:
        if name == "existing-cost-dataset":
            return SimpleNamespace(id="existing-id", name=name)
        raise Exception("404 dataset not found")

    def create_dataset(self, *, name: str, description: str | None = None, metadata: dict[str, Any] | None = None):
        payload = {"id": "lf-dataset-1", "name": name, "description": description, "metadata": metadata}
        self.created_dataset_calls.append(payload)
        return payload

    def create_dataset_item(self, **kwargs: Any):
        self.created_item_calls.append(kwargs)
        return {"id": f"item-{kwargs.get('source_trace_id') or kwargs.get('input')}"}


class LiteLLMCostBuilderServiceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.store = FakeLiteLLMStore()
        self.collector = FakeLangfuseCollector()
        self.service = LiteLLMCostBuilderService(self.collector, store=self.store)

    def test_preview_normalizes_rows_and_summary(self) -> None:
        preview = self.service.preview_candidates(LiteLLMCostFilters(limit=20))
        self.assertEqual(preview.summary.total_requests, 2)
        self.assertAlmostEqual(preview.summary.total_cost, 0.32)
        self.assertEqual(preview.summary.total_tokens, 240)
        self.assertEqual(preview.rows[0].langfuse_trace_id, "trace-1")

    def test_create_dataset_blocks_existing_name(self) -> None:
        result = self.service.create_dataset(
            LiteLLMCostDatasetRequest(
                dataset_name="existing-cost-dataset",
                rows=[LiteLLMCostCandidateRow(request_id="req-1")],
            )
        )
        self.assertTrue(result.errors)

    def test_create_dataset_writes_langfuse_items(self) -> None:
        preview = self.service.preview_candidates(LiteLLMCostFilters(limit=20))
        result = self.service.create_dataset(
            LiteLLMCostDatasetRequest(
                dataset_name="litellm-cost-dataset",
                rows=preview.rows,
                filters=LiteLLMCostFilters(limit=20),
            )
        )
        self.assertEqual(result.dataset_id, "lf-dataset-1")
        self.assertEqual(result.created_items, 2)
        self.assertEqual(self.collector.created_item_calls[0]["source_trace_id"], "trace-1")
        self.assertEqual(self.collector.created_item_calls[0]["expected_output"], {"answer": "world"})


class LiteLLMIngestionWriterServiceTest(unittest.TestCase):
    def test_ingest_rows_upserts_into_store(self) -> None:
        store = FakeLiteLLMStore()
        service = LiteLLMIngestionWriterService(store=store)
        result = service.ingest_rows(
            LiteLLMIngestionRequest(
                rows=[
                    {
                        "request_id": "req-3",
                        "model_name": "gpt-4.1",
                        "total_cost": 0.04,
                        "request_input": {"question": "hello"},
                    }
                ]
            )
        )
        self.assertEqual(result.requested_rows, 1)
        self.assertEqual(result.upserted_rows, 1)
        self.assertEqual(store.upserted_rows[0]["request_id"], "req-3")
        self.assertEqual(store.upserted_rows[0]["model_name"], "gpt-4.1")


class LiteLLMStoreConfigTest(unittest.TestCase):
    def test_store_config_uses_code_first_defaults(self) -> None:
        store = PostgresLiteLLMStore(
            dsn="postgresql://user:pass@localhost:5432/litellm",
            mapping=None,
        )
        config = store.get_config()
        self.assertTrue(config.enabled)
        self.assertEqual(config.mapping.table_name, DEFAULT_LITELLM_LOG_TABLE)
        self.assertEqual(config.mapping.id_column, "request_id")
        self.assertEqual(config.schema_mode, "code_first")
        self.assertTrue(config.auto_create_table)
        self.assertFalse(config.missing_fields)

    def test_create_table_sql_contains_canonical_columns(self) -> None:
        store = PostgresLiteLLMStore(
            dsn="postgresql://user:pass@localhost:5432/litellm",
            mapping=LiteLLMFieldMapping(),
        )
        ddl = store._create_table_sql()
        self.assertIn('CREATE TABLE IF NOT EXISTS "litellm_request_logs"', ddl)
        self.assertIn('"request_id" TEXT PRIMARY KEY', ddl)
        self.assertIn('"request_input" JSONB NULL', ddl)
        self.assertIn('"langfuse_trace_id" TEXT NULL', ddl)

    def test_create_index_statements_cover_query_columns(self) -> None:
        store = PostgresLiteLLMStore(
            dsn="postgresql://user:pass@localhost:5432/litellm",
            mapping=LiteLLMFieldMapping(),
        )
        statements = store._create_index_statements()
        joined = "\n".join(statements)
        self.assertIn('"idx_litellm_request_logs_created_at"', joined)
        self.assertIn('"idx_litellm_request_logs_model_name"', joined)
        self.assertIn('"idx_litellm_request_logs_langfuse_trace_id"', joined)


class LiteLLMCostBuilderPageHelpersTest(unittest.TestCase):
    def test_parse_json_object_accepts_dict_only(self) -> None:
        parsed, error = _parse_json_object('{"source":"litellm"}')
        self.assertEqual(parsed, {"source": "litellm"})
        self.assertIsNone(error)

    def test_parse_ingestion_rows_accepts_object_and_array(self) -> None:
        rows, error = _parse_ingestion_rows('{"request_id":"req-1","model_name":"gpt-4.1","total_cost":0.01}')
        self.assertIsNone(error)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].request_id, "req-1")

        rows, error = _parse_ingestion_rows('[{"request_id":"req-2","model_name":"gpt-4.1","total_cost":0.02}]')
        self.assertIsNone(error)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].request_id, "req-2")

    def test_selected_rows_from_state_filters_preview_rows(self) -> None:
        preview = LiteLLMCostDatasetPreview(
            config=LiteLLMStoreConfig(
                enabled=True,
                dsn_present=True,
                timeout_seconds=30,
                auto_create_table=True,
                schema_mode="code_first",
                table_bootstrapped=False,
                mapping=LiteLLMFieldMapping(),
                missing_fields=[],
                warnings=[],
            ),
            filters=LiteLLMCostFilters(),
            summary=LiteLLMCostPreviewSummary(
                total_requests=2,
                total_cost=0.0,
                total_tokens=0,
                avg_latency_ms=None,
            ),
            rows=[
                LiteLLMCostCandidateRow(request_id="req-1"),
                LiteLLMCostCandidateRow(request_id="req-2"),
            ],
            warnings=[],
        )
        fake_st = SimpleNamespace(session_state={"litellm_cost_selected_request_ids": ["req-2"]})
        with patch("src.frontend.pages.litellm_cost_builder.st", fake_st):
            selected = _selected_rows_from_state(preview)
        self.assertEqual([row.request_id for row in selected], ["req-2"])

    def test_send_dataset_to_experiment_studio_sets_pending_page(self) -> None:
        state: dict[str, Any] = {}
        fake_st = SimpleNamespace(session_state=state)
        with patch("src.frontend.pages.litellm_cost_builder.st", fake_st), patch(
            "src.frontend.pages.litellm_cost_builder.fetch_dataset_by_name",
            return_value=SimpleNamespace(dataset_name="litellm-cost-dataset"),
        ):
            _send_dataset_to_experiment_studio("litellm-cost-dataset")
        self.assertEqual(state["pending_active_page"], "Experiment Studio")
        self.assertEqual(state["studio_dataset_name"], "litellm-cost-dataset")


if __name__ == "__main__":
    unittest.main()
