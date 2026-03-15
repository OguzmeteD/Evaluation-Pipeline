from __future__ import annotations

import json
import unittest
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any

from src.core.langfuse_client import LangfuseCollectorClient
from src.core.metrics_analytics import MetricsAnalyticsService
from src.core.run_history import InMemoryRunHistoryStore
from src.frontend.pages.tool_judge import build_evaluator_selection
from src.schemas.experiment_runner import ExperimentMode, ExperimentRunRecord, ExperimentRunStatus, PromptSource, PromptType
from src.schemas.metrics_analytics import (
    MetricsTimeGranularity,
    PromptMetricsFilters,
    ToolJudgeFilters,
)


class FakeMetricsSDK:
    def __init__(self, rows: list[dict[str, Any]] | None = None) -> None:
        self.rows = rows or []
        self.queries: list[dict[str, Any]] = []
        self.api = SimpleNamespace(
            metrics_v_2=SimpleNamespace(metrics=self._metrics),
        )

    def _metrics(self, *, query: str, request_options: Any = None) -> Any:
        self.queries.append(json.loads(query))
        return SimpleNamespace(data=self.rows)


class FakeMetricsCollector:
    def __init__(self) -> None:
        self.observation_calls: list[str] = []
        self.numeric_calls: list[list[str]] = []
        self.categorical_calls: list[list[str]] = []

    def get_prompt_analytics(self, filters: PromptMetricsFilters) -> list[dict[str, Any]]:
        return [
            {
                "promptName": "support-task",
                "promptVersion": 3,
                "count": 5,
                "avgLatency": 200.0,
                "sumTotalCost": 1.5,
                "sumInputTokens": 100,
                "sumOutputTokens": 40,
                "sumTotalTokens": 140,
            }
        ]

    def get_run_prompt_analytics(self, filters: PromptMetricsFilters) -> list[dict[str, Any]]:
        return [
            {
                "time": "2026-03-09T00:00:00Z",
                "promptName": "support-task",
                "promptVersion": 3,
                "count": 5,
                "avgLatency": 200.0,
                "sumTotalCost": 1.5,
                "sumTotalTokens": 140,
            }
        ]

    def get_tool_observation_metrics(self, filters: ToolJudgeFilters, *, match_field: str = "name") -> list[dict[str, Any]]:
        self.observation_calls.append(match_field)
        if match_field == "name":
            return []
        return [
            {
                "name": "retrieval_span",
                "tags": ["rag_search"],
                "providedModelName": "text-embedding-3-large",
                "count": 4,
                "avgLatency": 150.0,
                "sumTotalCost": 0.4,
                "sumInputTokens": 40,
                "sumOutputTokens": 20,
                "sumTotalTokens": 60,
            }
        ]

    def get_tool_evaluator_metrics(
        self,
        filters: ToolJudgeFilters,
        *,
        observation_names: list[str],
        categorical: bool = False,
    ) -> list[dict[str, Any]]:
        if categorical:
            self.categorical_calls.append(observation_names)
            return [
                {
                    "name": "rag",
                    "observationName": "retrieval_span",
                    "source": "EVAL",
                    "stringValue": "pass",
                    "count": 2,
                }
            ]
        self.numeric_calls.append(observation_names)
        return [
            {
                "name": "embedding",
                "observationName": "retrieval_span",
                "source": "EVAL",
                "dataType": "NUMERIC",
                "count": 4,
                "avgValue": 0.85,
            }
        ]


class LangfuseCollectorMetricsQueryTest(unittest.TestCase):
    def test_get_prompt_analytics_builds_expected_query(self) -> None:
        sdk = FakeMetricsSDK()
        collector = LangfuseCollectorClient(sdk_client=sdk)
        collector.get_prompt_analytics(
            PromptMetricsFilters(
                prompt_name="support-task",
                prompt_versions=[2, 3],
                environment="production",
                model_name="gpt-4.1-mini",
                time_granularity=MetricsTimeGranularity.DAY,
                limit=25,
            )
        )
        query = sdk.queries[0]
        self.assertEqual(query["view"], "observations")
        self.assertEqual(query["config"]["row_limit"], 25)
        self.assertEqual(query["dimensions"][0]["field"], "promptName")
        self.assertTrue(any(metric["measure"] == "totalCost" for metric in query["metrics"]))
        self.assertTrue(any(f["column"] == "promptName" and f["value"] == "support-task" for f in query["filters"]))
        self.assertTrue(any(f["column"] == "promptVersion" for f in query["filters"]))


class MetricsAnalyticsServiceTest(unittest.TestCase):
    def setUp(self) -> None:
        history_store = InMemoryRunHistoryStore(
            records=[
                ExperimentRunRecord(
                    id="run-1",
                    created_at=datetime(2026, 3, 9, tzinfo=UTC),
                    mode=ExperimentMode.PROMPT_RUNNER,
                    dataset_name="support-dataset",
                    run_name="support-run",
                    description=None,
                    status=ExperimentRunStatus.SUCCEEDED,
                    task_prompt_source=PromptSource.LANGFUSE_PROMPT,
                    task_prompt_name="support-task",
                    task_prompt_version=3,
                    task_prompt_type=PromptType.TEXT,
                    task_prompt_fingerprint="abc123",
                    judge_prompt_source=PromptSource.CUSTOM_PROMPT,
                    judge_prompt_name=None,
                    judge_prompt_version=None,
                    judge_prompt_type=PromptType.TEXT,
                    judge_prompt_fingerprint="def456",
                    task_model="gpt-4.1-mini",
                    judge_model="gpt-4.1",
                    metric_names=["helpfulness"],
                    aggregate_metrics=[],
                    processed_items=5,
                    failed_items=0,
                    dataset_run_id="dataset-run-1",
                    dataset_run_url="https://langfuse.local/runs/dataset-run-1",
                    warnings=[],
                    errors=[],
                )
            ]
        )
        self.collector = FakeMetricsCollector()
        self.service = MetricsAnalyticsService(self.collector, history_store=history_store)

    def test_prompt_analytics_aggregates_metrics_and_enriches_history(self) -> None:
        dataset = self.service.get_prompt_analytics_dataset(
            PromptMetricsFilters(prompt_name="support-task", prompt_versions=[3], limit=20)
        )
        self.assertEqual(dataset.summary.total_observations, 5)
        self.assertAlmostEqual(dataset.summary.total_cost, 1.5)
        self.assertEqual(dataset.summary.total_tokens, 140)
        self.assertEqual(dataset.run_rows[0].matched_total_tokens, 140)
        self.assertEqual(dataset.run_rows[0].task_prompt_version, 3)

    def test_tool_judge_falls_back_to_tags_and_aggregates_scores(self) -> None:
        dataset = self.service.get_tool_judge_dataset(
            ToolJudgeFilters(
                tool_names=["rag_search"],
                evaluator_names=["rag", "embedding"],
                limit=20,
            )
        )
        self.assertEqual(self.collector.observation_calls, ["name", "tags"])
        self.assertTrue(dataset.warnings)
        self.assertEqual(dataset.tool_rows[0].tool_name, "rag_search")
        self.assertEqual(dataset.tool_rows[0].top_model, "text-embedding-3-large")
        embedding_row = next(row for row in dataset.evaluator_rows if row.evaluator_name == "embedding")
        self.assertAlmostEqual(embedding_row.average_score or 0.0, 0.85)
        rag_row = next(row for row in dataset.evaluator_rows if row.evaluator_name == "rag")
        self.assertEqual(rag_row.categorical_breakdown["pass"], 2)

    def test_build_evaluator_selection_supports_preset_and_custom(self) -> None:
        selected = build_evaluator_selection(["rag", "embedding"], "retrieval, rag, custom_eval")
        self.assertEqual(selected, ["rag", "embedding", "retrieval", "custom_eval"])


if __name__ == "__main__":
    unittest.main()
