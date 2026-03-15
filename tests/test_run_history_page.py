from __future__ import annotations

import unittest
from datetime import UTC, datetime

from src.frontend.pages.run_history import (
    build_aggregate_trend_rows,
    build_history_table_rows,
    filter_history_records,
)
from src.schemas.experiment_runner import (
    AggregateMetricResult,
    ExperimentMode,
    ExperimentRunRecord,
    ExperimentRunStatus,
    PromptSource,
    PromptType,
)


class RunHistoryPageTest(unittest.TestCase):
    def setUp(self) -> None:
        self.records = [
            ExperimentRunRecord(
                id="run-1",
                created_at=datetime(2026, 3, 9, 10, 0, tzinfo=UTC),
                mode=ExperimentMode.PROMPT_RUNNER,
                dataset_name="support-dataset",
                run_name="run-1",
                status=ExperimentRunStatus.SUCCEEDED,
                task_prompt_source=PromptSource.CUSTOM_PROMPT,
                task_prompt_name="task-a",
                task_prompt_version=1,
                task_prompt_type=PromptType.TEXT,
                judge_prompt_source=PromptSource.CUSTOM_PROMPT,
                judge_prompt_name="judge-a",
                judge_prompt_version=1,
                judge_prompt_type=PromptType.TEXT,
                metric_names=["helpfulness"],
                aggregate_metrics=[AggregateMetricResult(name="helpfulness", average_score=0.8, count=5)],
            ),
            ExperimentRunRecord(
                id="run-2",
                created_at=datetime(2026, 3, 9, 11, 0, tzinfo=UTC),
                mode=ExperimentMode.REEVALUATE_EXISTING,
                dataset_name="support-dataset",
                run_name="run-2",
                status=ExperimentRunStatus.FAILED,
                task_prompt_source=PromptSource.CUSTOM_PROMPT,
                task_prompt_name="task-b",
                task_prompt_version=2,
                task_prompt_type=PromptType.TEXT,
                judge_prompt_source=PromptSource.CUSTOM_PROMPT,
                judge_prompt_name="judge-b",
                judge_prompt_version=2,
                judge_prompt_type=PromptType.TEXT,
                metric_names=["correctness"],
                aggregate_metrics=[AggregateMetricResult(name="correctness", average_score=0.4, count=3)],
            ),
        ]

    def test_filter_history_records_by_status(self) -> None:
        filtered = filter_history_records(self.records, ["succeeded"])
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0].run_name, "run-1")

    def test_build_aggregate_trend_rows_flattens_metrics(self) -> None:
        rows = build_aggregate_trend_rows(self.records)
        self.assertEqual(len(rows), 8)
        self.assertEqual(rows[0]["series_name"], "helpfulness: average_score")
        self.assertEqual(rows[1]["series_name"], "helpfulness: count")
        self.assertEqual(rows[-1]["series_name"], "run: processed_items")
        self.assertEqual(rows[-1]["status"], "failed")

    def test_build_history_table_rows_includes_status_badge(self) -> None:
        rows = build_history_table_rows(self.records)
        self.assertEqual(rows[0]["status_badge"], "SUCCEEDED")
        self.assertEqual(rows[1]["status_badge"], "FAILED")


if __name__ == "__main__":
    unittest.main()
