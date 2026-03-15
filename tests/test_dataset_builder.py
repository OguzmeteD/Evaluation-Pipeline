from __future__ import annotations

import unittest
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

from src.core.dataset_builder import DatasetBuilderService
from src.frontend.pages.dataset_builder import (
    _build_metric_names,
    _parse_json_object,
    _selected_candidates_from_state,
    _send_dataset_to_experiment_studio,
)
from src.frontend.streamlit_app import _consume_pending_page_switch
from src.schemas.dataset_builder import (
    DatasetCandidateResult,
    DatasetCandidateTrace,
    DatasetBuilderFilters,
    DatasetCreationRequest,
    DatasetMetricThreshold,
)


class FakeDatasetCollector:
    def __init__(self) -> None:
        self.created_dataset_calls: list[dict[str, Any]] = []
        self.created_item_calls: list[dict[str, Any]] = []
        self.fail_item_trace_ids: set[str] = set()

    def list_scores(self, filters) -> list[dict[str, Any]]:
        if filters.score_names == ["helpfulness"]:
            return [
                {
                    "id": "score-1",
                    "trace_id": "trace-1",
                    "name": "helpfulness",
                    "value": 0.92,
                    "metadata": {"judge_name": "judge-a"},
                    "created_at": datetime(2026, 3, 9, tzinfo=UTC),
                },
                {
                    "id": "score-2",
                    "trace_id": "trace-2",
                    "name": "helpfulness",
                    "value": 0.88,
                    "metadata": {"judge_name": "judge-a"},
                    "created_at": datetime(2026, 3, 9, tzinfo=UTC),
                },
            ]
        if filters.score_names == ["correctness"]:
            return [
                {
                    "id": "score-3",
                    "trace_id": "trace-1",
                    "name": "correctness",
                    "value": 0.91,
                    "metadata": {"judge_name": "judge-b"},
                    "created_at": datetime(2026, 3, 9, tzinfo=UTC),
                }
            ]
        return []

    def get_trace(self, trace_id: str) -> dict[str, Any]:
        traces = {
            "trace-1": {
                "id": "trace-1",
                "name": "Trace One",
                "input": {"question": "How to wash?"},
                "output": {"answer": "Use quick wash."},
                "session_id": "session-1",
                "metadata": {"experiment_id": "exp-1"},
            },
            "trace-2": {
                "id": "trace-2",
                "name": "Trace Two",
                "input": None,
                "output": None,
                "session_id": "session-2",
                "metadata": {"experiment_id": "exp-2"},
            },
        }
        return traces[trace_id]

    def list_observations(self, filters) -> tuple[list[dict[str, Any]], None]:
        if filters.trace_ids == ["trace-2"]:
            return (
                [
                    {
                        "id": "obs-2",
                        "trace_id": "trace-2",
                        "input": {"question": "fallback input"},
                        "output": {"answer": "fallback output"},
                    }
                ],
                None,
            )
        return ([], None)

    def get_dataset(self, name: str) -> Any:
        if name == "existing-dataset":
            return SimpleNamespace(id="existing-id", name=name)
        raise Exception("404 dataset not found")

    def create_dataset(self, *, name: str, description: str | None = None, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        payload = {"id": "new-dataset-id", "name": name, "description": description, "metadata": metadata}
        self.created_dataset_calls.append(payload)
        return payload

    def create_dataset_item(self, **kwargs: Any) -> dict[str, Any]:
        self.created_item_calls.append(kwargs)
        if kwargs.get("source_trace_id") in self.fail_item_trace_ids:
            raise RuntimeError("item create failed")
        return {"id": f"item-{kwargs.get('source_trace_id')}"}


class DatasetBuilderServiceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.collector = FakeDatasetCollector()
        self.service = DatasetBuilderService(self.collector)

    def test_preview_candidates_requires_all_metrics_to_match(self) -> None:
        result = self.service.preview_candidate_traces(
            DatasetBuilderFilters(
                metric_thresholds=[
                    DatasetMetricThreshold(metric_name="helpfulness", min_score=0.8),
                    DatasetMetricThreshold(metric_name="correctness", min_score=0.8),
                ],
                limit=10,
            )
        )
        self.assertEqual(result.total_candidates, 1)
        self.assertEqual(result.candidates[0].trace_id, "trace-1")
        self.assertEqual(result.candidates[0].matched_metrics, ["helpfulness", "correctness"])

    def test_preview_candidates_uses_observation_fallback_for_missing_trace_io(self) -> None:
        result = self.service.preview_candidate_traces(
            DatasetBuilderFilters(
                metric_thresholds=[DatasetMetricThreshold(metric_name="helpfulness", min_score=0.8)],
                limit=10,
            )
        )
        candidate = next(row for row in result.candidates if row.trace_id == "trace-2")
        self.assertEqual(candidate.input_payload, {"question": "fallback input"})
        self.assertEqual(candidate.output_payload, {"answer": "fallback output"})

    def test_create_dataset_blocks_when_name_exists(self) -> None:
        result = self.service.create_dataset_from_candidates(
            DatasetCreationRequest(
                dataset_name="existing-dataset",
                candidates=[],
            )
        )
        self.assertTrue(result.errors)
        self.assertIn("zaten mevcut", result.errors[0])

    def test_create_dataset_reports_partial_item_failures(self) -> None:
        preview = self.service.preview_candidate_traces(
            DatasetBuilderFilters(
                metric_thresholds=[DatasetMetricThreshold(metric_name="helpfulness", min_score=0.8)],
                limit=10,
            )
        )
        self.collector.fail_item_trace_ids.add("trace-2")
        result = self.service.create_dataset_from_candidates(
            DatasetCreationRequest(
                dataset_name="fresh-dataset",
                description="Created from high scores",
                candidates=preview.candidates,
            )
        )
        self.assertEqual(result.dataset_id, "new-dataset-id")
        self.assertEqual(result.created_items, 1)
        self.assertEqual(result.failed_items, 1)
        self.assertTrue(result.warnings)
        self.assertEqual(self.collector.created_item_calls[0]["expected_output"], {"answer": "Use quick wash."})

    def test_preview_candidates_returns_warning_when_metric_fetch_fails(self) -> None:
        failing_collector = FakeDatasetCollector()

        def _raise_timeout(filters):
            raise TimeoutError("read timed out")

        failing_collector.list_scores = _raise_timeout  # type: ignore[method-assign]
        service = DatasetBuilderService(failing_collector)
        result = service.preview_candidate_traces(
            DatasetBuilderFilters(
                metric_thresholds=[DatasetMetricThreshold(metric_name="helpfulness", min_score=0.8)],
                limit=10,
            )
        )
        self.assertEqual(result.total_candidates, 0)
        self.assertTrue(any("score fetch basarisiz" in warning for warning in result.warnings))


class DatasetBuilderPageHelpersTest(unittest.TestCase):
    def test_parse_json_object_accepts_dict_only(self) -> None:
        parsed, error = _parse_json_object('{"source":"builder"}')
        self.assertEqual(parsed, {"source": "builder"})
        self.assertIsNone(error)
        parsed, error = _parse_json_object('["bad"]')
        self.assertIsNone(parsed)
        self.assertIn("object", error or "")

    def test_build_metric_names_deduplicates_custom(self) -> None:
        self.assertEqual(
            _build_metric_names(["helpfulness", "correctness"], "helpfulness"),
            ["helpfulness", "correctness"],
        )
        self.assertEqual(
            _build_metric_names(["helpfulness"], "brand_safety"),
            ["helpfulness", "brand_safety"],
        )

    def test_selected_candidates_from_state_filters_preview_rows(self) -> None:
        preview = DatasetCandidateResult(
            filters=DatasetBuilderFilters(),
            candidates=[
                DatasetCandidateTrace(trace_id="trace-1"),
                DatasetCandidateTrace(trace_id="trace-2"),
            ],
            total_candidates=2,
        )
        fake_st = SimpleNamespace(session_state={"dataset_builder_selected_trace_ids": ["trace-2"]})
        with patch("src.frontend.pages.dataset_builder.st", fake_st):
            selected = _selected_candidates_from_state(preview)
        self.assertEqual([candidate.trace_id for candidate in selected], ["trace-2"])

    def test_send_dataset_to_experiment_studio_sets_pending_page_and_dataset(self) -> None:
        state: dict[str, Any] = {}
        fake_st = SimpleNamespace(session_state=state)
        with patch("src.frontend.pages.dataset_builder.st", fake_st), patch(
            "src.frontend.pages.dataset_builder.fetch_dataset_by_name",
            return_value=SimpleNamespace(dataset_name="created-dataset"),
        ):
            _send_dataset_to_experiment_studio("created-dataset")
        self.assertEqual(state["studio_dataset_name"], "created-dataset")
        self.assertEqual(state["pending_active_page"], "Experiment Studio")
        self.assertIn("yuklendi", state["studio_prompt_apply_message"])

    def test_consume_pending_page_switch_updates_active_page(self) -> None:
        state = {"pending_active_page": "Experiment Studio"}
        fake_st = SimpleNamespace(session_state=state)
        with patch("src.frontend.streamlit_app.st", fake_st):
            _consume_pending_page_switch()
        self.assertEqual(state["active_page"], "Experiment Studio")
        self.assertNotIn("pending_active_page", state)


if __name__ == "__main__":
    unittest.main()
