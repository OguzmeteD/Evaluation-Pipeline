from __future__ import annotations

import json
import unittest
from datetime import UTC, datetime
from types import SimpleNamespace

import httpx

from src.core.langfuse_client import LangfuseCollectorClient
from src.schemas.evaluation_dataset import JudgeDatasetFilters
from src.schemas.metrics_analytics import PromptMetricsFilters


class FakeScoreSDK:
    def __init__(self) -> None:
        self.calls: list[dict[str, int]] = []
        self.api = SimpleNamespace(
            score_v_2=SimpleNamespace(get=self._get_scores),
        )

    def _get_scores(self, *, page: int, limit: int, **kwargs):
        if limit > 100:
            raise AssertionError("score_v_2.get received limit > 100")
        self.calls.append({"page": page, "limit": limit})
        if page == 1:
            rows = [
                {"id": f"score-{index}", "trace_id": f"trace-{index}", "name": "helpfulness", "value": 0.9}
                for index in range(100)
            ]
        elif page == 2:
            rows = [
                {"id": f"score-{100 + index}", "trace_id": f"trace-{100 + index}", "name": "helpfulness", "value": 0.9}
                for index in range(30)
            ]
        else:
            rows = []
        return SimpleNamespace(data=rows)


class FlakyScoreSDK:
    def __init__(self) -> None:
        self.calls = 0
        self.api = SimpleNamespace(
            score_v_2=SimpleNamespace(get=self._get_scores),
        )

    def _get_scores(self, **kwargs):
        self.calls += 1
        if self.calls == 1:
            raise httpx.ReadTimeout("timed out")
        return SimpleNamespace(data=[])


class ObservationSDK:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []
        self.api = SimpleNamespace(
            observations_v_2=SimpleNamespace(get_many=self._get_many),
        )

    def _get_many(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(data=[], meta=SimpleNamespace(next_cursor=None))


class MetricsSDK:
    def __init__(self) -> None:
        self.queries: list[dict[str, object]] = []
        self.api = SimpleNamespace(
            metrics_v_2=SimpleNamespace(metrics=self._metrics),
        )

    def _metrics(self, *, query: str, request_options=None):
        self.queries.append(json.loads(query))
        return SimpleNamespace(data=[])


class LangfuseClientTest(unittest.TestCase):
    def test_list_scores_clamps_limit_and_paginates(self) -> None:
        sdk = FakeScoreSDK()
        collector = LangfuseCollectorClient(sdk_client=sdk)
        scores = collector.list_scores(
            JudgeDatasetFilters(
                score_names=["helpfulness"],
                limit=130,
            )
        )
        self.assertEqual(len(scores), 130)
        self.assertEqual(
            sdk.calls,
            [{"page": 1, "limit": 50}, {"page": 2, "limit": 50}],
        )

    def test_list_scores_retries_read_timeout(self) -> None:
        sdk = FlakyScoreSDK()
        collector = LangfuseCollectorClient(sdk_client=sdk)
        scores = collector.list_scores(
            JudgeDatasetFilters(
                score_names=["helpfulness"],
                limit=10,
            )
        )
        self.assertEqual(scores, [])
        self.assertEqual(sdk.calls, 2)

    def test_list_observations_does_not_send_parse_io_as_json(self) -> None:
        sdk = ObservationSDK()
        collector = LangfuseCollectorClient(sdk_client=sdk)
        observations, next_cursor = collector.list_observations(JudgeDatasetFilters(limit=10))
        self.assertEqual(observations, [])
        self.assertIsNone(next_cursor)
        self.assertNotIn("parse_io_as_json", sdk.calls[0])

    def test_metrics_query_serializes_timezone_aware_iso_datetimes(self) -> None:
        sdk = MetricsSDK()
        collector = LangfuseCollectorClient(sdk_client=sdk)
        collector.get_prompt_analytics(
            PromptMetricsFilters(
                prompt_name="support-task",
                from_date=datetime(2026, 3, 16, 10, 0, 0, tzinfo=UTC),
                to_date=datetime(2026, 3, 16, 12, 0, 0),
                limit=10,
            )
        )
        query = sdk.queries[0]
        self.assertEqual(query["fromTimestamp"], "2026-03-16T10:00:00Z")
        self.assertEqual(query["toTimestamp"], "2026-03-16T12:00:00Z")


if __name__ == "__main__":
    unittest.main()
