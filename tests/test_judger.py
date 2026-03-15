from __future__ import annotations

import unittest
from datetime import datetime, timedelta
from typing import Any

from src.core.judger import LangfuseJudgeService
from src.schemas.evaluation_dataset import JudgeDatasetFilters, PromptSource


class FakeCollector:
    def __init__(
        self,
        *,
        traces: list[dict[str, Any]],
        observations: list[dict[str, Any]],
        scores: list[dict[str, Any]],
    ) -> None:
        self._traces = traces
        self._observations = observations
        self._scores = scores

    def list_traces(self, filters: JudgeDatasetFilters) -> list[dict[str, Any]]:
        return self._traces

    def list_observations(
        self, filters: JudgeDatasetFilters
    ) -> tuple[list[dict[str, Any]], str | None]:
        return self._observations, None

    def list_scores(self, filters: JudgeDatasetFilters) -> list[dict[str, Any]]:
        return self._scores


class LangfuseJudgeServiceTest(unittest.TestCase):
    def setUp(self) -> None:
        base_time = datetime(2026, 3, 8, 12, 0, 0)
        self.trace = {
            "id": "trace-1",
            "name": "support-flow",
            "session_id": "session-1",
            "timestamp": base_time,
            "input": {
                "messages": [
                    {"role": "system", "content": "Fallback system prompt"},
                    {"role": "user", "content": "Need help"},
                ]
            },
            "output": {"summary": "Trace level summary"},
            "metadata": {"experiment_id": "exp-1"},
            "latency": 321.0,
            "total_cost": 0.33,
        }
        self.observation = {
            "id": "obs-1",
            "trace_id": "trace-1",
            "name": "model-call",
            "type": "generation",
            "start_time": base_time,
            "end_time": base_time + timedelta(seconds=2),
            "model": "claude-judge",
            "input": {
                "messages": [
                    {"role": "system", "content": "Official system prompt"},
                    {"role": "user", "content": "How do I reset my password?"},
                ]
            },
            "output": {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "Use the reset password link on the sign-in page.",
                        }
                    }
                ]
            },
            "usage": {"input_tokens": 10, "output_tokens": 12},
            "metadata": {"experiment_id": "exp-1"},
            "latency": 200.0,
            "calculated_total_cost": 0.12,
        }
        self.trace_score = {
            "id": "score-trace-1",
            "trace_id": "trace-1",
            "name": "trace_quality",
            "value": 0.9,
            "comment": "Strong overall result",
            "source": "LLM_AS_A_JUDGE",
            "metadata": {"judge_name": "trace-judge"},
            "created_at": base_time,
            "updated_at": base_time,
        }
        self.observation_score = {
            "id": "score-obs-1",
            "trace_id": "trace-1",
            "observation_id": "obs-1",
            "name": "helpfulness",
            "value": 0.7,
            "comment": "Mostly useful",
            "source": "LLM_AS_A_JUDGE",
            "metadata": {"judge_name": "helpfulness-judge"},
            "created_at": base_time,
            "updated_at": base_time,
        }

    def test_prefers_observation_prompt_over_trace_fallback(self) -> None:
        service = LangfuseJudgeService(
            FakeCollector(
                traces=[self.trace],
                observations=[self.observation],
                scores=[self.observation_score],
            )
        )

        dataset = service.get_evaluation_dataset()

        self.assertEqual(dataset.rows[0].system_prompt, "Official system prompt")
        self.assertEqual(dataset.rows[0].prompt_source, PromptSource.OBSERVATION_INPUT)

    def test_generation_extraction_and_score_binding(self) -> None:
        service = LangfuseJudgeService(
            FakeCollector(
                traces=[self.trace],
                observations=[self.observation],
                scores=[self.observation_score, self.trace_score],
            )
        )

        dataset = service.get_evaluation_dataset()

        self.assertEqual(
            dataset.rows[0].generation_text,
            "Use the reset password link on the sign-in page.",
        )
        self.assertEqual(len(dataset.rows[0].judge_scores), 1)
        self.assertEqual(dataset.rows[0].judge_scores[0].score_name, "helpfulness")
        self.assertEqual(len(dataset.traces[0].trace_scores), 1)

    def test_trace_summary_aggregates_scores_and_deduplicates(self) -> None:
        duplicated_score = dict(self.observation_score)
        service = LangfuseJudgeService(
            FakeCollector(
                traces=[self.trace],
                observations=[self.observation],
                scores=[self.observation_score, duplicated_score, self.trace_score],
            )
        )

        dataset = service.get_evaluation_dataset()

        self.assertEqual(dataset.meta.counts.observation_scores, 1)
        self.assertAlmostEqual(dataset.traces[0].avg_score or 0.0, 0.8, places=3)
        self.assertEqual(dataset.traces[0].observation_count, 1)

    def test_trace_input_prompt_fallback_when_observation_prompt_missing(self) -> None:
        observation = dict(self.observation)
        observation["input"] = {"messages": [{"role": "user", "content": "Only user input"}]}
        service = LangfuseJudgeService(
            FakeCollector(
                traces=[self.trace],
                observations=[observation],
                scores=[self.observation_score],
            )
        )

        dataset = service.get_evaluation_dataset()

        self.assertEqual(dataset.rows[0].system_prompt, "Fallback system prompt")
        self.assertEqual(dataset.rows[0].prompt_source, PromptSource.TRACE_INPUT)

    def test_missing_prompt_is_renderable_and_warned(self) -> None:
        trace = dict(self.trace)
        trace["input"] = {"messages": [{"role": "user", "content": "No system prompt"}]}
        observation = dict(self.observation)
        observation["input"] = {"messages": [{"role": "user", "content": "Question only"}]}
        service = LangfuseJudgeService(
            FakeCollector(
                traces=[trace],
                observations=[observation],
                scores=[self.observation_score],
            )
        )

        dataset = service.get_evaluation_dataset()

        self.assertIsNone(dataset.rows[0].system_prompt)
        self.assertTrue(dataset.meta.warnings)


if __name__ == "__main__":
    unittest.main()
