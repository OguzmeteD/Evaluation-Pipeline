from __future__ import annotations

import asyncio
import json
import unittest
from dataclasses import dataclass
from datetime import datetime, timedelta
from types import SimpleNamespace
from typing import Any

from src.core.experiment_runner import ExperimentRunnerService
from src.core.prompt_registry import PromptResolverService
from src.core.run_history import InMemoryRunHistoryStore
from src.frontend.pages.experiment_studio import (
    available_scopes,
    build_metric_specs_from_form,
    count_missing_scope_ids,
    validate_run_form,
)
from src.schemas.endpoint_runner import EndpointConfig, EndpointExecutionResult as EndpointExecutionOutput, EndpointItemResult, EndpointResponseMapping
from src.schemas.openreward_runner import OpenRewardConfig, OpenRewardExecutionResult as OpenRewardExecutionOutput, OpenRewardItemResult
from src.schemas.experiment_runner import (
    DatasetFetchResult,
    EvaluationScope,
    EvaluatorMetricSpec,
    ExperimentExecutionRequest,
    ExperimentMode,
    ExperimentRunStatus,
    JudgeMetricOutput,
    PromptPublishTarget,
    PromptResolutionRequest,
    PromptSource,
    PromptTarget,
    PromptType,
    PublishedPromptRequest,
)


@dataclass
class FakeDatasetItem:
    id: str
    input: Any
    expected_output: Any = None
    metadata: dict[str, Any] | None = None
    status: str = "ACTIVE"
    source_trace_id: str | None = None
    source_observation_id: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None


@dataclass
class FakeDataset:
    id: str
    name: str
    description: str | None
    metadata: dict[str, Any] | None
    items: list[FakeDatasetItem]


@dataclass
class FakeEvaluation:
    name: str
    value: float
    comment: str | None = None
    metadata: dict[str, Any] | None = None


@dataclass
class FakeItemResult:
    item: FakeDatasetItem
    output: Any
    evaluations: list[FakeEvaluation]
    trace_id: str | None
    dataset_run_id: str | None


@dataclass
class FakeExperimentResult:
    name: str
    run_name: str
    description: str | None
    item_results: list[FakeItemResult]
    run_evaluations: list[FakeEvaluation]
    dataset_run_id: str | None
    dataset_run_url: str | None


@dataclass
class FakeEvaluatorStat:
    name: str
    successful_runs: int
    failed_runs: int
    total_scores_created: int


@dataclass
class FakeBatchResult:
    total_items_fetched: int
    total_items_processed: int
    total_items_failed: int
    total_scores_created: int
    total_composite_scores_created: int
    total_evaluations_failed: int
    evaluator_stats: list[FakeEvaluatorStat]
    resume_token: Any
    completed: bool
    duration_seconds: float
    failed_item_ids: list[str]
    error_summary: dict[str, int]
    has_more_items: bool
    item_evaluations: dict[str, list[FakeEvaluation]]


class FakeTextPromptClient:
    def __init__(self, name: str, prompt: str, version: int = 1, is_fallback: bool = False) -> None:
        self.name = name
        self.prompt = prompt
        self.version = version
        self.is_fallback = is_fallback
        self.variables = ["customer"] if "{{customer}}" in prompt else []

    def compile(self) -> str:
        return self.prompt


class FakeChatPromptClient:
    def __init__(self, name: str, prompt: list[dict[str, Any]], version: int = 1) -> None:
        self.name = name
        self.prompt = prompt
        self.version = version
        self.is_fallback = False
        self.variables = []

    def compile(self) -> list[dict[str, Any]]:
        return self.prompt


class FakeLLMGateway:
    def __init__(self) -> None:
        self.task_calls: list[dict[str, Any]] = []
        self.eval_calls: list[dict[str, Any]] = []

    def generate_task_output(self, *, model_name: str, system_prompt: str | None, item_input: Any) -> Any:
        self.task_calls.append({"model_name": model_name, "system_prompt": system_prompt, "item_input": item_input})
        return f"generated::{item_input}"

    async def agenerate_task_output(self, *, model_name: str, system_prompt: str | None, item_input: Any) -> Any:
        return self.generate_task_output(model_name=model_name, system_prompt=system_prompt, item_input=item_input)

    def evaluate_metric(
        self,
        *,
        model_name: str,
        judge_prompt: str,
        metric: EvaluatorMetricSpec,
        item_input: Any,
        output: Any,
        expected_output: Any,
        metadata: dict[str, Any] | None,
    ) -> JudgeMetricOutput:
        self.eval_calls.append({"model_name": model_name, "judge_prompt": judge_prompt, "metric": metric.name})
        return JudgeMetricOutput(score=0.75, comment=f"{metric.name} ok")

    async def aevaluate_metric(
        self,
        *,
        model_name: str,
        judge_prompt: str,
        metric: EvaluatorMetricSpec,
        item_input: Any,
        output: Any,
        expected_output: Any,
        metadata: dict[str, Any] | None,
    ) -> JudgeMetricOutput:
        return self.evaluate_metric(
            model_name=model_name,
            judge_prompt=judge_prompt,
            metric=metric,
            item_input=item_input,
            output=output,
            expected_output=expected_output,
            metadata=metadata,
        )


class FakeEndpointRunner:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def run_endpoint_execution(self, *, items: list[Any], request: Any) -> EndpointExecutionOutput:
        self.calls.append({"items": items, "request": request})
        return EndpointExecutionOutput(
            processed_items=len(items),
            failed_items=1,
            item_results=[
                EndpointItemResult(
                    dataset_item_id=items[0].id,
                    request_payload={"input": items[0].input},
                    raw_response={"result": {"text": "ok"}},
                    output="ok",
                    trace_id="trace-endpoint",
                    observation_id="obs-endpoint",
                    status_code=200,
                    latency_ms=123.0,
                    error=None,
                    response_metadata={"source": "endpoint"},
                ),
                EndpointItemResult(
                    dataset_item_id=items[1].id,
                    request_payload={"input": items[1].input},
                    raw_response="boom",
                    output=None,
                    trace_id=None,
                    observation_id=None,
                    status_code=500,
                    latency_ms=50.0,
                    error="HTTP 500",
                    response_metadata=None,
                ),
            ],
            warnings=["1 endpoint request failed"],
            summary={"success_rate": 0.5, "avg_latency_ms": 86.5},
        )


class FakeOpenRewardRunner:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def run_openreward_execution(self, *, items: list[Any], request: Any) -> OpenRewardExecutionOutput:
        self.calls.append({"items": items, "request": request})
        return OpenRewardExecutionOutput(
            processed_items=len(items),
            failed_items=1,
            item_results=[
                OpenRewardItemResult(
                    dataset_item_id=items[0].id,
                    session_id="sid-1",
                    prompt_blocks=[{"type": "text", "text": "Solve the task"}],
                    available_tools=["submit"],
                    tool_input={"answer": items[0].expected_output},
                    tool_output={"blocks": [{"type": "text", "text": "Correct"}], "reward": 1.0, "finished": True},
                    output="Correct",
                    reward=1.0,
                    finished=True,
                    rollout_id="rollout-1",
                    rollout_url="https://openreward.ai/rollout/rollout-1",
                    latency_ms=80.0,
                    metadata={"judge": "pass"},
                ),
                OpenRewardItemResult(
                    dataset_item_id=items[1].id,
                    session_id="sid-2",
                    prompt_blocks=[],
                    available_tools=["submit"],
                    tool_input={"answer": items[1].expected_output},
                    output=None,
                    reward=None,
                    finished=None,
                    rollout_id=None,
                    rollout_url=None,
                    latency_ms=40.0,
                    error="tool missing",
                ),
            ],
            warnings=["1 openreward item failed"],
            summary={"success_rate": 0.5, "average_reward": 1.0, "finished_rate": 1.0, "avg_latency_ms": 60.0},
        )


class FakeCollector:
    def __init__(self, dataset: FakeDataset, prompts: dict[tuple[str, str], Any] | None = None) -> None:
        self._dataset = dataset
        self._prompts = prompts or {}
        self.run_experiment_calls: list[dict[str, Any]] = []
        self.run_batched_calls: list[dict[str, Any]] = []
        self.mapped_inputs: list[dict[str, Any]] = []
        self.created_prompts: list[dict[str, Any]] = []
        self.updated_prompts: list[dict[str, Any]] = []
        self.sdk_client = SimpleNamespace(
            run_experiment=self._run_experiment,
            run_batched_evaluation=self._run_batched_evaluation,
        )

    def get_dataset(self, name: str) -> FakeDataset:
        if name != self._dataset.name:
            raise ValueError("dataset not found")
        return self._dataset

    def get_prompt(
        self,
        name: str,
        *,
        version: int | None = None,
        label: str | None = None,
        type: str = "text",
        fallback: Any = None,
    ) -> Any:
        key = (name, type)
        if key not in self._prompts:
            raise ValueError("prompt not found")
        return self._prompts[key]

    def create_or_update_prompt_version(
        self,
        *,
        name: str,
        prompt: str | list[dict[str, Any]],
        prompt_type: str = "text",
        label: str | None = None,
        commit_message: str | None = None,
        tags: list[str] | None = None,
    ) -> Any:
        self.created_prompts.append(
            {
                "name": name,
                "prompt": prompt,
                "prompt_type": prompt_type,
                "label": label,
                "commit_message": commit_message,
                "tags": tags,
            }
        )
        version = len(self.created_prompts)
        client = FakeTextPromptClient(name=name, prompt=prompt if isinstance(prompt, str) else json.dumps(prompt), version=version)
        self._prompts[(name, prompt_type)] = client
        if label:
            self.updated_prompts.append({"name": name, "version": version, "labels": [label]})
        return client

    def _run_experiment(self, **kwargs: Any) -> FakeExperimentResult:
        self.run_experiment_calls.append(kwargs)
        item_results = []
        for item in kwargs["data"]:
            output = kwargs["task"](item=item)
            if asyncio.iscoroutine(output):
                output = asyncio.run(output)
            evaluations = []
            for evaluator in kwargs["evaluators"]:
                evaluation = evaluator(
                    input=item.input,
                    output=output,
                    expected_output=item.expected_output,
                    metadata=item.metadata,
                )
                if asyncio.iscoroutine(evaluation):
                    evaluation = asyncio.run(evaluation)
                evaluations.append(evaluation)
            item_results.append(
                FakeItemResult(
                    item=item,
                    output=output,
                    evaluations=evaluations,
                    trace_id=f"trace-{item.id}",
                    dataset_run_id="dataset-run-1",
                )
            )
        return FakeExperimentResult(
            name=kwargs["name"],
            run_name=kwargs["run_name"],
            description=kwargs.get("description"),
            item_results=item_results,
            run_evaluations=[],
            dataset_run_id="dataset-run-1",
            dataset_run_url="https://langfuse.local/runs/dataset-run-1",
        )

    def _run_batched_evaluation(self, **kwargs: Any) -> FakeBatchResult:
        self.run_batched_calls.append(kwargs)
        relevant_items = [
            item
            for item in self._dataset.items
            if item.source_observation_id
        ]
        if relevant_items:
            mapped = kwargs["mapper"](
                item=SimpleNamespace(
                    id=relevant_items[0].source_observation_id,
                    input=relevant_items[0].input,
                    output=f"observed::{relevant_items[0].input}",
                    trace_id="trace-obs-1",
                    metadata={"trace": "obs-trace"},
                ),
                extra_context={"source": "test"},
            )
            self.mapped_inputs.append(
                {
                    "input": mapped.input,
                    "output": mapped.output,
                    "expected_output": mapped.expected_output,
                    "metadata": mapped.metadata,
                }
            )
        return FakeBatchResult(
            total_items_fetched=1,
            total_items_processed=1,
            total_items_failed=0,
            total_scores_created=len(kwargs["evaluators"]),
            total_composite_scores_created=0,
            total_evaluations_failed=0,
            evaluator_stats=[FakeEvaluatorStat(name="helpfulness", successful_runs=1, failed_runs=0, total_scores_created=1)],
            resume_token=None,
            completed=True,
            duration_seconds=0.4,
            failed_item_ids=[],
            error_summary={},
            has_more_items=False,
            item_evaluations={"obs-1": [FakeEvaluation(name="helpfulness", value=0.9, comment="good")]},
        )


class ExperimentRunnerServiceTest(unittest.TestCase):
    def setUp(self) -> None:
        dataset = FakeDataset(
            id="dataset-1",
            name="support-dataset",
            description="Support prompts",
            metadata={"team": "support"},
            items=[
                FakeDatasetItem(
                    id="item-1",
                    input={"question": "How do I reset my password?"},
                    expected_output="Use the reset link.",
                    metadata={"priority": "high"},
                    source_trace_id="trace-1",
                    source_observation_id="obs-1",
                ),
                FakeDatasetItem(
                    id="item-2",
                    input="Explain pricing",
                    expected_output="Pricing depends on plan.",
                ),
            ],
        )
        prompts = {
            ("task-support", "text"): FakeTextPromptClient("task-support", "Support task prompt"),
            ("judge-support", "text"): FakeTextPromptClient("judge-support", "Judge support answers"),
            (
                "judge-chat",
                "chat",
            ): FakeChatPromptClient(
                "judge-chat",
                [
                    {"role": "system", "content": "Evaluate the answer carefully."},
                    {"role": "user", "content": "Use rubric strictly."},
                ],
            ),
        }
        self.collector = FakeCollector(dataset, prompts=prompts)
        self.gateway = FakeLLMGateway()
        self.endpoint_runner = FakeEndpointRunner()
        self.openreward_runner = FakeOpenRewardRunner()
        self.history_store = InMemoryRunHistoryStore()
        self.prompt_resolver = PromptResolverService(self.collector)
        self.service = ExperimentRunnerService(
            self.collector,
            llm_gateway=self.gateway,
            endpoint_runner=self.endpoint_runner,
            openreward_runner=self.openreward_runner,
            prompt_resolver=self.prompt_resolver,
            history_store=self.history_store,
        )

    def test_fetch_dataset_by_name_normalizes_items(self) -> None:
        result = self.service.fetch_dataset_by_name("support-dataset")
        self.assertIsInstance(result, DatasetFetchResult)
        self.assertEqual(result.total_items, 2)
        self.assertEqual(result.items[0].source_observation_id, "obs-1")
        self.assertTrue(result.warnings)

    def test_resolve_text_prompt_from_langfuse(self) -> None:
        result = self.service.resolve_prompt(
            PromptResolutionRequest(
                source=PromptSource.LANGFUSE_PROMPT,
                target=PromptTarget.TASK,
                prompt_name="task-support",
                prompt_type=PromptType.TEXT,
            )
        )
        self.assertTrue(result.found)
        self.assertEqual(result.resolved_prompt.compiled_text, "Support task prompt")
        self.assertEqual(result.resolved_prompt.source, PromptSource.LANGFUSE_PROMPT)

    def test_resolve_chat_prompt_normalizes_messages(self) -> None:
        result = self.service.resolve_prompt(
            PromptResolutionRequest(
                source=PromptSource.LANGFUSE_PROMPT,
                target=PromptTarget.JUDGE,
                prompt_name="judge-chat",
                prompt_type=PromptType.CHAT,
            )
        )
        self.assertEqual(result.resolved_prompt.prompt_type, PromptType.CHAT)
        self.assertEqual(len(result.resolved_prompt.messages), 2)
        self.assertIn("[system] Evaluate the answer carefully.", result.resolved_prompt.compiled_text)

    def test_prompt_fallback_to_custom_when_langfuse_missing(self) -> None:
        result = self.service.resolve_prompt(
            PromptResolutionRequest(
                source=PromptSource.LANGFUSE_PROMPT,
                target=PromptTarget.JUDGE,
                prompt_name="missing-prompt",
                prompt_type=PromptType.TEXT,
                custom_prompt="Manual judge prompt",
            )
        )
        self.assertFalse(result.found)
        self.assertEqual(result.resolved_prompt.source, PromptSource.CUSTOM_PROMPT)
        self.assertTrue(result.resolved_prompt.is_fallback)

    def test_build_metric_specs_supports_preset_and_custom(self) -> None:
        metrics = build_metric_specs_from_form(["helpfulness", "correctness"], "brand_safety", "Avoid risky advice")
        self.assertEqual([metric.name for metric in metrics], ["helpfulness", "correctness", "brand_safety"])
        self.assertTrue(metrics[-1].is_custom)

    def test_run_prompt_experiment_with_prompt_names_and_history(self) -> None:
        request = ExperimentExecutionRequest(
            dataset_name="support-dataset",
            mode=ExperimentMode.PROMPT_RUNNER,
            judge_prompt=None,
            judge_model="judge-model",
            task_model="task-model",
            metrics=[EvaluatorMetricSpec(name="helpfulness", rubric="Be helpful")],
            run_name="support-run",
            max_concurrency=3,
            task_prompt_source=PromptSource.LANGFUSE_PROMPT,
            task_prompt_name="task-support",
            task_prompt_type=PromptType.TEXT,
            judge_prompt_source=PromptSource.LANGFUSE_PROMPT,
            judge_prompt_name="judge-support",
            judge_prompt_type=PromptType.TEXT,
        )
        result = self.service.run_prompt_experiment(request)
        call = self.collector.run_experiment_calls[0]
        self.assertEqual(call["run_name"], "support-run")
        self.assertEqual(call["max_concurrency"], 3)
        self.assertEqual(self.gateway.task_calls[0]["system_prompt"], "Support task prompt")
        self.assertEqual(self.gateway.eval_calls[0]["judge_prompt"], "Judge support answers")
        self.assertEqual(result.processed_items, 2)
        self.assertEqual(result.status, ExperimentRunStatus.SUCCEEDED)
        self.assertEqual(len(self.history_store.records), 1)
        history_record = self.history_store.records[0]
        payload = history_record.model_dump()
        self.assertNotIn("compiled_text", payload)
        self.assertIsNotNone(history_record.task_prompt_fingerprint)
        self.assertIsNotNone(result.history_record_id)

    def test_run_prompt_experiment_uses_custom_fallback_when_prompt_missing(self) -> None:
        request = ExperimentExecutionRequest(
            dataset_name="support-dataset",
            mode=ExperimentMode.PROMPT_RUNNER,
            judge_prompt="Manual judge prompt",
            judge_model="judge-model",
            task_system_prompt="Manual task prompt",
            task_model="task-model",
            metrics=[EvaluatorMetricSpec(name="helpfulness", rubric="Be helpful")],
            task_prompt_source=PromptSource.LANGFUSE_PROMPT,
            task_prompt_name="missing-task-prompt",
            task_prompt_type=PromptType.TEXT,
            judge_prompt_source=PromptSource.CUSTOM_PROMPT,
        )
        result = self.service.run_prompt_experiment(request)
        self.assertEqual(result.status, ExperimentRunStatus.SUCCEEDED)
        self.assertEqual(self.gateway.task_calls[0]["system_prompt"], "Manual task prompt")
        self.assertTrue(result.warnings)

    def test_run_dataset_reevaluation_builds_scope_filter_and_history(self) -> None:
        request = ExperimentExecutionRequest(
            dataset_name="support-dataset",
            mode=ExperimentMode.REEVALUATE_EXISTING,
            judge_prompt="Manual judge prompt",
            judge_model="judge-model",
            metrics=[EvaluatorMetricSpec(name="helpfulness", rubric="Be helpful")],
            scope=EvaluationScope.OBSERVATIONS,
            run_name="rejudge-run",
            judge_prompt_source=PromptSource.CUSTOM_PROMPT,
        )
        result = self.service.run_dataset_reevaluation(request)
        call = self.collector.run_batched_calls[0]
        self.assertEqual(call["scope"], "observations")
        self.assertEqual(
            json.loads(call["filter"]),
            [{"type": "string", "column": "id", "operator": "=", "value": "obs-1"}],
        )
        self.assertEqual(result.processed_items, 1)
        self.assertTrue(result.warnings)
        self.assertEqual(result.item_results[0].entity_id, "obs-1")
        self.assertEqual(result.item_results[0].trace_id, "trace-obs-1")
        self.assertEqual(result.item_results[0].output, "observed::{'question': 'How do I reset my password?'}")
        self.assertEqual(self.collector.mapped_inputs[0]["expected_output"], "Use the reset link.")
        self.assertEqual(self.collector.mapped_inputs[0]["metadata"]["dataset_item_id"], "item-1")
        self.assertEqual(self.collector.mapped_inputs[0]["metadata"]["trace"], "obs-trace")
        self.assertEqual(len(self.history_store.records), 1)

    def test_publish_prompt_creates_new_version_and_returns_metadata(self) -> None:
        result = self.service.publish_prompt(
            PublishedPromptRequest(
                target=PromptPublishTarget.JUDGE,
                prompt_name="judge-support",
                prompt_type=PromptType.TEXT,
                prompt_text="Judge support answers strictly.",
                label="production",
                source=PromptSource.CUSTOM_PROMPT,
            )
        )
        self.assertEqual(result.prompt_name, "judge-support")
        self.assertEqual(result.prompt_version, 1)
        self.assertEqual(result.prompt_label, "production")
        self.assertEqual(self.collector.created_prompts[0]["prompt"], "Judge support answers strictly.")

    def test_run_prompt_experiment_uses_published_prompt_metadata(self) -> None:
        self.collector.create_or_update_prompt_version(
            name="task-published",
            prompt="Published task prompt",
            prompt_type="text",
            label="production",
        )
        self.collector.create_or_update_prompt_version(
            name="judge-published",
            prompt="Published judge prompt",
            prompt_type="text",
            label="production",
        )
        request = ExperimentExecutionRequest(
            dataset_name="support-dataset",
            mode=ExperimentMode.PROMPT_RUNNER,
            judge_model="judge-model",
            task_model="task-model",
            metrics=[EvaluatorMetricSpec(name="helpfulness", rubric="Be helpful")],
            task_prompt_source=PromptSource.CUSTOM_PROMPT,
            judge_prompt_source=PromptSource.CUSTOM_PROMPT,
            task_prompt_name="task-published",
            judge_prompt_name="judge-published",
            use_published_task_prompt=True,
            use_published_judge_prompt=True,
        )
        result = self.service.run_prompt_experiment(request)
        self.assertEqual(result.status, ExperimentRunStatus.SUCCEEDED)
        self.assertEqual(self.gateway.task_calls[0]["system_prompt"], "Published task prompt")
        self.assertEqual(self.gateway.eval_calls[0]["judge_prompt"], "Published judge prompt")
        self.assertIsNotNone(result.published_task_prompt)
        self.assertEqual(result.published_task_prompt.prompt_name, "task-published")
        self.assertIsNotNone(result.published_judge_prompt)
        self.assertEqual(result.item_results[0].evaluations[0].metadata["judge_prompt_name"], "judge-published")
        self.assertEqual(self.history_store.records[0].task_prompt_name, "task-published")

    def test_build_batch_filter_uses_string_options_for_multiple_ids(self) -> None:
        filter_payload = self.service._build_batch_filter(["obs-1", "obs-2"])
        self.assertEqual(
            json.loads(filter_payload),
            [{"type": "stringOptions", "column": "id", "operator": "any of", "value": ["obs-1", "obs-2"]}],
        )

    def test_list_recent_runs_supports_limit_and_filters(self) -> None:
        now = datetime.now()
        request1 = ExperimentExecutionRequest(
            dataset_name="support-dataset",
            mode=ExperimentMode.REEVALUATE_EXISTING,
            judge_prompt="Manual judge prompt",
            judge_model="judge-model",
            metrics=[EvaluatorMetricSpec(name="helpfulness")],
            scope=EvaluationScope.OBSERVATIONS,
            run_name="run-1",
            judge_prompt_source=PromptSource.CUSTOM_PROMPT,
        )
        request2 = ExperimentExecutionRequest(
            dataset_name="support-dataset",
            mode=ExperimentMode.PROMPT_RUNNER,
            judge_prompt="Manual judge prompt",
            judge_model="judge-model",
            task_system_prompt="Manual task prompt",
            task_model="task-model",
            metrics=[EvaluatorMetricSpec(name="correctness")],
            run_name="run-2",
            task_prompt_source=PromptSource.CUSTOM_PROMPT,
            judge_prompt_source=PromptSource.CUSTOM_PROMPT,
        )
        self.service.run_dataset_reevaluation(request1)
        self.service.run_prompt_experiment(request2)
        history = self.service.list_recent_experiment_runs(limit=1, mode=ExperimentMode.PROMPT_RUNNER)
        self.assertEqual(history.total_runs, 1)
        self.assertEqual(len(history.records), 1)
        self.assertEqual(history.records[0].mode, ExperimentMode.PROMPT_RUNNER)

    def test_available_scopes_and_missing_scope_count(self) -> None:
        dataset = self.service.fetch_dataset_by_name("support-dataset")
        scopes = available_scopes(dataset.items)
        self.assertIn(EvaluationScope.OBSERVATIONS.value, scopes)
        self.assertIn(EvaluationScope.TRACES.value, scopes)
        self.assertEqual(count_missing_scope_ids(dataset.items, EvaluationScope.OBSERVATIONS), 1)

    def test_validate_run_form_requires_resolved_prompts(self) -> None:
        errors = validate_run_form(
            dataset=None,
            mode=ExperimentMode.PROMPT_RUNNER,
            metrics=[],
            task_prompt={"is_ready": False},
            judge_prompt={"is_ready": False},
            task_model=None,
            judge_model=None,
            endpoint_config=None,
            enable_endpoint_judging=False,
            openreward_config=None,
            enable_openreward_judging=False,
        )
        self.assertTrue(errors)

    def test_validate_run_form_allows_endpoint_without_judge(self) -> None:
        errors = validate_run_form(
            dataset=SimpleNamespace(),
            mode=ExperimentMode.ENDPOINT_RUN,
            metrics=[],
            task_prompt={"is_ready": False},
            judge_prompt={"is_ready": False},
            task_model=None,
            judge_model=None,
            endpoint_config=EndpointConfig(url="https://example.com/run"),
            enable_endpoint_judging=False,
            openreward_config=None,
            enable_openreward_judging=False,
        )
        self.assertEqual(errors, [])

    def test_validate_run_form_allows_openreward_without_judge(self) -> None:
        errors = validate_run_form(
            dataset=SimpleNamespace(),
            mode=ExperimentMode.OPENREWARD_RUN,
            metrics=[],
            task_prompt={"is_ready": False},
            judge_prompt={"is_ready": False},
            task_model=None,
            judge_model=None,
            endpoint_config=None,
            enable_endpoint_judging=False,
            openreward_config=OpenRewardConfig(environment_name="owner/env", tool_name="submit"),
            enable_openreward_judging=False,
        )
        self.assertEqual(errors, [])

    def test_run_endpoint_evaluation_persists_history_and_metrics(self) -> None:
        result = self.service.run_endpoint_evaluation(
            ExperimentExecutionRequest(
                dataset_name="support-dataset",
                mode=ExperimentMode.ENDPOINT_RUN,
                metrics=[EvaluatorMetricSpec(name="helpfulness")],
                run_name="endpoint-run",
                endpoint_config=EndpointConfig(url="https://example.com/run", method="POST"),
                endpoint_response_mapping=EndpointResponseMapping(response_type="json"),
                enable_endpoint_judging=False,
            )
        )
        self.assertEqual(result.mode, ExperimentMode.ENDPOINT_RUN)
        self.assertEqual(result.processed_items, 2)
        self.assertEqual(result.failed_items, 1)
        self.assertEqual(result.item_results[0].status_code, 200)
        self.assertEqual(result.item_results[0].request_payload, {"input": {"question": "How do I reset my password?"}})
        self.assertTrue(any(metric.name == "success_rate" for metric in result.aggregate_metrics))
        self.assertEqual(self.history_store.records[0].endpoint_url, "https://example.com/run")
        self.assertEqual(self.history_store.records[0].mode, ExperimentMode.ENDPOINT_RUN)

    def test_run_openreward_evaluation_persists_history_and_metrics(self) -> None:
        result = self.service.run_openreward_evaluation(
            ExperimentExecutionRequest(
                dataset_name="support-dataset",
                mode=ExperimentMode.OPENREWARD_RUN,
                metrics=[EvaluatorMetricSpec(name="correctness")],
                run_name="openreward-run",
                openreward_config=OpenRewardConfig(environment_name="owner/env", tool_name="submit"),
                enable_openreward_judging=False,
            )
        )
        self.assertEqual(result.mode, ExperimentMode.OPENREWARD_RUN)
        self.assertEqual(result.processed_items, 2)
        self.assertEqual(result.failed_items, 1)
        self.assertEqual(result.item_results[0].entity_type, "openreward_session")
        self.assertEqual(result.item_results[0].request_payload, {"answer": "Use the reset link."})
        self.assertTrue(any(metric.name == "average_reward" for metric in result.aggregate_metrics))
        self.assertEqual(self.history_store.records[0].openreward_environment_name, "owner/env")
        self.assertEqual(self.history_store.records[0].openreward_tool_name, "submit")
        self.assertEqual(self.history_store.records[0].mode, ExperimentMode.OPENREWARD_RUN)


if __name__ == "__main__":
    unittest.main()
