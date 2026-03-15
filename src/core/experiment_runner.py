from __future__ import annotations

import asyncio
import json
import os
from datetime import UTC, datetime
from typing import Any, Protocol

from langfuse.batch_evaluation import EvaluatorInputs
from langfuse.experiment import Evaluation

from src.core.langfuse_client import LangfuseCollectorClient
from src.core.prompt_registry import PromptResolverService
from src.core.run_history import PostgresRunHistoryStore, RunHistoryStore, build_run_record_id
from src.schemas.experiment_runner import (
    AggregateMetricResult,
    CustomEvaluatorSpec,
    DatasetFetchResult,
    EvaluationScope,
    EvaluatorMetricSpec,
    ExperimentExecutionRequest,
    ExperimentExecutionResult,
    ExperimentItemResultView,
    ExperimentMode,
    ExperimentRunHistoryResult,
    ExperimentRunRecord,
    ExperimentRunStatus,
    JudgeMetricOutput,
    NormalizedDatasetItem,
    NormalizedEvaluationResult,
    PRESET_METRIC_RUBRICS,
    PromptPublishTarget,
    PromptResolutionRequest,
    PromptResolutionResult,
    PromptSource,
    PromptTarget,
    PromptType,
    PublishedPromptRequest,
    PublishedPromptResult,
    ResolvedPrompt,
)


class LLMGateway(Protocol):
    def generate_task_output(
        self,
        *,
        model_name: str,
        system_prompt: str | None,
        item_input: Any,
    ) -> Any: ...

    async def agenerate_task_output(
        self,
        *,
        model_name: str,
        system_prompt: str | None,
        item_input: Any,
    ) -> Any: ...

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
    ) -> JudgeMetricOutput: ...

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
    ) -> JudgeMetricOutput: ...


class PydanticAIGateway:
    def generate_task_output(
        self,
        *,
        model_name: str,
        system_prompt: str | None,
        item_input: Any,
    ) -> Any:
        from pydantic_ai import Agent

        self._require_model(model_name, env_var="EXPERIMENT_TASK_MODEL")
        agent = Agent(
            model_name,
            output_type=str,
            system_prompt=system_prompt or "You are a task runner. Produce the best answer for the given input.",
            retries=1,
        )
        return agent.run_sync(_stringify_payload(item_input)).output

    async def agenerate_task_output(
        self,
        *,
        model_name: str,
        system_prompt: str | None,
        item_input: Any,
    ) -> Any:
        from pydantic_ai import Agent

        self._require_model(model_name, env_var="EXPERIMENT_TASK_MODEL")
        agent = Agent(
            model_name,
            output_type=str,
            system_prompt=system_prompt or "You are a task runner. Produce the best answer for the given input.",
            retries=1,
        )
        return (await agent.run(_stringify_payload(item_input))).output

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
        from pydantic_ai import Agent

        self._require_model(model_name, env_var="EXPERIMENT_JUDGE_MODEL")
        rubric = (
            metric.rubric
            or PRESET_METRIC_RUBRICS.get(metric.name)
            or "Use the custom judge prompt to score this output."
        )
        system_prompt = (
            "You are an LLM evaluation judge. "
            "Return a score between 0.0 and 1.0 and a short comment. "
            "Higher is better. Be strict and consistent.\n\n"
            f"Global judge prompt:\n{judge_prompt}\n\n"
            f"Metric: {metric.name}\n"
            f"Rubric: {rubric}"
        )
        user_prompt = (
            f"Input:\n{_stringify_payload(item_input)}\n\n"
            f"Output:\n{_stringify_payload(output)}\n\n"
            f"Expected output:\n{_stringify_payload(expected_output)}\n\n"
            f"Metadata:\n{_stringify_payload(metadata or {})}"
        )
        agent = Agent(
            model_name,
            output_type=JudgeMetricOutput,
            system_prompt=system_prompt,
            retries=2,
        )
        return agent.run_sync(user_prompt).output

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
        from pydantic_ai import Agent

        self._require_model(model_name, env_var="EXPERIMENT_JUDGE_MODEL")
        rubric = (
            metric.rubric
            or PRESET_METRIC_RUBRICS.get(metric.name)
            or "Use the custom judge prompt to score this output."
        )
        system_prompt = (
            "You are an LLM evaluation judge. "
            "Return a score between 0.0 and 1.0 and a short comment. "
            "Higher is better. Be strict and consistent.\n\n"
            f"Global judge prompt:\n{judge_prompt}\n\n"
            f"Metric: {metric.name}\n"
            f"Rubric: {rubric}"
        )
        user_prompt = (
            f"Input:\n{_stringify_payload(item_input)}\n\n"
            f"Output:\n{_stringify_payload(output)}\n\n"
            f"Expected output:\n{_stringify_payload(expected_output)}\n\n"
            f"Metadata:\n{_stringify_payload(metadata or {})}"
        )
        agent = Agent(
            model_name,
            output_type=JudgeMetricOutput,
            system_prompt=system_prompt,
            retries=2,
        )
        return (await agent.run(user_prompt)).output

    @staticmethod
    def _require_model(model_name: str | None, *, env_var: str) -> None:
        if model_name:
            return
        raise ValueError(f"Model name is required. Set it in the UI or via {env_var}.")


class ExperimentRunnerService:
    def __init__(
        self,
        collector: LangfuseCollectorClient,
        llm_gateway: LLMGateway | None = None,
        prompt_resolver: PromptResolverService | None = None,
        history_store: RunHistoryStore | None = None,
    ) -> None:
        self.collector = collector
        self.llm_gateway = llm_gateway or PydanticAIGateway()
        self.prompt_resolver = prompt_resolver or PromptResolverService(collector)
        self.history_store = history_store or PostgresRunHistoryStore()

    def fetch_dataset_by_name(self, name: str) -> DatasetFetchResult:
        dataset = self.collector.get_dataset(name)
        items = [self._normalize_dataset_item(item) for item in dataset.items]
        warnings: list[str] = []
        if not items:
            warnings.append("Dataset bulundu ancak item icermiyor.")
        missing_source_count = sum(
            1 for item in items if not (item.source_trace_id or item.source_observation_id)
        )
        if missing_source_count == len(items) and items:
            warnings.append("Dataset item'larinda source trace veya source observation baglantisi yok.")
        elif missing_source_count:
            warnings.append(
                f"{missing_source_count} dataset item source trace/source observation baglantisi tasimiyor."
            )
        return DatasetFetchResult(
            dataset_id=dataset.id,
            dataset_name=dataset.name,
            description=dataset.description,
            metadata=dataset.metadata if isinstance(dataset.metadata, dict) else None,
            items=items,
            total_items=len(items),
            warnings=warnings,
        )

    def build_custom_evaluators(self, spec: CustomEvaluatorSpec) -> list[Any]:
        return [self._build_single_evaluator(spec=spec, metric=metric) for metric in spec.metrics]

    def resolve_prompt(self, request: PromptResolutionRequest) -> PromptResolutionResult:
        return self.prompt_resolver.resolve_prompt(request)

    def publish_prompt(self, request: PublishedPromptRequest) -> PublishedPromptResult:
        return self.prompt_resolver.publish_prompt(request)

    def list_prompts(
        self,
        *,
        name: str | None = None,
        label: str | None = None,
        tag: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        return self.prompt_resolver.list_prompts(name=name, label=label, tag=tag, limit=limit)

    def list_recent_experiment_runs(
        self,
        *,
        limit: int = 20,
        dataset_name: str | None = None,
        mode: ExperimentMode | None = None,
    ) -> ExperimentRunHistoryResult:
        return self.history_store.list_recent_runs(limit=limit, dataset_name=dataset_name, mode=mode)

    def run_prompt_experiment(self, request: ExperimentExecutionRequest) -> ExperimentExecutionResult:
        run_name = request.run_name or self._default_run_name(request.dataset_name, request.mode)
        try:
            dataset = self.collector.get_dataset(request.dataset_name)
            if not dataset.items:
                failed = self._build_failed_result(
                    request,
                    errors=["Dataset item icermedigi icin experiment calistirilamadi."],
                    run_name=run_name,
                )
                return self._persist_result(failed)

            request, prompt_warnings = self._resolve_request_prompts(request, require_task_prompt=True)
            evaluator_spec = CustomEvaluatorSpec(
                judge_prompt=request.resolved_judge_prompt.compiled_text,
                metrics=request.metrics,
                judge_model=request.judge_model or "",
                judge_prompt_name=request.resolved_judge_prompt.prompt_name,
                judge_prompt_label=request.resolved_judge_prompt.prompt_label,
                judge_prompt_version=request.resolved_judge_prompt.prompt_version,
                judge_prompt_fingerprint=request.resolved_judge_prompt.fingerprint,
                judge_prompt_source=request.resolved_judge_prompt.source,
            )
            evaluators = self.build_custom_evaluators(evaluator_spec)

            async def task(*, item: Any, **_: Any) -> Any:
                item_input = getattr(item, "input", None)
                return await self.llm_gateway.agenerate_task_output(
                    model_name=request.task_model or "",
                    system_prompt=request.resolved_task_prompt.compiled_text,
                    item_input=item_input,
                )

            result = self.collector.sdk_client.run_experiment(
                name=request.dataset_name,
                run_name=run_name,
                description=request.description,
                data=dataset.items,
                task=task,
                evaluators=evaluators,
                max_concurrency=request.max_concurrency,
                metadata=request.metadata,
            )
            normalized = self._normalize_prompt_experiment_result(
                dataset_name=request.dataset_name,
                run_name=run_name,
                description=request.description,
                result=result,
                task_prompt=request.resolved_task_prompt,
                judge_prompt=request.resolved_judge_prompt,
                task_model=request.task_model,
                judge_model=request.judge_model,
            )
            normalized.published_task_prompt = self._published_prompt_from_summary(
                task_prompt=request.resolved_task_prompt,
                target=PromptPublishTarget.TASK,
            )
            normalized.published_judge_prompt = self._published_prompt_from_summary(
                task_prompt=request.resolved_judge_prompt,
                target=PromptPublishTarget.JUDGE,
            )
            normalized.warnings.extend(prompt_warnings)
            return self._persist_result(normalized)
        except Exception as exc:
            failed = self._build_failed_result(request, errors=[str(exc)], run_name=run_name)
            return self._persist_result(failed)

    def run_dataset_reevaluation(self, request: ExperimentExecutionRequest) -> ExperimentExecutionResult:
        run_name = request.run_name or self._default_run_name(request.dataset_name, request.mode)
        try:
            dataset = self.fetch_dataset_by_name(request.dataset_name)
            request, prompt_warnings = self._resolve_request_prompts(request, require_task_prompt=False)
            metric_spec = CustomEvaluatorSpec(
                judge_prompt=request.resolved_judge_prompt.compiled_text,
                metrics=request.metrics,
                judge_model=request.judge_model or "",
                judge_prompt_name=request.resolved_judge_prompt.prompt_name,
                judge_prompt_label=request.resolved_judge_prompt.prompt_label,
                judge_prompt_version=request.resolved_judge_prompt.prompt_version,
                judge_prompt_fingerprint=request.resolved_judge_prompt.fingerprint,
                judge_prompt_source=request.resolved_judge_prompt.source,
            )
            evaluators = self.build_custom_evaluators(metric_spec)
            target_items, warnings = self._select_relevant_items(dataset.items, request.scope)
            if not target_items:
                failed = self._build_failed_result(
                    request,
                    errors=["Secilen scope icin source id tasiyan dataset item bulunamadi."],
                    warnings=prompt_warnings + warnings,
                    run_name=run_name,
                )
                return self._persist_result(failed)

            target_map = {
                self._scope_entity_id(item, request.scope): item
                for item in target_items
                if self._scope_entity_id(item, request.scope)
            }
            target_ids = [entity_id for entity_id in target_map if entity_id]
            filter_payload = self._build_batch_filter(target_ids)
            entity_snapshot_map: dict[str, dict[str, Any]] = {}

            def mapper(*, item: Any, **kwargs: Any) -> EvaluatorInputs:
                entity_id = getattr(item, "id", None)
                dataset_item = target_map.get(entity_id)
                combined_metadata = {}
                entity_metadata = getattr(item, "metadata", None)
                if isinstance(entity_metadata, dict):
                    combined_metadata.update(entity_metadata)
                if dataset_item and dataset_item.metadata:
                    combined_metadata["dataset_item_metadata"] = dataset_item.metadata
                if dataset_item:
                    combined_metadata["dataset_item_id"] = dataset_item.id
                if entity_id:
                    entity_snapshot_map[entity_id] = {
                        "input": getattr(item, "input", None),
                        "output": getattr(item, "output", None),
                        "trace_id": getattr(item, "trace_id", None),
                        "metadata": combined_metadata or None,
                    }
                return EvaluatorInputs(
                    input=getattr(item, "input", None),
                    output=getattr(item, "output", None),
                    expected_output=dataset_item.expected_output if dataset_item else None,
                    metadata=combined_metadata or None,
                )

            batch_result = self.collector.sdk_client.run_batched_evaluation(
                scope=request.scope.value,
                mapper=mapper,
                filter=filter_payload,
                evaluators=evaluators,
                max_items=len(target_ids),
                max_concurrency=request.max_concurrency,
                metadata={
                    **request.metadata,
                    "dataset_name": request.dataset_name,
                    "run_name": run_name,
                },
            )
            normalized = self._normalize_batch_result(
                dataset_name=request.dataset_name,
                run_name=run_name,
                scope=request.scope,
                batch_result=batch_result,
                target_map=target_map,
                entity_snapshot_map=entity_snapshot_map,
                warnings=warnings + prompt_warnings,
                judge_prompt=request.resolved_judge_prompt,
                judge_model=request.judge_model,
            )
            normalized.published_judge_prompt = self._published_prompt_from_summary(
                task_prompt=request.resolved_judge_prompt,
                target=PromptPublishTarget.JUDGE,
            )
            return self._persist_result(normalized)
        except Exception as exc:
            failed = self._build_failed_result(
                request,
                errors=[str(exc)],
                run_name=run_name,
            )
            return self._persist_result(failed)

    @staticmethod
    def _build_batch_filter(target_ids: list[str]) -> str | None:
        if not target_ids:
            return None
        if len(target_ids) == 1:
            return json.dumps(
                [
                    {
                        "type": "string",
                        "column": "id",
                        "operator": "=",
                        "value": target_ids[0],
                    }
                ]
            )
        return json.dumps(
            [
                {
                    "type": "stringOptions",
                    "column": "id",
                    "operator": "any of",
                    "value": target_ids,
                }
            ]
        )

    def _resolve_request_prompts(
        self,
        request: ExperimentExecutionRequest,
        *,
        require_task_prompt: bool,
    ) -> tuple[ExperimentExecutionRequest, list[str]]:
        warnings: list[str] = []
        task_prompt = request.resolved_task_prompt
        task_source = PromptSource.LANGFUSE_PROMPT if request.use_published_task_prompt else request.task_prompt_source
        judge_source = PromptSource.LANGFUSE_PROMPT if request.use_published_judge_prompt else request.judge_prompt_source
        if require_task_prompt or task_source == PromptSource.LANGFUSE_PROMPT or (request.task_system_prompt or "").strip():
            task_result = self.prompt_resolver.resolve_prompt(
                PromptResolutionRequest(
                    source=task_source,
                    target=PromptTarget.TASK,
                    prompt_name=request.task_prompt_name,
                    prompt_label=request.task_prompt_label,
                    prompt_version=request.task_prompt_version,
                    prompt_type=request.task_prompt_type,
                    custom_prompt=request.task_system_prompt,
                )
            )
            task_prompt = task_result.resolved_prompt
            warnings.extend(task_result.warnings)
        elif not request.resolved_task_prompt:
            task_prompt = ResolvedPrompt(
                source=PromptSource.CUSTOM_PROMPT,
                target=PromptTarget.TASK,
                prompt_type=request.task_prompt_type,
                compiled_text="",
                fingerprint=None,
            )

        judge_result = self.prompt_resolver.resolve_prompt(
            PromptResolutionRequest(
                source=judge_source,
                target=PromptTarget.JUDGE,
                prompt_name=request.judge_prompt_name,
                prompt_label=request.judge_prompt_label,
                prompt_version=request.judge_prompt_version,
                prompt_type=request.judge_prompt_type,
                custom_prompt=request.judge_prompt,
            )
        )
        warnings.extend(judge_result.warnings)
        data = request.model_dump()
        data["task_prompt_source"] = task_source
        data["judge_prompt_source"] = judge_source
        data["judge_model"] = request.judge_model or os.getenv("EXPERIMENT_JUDGE_MODEL")
        data["task_model"] = request.task_model or os.getenv("EXPERIMENT_TASK_MODEL")
        data["resolved_task_prompt"] = task_prompt.model_dump() if task_prompt else None
        data["resolved_judge_prompt"] = judge_result.resolved_prompt.model_dump()
        return ExperimentExecutionRequest.model_validate(data), warnings

    def _persist_result(self, result: ExperimentExecutionResult) -> ExperimentExecutionResult:
        record = self._to_run_record(result)
        if record is None:
            return result
        try:
            record_id = self.history_store.save_run(record)
            result.history_record_id = record_id
            if record_id:
                result.warnings.append(f"Run ozeti PostgreSQL history tablosuna kaydedildi: {record_id}")
            elif not self.history_store.is_enabled():
                result.warnings.append("Run history kaydi atlandi: PostgreSQL baglantisi tanimli degil.")
        except Exception as exc:
            result.warnings.append(f"Run history kaydedilemedi: {exc}")
        return result

    def _to_run_record(self, result: ExperimentExecutionResult) -> ExperimentRunRecord | None:
        if result.task_prompt_summary is None or result.judge_prompt_summary is None:
            return None
        return ExperimentRunRecord(
            id=build_run_record_id(),
            created_at=datetime.now(UTC),
            mode=result.mode,
            dataset_name=result.dataset_name,
            run_name=result.run_name,
            description=result.description,
            status=result.status,
            task_prompt_source=result.task_prompt_summary.source,
            task_prompt_name=result.task_prompt_summary.prompt_name,
            task_prompt_label=result.task_prompt_summary.prompt_label,
            task_prompt_version=result.task_prompt_summary.prompt_version,
            task_prompt_type=result.task_prompt_summary.prompt_type,
            task_prompt_fingerprint=result.task_prompt_summary.fingerprint,
            judge_prompt_source=result.judge_prompt_summary.source,
            judge_prompt_name=result.judge_prompt_summary.prompt_name,
            judge_prompt_label=result.judge_prompt_summary.prompt_label,
            judge_prompt_version=result.judge_prompt_summary.prompt_version,
            judge_prompt_type=result.judge_prompt_summary.prompt_type,
            judge_prompt_fingerprint=result.judge_prompt_summary.fingerprint,
            published_from_custom=bool(
                (result.published_task_prompt and result.published_task_prompt.source == PromptSource.CUSTOM_PROMPT)
                or (result.published_judge_prompt and result.published_judge_prompt.source == PromptSource.CUSTOM_PROMPT)
            ),
            published_at=datetime.now(UTC)
            if result.published_task_prompt or result.published_judge_prompt
            else None,
            task_model=result.raw_summary.get("task_model"),
            judge_model=result.raw_summary.get("judge_model"),
            metric_names=[metric.name for metric in result.aggregate_metrics] or list({evaluation.name for item in result.item_results for evaluation in item.evaluations}),
            aggregate_metrics=result.aggregate_metrics,
            processed_items=result.processed_items,
            failed_items=result.failed_items,
            dataset_run_id=result.dataset_run_id,
            dataset_run_url=result.dataset_run_url,
            warnings=result.warnings,
            errors=result.errors,
        )

    def _build_single_evaluator(
        self,
        *,
        spec: CustomEvaluatorSpec,
        metric: EvaluatorMetricSpec,
    ) -> Any:
        async def evaluator(
            *,
            input: Any,
            output: Any,
            expected_output: Any = None,
            metadata: dict[str, Any] | None = None,
            **_: Any,
        ) -> Evaluation:
            judged = await self.llm_gateway.aevaluate_metric(
                model_name=spec.judge_model,
                judge_prompt=spec.judge_prompt,
                metric=metric,
                item_input=input,
                output=output,
                expected_output=expected_output,
                metadata=metadata,
            )
            return Evaluation(
                name=metric.name,
                value=judged.score,
                comment=judged.comment,
                metadata={
                    "rubric": metric.rubric,
                    "is_custom": metric.is_custom,
                    "judge_prompt_name": spec.judge_prompt_name,
                    "judge_prompt_label": spec.judge_prompt_label,
                    "judge_prompt_version": spec.judge_prompt_version,
                    "judge_prompt_fingerprint": spec.judge_prompt_fingerprint,
                    "judge_prompt_source": spec.judge_prompt_source.value,
                },
            )

        return evaluator

    def _build_failed_result(
        self,
        request: ExperimentExecutionRequest,
        *,
        errors: list[str],
        warnings: list[str] | None = None,
        run_name: str | None = None,
    ) -> ExperimentExecutionResult:
        task_prompt_summary = request.resolved_task_prompt or self._fallback_prompt_summary(
            source=request.task_prompt_source,
            target=PromptTarget.TASK,
            prompt_name=request.task_prompt_name,
            prompt_label=request.task_prompt_label,
            prompt_version=request.task_prompt_version,
            prompt_type=request.task_prompt_type,
            raw_prompt=request.task_system_prompt,
        )
        judge_prompt_summary = request.resolved_judge_prompt or self._fallback_prompt_summary(
            source=request.judge_prompt_source,
            target=PromptTarget.JUDGE,
            prompt_name=request.judge_prompt_name,
            prompt_label=request.judge_prompt_label,
            prompt_version=request.judge_prompt_version,
            prompt_type=request.judge_prompt_type,
            raw_prompt=request.judge_prompt,
        )
        return ExperimentExecutionResult(
            mode=request.mode,
            dataset_name=request.dataset_name,
            run_name=run_name or request.run_name or self._default_run_name(request.dataset_name, request.mode),
            description=request.description,
            status=ExperimentRunStatus.FAILED,
            total_items=0,
            processed_items=0,
            failed_items=0,
            warnings=list(warnings or []),
            errors=errors,
            aggregate_metrics=[],
            item_results=[],
            evaluator_stats=[],
            raw_summary={
                "task_model": request.task_model or os.getenv("EXPERIMENT_TASK_MODEL"),
                "judge_model": request.judge_model or os.getenv("EXPERIMENT_JUDGE_MODEL"),
            },
            task_prompt_summary=task_prompt_summary,
            judge_prompt_summary=judge_prompt_summary,
        )

    @staticmethod
    def _published_prompt_from_summary(
        *,
        task_prompt: ResolvedPrompt | None,
        target: PromptPublishTarget,
    ) -> PublishedPromptResult | None:
        if not task_prompt or task_prompt.source != PromptSource.LANGFUSE_PROMPT or not task_prompt.prompt_name:
            return None
        return PublishedPromptResult(
            target=target,
            prompt_name=task_prompt.prompt_name,
            prompt_version=task_prompt.prompt_version,
            prompt_label=task_prompt.prompt_label,
            prompt_type=task_prompt.prompt_type or PromptType.TEXT,
            source=task_prompt.source,
            fingerprint=task_prompt.fingerprint,
            use_for_next_run=True,
        )

    @staticmethod
    def _normalize_dataset_item(item: Any) -> NormalizedDatasetItem:
        metadata = getattr(item, "metadata", None)
        return NormalizedDatasetItem(
            id=getattr(item, "id"),
            input=getattr(item, "input", None),
            expected_output=getattr(item, "expected_output", None),
            metadata=metadata if isinstance(metadata, dict) else None,
            status=str(getattr(item, "status", "")) or None,
            source_trace_id=getattr(item, "source_trace_id", None),
            source_observation_id=getattr(item, "source_observation_id", None),
            created_at=getattr(item, "created_at", None),
            updated_at=getattr(item, "updated_at", None),
        )

    def _normalize_prompt_experiment_result(
        self,
        *,
        dataset_name: str,
        run_name: str,
        description: str | None,
        result: Any,
        task_prompt: ResolvedPrompt,
        judge_prompt: ResolvedPrompt,
        task_model: str | None,
        judge_model: str | None,
    ) -> ExperimentExecutionResult:
        item_results = [self._normalize_item_result(item_result) for item_result in result.item_results]
        aggregate_metrics = self._aggregate_metrics_from_rows(item_results)
        return ExperimentExecutionResult(
            mode=ExperimentMode.PROMPT_RUNNER,
            dataset_name=dataset_name,
            run_name=run_name,
            description=description,
            status=ExperimentRunStatus.SUCCEEDED,
            dataset_run_id=getattr(result, "dataset_run_id", None),
            dataset_run_url=getattr(result, "dataset_run_url", None),
            total_items=len(item_results),
            processed_items=len(item_results),
            failed_items=0,
            warnings=[],
            aggregate_metrics=aggregate_metrics,
            item_results=item_results,
            raw_summary={
                "experiment_name": getattr(result, "name", dataset_name),
                "run_evaluations": [self._normalize_evaluation(ev).model_dump() for ev in getattr(result, "run_evaluations", [])],
                "task_model": task_model or os.getenv("EXPERIMENT_TASK_MODEL"),
                "judge_model": judge_model or os.getenv("EXPERIMENT_JUDGE_MODEL"),
            },
            task_prompt_summary=task_prompt,
            judge_prompt_summary=judge_prompt,
        )

    def _normalize_batch_result(
        self,
        *,
        dataset_name: str,
        run_name: str,
        scope: EvaluationScope,
        batch_result: Any,
        target_map: dict[str, NormalizedDatasetItem],
        entity_snapshot_map: dict[str, dict[str, Any]],
        warnings: list[str],
        judge_prompt: ResolvedPrompt,
        judge_model: str | None,
    ) -> ExperimentExecutionResult:
        rows: list[ExperimentItemResultView] = []
        aggregate_source: list[ExperimentItemResultView] = []
        for entity_id, evaluations in getattr(batch_result, "item_evaluations", {}).items():
            dataset_item = target_map.get(entity_id)
            entity_snapshot = entity_snapshot_map.get(entity_id, {})
            row = ExperimentItemResultView(
                dataset_item_id=dataset_item.id if dataset_item else None,
                entity_id=entity_id,
                entity_type=scope.value,
                trace_id=entity_snapshot.get("trace_id") or (entity_id if scope == EvaluationScope.TRACES else None),
                input=dataset_item.input if dataset_item else None,
                expected_output=dataset_item.expected_output if dataset_item else None,
                output=entity_snapshot.get("output"),
                evaluations=[self._normalize_evaluation(ev) for ev in evaluations],
            )
            rows.append(row)
            aggregate_source.append(row)
        evaluator_stats = [
            {
                "name": getattr(stat, "name", None),
                "successful_runs": getattr(stat, "successful_runs", None),
                "failed_runs": getattr(stat, "failed_runs", None),
                "total_scores_created": getattr(stat, "total_scores_created", None),
            }
            for stat in getattr(batch_result, "evaluator_stats", [])
        ]
        error_summary = getattr(batch_result, "error_summary", {}) or {}
        errors = [f"{key}: {value}" for key, value in error_summary.items()]
        return ExperimentExecutionResult(
            mode=ExperimentMode.REEVALUATE_EXISTING,
            dataset_name=dataset_name,
            run_name=run_name,
            status=ExperimentRunStatus.FAILED if errors else ExperimentRunStatus.SUCCEEDED,
            total_items=getattr(batch_result, "total_items_fetched", len(rows)),
            processed_items=getattr(batch_result, "total_items_processed", len(rows)),
            failed_items=getattr(batch_result, "total_items_failed", 0),
            warnings=warnings,
            errors=errors,
            aggregate_metrics=self._aggregate_metrics_from_rows(aggregate_source),
            item_results=rows,
            evaluator_stats=evaluator_stats,
            raw_summary={
                "completed": getattr(batch_result, "completed", None),
                "duration_seconds": getattr(batch_result, "duration_seconds", None),
                "total_scores_created": getattr(batch_result, "total_scores_created", None),
                "failed_item_ids": getattr(batch_result, "failed_item_ids", []),
                "has_more_items": getattr(batch_result, "has_more_items", False),
                "task_model": None,
                "judge_model": judge_model or os.getenv("EXPERIMENT_JUDGE_MODEL"),
            },
            task_prompt_summary=ResolvedPrompt(
                source=PromptSource.CUSTOM_PROMPT,
                target=PromptTarget.TASK,
                compiled_text="reevaluation",
                fingerprint=None,
            ),
            judge_prompt_summary=judge_prompt,
        )

    @staticmethod
    def _normalize_item_result(item_result: Any) -> ExperimentItemResultView:
        item = getattr(item_result, "item", None)
        evaluations = [ExperimentRunnerService._normalize_evaluation(ev) for ev in getattr(item_result, "evaluations", [])]
        return ExperimentItemResultView(
            dataset_item_id=getattr(item, "id", None),
            entity_id=getattr(item, "id", None),
            entity_type="dataset_item",
            trace_id=getattr(item_result, "trace_id", None),
            dataset_run_id=getattr(item_result, "dataset_run_id", None),
            input=getattr(item, "input", None),
            expected_output=getattr(item, "expected_output", None),
            output=getattr(item_result, "output", None),
            evaluations=evaluations,
        )

    @staticmethod
    def _normalize_evaluation(evaluation: Any) -> NormalizedEvaluationResult:
        metadata = getattr(evaluation, "metadata", None)
        return NormalizedEvaluationResult(
            name=getattr(evaluation, "name"),
            value=getattr(evaluation, "value", None),
            comment=getattr(evaluation, "comment", None),
            metadata=metadata if isinstance(metadata, dict) else None,
        )

    @staticmethod
    def _aggregate_metrics_from_rows(rows: list[ExperimentItemResultView]) -> list[AggregateMetricResult]:
        grouped: dict[str, list[float]] = {}
        for row in rows:
            for evaluation in row.evaluations:
                if isinstance(evaluation.value, (int, float)) and not isinstance(evaluation.value, bool):
                    grouped.setdefault(evaluation.name, []).append(float(evaluation.value))
        metrics = [
            AggregateMetricResult(
                name=name,
                average_score=(sum(values) / len(values)) if values else None,
                count=len(values),
            )
            for name, values in grouped.items()
        ]
        metrics.sort(key=lambda metric: metric.name)
        return metrics

    @staticmethod
    def _select_relevant_items(
        items: list[NormalizedDatasetItem],
        scope: EvaluationScope,
    ) -> tuple[list[NormalizedDatasetItem], list[str]]:
        selected: list[NormalizedDatasetItem] = []
        skipped = 0
        for item in items:
            entity_id = ExperimentRunnerService._scope_entity_id(item, scope)
            if entity_id:
                selected.append(item)
            else:
                skipped += 1
        warnings: list[str] = []
        if skipped:
            warnings.append(
                f"{skipped} dataset item secilen scope icin uygun source id icermedigi icin atlandi."
            )
        return selected, warnings

    @staticmethod
    def _scope_entity_id(item: NormalizedDatasetItem, scope: EvaluationScope) -> str | None:
        if scope == EvaluationScope.TRACES:
            return item.source_trace_id
        return item.source_observation_id

    @staticmethod
    def _default_run_name(dataset_name: str, mode: ExperimentMode) -> str:
        suffix = datetime.now().strftime("%Y%m%d%H%M")
        return f"{dataset_name}-{mode.value}-{suffix}"

    @staticmethod
    def _fallback_prompt_summary(
        *,
        source: PromptSource,
        target: PromptTarget,
        prompt_name: str | None,
        prompt_label: str | None,
        prompt_version: int | None,
        prompt_type: Any,
        raw_prompt: str | None,
    ) -> ResolvedPrompt:
        compiled_text = (raw_prompt or "").strip()
        return ResolvedPrompt(
            source=source,
            target=target,
            prompt_name=prompt_name,
            prompt_label=prompt_label,
            prompt_version=prompt_version,
            prompt_type=prompt_type,
            compiled_text=compiled_text,
            messages=[],
            variables=[],
            is_fallback=(source == PromptSource.LANGFUSE_PROMPT),
            fingerprint=None,
        )



def _stringify_payload(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=True, indent=2, default=str)
    except TypeError:
        return str(value)


_DEFAULT_SERVICE: ExperimentRunnerService | None = None


def _get_service() -> ExperimentRunnerService:
    global _DEFAULT_SERVICE
    if _DEFAULT_SERVICE is None:
        _DEFAULT_SERVICE = ExperimentRunnerService(LangfuseCollectorClient())
    return _DEFAULT_SERVICE



def fetch_dataset_by_name(name: str) -> DatasetFetchResult:
    return _get_service().fetch_dataset_by_name(name)



def build_custom_evaluators(spec: CustomEvaluatorSpec) -> list[Any]:
    return _get_service().build_custom_evaluators(spec)



def resolve_prompt(request: PromptResolutionRequest) -> PromptResolutionResult:
    return _get_service().resolve_prompt(request)


def publish_prompt(request: PublishedPromptRequest) -> PublishedPromptResult:
    return _get_service().publish_prompt(request)


def list_prompts(
    *,
    name: str | None = None,
    label: str | None = None,
    tag: str | None = None,
    limit: int = 50,
) -> list[dict[str, Any]]:
    return _get_service().list_prompts(name=name, label=label, tag=tag, limit=limit)



def run_prompt_experiment(request: ExperimentExecutionRequest) -> ExperimentExecutionResult:
    return _get_service().run_prompt_experiment(request)



def run_dataset_reevaluation(request: ExperimentExecutionRequest) -> ExperimentExecutionResult:
    return _get_service().run_dataset_reevaluation(request)


def run_llm_judge_on_existing_results(request: ExperimentExecutionRequest) -> ExperimentExecutionResult:
    return _get_service().run_dataset_reevaluation(request)



def list_recent_experiment_runs(
    limit: int = 20,
    dataset_name: str | None = None,
    mode: ExperimentMode | None = None,
) -> ExperimentRunHistoryResult:
    return _get_service().list_recent_experiment_runs(limit=limit, dataset_name=dataset_name, mode=mode)
