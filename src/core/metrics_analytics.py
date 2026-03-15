from __future__ import annotations

from collections import defaultdict
from datetime import date, datetime, time, timedelta
from statistics import mean
from typing import Any

from src.core.langfuse_client import LangfuseCollectorClient
from src.core.run_history import PostgresRunHistoryStore, RunHistoryStore
from src.schemas.metrics_analytics import (
    PromptAnalyticsDataset,
    PromptMetricsFilters,
    PromptMetricsSummary,
    PromptRunRow,
    PromptTrendRow,
    PromptVersionRow,
    ToolEvaluatorRow,
    ToolJudgeDataset,
    ToolJudgeFilters,
    ToolMatchStrategy,
    ToolMetricsRow,
    ToolMetricsSummary,
)


class MetricsAnalyticsService:
    def __init__(
        self,
        collector: LangfuseCollectorClient,
        history_store: RunHistoryStore | None = None,
    ) -> None:
        self.collector = collector
        self.history_store = history_store or PostgresRunHistoryStore()

    def get_prompt_analytics_dataset(
        self,
        filters: PromptMetricsFilters | None = None,
    ) -> PromptAnalyticsDataset:
        filters = filters or _default_prompt_filters()
        version_rows = self._build_prompt_version_rows(self.collector.get_prompt_analytics(filters))
        trend_rows = self._build_prompt_trend_rows(self.collector.get_run_prompt_analytics(filters))
        run_rows = self._build_prompt_run_rows(filters, version_rows)
        warnings: list[str] = []
        if not version_rows:
            warnings.append("Secilen filtrelerle prompt metrics verisi bulunamadi.")
        if not trend_rows:
            warnings.append("Prompt trend verisi bulunamadi.")
        if not run_rows:
            warnings.append("Prompt analytics icin eslesen run history kaydi bulunamadi.")
        return PromptAnalyticsDataset(
            summary=self._build_prompt_summary(version_rows),
            version_rows=version_rows,
            trend_rows=trend_rows,
            run_rows=run_rows,
            warnings=warnings,
        )

    def get_tool_judge_dataset(
        self,
        filters: ToolJudgeFilters | None = None,
    ) -> ToolJudgeDataset:
        filters = filters or _default_tool_filters()
        warnings: list[str] = []

        raw_name_rows = self.collector.get_tool_observation_metrics(filters, match_field="name")
        strategy = ToolMatchStrategy.NAME
        raw_rows = raw_name_rows
        if filters.tool_names and not raw_name_rows:
            raw_rows = self.collector.get_tool_observation_metrics(filters, match_field="tags")
            strategy = ToolMatchStrategy.TAGS
            if raw_rows:
                warnings.append("Observation name ile eslesme bulunamadi; tags fallback kullanildi.")

        tool_rows = self._build_tool_rows(raw_rows, filters.tool_names, strategy)
        observation_names = sorted(
            {
                observation_name
                for row in tool_rows
                for observation_name in row.observation_names
            }
        )

        numeric_rows = self.collector.get_tool_evaluator_metrics(
            filters,
            observation_names=observation_names,
            categorical=False,
        )
        categorical_rows = self.collector.get_tool_evaluator_metrics(
            filters,
            observation_names=observation_names,
            categorical=True,
        )
        evaluator_rows = self._build_tool_evaluator_rows(
            numeric_rows=numeric_rows,
            categorical_rows=categorical_rows,
            tool_rows=tool_rows,
            matched_by=strategy,
        )

        if not tool_rows:
            warnings.append("Secilen tool filtreleriyle observation metrics verisi bulunamadi.")
        if not evaluator_rows:
            warnings.append("Secilen evaluator filtreleriyle score metrics verisi bulunamadi.")

        return ToolJudgeDataset(
            summary=self._build_tool_summary(tool_rows),
            tool_rows=tool_rows,
            evaluator_rows=evaluator_rows,
            warnings=warnings,
        )

    def _build_prompt_version_rows(self, raw_rows: list[dict[str, Any]]) -> list[PromptVersionRow]:
        grouped: dict[tuple[str | None, int | None], dict[str, Any]] = {}
        for row in raw_rows:
            prompt_name = _first_value(row, "promptName")
            prompt_version = _coerce_int(_first_value(row, "promptVersion"))
            key = (prompt_name, prompt_version)
            bucket = grouped.setdefault(
                key,
                {
                    "count": 0,
                    "latency_weighted_sum": 0.0,
                    "cost": 0.0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0,
                    "models": set(),
                },
            )
            count = _coerce_int(_metric_value(row, "count", "count")) or 0
            latency = _coerce_float(_metric_value(row, "latency", "avg"))
            bucket["count"] += count
            bucket["cost"] += _coerce_float(_metric_value(row, "totalCost", "sum")) or 0.0
            bucket["input_tokens"] += _coerce_int(_metric_value(row, "inputTokens", "sum")) or 0
            bucket["output_tokens"] += _coerce_int(_metric_value(row, "outputTokens", "sum")) or 0
            bucket["total_tokens"] += _coerce_int(_metric_value(row, "totalTokens", "sum")) or 0
            if latency is not None and count:
                bucket["latency_weighted_sum"] += latency * count
            model_name = _first_value(row, "providedModelName")
            if model_name:
                bucket["models"].add(model_name)

        result = [
            PromptVersionRow(
                prompt_name=prompt_name,
                prompt_version=prompt_version,
                observation_count=data["count"],
                avg_latency_ms=(data["latency_weighted_sum"] / data["count"]) if data["count"] else None,
                total_cost=data["cost"],
                input_tokens=data["input_tokens"],
                output_tokens=data["output_tokens"],
                total_tokens=data["total_tokens"],
                model_name=next(iter(data["models"])) if len(data["models"]) == 1 else None,
            )
            for (prompt_name, prompt_version), data in grouped.items()
        ]
        result.sort(key=lambda row: ((row.prompt_name or ""), -(row.prompt_version or 0)))
        return result

    def _build_prompt_trend_rows(self, raw_rows: list[dict[str, Any]]) -> list[PromptTrendRow]:
        result: list[PromptTrendRow] = []
        for row in raw_rows:
            result.append(
                PromptTrendRow(
                    bucket=str(
                        _first_value(
                            row,
                            "time",
                            "timestamp",
                            "startTime",
                            "startTimeMonth",
                            "bucket",
                        )
                        or "unknown"
                    ),
                    prompt_name=_first_value(row, "promptName"),
                    prompt_version=_coerce_int(_first_value(row, "promptVersion")),
                    observation_count=_coerce_int(_metric_value(row, "count", "count")) or 0,
                    avg_latency_ms=_coerce_float(_metric_value(row, "latency", "avg")),
                    total_cost=_coerce_float(_metric_value(row, "totalCost", "sum")) or 0.0,
                    total_tokens=_coerce_int(_metric_value(row, "totalTokens", "sum")) or 0,
                )
            )
        result.sort(key=lambda row: row.bucket)
        return result

    def _build_prompt_run_rows(
        self,
        filters: PromptMetricsFilters,
        version_rows: list[PromptVersionRow],
    ) -> list[PromptRunRow]:
        history = self.history_store.list_recent_runs(limit=filters.limit)
        version_map = {
            (row.prompt_name, row.prompt_version): row
            for row in version_rows
        }
        result: list[PromptRunRow] = []
        for record in history.records:
            if filters.dataset_name and record.dataset_name != filters.dataset_name:
                continue
            if filters.run_name and filters.run_name.lower() not in record.run_name.lower():
                continue
            if filters.prompt_name and not (
                record.task_prompt_name == filters.prompt_name
                or record.judge_prompt_name == filters.prompt_name
            ):
                continue
            if filters.prompt_versions and not (
                record.task_prompt_version in filters.prompt_versions
                or record.judge_prompt_version in filters.prompt_versions
            ):
                continue
            matching_rows = []
            for key in {
                (record.task_prompt_name, record.task_prompt_version),
                (record.judge_prompt_name, record.judge_prompt_version),
            }:
                row = version_map.get(key)
                if row is not None:
                    matching_rows.append(row)
            matched_total_cost = sum(row.total_cost for row in matching_rows) if matching_rows else None
            matched_total_tokens = sum(row.total_tokens for row in matching_rows) if matching_rows else None
            matched_avg_latency_ms = (
                sum((row.avg_latency_ms or 0.0) * row.observation_count for row in matching_rows)
                / sum(row.observation_count for row in matching_rows)
                if matching_rows and sum(row.observation_count for row in matching_rows)
                else None
            )
            result.append(
                PromptRunRow(
                    created_at=record.created_at,
                    run_name=record.run_name,
                    dataset_name=record.dataset_name,
                    status=record.status.value,
                    mode=record.mode.value,
                    task_prompt_name=record.task_prompt_name,
                    task_prompt_version=record.task_prompt_version,
                    judge_prompt_name=record.judge_prompt_name,
                    judge_prompt_version=record.judge_prompt_version,
                    task_model=record.task_model,
                    judge_model=record.judge_model,
                    processed_items=record.processed_items,
                    failed_items=record.failed_items,
                    matched_total_cost=matched_total_cost,
                    matched_avg_latency_ms=matched_avg_latency_ms,
                    matched_total_tokens=matched_total_tokens,
                    dataset_run_url=record.dataset_run_url,
                )
            )
        result.sort(key=lambda row: row.created_at, reverse=True)
        return result

    @staticmethod
    def _build_prompt_summary(version_rows: list[PromptVersionRow]) -> PromptMetricsSummary:
        total_observations = sum(row.observation_count for row in version_rows)
        return PromptMetricsSummary(
            total_observations=total_observations,
            total_cost=sum(row.total_cost for row in version_rows),
            avg_latency_ms=(
                sum((row.avg_latency_ms or 0.0) * row.observation_count for row in version_rows)
                / total_observations
                if total_observations
                else None
            ),
            total_input_tokens=sum(row.input_tokens for row in version_rows),
            total_output_tokens=sum(row.output_tokens for row in version_rows),
            total_tokens=sum(row.total_tokens for row in version_rows),
        )

    def _build_tool_rows(
        self,
        raw_rows: list[dict[str, Any]],
        requested_tool_names: list[str],
        matched_by: ToolMatchStrategy,
    ) -> list[ToolMetricsRow]:
        grouped: dict[str, dict[str, Any]] = {}
        requested_lookup = {name.lower(): name for name in requested_tool_names}
        for row in raw_rows:
            observation_name = _first_value(row, "name")
            tags = _as_list(_first_value(row, "tags"))
            matched_tools: list[str]
            if matched_by == ToolMatchStrategy.NAME:
                if not observation_name:
                    continue
                matched_tools = [requested_lookup.get(str(observation_name).lower(), str(observation_name))]
            else:
                matched_tools = [
                    requested_lookup.get(tag.lower(), tag)
                    for tag in tags
                    if tag.lower() in requested_lookup
                ]
                if not matched_tools and requested_tool_names:
                    continue
                if not matched_tools and observation_name:
                    matched_tools = [str(observation_name)]

            count = _coerce_int(_metric_value(row, "count", "count")) or 0
            latency = _coerce_float(_metric_value(row, "latency", "avg"))
            total_cost = _coerce_float(_metric_value(row, "totalCost", "sum")) or 0.0
            input_tokens = _coerce_int(_metric_value(row, "inputTokens", "sum")) or 0
            output_tokens = _coerce_int(_metric_value(row, "outputTokens", "sum")) or 0
            total_tokens = _coerce_int(_metric_value(row, "totalTokens", "sum")) or 0
            model_name = _first_value(row, "providedModelName")
            for tool_name in matched_tools:
                bucket = grouped.setdefault(
                    tool_name,
                    {
                        "count": 0,
                        "latency_weighted_sum": 0.0,
                        "cost": 0.0,
                        "input_tokens": 0,
                        "output_tokens": 0,
                        "total_tokens": 0,
                        "model_counts": defaultdict(int),
                        "observation_names": set(),
                    },
                )
                bucket["count"] += count
                bucket["cost"] += total_cost
                bucket["input_tokens"] += input_tokens
                bucket["output_tokens"] += output_tokens
                bucket["total_tokens"] += total_tokens
                if latency is not None and count:
                    bucket["latency_weighted_sum"] += latency * count
                if model_name:
                    bucket["model_counts"][model_name] += count
                if observation_name:
                    bucket["observation_names"].add(str(observation_name))

        result = [
            ToolMetricsRow(
                tool_name=tool_name,
                matched_by=matched_by,
                observation_count=data["count"],
                avg_latency_ms=(data["latency_weighted_sum"] / data["count"]) if data["count"] else None,
                total_cost=data["cost"],
                input_tokens=data["input_tokens"],
                output_tokens=data["output_tokens"],
                total_tokens=data["total_tokens"],
                top_model=max(data["model_counts"], key=data["model_counts"].get) if data["model_counts"] else None,
                observation_names=sorted(data["observation_names"]),
            )
            for tool_name, data in grouped.items()
        ]
        result.sort(key=lambda row: (-row.observation_count, row.tool_name))
        return result

    def _build_tool_evaluator_rows(
        self,
        *,
        numeric_rows: list[dict[str, Any]],
        categorical_rows: list[dict[str, Any]],
        tool_rows: list[ToolMetricsRow],
        matched_by: ToolMatchStrategy,
    ) -> list[ToolEvaluatorRow]:
        observation_to_tools: dict[str, list[str]] = defaultdict(list)
        for row in tool_rows:
            for observation_name in row.observation_names:
                observation_to_tools[observation_name].append(row.tool_name)

        grouped: dict[tuple[str, str, str | None, str | None], dict[str, Any]] = {}
        for row in numeric_rows:
            observation_name = _first_value(row, "observationName")
            evaluator_name = _first_value(row, "name")
            if not observation_name or not evaluator_name:
                continue
            tools = observation_to_tools.get(str(observation_name), [])
            count = _coerce_int(_metric_value(row, "count", "count")) or 0
            avg_score = _coerce_float(_metric_value(row, "value", "avg"))
            for tool_name in tools:
                key = (tool_name, str(evaluator_name), _first_value(row, "source"), _first_value(row, "dataType"))
                bucket = grouped.setdefault(
                    key,
                    {"count": 0, "score_weighted_sum": 0.0, "categorical_breakdown": {}},
                )
                bucket["count"] += count
                if avg_score is not None and count:
                    bucket["score_weighted_sum"] += avg_score * count

        for row in categorical_rows:
            observation_name = _first_value(row, "observationName")
            evaluator_name = _first_value(row, "name")
            category = _first_value(row, "stringValue")
            if not observation_name or not evaluator_name:
                continue
            tools = observation_to_tools.get(str(observation_name), [])
            count = _coerce_int(_metric_value(row, "count", "count")) or 0
            for tool_name in tools:
                key = (tool_name, str(evaluator_name), _first_value(row, "source"), "CATEGORICAL")
                bucket = grouped.setdefault(
                    key,
                    {"count": 0, "score_weighted_sum": 0.0, "categorical_breakdown": {}},
                )
                bucket["count"] += count
                if category:
                    bucket["categorical_breakdown"][str(category)] = (
                        bucket["categorical_breakdown"].get(str(category), 0) + count
                    )

        result = [
            ToolEvaluatorRow(
                tool_name=tool_name,
                evaluator_name=evaluator_name,
                matched_by=matched_by,
                average_score=(data["score_weighted_sum"] / data["count"]) if data["count"] and data["score_weighted_sum"] else None,
                count=data["count"],
                score_source=source,
                data_type=data_type,
                categorical_breakdown=data["categorical_breakdown"],
            )
            for (tool_name, evaluator_name, source, data_type), data in grouped.items()
        ]
        result.sort(key=lambda row: (row.tool_name, row.evaluator_name, row.data_type or ""))
        return result

    @staticmethod
    def _build_tool_summary(tool_rows: list[ToolMetricsRow]) -> ToolMetricsSummary:
        observation_counts = [row.observation_count for row in tool_rows]
        return ToolMetricsSummary(
            tool_count=len(tool_rows),
            observation_count=sum(observation_counts),
            total_cost=sum(row.total_cost for row in tool_rows),
            avg_latency_ms=mean([row.avg_latency_ms for row in tool_rows if row.avg_latency_ms is not None])
            if any(row.avg_latency_ms is not None for row in tool_rows)
            else None,
            total_tokens=sum(row.total_tokens for row in tool_rows),
        )


def _first_value(row: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in row:
            return row.get(key)
    return None


def _metric_value(row: dict[str, Any], measure: str, aggregation: str) -> Any:
    candidate_keys = [
        f"{measure}_{aggregation}",
        f"{aggregation}_{measure}",
        f"{aggregation}({measure})",
        f"{measure}:{aggregation}",
        f"{aggregation}{measure[0].upper()}{measure[1:]}",
        f"{measure}{aggregation[0].upper()}{aggregation[1:]}",
    ]
    if measure == "count":
        candidate_keys.extend(["count", "row_count"])
    for key in candidate_keys:
        if key in row:
            return row.get(key)
    return None


def _coerce_float(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return None


def _coerce_int(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return None


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(value)]


def _default_prompt_filters() -> PromptMetricsFilters:
    today = date.today()
    return PromptMetricsFilters(
        from_date=datetime.combine(today - timedelta(days=14), time.min),
        to_date=datetime.combine(today, time.max),
        limit=100,
    )


def _default_tool_filters() -> ToolJudgeFilters:
    today = date.today()
    return ToolJudgeFilters(
        from_date=datetime.combine(today - timedelta(days=14), time.min),
        to_date=datetime.combine(today, time.max),
        limit=100,
    )


_DEFAULT_METRICS_SERVICE: MetricsAnalyticsService | None = None


def _get_service() -> MetricsAnalyticsService:
    global _DEFAULT_METRICS_SERVICE
    if _DEFAULT_METRICS_SERVICE is None:
        _DEFAULT_METRICS_SERVICE = MetricsAnalyticsService(LangfuseCollectorClient())
    return _DEFAULT_METRICS_SERVICE


def get_prompt_analytics_dataset(
    filters: PromptMetricsFilters | None = None,
) -> PromptAnalyticsDataset:
    return _get_service().get_prompt_analytics_dataset(filters)


def get_tool_judge_dataset(
    filters: ToolJudgeFilters | None = None,
) -> ToolJudgeDataset:
    return _get_service().get_tool_judge_dataset(filters)
