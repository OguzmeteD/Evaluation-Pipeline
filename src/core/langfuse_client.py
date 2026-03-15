from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import httpx
from langfuse import Langfuse

from src.schemas.evaluation_dataset import JudgeDatasetFilters
from src.schemas.metrics_analytics import PromptMetricsFilters, ToolJudgeFilters


@dataclass(slots=True)
class LangfuseConfig:
    public_key: str | None
    secret_key: str | None
    host: str | None
    project_id: str | None = None
    default_judge_names: tuple[str, ...] = ()
    default_score_names: tuple[str, ...] = ()
    timeout_seconds: int = 60
    score_page_size: int = 50
    api_retries: int = 2

    @classmethod
    def from_env(cls) -> "LangfuseConfig":
        def _split_csv(value: str | None) -> tuple[str, ...]:
            if not value:
                return ()
            return tuple(item.strip() for item in value.split(",") if item.strip())

        return cls(
            public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
            secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
            host=os.getenv("LANGFUSE_HOST"),
            project_id=os.getenv("LANGFUSE_PROJECT_ID"),
            default_judge_names=_split_csv(os.getenv("DEFAULT_JUDGE_NAMES")),
            default_score_names=_split_csv(os.getenv("DEFAULT_SCORE_NAMES")),
            timeout_seconds=max(10, int(os.getenv("LANGFUSE_TIMEOUT_SECONDS", "60"))),
            score_page_size=max(10, min(int(os.getenv("LANGFUSE_SCORE_PAGE_SIZE", "50")), 100)),
            api_retries=max(0, int(os.getenv("LANGFUSE_API_RETRIES", "2"))),
        )

    def build_client(self) -> Langfuse:
        if not self.public_key or not self.secret_key:
            raise ValueError(
                "LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY must be configured."
            )

        kwargs: dict[str, Any] = {
            "public_key": self.public_key,
            "secret_key": self.secret_key,
        }
        if self.host:
            kwargs["host"] = self.host
        kwargs["timeout"] = self.timeout_seconds

        return Langfuse(**kwargs)


class LangfuseCollectorClient:
    """Thin wrapper around the Langfuse SDK with local filtering helpers."""

    def __init__(
        self,
        sdk_client: Langfuse | None = None,
        config: LangfuseConfig | None = None,
    ) -> None:
        self.config = config or LangfuseConfig.from_env()
        self.sdk_client = sdk_client or self.config.build_client()

    def list_traces(self, filters: JudgeDatasetFilters) -> list[dict[str, Any]]:
        page = self._page_from_cursor(filters.cursor)
        response = self.sdk_client.api.trace.list(
            page=page,
            limit=filters.limit,
            session_id=filters.session_ids[0] if len(filters.session_ids) == 1 else None,
            from_timestamp=filters.from_date,
            to_timestamp=filters.to_date,
        )
        traces = [self._to_dict(trace) for trace in response.data]
        return self._filter_traces_locally(traces, filters)

    def list_observations(
        self, filters: JudgeDatasetFilters
    ) -> tuple[list[dict[str, Any]], str | None]:
        response = self.sdk_client.api.observations_v_2.get_many(
            limit=filters.limit,
            cursor=filters.cursor,
            from_start_time=filters.from_date,
            to_start_time=filters.to_date,
            trace_id=filters.trace_ids[0] if len(filters.trace_ids) == 1 else None,
        )
        observations = [self._to_dict(item) for item in response.data]
        filtered = self._filter_observations_locally(observations, filters)
        next_cursor = getattr(response.meta, "next_cursor", None)
        return filtered, next_cursor

    def list_scores(self, filters: JudgeDatasetFilters) -> list[dict[str, Any]]:
        page = self._page_from_cursor(filters.cursor) or 1
        score_names = filters.score_names or list(self.config.default_score_names)
        trace_id = filters.trace_ids[0] if len(filters.trace_ids) == 1 else None
        page_size = max(1, min(filters.limit, self.config.score_page_size, 100))
        collected: list[dict[str, Any]] = []
        while len(collected) < max(filters.limit, 1):
            response = self._score_get_with_retry(
                page=page,
                limit=page_size,
                name=score_names[0] if len(score_names) == 1 else None,
                session_id=filters.session_ids[0] if len(filters.session_ids) == 1 else None,
                trace_id=trace_id,
                from_timestamp=filters.from_date,
                to_timestamp=filters.to_date,
            )
            scores = [self._to_dict(score) for score in response.data]
            collected.extend(self._filter_scores_locally(scores, filters))
            if len(scores) < page_size:
                break
            page += 1
        return collected[: filters.limit]

    def _score_get_with_retry(self, **kwargs: Any) -> Any:
        attempts = self.config.api_retries + 1
        last_exc: Exception | None = None
        for attempt in range(attempts):
            try:
                return self.sdk_client.api.score_v_2.get(
                    request_options={"timeout_in_seconds": self.config.timeout_seconds},
                    **kwargs,
                )
            except httpx.ReadTimeout as exc:
                last_exc = exc
                if attempt == attempts - 1:
                    raise
            except Exception:
                raise
        if last_exc is not None:
            raise last_exc
        raise RuntimeError("Unexpected score fetch retry state")

    def query_metrics(self, query: str) -> list[dict[str, Any]]:
        response = self.sdk_client.api.metrics_v_2.metrics(query=query)
        return [self._to_dict(item) for item in response.data]

    def get_dataset(self, name: str) -> Any:
        return self.sdk_client.get_dataset(name)

    def get_trace(self, trace_id: str) -> dict[str, Any]:
        return self._to_dict(self.sdk_client.api.trace.get(trace_id=trace_id))

    def create_dataset(
        self,
        *,
        name: str,
        description: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return self._to_dict(
            self.sdk_client.create_dataset(
                name=name,
                description=description,
                metadata=metadata,
            )
        )

    def create_dataset_item(
        self,
        *,
        dataset_name: str,
        input: Any = None,
        expected_output: Any = None,
        metadata: dict[str, Any] | None = None,
        source_trace_id: str | None = None,
        source_observation_id: str | None = None,
        status: str | None = None,
        item_id: str | None = None,
    ) -> dict[str, Any]:
        return self._to_dict(
            self.sdk_client.create_dataset_item(
                dataset_name=dataset_name,
                input=input,
                expected_output=expected_output,
                metadata=metadata,
                source_trace_id=source_trace_id,
                source_observation_id=source_observation_id,
                status=status,
                id=item_id,
            )
        )

    def get_prompt(
        self,
        name: str,
        *,
        version: int | None = None,
        label: str | None = None,
        type: str = "text",
        fallback: Any = None,
    ) -> Any:
        kwargs: dict[str, Any] = {"name": name, "type": type}
        if version is not None:
            kwargs["version"] = version
        if label is not None:
            kwargs["label"] = label
        if fallback is not None:
            kwargs["fallback"] = fallback
        return self.sdk_client.get_prompt(**kwargs)

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
        prompt_client = self.sdk_client.create_prompt(
            name=name,
            prompt=prompt,
            type=prompt_type,
            labels=[],
            tags=tags,
            commit_message=commit_message,
        )
        if label:
            version = getattr(prompt_client, "version", None)
            if version is not None:
                self.sdk_client.update_prompt(name=name, version=version, new_labels=[label])
                prompt_client = self.sdk_client.get_prompt(name=name, version=version, type=prompt_type)
        return prompt_client

    def list_prompts(
        self,
        *,
        name: str | None = None,
        label: str | None = None,
        tag: str | None = None,
        limit: int = 50,
        page: int = 1,
    ) -> list[dict[str, Any]]:
        response = self.sdk_client.api.prompts.list(
            name=name or None,
            label=label or None,
            tag=tag or None,
            limit=limit,
            page=page,
        )
        return [self._to_dict(item) for item in response.data]

    def get_prompt_analytics(self, filters: PromptMetricsFilters) -> list[dict[str, Any]]:
        return self._metrics_query(
            view="observations",
            dimensions=["promptName", "promptVersion", "providedModelName"],
            metrics=[
                {"measure": "count", "aggregation": "count"},
                {"measure": "latency", "aggregation": "avg"},
                {"measure": "totalCost", "aggregation": "sum"},
                {"measure": "inputTokens", "aggregation": "sum"},
                {"measure": "outputTokens", "aggregation": "sum"},
                {"measure": "totalTokens", "aggregation": "sum"},
            ],
            filters=self._build_prompt_metrics_filters(filters),
            from_timestamp=filters.from_date,
            to_timestamp=filters.to_date,
            row_limit=filters.limit,
            order_by=[{"field": "promptVersion", "direction": "desc"}],
        )

    def get_run_prompt_analytics(self, filters: PromptMetricsFilters) -> list[dict[str, Any]]:
        return self._metrics_query(
            view="observations",
            dimensions=["promptName", "promptVersion"],
            metrics=[
                {"measure": "count", "aggregation": "count"},
                {"measure": "latency", "aggregation": "avg"},
                {"measure": "totalCost", "aggregation": "sum"},
                {"measure": "totalTokens", "aggregation": "sum"},
            ],
            filters=self._build_prompt_metrics_filters(filters),
            from_timestamp=filters.from_date,
            to_timestamp=filters.to_date,
            time_dimension=filters.time_granularity.value if filters.time_granularity else None,
            row_limit=filters.limit,
            order_by=[{"field": "promptVersion", "direction": "desc"}],
        )

    def get_tool_observation_metrics(
        self,
        filters: ToolJudgeFilters,
        *,
        match_field: str = "name",
    ) -> list[dict[str, Any]]:
        query_filters = self._build_tool_observation_filters(filters)
        tool_filter = self._build_tool_filter(filters.tool_names, match_field=match_field)
        if tool_filter:
            query_filters.append(tool_filter)
        return self._metrics_query(
            view="observations",
            dimensions=["name", "tags", "providedModelName"],
            metrics=[
                {"measure": "count", "aggregation": "count"},
                {"measure": "latency", "aggregation": "avg"},
                {"measure": "totalCost", "aggregation": "sum"},
                {"measure": "inputTokens", "aggregation": "sum"},
                {"measure": "outputTokens", "aggregation": "sum"},
                {"measure": "totalTokens", "aggregation": "sum"},
            ],
            filters=query_filters,
            from_timestamp=filters.from_date,
            to_timestamp=filters.to_date,
            row_limit=filters.limit,
            order_by=[{"field": "count", "direction": "desc"}],
        )

    def get_tool_evaluator_metrics(
        self,
        filters: ToolJudgeFilters,
        *,
        observation_names: list[str],
        categorical: bool = False,
    ) -> list[dict[str, Any]]:
        query_filters = self._build_tool_evaluator_filters(filters, observation_names)
        dimensions = ["name", "observationName", "source"]
        metrics = [{"measure": "count", "aggregation": "count"}]
        if categorical:
            dimensions.append("stringValue")
        else:
            dimensions.append("dataType")
            metrics.append({"measure": "value", "aggregation": "avg"})
        return self._metrics_query(
            view="scores-categorical" if categorical else "scores-numeric",
            dimensions=dimensions,
            metrics=metrics,
            filters=query_filters,
            from_timestamp=filters.from_date,
            to_timestamp=filters.to_date,
            row_limit=filters.limit,
            order_by=[{"field": "count", "direction": "desc"}],
        )

    @staticmethod
    def _page_from_cursor(cursor: str | None) -> int | None:
        if cursor is None:
            return None
        try:
            return max(1, int(cursor))
        except ValueError:
            return None

    def _filter_traces_locally(
        self, traces: list[dict[str, Any]], filters: JudgeDatasetFilters
    ) -> list[dict[str, Any]]:
        filtered: list[dict[str, Any]] = []
        trace_ids = set(filters.trace_ids)
        session_ids = set(filters.session_ids)
        for trace in traces:
            trace_id = trace.get("id")
            session_id = trace.get("session_id") or trace.get("sessionId")
            experiment_id = self._extract_experiment_id(trace)
            if trace_ids and trace_id not in trace_ids:
                continue
            if session_ids and session_id not in session_ids:
                continue
            if filters.experiment_id and experiment_id != filters.experiment_id:
                continue
            filtered.append(trace)
        return filtered

    def _filter_observations_locally(
        self, observations: list[dict[str, Any]], filters: JudgeDatasetFilters
    ) -> list[dict[str, Any]]:
        trace_ids = set(filters.trace_ids)
        filtered: list[dict[str, Any]] = []
        for observation in observations:
            trace_id = observation.get("trace_id") or observation.get("traceId")
            if trace_ids and trace_id not in trace_ids:
                continue
            filtered.append(observation)
        return filtered

    def _filter_scores_locally(
        self, scores: list[dict[str, Any]], filters: JudgeDatasetFilters
    ) -> list[dict[str, Any]]:
        trace_ids = set(filters.trace_ids)
        session_ids = set(filters.session_ids)
        judge_names = set(filters.judge_names or self.config.default_judge_names)
        score_names = set(filters.score_names or self.config.default_score_names)
        filtered: list[dict[str, Any]] = []
        for score in scores:
            trace_id = score.get("trace_id") or score.get("traceId")
            session_id = score.get("session_id") or score.get("sessionId")
            score_name = score.get("name")
            judge_name = self._extract_judge_name(score)
            numeric_value = self._extract_numeric_value(score)
            if trace_ids and trace_id not in trace_ids:
                continue
            if session_ids and session_id not in session_ids:
                continue
            if score_names and score_name not in score_names:
                continue
            if judge_names and judge_name not in judge_names:
                continue
            if filters.min_score is not None and numeric_value is not None:
                if numeric_value < filters.min_score:
                    continue
            filtered.append(score)
        return filtered

    @staticmethod
    def _to_dict(value: Any) -> dict[str, Any]:
        if value is None:
            return {}
        if isinstance(value, dict):
            return value
        if hasattr(value, "model_dump"):
            return value.model_dump()
        if hasattr(value, "dict"):
            return value.dict()
        if hasattr(value, "__dict__"):
            return dict(value.__dict__)
        raise TypeError(f"Unsupported payload type: {type(value)!r}")

    @staticmethod
    def _extract_judge_name(score: dict[str, Any]) -> str:
        metadata = score.get("metadata") or {}
        return (
            metadata.get("judge_name")
            or metadata.get("judgeName")
            or score.get("author_user_id")
            or score.get("authorUserId")
            or score.get("name")
            or "unknown"
        )

    @staticmethod
    def _extract_numeric_value(score: dict[str, Any]) -> float | None:
        value = score.get("value")
        if isinstance(value, bool):
            return float(value)
        if isinstance(value, (int, float)):
            return float(value)
        return None

    @staticmethod
    def _extract_experiment_id(payload: dict[str, Any]) -> str | None:
        metadata = payload.get("metadata") or {}
        return (
            metadata.get("experiment_id")
            or metadata.get("experimentId")
            or payload.get("experiment_id")
            or payload.get("experimentId")
            or payload.get("dataset_run_id")
            or payload.get("datasetRunId")
        )

    def _metrics_query(
        self,
        *,
        view: str,
        dimensions: list[str],
        metrics: list[dict[str, Any]],
        filters: list[dict[str, Any]],
        from_timestamp: datetime | None,
        to_timestamp: datetime | None,
        row_limit: int,
        order_by: list[dict[str, Any]] | None = None,
        time_dimension: str | None = None,
    ) -> list[dict[str, Any]]:
        query: dict[str, Any] = {
            "view": view,
            "dimensions": [{"field": field} for field in dimensions],
            "metrics": metrics,
            "filters": filters,
            "config": {"row_limit": max(1, min(row_limit, 1000))},
        }
        if from_timestamp is not None:
            query["fromTimestamp"] = self._to_langfuse_iso_datetime(from_timestamp)
        if to_timestamp is not None:
            query["toTimestamp"] = self._to_langfuse_iso_datetime(to_timestamp)
        if order_by:
            query["orderBy"] = order_by
        if time_dimension:
            query["timeDimension"] = {"granularity": time_dimension}
        return self.query_metrics(json.dumps(query))

    @staticmethod
    def _to_langfuse_iso_datetime(value: datetime) -> str:
        if value.tzinfo is None:
            value = value.replace(tzinfo=UTC)
        return value.isoformat().replace("+00:00", "Z")

    @staticmethod
    def _build_prompt_metrics_filters(filters: PromptMetricsFilters) -> list[dict[str, Any]]:
        query_filters: list[dict[str, Any]] = []
        if filters.prompt_name:
            query_filters.append(
                {
                    "column": "promptName",
                    "operator": "=",
                    "type": "string",
                    "value": filters.prompt_name,
                }
            )
        if filters.prompt_versions:
            query_filters.append(
                {
                    "column": "promptVersion",
                    "operator": "any of",
                    "type": "categoryOptions",
                    "value": filters.prompt_versions,
                }
            )
        if filters.environment:
            query_filters.append(
                {
                    "column": "environment",
                    "operator": "=",
                    "type": "string",
                    "value": filters.environment,
                }
            )
        if filters.model_name:
            query_filters.append(
                {
                    "column": "providedModelName",
                    "operator": "=",
                    "type": "string",
                    "value": filters.model_name,
                }
            )
        return query_filters

    @staticmethod
    def _build_tool_filter(tool_names: list[str], *, match_field: str) -> dict[str, Any] | None:
        if not tool_names:
            return None
        if match_field == "tags":
            return {
                "column": "tags",
                "operator": "any of",
                "type": "arrayOptions",
                "value": tool_names,
            }
        return {
            "column": "name",
            "operator": "any of",
            "type": "categoryOptions",
            "value": tool_names,
        }

    @staticmethod
    def _build_tool_observation_filters(filters: ToolJudgeFilters) -> list[dict[str, Any]]:
        query_filters: list[dict[str, Any]] = []
        if filters.environment:
            query_filters.append(
                {
                    "column": "environment",
                    "operator": "=",
                    "type": "string",
                    "value": filters.environment,
                }
            )
        if filters.observation_types:
            query_filters.append(
                {
                    "column": "type",
                    "operator": "any of",
                    "type": "categoryOptions",
                    "value": filters.observation_types,
                }
            )
        return query_filters

    @staticmethod
    def _build_tool_evaluator_filters(
        filters: ToolJudgeFilters,
        observation_names: list[str],
    ) -> list[dict[str, Any]]:
        query_filters: list[dict[str, Any]] = []
        if filters.evaluator_names:
            query_filters.append(
                {
                    "column": "name",
                    "operator": "any of",
                    "type": "categoryOptions",
                    "value": filters.evaluator_names,
                }
            )
        if observation_names:
            query_filters.append(
                {
                    "column": "observationName",
                    "operator": "any of",
                    "type": "categoryOptions",
                    "value": observation_names,
                }
            )
        if filters.environment:
            query_filters.append(
                {
                    "column": "environment",
                    "operator": "=",
                    "type": "string",
                    "value": filters.environment,
                }
            )
        return query_filters
