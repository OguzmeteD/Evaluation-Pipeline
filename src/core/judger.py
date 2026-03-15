from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from statistics import mean
from typing import Any

from src.core.langfuse_client import LangfuseCollectorClient
from src.schemas.evaluation_dataset import (
    DatasetCounts,
    DatasetMeta,
    EvaluationDataset,
    GenerationRow,
    JudgeDatasetFilters,
    JudgeScoreRecord,
    PromptSource,
    ScoreBreakdown,
    TraceSummary,
)


@dataclass(slots=True)
class PromptExtractionResult:
    system_prompt: str | None
    prompt_messages: list[dict[str, Any]]
    source: PromptSource


class LangfuseJudgeService:
    """Collects Langfuse judge data and exposes a UI-friendly normalized dataset."""

    def __init__(self, collector: LangfuseCollectorClient) -> None:
        self.collector = collector

    def get_evaluation_dataset(
        self, filters: JudgeDatasetFilters | None = None
    ) -> EvaluationDataset:
        filters = filters or JudgeDatasetFilters()
        traces = self.collector.list_traces(filters)
        observations, next_cursor = self.collector.list_observations(filters)
        scores = self.collector.list_scores(filters)

        trace_map = {trace.get("id"): trace for trace in traces if trace.get("id")}
        scores = self._deduplicate_scores(scores)

        for observation in observations:
            trace_id = self._pick(observation, "trace_id", "traceId")
            if trace_id and trace_id not in trace_map:
                trace_map[trace_id] = {}

        rows = self._build_generation_rows(
            observations=observations,
            trace_map=trace_map,
            scores=scores,
            filters=filters,
        )
        traces_out = self._build_trace_summaries(rows=rows, trace_map=trace_map, scores=scores)
        meta = self._build_meta(
            filters=filters,
            rows=rows,
            traces=traces_out,
            scores=scores,
            next_cursor=next_cursor,
            warnings=self._build_warnings(traces, observations, scores),
        )

        return EvaluationDataset(traces=traces_out, rows=rows, meta=meta)

    def _build_generation_rows(
        self,
        *,
        observations: list[dict[str, Any]],
        trace_map: dict[str, dict[str, Any]],
        scores: list[dict[str, Any]],
        filters: JudgeDatasetFilters,
    ) -> list[GenerationRow]:
        observation_scores = self._group_scores_by_observation(scores)
        rows: list[GenerationRow] = []
        for observation in observations:
            trace_id = self._pick(observation, "trace_id", "traceId")
            if not trace_id:
                continue
            trace = trace_map.get(trace_id, {})
            experiment_id = self._extract_experiment_id(observation, trace)
            if filters.experiment_id and experiment_id != filters.experiment_id:
                continue

            prompt = self._extract_prompt(observation, trace)
            generation_text = self._extract_generation_text(observation, trace)
            row_scores = [
                self._normalize_score(score, experiment_id=experiment_id)
                for score in observation_scores.get(observation.get("id"), [])
            ]
            rows.append(
                GenerationRow(
                    trace_id=trace_id,
                    observation_id=observation.get("id"),
                    session_id=self._pick(trace, "session_id", "sessionId"),
                    experiment_id=experiment_id,
                    trace_name=trace.get("name"),
                    observation_name=observation.get("name"),
                    model=self._pick(observation, "model", "model_id", "modelId"),
                    latency_ms=self._extract_latency_ms(observation, trace),
                    total_tokens=self._extract_total_tokens(observation),
                    total_cost=self._extract_total_cost(observation, trace),
                    system_prompt=prompt.system_prompt,
                    prompt_messages=prompt.prompt_messages,
                    generation_text=generation_text,
                    judge_scores=row_scores,
                    input_payload=observation.get("input"),
                    output_payload=observation.get("output"),
                    started_at=self._pick(
                        observation, "start_time", "startTime", default=None
                    ),
                    ended_at=self._pick(observation, "end_time", "endTime", default=None),
                    prompt_source=prompt.source,
                    has_prompt=prompt.system_prompt is not None,
                    has_generation=generation_text is not None,
                )
            )
        rows.sort(
            key=lambda row: (
                row.started_at or datetime.min,
                row.trace_id,
                row.observation_id or "",
            )
        )
        return rows

    def _build_trace_summaries(
        self,
        *,
        rows: list[GenerationRow],
        trace_map: dict[str, dict[str, Any]],
        scores: list[dict[str, Any]],
    ) -> list[TraceSummary]:
        trace_scores = self._group_trace_scores(scores)
        rows_by_trace: dict[str, list[GenerationRow]] = defaultdict(list)
        for row in rows:
            rows_by_trace[row.trace_id].append(row)

        summaries: list[TraceSummary] = []
        for trace_id, trace in trace_map.items():
            trace_rows = rows_by_trace.get(trace_id, [])
            normalized_trace_scores = [
                self._normalize_score(
                    score, experiment_id=self._extract_experiment_id(trace)
                )
                for score in trace_scores.get(trace_id, [])
            ]
            all_scores = [
                score.score_value
                for row in trace_rows
                for score in row.judge_scores
                if score.score_value is not None
            ] + [
                score.score_value
                for score in normalized_trace_scores
                if score.score_value is not None
            ]
            summaries.append(
                TraceSummary(
                    trace_id=trace_id,
                    session_id=self._pick(trace, "session_id", "sessionId"),
                    experiment_id=self._extract_experiment_id(trace),
                    trace_name=trace.get("name"),
                    observation_count=len(trace_rows),
                    avg_score=mean(all_scores) if all_scores else None,
                    score_breakdown=self._build_score_breakdown(
                        [score for row in trace_rows for score in row.judge_scores]
                        + normalized_trace_scores
                    ),
                    trace_scores=normalized_trace_scores,
                    has_prompt=any(row.has_prompt for row in trace_rows),
                    has_generation=any(row.has_generation for row in trace_rows),
                    started_at=min(
                        (row.started_at for row in trace_rows if row.started_at),
                        default=self._pick(trace, "timestamp", default=None),
                    ),
                    ended_at=max(
                        (row.ended_at for row in trace_rows if row.ended_at),
                        default=None,
                    ),
                    latency_ms=self._extract_trace_latency_ms(trace),
                    total_cost=self._extract_trace_total_cost(trace, trace_rows),
                    total_tokens=sum(row.total_tokens or 0 for row in trace_rows) or None,
                    observation_ids=[
                        row.observation_id for row in trace_rows if row.observation_id
                    ],
                )
            )
        summaries.sort(key=lambda trace: (trace.started_at or datetime.min, trace.trace_id))
        return summaries

    def _build_meta(
        self,
        *,
        filters: JudgeDatasetFilters,
        rows: list[GenerationRow],
        traces: list[TraceSummary],
        scores: list[dict[str, Any]],
        next_cursor: str | None,
        warnings: list[str],
    ) -> DatasetMeta:
        prompt_coverage = (
            sum(1 for row in rows if row.has_prompt) / len(rows) if rows else 0.0
        )
        generation_coverage = (
            sum(1 for row in rows if row.has_generation) / len(rows) if rows else 0.0
        )
        score_values = [
            score.score_value
            for row in rows
            for score in row.judge_scores
            if score.score_value is not None
        ] + [
            score.score_value
            for trace in traces
            for score in trace.trace_scores
            if score.score_value is not None
        ]
        counts = DatasetCounts(
            traces=len(traces),
            rows=len(rows),
            observation_scores=sum(len(row.judge_scores) for row in rows),
            trace_scores=sum(len(trace.trace_scores) for trace in traces),
        )
        metrics = {
            "total_cost": sum(trace.total_cost or 0.0 for trace in traces),
            "average_trace_latency_ms": mean(
                [trace.latency_ms for trace in traces if trace.latency_ms is not None]
            )
            if any(trace.latency_ms is not None for trace in traces)
            else None,
            "total_tokens": sum(trace.total_tokens or 0 for trace in traces),
            "raw_score_count": len(scores),
        }
        return DatasetMeta(
            filters=filters,
            counts=counts,
            prompt_coverage=prompt_coverage,
            generation_coverage=generation_coverage,
            average_score=mean(score_values) if score_values else None,
            warnings=warnings,
            next_cursor=next_cursor,
            metrics=metrics,
        )

    def _build_warnings(
        self,
        traces: list[dict[str, Any]],
        observations: list[dict[str, Any]],
        scores: list[dict[str, Any]],
    ) -> list[str]:
        warnings: list[str] = []
        if not traces:
            warnings.append("No traces matched the requested filters.")
        if not observations:
            warnings.append("No observations were returned by Langfuse.")
        if not scores:
            warnings.append("No judge scores were returned by Langfuse.")
        missing_prompt_count = sum(
            1
            for observation in observations
            if self._extract_prompt(observation, {}).source == PromptSource.UNAVAILABLE
        )
        if missing_prompt_count:
            warnings.append(
                f"{missing_prompt_count} observation(s) did not expose a system prompt."
            )
        return warnings

    def _build_score_breakdown(
        self, scores: list[JudgeScoreRecord]
    ) -> list[ScoreBreakdown]:
        grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
        for score in scores:
            if score.score_value is None:
                continue
            grouped[(score.score_name, score.judge_name)].append(score.score_value)
        breakdown = [
            ScoreBreakdown(
                score_name=score_name,
                judge_name=judge_name,
                average_score=mean(values) if values else None,
                count=len(values),
            )
            for (score_name, judge_name), values in grouped.items()
        ]
        breakdown.sort(key=lambda item: (item.score_name, item.judge_name))
        return breakdown

    def _group_scores_by_observation(
        self, scores: list[dict[str, Any]]
    ) -> dict[str, list[dict[str, Any]]]:
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for score in scores:
            observation_id = self._pick(score, "observation_id", "observationId")
            if observation_id:
                grouped[observation_id].append(score)
        return grouped

    def _group_trace_scores(
        self, scores: list[dict[str, Any]]
    ) -> dict[str, list[dict[str, Any]]]:
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for score in scores:
            trace_id = self._pick(score, "trace_id", "traceId")
            if trace_id and not self._pick(score, "observation_id", "observationId"):
                grouped[trace_id].append(score)
        return grouped

    def _normalize_score(
        self, score: dict[str, Any], *, experiment_id: str | None
    ) -> JudgeScoreRecord:
        metadata = score.get("metadata") or {}
        return JudgeScoreRecord(
            score_id=score.get("id", "unknown-score"),
            trace_id=self._pick(score, "trace_id", "traceId"),
            observation_id=self._pick(score, "observation_id", "observationId"),
            session_id=self._pick(score, "session_id", "sessionId"),
            experiment_id=experiment_id
            or metadata.get("experiment_id")
            or metadata.get("experimentId")
            or score.get("dataset_run_id")
            or score.get("datasetRunId"),
            judge_name=metadata.get("judge_name")
            or metadata.get("judgeName")
            or self._pick(score, "author_user_id", "authorUserId")
            or score.get("name", "unknown"),
            score_name=score.get("name", "unknown"),
            score_value=self._coerce_float(score.get("value")),
            score_label=score.get("string_value") or score.get("stringValue"),
            score_comment=score.get("comment"),
            score_source=self._stringify(score.get("source")),
            author_user_id=self._pick(score, "author_user_id", "authorUserId"),
            data_type=score.get("data_type") or score.get("dataType"),
            created_at=self._pick(score, "created_at", "createdAt", default=None),
            updated_at=self._pick(score, "updated_at", "updatedAt", default=None),
            metadata=metadata if isinstance(metadata, dict) else None,
        )

    def _deduplicate_scores(self, scores: list[dict[str, Any]]) -> list[dict[str, Any]]:
        deduped: dict[str, dict[str, Any]] = {}
        for score in scores:
            score_id = score.get("id")
            key = score_id or "|".join(
                [
                    str(self._pick(score, "trace_id", "traceId", default="")),
                    str(self._pick(score, "observation_id", "observationId", default="")),
                    str(score.get("name", "")),
                    str(score.get("value", "")),
                    str(score.get("comment", "")),
                ]
            )
            deduped[key] = score
        return list(deduped.values())

    def _extract_prompt(
        self, observation: dict[str, Any], trace: dict[str, Any]
    ) -> PromptExtractionResult:
        prompt_payload = (
            observation.get("prompt")
            or observation.get("prompt_template")
            or observation.get("promptTemplate")
            or observation.get("prompt_details")
            or observation.get("promptDetails")
        )
        if prompt_payload:
            messages = self._extract_messages(prompt_payload)
            system_prompt = self._extract_system_prompt_from_messages(messages)
            if system_prompt:
                return PromptExtractionResult(
                    system_prompt=system_prompt,
                    prompt_messages=messages,
                    source=PromptSource.LANGFUSE_PROMPT,
                )

        observation_messages = self._extract_messages(observation.get("input"))
        observation_prompt = self._extract_system_prompt_from_messages(observation_messages)
        if observation_prompt:
            return PromptExtractionResult(
                system_prompt=observation_prompt,
                prompt_messages=observation_messages,
                source=PromptSource.OBSERVATION_INPUT,
            )

        trace_messages = self._extract_messages(trace.get("input"))
        trace_prompt = self._extract_system_prompt_from_messages(trace_messages)
        if trace_prompt:
            return PromptExtractionResult(
                system_prompt=trace_prompt,
                prompt_messages=trace_messages,
                source=PromptSource.TRACE_INPUT,
            )

        return PromptExtractionResult(
            system_prompt=None,
            prompt_messages=[],
            source=PromptSource.UNAVAILABLE,
        )

    def _extract_generation_text(
        self, observation: dict[str, Any], trace: dict[str, Any]
    ) -> str | None:
        for payload in (observation.get("output"), trace.get("output")):
            text = self._extract_text_candidate(payload)
            if text:
                return text
        return None

    def _extract_text_candidate(self, payload: Any) -> str | None:
        if payload is None:
            return None
        if isinstance(payload, str):
            return payload
        if isinstance(payload, list):
            joined = "\n".join(
                [candidate for item in payload if (candidate := self._extract_text_candidate(item))]
            )
            return joined or None
        if isinstance(payload, dict):
            for key in (
                "text",
                "content",
                "completion",
                "output",
                "response",
                "generation",
                "message",
                "answer",
                "summary",
            ):
                if key in payload:
                    text = self._extract_text_candidate(payload[key])
                    if text:
                        return text
            choices = payload.get("choices")
            if isinstance(choices, list):
                for choice in choices:
                    text = self._extract_text_candidate(choice)
                    if text:
                        return text
            role = payload.get("role")
            content = payload.get("content")
            if isinstance(role, str) and isinstance(content, str):
                return content
        return None

    def _extract_messages(self, payload: Any) -> list[dict[str, Any]]:
        if payload is None:
            return []
        if isinstance(payload, list):
            messages: list[dict[str, Any]] = []
            for item in payload:
                messages.extend(self._extract_messages(item))
            return messages
        if isinstance(payload, str):
            return []
        if isinstance(payload, dict):
            if "messages" in payload and isinstance(payload["messages"], list):
                return self._extract_messages(payload["messages"])
            role = payload.get("role")
            content = payload.get("content")
            if isinstance(role, str) and content is not None:
                normalized_content = self._extract_text_candidate(content) or self._stringify(
                    content
                )
                return [{"role": role, "content": normalized_content}]
            candidates: list[dict[str, Any]] = []
            for key in ("input", "prompt", "conversation", "items"):
                if key in payload:
                    candidates.extend(self._extract_messages(payload[key]))
            return candidates
        return []

    @staticmethod
    def _extract_system_prompt_from_messages(messages: list[dict[str, Any]]) -> str | None:
        for message in messages:
            if message.get("role") == "system":
                content = message.get("content")
                if isinstance(content, str) and content.strip():
                    return content
        return None

    def _extract_total_tokens(self, observation: dict[str, Any]) -> int | None:
        usage = observation.get("usage") or {}
        usage_details = observation.get("usage_details") or observation.get("usageDetails") or {}
        candidates = [
            usage.get("total_tokens"),
            usage.get("totalTokens"),
            usage_details.get("total"),
            usage_details.get("total_tokens"),
        ]
        for candidate in candidates:
            if isinstance(candidate, int):
                return candidate
        input_tokens = usage.get("input") or usage.get("input_tokens") or 0
        output_tokens = usage.get("output") or usage.get("output_tokens") or 0
        if isinstance(input_tokens, int) and isinstance(output_tokens, int):
            total = input_tokens + output_tokens
            return total or None
        return None

    def _extract_total_cost(
        self, observation: dict[str, Any], trace: dict[str, Any]
    ) -> float | None:
        candidates = [
            observation.get("total_price"),
            observation.get("totalPrice"),
            observation.get("calculated_total_cost"),
            observation.get("calculatedTotalCost"),
            (observation.get("cost_details") or {}).get("total"),
            trace.get("total_cost"),
            trace.get("totalCost"),
        ]
        for candidate in candidates:
            value = self._coerce_float(candidate)
            if value is not None:
                return value
        return None

    def _extract_latency_ms(
        self, observation: dict[str, Any], trace: dict[str, Any]
    ) -> float | None:
        latency = observation.get("latency")
        if latency is not None:
            return self._coerce_float(latency)
        start_time = self._pick(observation, "start_time", "startTime", default=None)
        end_time = self._pick(observation, "end_time", "endTime", default=None)
        if isinstance(start_time, datetime) and isinstance(end_time, datetime):
            return (end_time - start_time).total_seconds() * 1000
        return self._extract_trace_latency_ms(trace)

    def _extract_trace_latency_ms(self, trace: dict[str, Any]) -> float | None:
        latency = trace.get("latency")
        if latency is None:
            return None
        return self._coerce_float(latency)

    def _extract_trace_total_cost(
        self, trace: dict[str, Any], rows: list[GenerationRow]
    ) -> float | None:
        trace_cost = self._coerce_float(trace.get("total_cost") or trace.get("totalCost"))
        if trace_cost is not None:
            return trace_cost
        total = sum(row.total_cost or 0.0 for row in rows)
        return total or None

    def _extract_experiment_id(
        self, payload: dict[str, Any], trace: dict[str, Any] | None = None
    ) -> str | None:
        trace = trace or {}
        for candidate in (payload, trace):
            metadata = candidate.get("metadata") or {}
            experiment_id = (
                metadata.get("experiment_id")
                or metadata.get("experimentId")
                or candidate.get("experiment_id")
                or candidate.get("experimentId")
                or candidate.get("dataset_run_id")
                or candidate.get("datasetRunId")
            )
            if experiment_id:
                return experiment_id
        return None

    @staticmethod
    def _pick(payload: dict[str, Any], *keys: str, default: Any = None) -> Any:
        for key in keys:
            if key in payload:
                return payload[key]
        return default

    @staticmethod
    def _coerce_float(value: Any) -> float | None:
        if isinstance(value, bool):
            return float(value)
        if isinstance(value, (int, float)):
            return float(value)
        return None

    @staticmethod
    def _stringify(value: Any) -> str | None:
        if value is None:
            return None
        if isinstance(value, str):
            return value
        return str(value)


def get_evaluation_dataset(
    filters: JudgeDatasetFilters | None = None,
    *,
    collector: LangfuseCollectorClient | None = None,
) -> EvaluationDataset:
    service = LangfuseJudgeService(collector or LangfuseCollectorClient())
    return service.get_evaluation_dataset(filters)
