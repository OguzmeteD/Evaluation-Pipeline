from __future__ import annotations

from statistics import mean
from typing import Any

from src.core.langfuse_client import LangfuseCollectorClient
from src.schemas.dataset_builder import (
    DatasetBuilderFilters,
    DatasetCandidateResult,
    DatasetCandidateScore,
    DatasetCandidateTrace,
    DatasetCreationRequest,
    DatasetCreationResult,
    DatasetMetricThreshold,
)
from src.schemas.evaluation_dataset import JudgeDatasetFilters


class DatasetBuilderService:
    def __init__(self, collector: LangfuseCollectorClient) -> None:
        self.collector = collector

    def preview_candidate_traces(self, filters: DatasetBuilderFilters) -> DatasetCandidateResult:
        warnings: list[str] = []
        if not filters.metric_thresholds:
            return DatasetCandidateResult(
                filters=filters,
                warnings=["En az bir metric threshold secilmelidir."],
            )

        grouped_scores: dict[str, dict[str, DatasetCandidateScore]] = {}
        all_trace_ids: set[str] = set()

        for threshold in filters.metric_thresholds:
            try:
                metric_scores = self.collector.list_scores(
                    JudgeDatasetFilters(
                        from_date=filters.from_date,
                        to_date=filters.to_date,
                        experiment_id=filters.experiment_id,
                        session_ids=filters.session_ids,
                        judge_names=[threshold.judge_name] if threshold.judge_name else [],
                        score_names=[threshold.metric_name],
                        min_score=threshold.min_score,
                        limit=filters.limit,
                    )
                )
            except Exception as exc:
                warnings.append(
                    f"'{threshold.metric_name}' metric'i icin score fetch basarisiz oldu: {exc}"
                )
                continue
            if not metric_scores:
                warnings.append(
                    f"'{threshold.metric_name}' metric'i icin esigi gecen score bulunamadi."
                )
                continue
            for score in metric_scores:
                trace_id = score.get("trace_id") or score.get("traceId")
                numeric_value = self._extract_numeric_value(score)
                if not trace_id or numeric_value is None:
                    continue
                all_trace_ids.add(trace_id)
                metric_key = self._metric_key(threshold.metric_name, threshold.judge_name)
                trace_scores = grouped_scores.setdefault(trace_id, {})
                candidate_score = DatasetCandidateScore(
                    metric_name=threshold.metric_name,
                    judge_name=threshold.judge_name or self._extract_judge_name(score),
                    score_value=numeric_value,
                    score_id=score.get("id"),
                    created_at=score.get("created_at") or score.get("createdAt"),
                    comment=score.get("comment"),
                )
                existing = trace_scores.get(metric_key)
                if existing is None or candidate_score.score_value > existing.score_value:
                    trace_scores[metric_key] = candidate_score

        candidates: list[DatasetCandidateTrace] = []
        if not all_trace_ids:
            warnings.append("Secilen filtrelerle aday trace bulunamadi.")
            return DatasetCandidateResult(filters=filters, warnings=warnings)

        for trace_id in sorted(all_trace_ids):
            trace_scores = grouped_scores.get(trace_id, {})
            if not self._matches_all_metrics(trace_scores, filters.metric_thresholds):
                continue
            try:
                trace = self.collector.get_trace(trace_id)
            except Exception as exc:
                warnings.append(f"Trace {trace_id} cekilemedi: {exc}")
                continue

            input_payload, output_payload = self._resolve_trace_io(trace)
            if input_payload is None or output_payload is None:
                warnings.append(
                    f"Trace {trace_id} icin trace IO eksik; observation fallback kullanildi."
                )

            score_summary = list(trace_scores.values())
            candidates.append(
                DatasetCandidateTrace(
                    trace_id=trace_id,
                    trace_name=trace.get("name"),
                    input_payload=input_payload,
                    output_payload=output_payload,
                    score_summary=score_summary,
                    matched_metrics=[score.metric_name for score in score_summary],
                    avg_score=mean(score.score_value for score in score_summary) if score_summary else None,
                    session_id=trace.get("session_id") or trace.get("sessionId"),
                    experiment_id=self._extract_experiment_id(trace),
                    metadata=self._build_candidate_metadata(trace, score_summary),
                )
            )

        candidates.sort(key=lambda row: (-(row.avg_score or 0.0), row.trace_id))
        limited = candidates[: filters.limit]
        if candidates and len(candidates) > len(limited):
            warnings.append(f"Preview sonucu {filters.limit} trace ile sinirlandi.")
        if not limited:
            warnings.append("Tum metric threshold kosullarini saglayan trace bulunamadi.")
        return DatasetCandidateResult(
            filters=filters,
            candidates=limited,
            total_candidates=len(limited),
            warnings=warnings,
        )

    def create_dataset_from_candidates(self, request: DatasetCreationRequest) -> DatasetCreationResult:
        if not request.dataset_name.strip():
            return DatasetCreationResult(
                dataset_name=request.dataset_name,
                errors=["Dataset name zorunludur."],
            )
        if self._dataset_exists(request.dataset_name):
            return DatasetCreationResult(
                dataset_name=request.dataset_name,
                errors=[f"'{request.dataset_name}' adli dataset zaten mevcut."],
            )
        if not request.candidates:
            return DatasetCreationResult(
                dataset_name=request.dataset_name,
                errors=["Olusturmak icin once candidate preview alinmalidir."],
            )

        dataset = self.collector.create_dataset(
            name=request.dataset_name,
            description=request.description,
            metadata=request.metadata,
        )
        item_ids: list[str] = []
        errors: list[str] = []
        warnings: list[str] = []
        created_items = 0

        for candidate in request.candidates:
            try:
                item = self.collector.create_dataset_item(
                    dataset_name=request.dataset_name,
                    input=candidate.input_payload,
                    expected_output=candidate.output_payload,
                    metadata=self._build_dataset_item_metadata(candidate),
                    source_trace_id=candidate.trace_id,
                )
                created_items += 1
                if item.get("id"):
                    item_ids.append(item["id"])
            except Exception as exc:
                errors.append(f"Trace {candidate.trace_id} dataset item olarak eklenemedi: {exc}")

        if errors and created_items:
            warnings.append("Dataset olustu ancak bazi item'lar eklenemedi.")
        return DatasetCreationResult(
            dataset_id=dataset.get("id"),
            dataset_name=request.dataset_name,
            created_items=created_items,
            failed_items=len(errors),
            warnings=warnings,
            errors=errors,
            item_ids=item_ids,
        )

    def _resolve_trace_io(self, trace: dict[str, Any]) -> tuple[Any, Any]:
        input_payload = trace.get("input")
        output_payload = trace.get("output")
        if input_payload is not None and output_payload is not None:
            return input_payload, output_payload

        observations, _ = self.collector.list_observations(
            JudgeDatasetFilters(trace_ids=[trace.get("id")], limit=50)
        )
        fallback_input = input_payload
        fallback_output = output_payload
        for observation in observations:
            if fallback_input is None and observation.get("input") is not None:
                fallback_input = observation.get("input")
            if fallback_output is None and observation.get("output") is not None:
                fallback_output = observation.get("output")
            if fallback_input is not None and fallback_output is not None:
                break
        return fallback_input, fallback_output

    @staticmethod
    def _metric_key(metric_name: str, judge_name: str | None) -> str:
        return f"{metric_name}::{judge_name or '*'}"

    @staticmethod
    def _matches_all_metrics(
        trace_scores: dict[str, DatasetCandidateScore],
        thresholds: list[DatasetMetricThreshold],
    ) -> bool:
        for threshold in thresholds:
            if DatasetBuilderService._metric_key(threshold.metric_name, threshold.judge_name) not in trace_scores:
                return False
        return True

    def _dataset_exists(self, dataset_name: str) -> bool:
        try:
            self.collector.get_dataset(dataset_name)
            return True
        except Exception as exc:
            message = str(exc).lower()
            if "not found" in message or "404" in message:
                return False
            raise

    @staticmethod
    def _extract_numeric_value(score: dict[str, Any]) -> float | None:
        value = score.get("value")
        if isinstance(value, bool):
            return float(value)
        if isinstance(value, (int, float)):
            return float(value)
        return None

    @staticmethod
    def _extract_judge_name(score: dict[str, Any]) -> str | None:
        metadata = score.get("metadata") or {}
        return (
            metadata.get("judge_name")
            or metadata.get("judgeName")
            or score.get("author_user_id")
            or score.get("authorUserId")
            or None
        )

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

    def _build_candidate_metadata(
        self,
        trace: dict[str, Any],
        score_summary: list[DatasetCandidateScore],
    ) -> dict[str, Any]:
        metadata = trace.get("metadata") if isinstance(trace.get("metadata"), dict) else {}
        return {
            **metadata,
            "selection_reason": "score_threshold_match",
            "trace_name": trace.get("name"),
            "score_summary": [score.model_dump(mode="json") for score in score_summary],
        }

    @staticmethod
    def _build_dataset_item_metadata(candidate: DatasetCandidateTrace) -> dict[str, Any]:
        return {
            "selection_reason": "score_threshold_match",
            "trace_id": candidate.trace_id,
            "trace_name": candidate.trace_name,
            "session_id": candidate.session_id,
            "experiment_id": candidate.experiment_id,
            "matched_metrics": candidate.matched_metrics,
            "score_summary": [score.model_dump(mode="json") for score in candidate.score_summary],
        }


_DEFAULT_SERVICE: DatasetBuilderService | None = None


def _get_service() -> DatasetBuilderService:
    global _DEFAULT_SERVICE
    if _DEFAULT_SERVICE is None:
        _DEFAULT_SERVICE = DatasetBuilderService(LangfuseCollectorClient())
    return _DEFAULT_SERVICE


def preview_dataset_candidates(filters: DatasetBuilderFilters) -> DatasetCandidateResult:
    return _get_service().preview_candidate_traces(filters)


def create_langfuse_dataset(request: DatasetCreationRequest) -> DatasetCreationResult:
    return _get_service().create_dataset_from_candidates(request)
