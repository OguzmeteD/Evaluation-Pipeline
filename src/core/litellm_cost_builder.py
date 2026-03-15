from __future__ import annotations

from statistics import mean
from typing import Any

from src.core.langfuse_client import LangfuseCollectorClient
from src.core.litellm_store import LiteLLMStore, PostgresLiteLLMStore
from src.schemas.litellm_cost_builder import (
    LiteLLMCostCandidateRow,
    LiteLLMCostDatasetPreview,
    LiteLLMCostDatasetRequest,
    LiteLLMCostDatasetResult,
    LiteLLMCostFilters,
    LiteLLMCostPreviewSummary,
    LiteLLMStoreConfig,
)


class LiteLLMCostBuilderService:
    def __init__(self, collector: LangfuseCollectorClient, store: LiteLLMStore | None = None) -> None:
        self.collector = collector
        self.store = store or PostgresLiteLLMStore()

    def get_store_config(self) -> LiteLLMStoreConfig:
        return self.store.get_config()

    def preview_candidates(self, filters: LiteLLMCostFilters) -> LiteLLMCostDatasetPreview:
        config = self.store.get_config()
        rows, warnings = self.store.list_requests(filters)
        normalized_rows = [self._normalize_row(row) for row in rows]
        if not normalized_rows:
            warnings = [*warnings, "Secilen filtrelerle LiteLLM request kaydi bulunamadi."]
        return LiteLLMCostDatasetPreview(
            config=config,
            filters=filters,
            summary=self._build_summary(normalized_rows),
            rows=normalized_rows,
            warnings=warnings,
        )

    def create_dataset(self, request: LiteLLMCostDatasetRequest) -> LiteLLMCostDatasetResult:
        if not request.dataset_name.strip():
            return LiteLLMCostDatasetResult(dataset_name=request.dataset_name, errors=["Dataset name zorunludur."])
        if not request.rows:
            return LiteLLMCostDatasetResult(
                dataset_name=request.dataset_name,
                errors=["Olusturmak icin once LiteLLM preview alinmalidir."],
            )
        if self._dataset_exists(request.dataset_name):
            return LiteLLMCostDatasetResult(
                dataset_name=request.dataset_name,
                errors=[f"'{request.dataset_name}' adli dataset zaten mevcut."],
            )

        dataset = self.collector.create_dataset(
            name=request.dataset_name,
            description=request.description,
            metadata=request.metadata,
        )
        created_items = 0
        item_ids: list[str] = []
        errors: list[str] = []
        warnings: list[str] = []

        for row in request.rows:
            try:
                item = self.collector.create_dataset_item(
                    dataset_name=request.dataset_name,
                    input=row.request_input,
                    expected_output=row.request_output,
                    metadata=self._build_item_metadata(row, request.filters),
                    source_trace_id=row.langfuse_trace_id,
                    source_observation_id=row.langfuse_observation_id,
                )
                created_items += 1
                if item.get("id"):
                    item_ids.append(item["id"])
            except Exception as exc:
                errors.append(f"LiteLLM request {row.request_id} dataset item olarak eklenemedi: {exc}")

        if created_items and errors:
            warnings.append("Dataset olustu ancak bazi LiteLLM request kayitlari item olarak eklenemedi.")
        return LiteLLMCostDatasetResult(
            dataset_id=dataset.get("id"),
            dataset_name=request.dataset_name,
            created_items=created_items,
            failed_items=len(errors),
            warnings=warnings,
            errors=errors,
            item_ids=item_ids,
        )

    def _dataset_exists(self, dataset_name: str) -> bool:
        try:
            self.collector.get_dataset(dataset_name)
            return True
        except Exception as exc:
            message = str(exc).lower()
            if "404" in message or "not found" in message:
                return False
            raise

    @staticmethod
    def _normalize_row(row: dict[str, Any]) -> LiteLLMCostCandidateRow:
        metadata = row.get("metadata")
        if isinstance(metadata, str):
            metadata = {"raw_metadata": metadata}
        if not isinstance(metadata, dict):
            metadata = None

        input_tokens = LiteLLMCostBuilderService._to_int(row.get("input_tokens"))
        output_tokens = LiteLLMCostBuilderService._to_int(row.get("output_tokens"))
        total_tokens = LiteLLMCostBuilderService._to_int(row.get("total_tokens"))
        if total_tokens == 0 and (input_tokens or output_tokens):
            total_tokens = input_tokens + output_tokens

        return LiteLLMCostCandidateRow(
            request_id=str(row.get("request_id")),
            created_at=row.get("created_at"),
            model_name=LiteLLMCostBuilderService._to_str(row.get("model_name")),
            provider=LiteLLMCostBuilderService._to_str(row.get("provider")),
            total_cost=LiteLLMCostBuilderService._to_float(row.get("total_cost")),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            latency_ms=LiteLLMCostBuilderService._nullable_float(row.get("latency_ms")),
            status=LiteLLMCostBuilderService._to_str(row.get("status")),
            request_input=row.get("request_input"),
            request_output=row.get("request_output"),
            langfuse_trace_id=LiteLLMCostBuilderService._to_str(row.get("langfuse_trace_id")),
            langfuse_observation_id=LiteLLMCostBuilderService._to_str(row.get("langfuse_observation_id")),
            metadata=metadata,
        )

    @staticmethod
    def _build_summary(rows: list[LiteLLMCostCandidateRow]) -> LiteLLMCostPreviewSummary:
        latencies = [row.latency_ms for row in rows if row.latency_ms is not None]
        return LiteLLMCostPreviewSummary(
            total_requests=len(rows),
            total_cost=sum(row.total_cost for row in rows),
            total_tokens=sum(row.total_tokens for row in rows),
            avg_latency_ms=mean(latencies) if latencies else None,
        )

    @staticmethod
    def _build_item_metadata(
        row: LiteLLMCostCandidateRow,
        filters: LiteLLMCostFilters | None,
    ) -> dict[str, Any]:
        return {
            "source": "litellm_cost_builder",
            "request_id": row.request_id,
            "model_name": row.model_name,
            "provider": row.provider,
            "total_cost": row.total_cost,
            "input_tokens": row.input_tokens,
            "output_tokens": row.output_tokens,
            "total_tokens": row.total_tokens,
            "latency_ms": row.latency_ms,
            "status": row.status,
            "langfuse_trace_id": row.langfuse_trace_id,
            "langfuse_observation_id": row.langfuse_observation_id,
            "filters": filters.model_dump(mode="json") if filters else None,
            "metadata": row.metadata,
        }

    @staticmethod
    def _to_int(value: Any) -> int:
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            return int(value)
        if isinstance(value, str) and value.strip():
            try:
                return int(float(value))
            except ValueError:
                return 0
        return 0

    @staticmethod
    def _to_float(value: Any) -> float:
        if isinstance(value, bool):
            return float(value)
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str) and value.strip():
            try:
                return float(value)
            except ValueError:
                return 0.0
        return 0.0

    @staticmethod
    def _nullable_float(value: Any) -> float | None:
        if value is None:
            return None
        parsed = LiteLLMCostBuilderService._to_float(value)
        return parsed

    @staticmethod
    def _to_str(value: Any) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        return text or None


_DEFAULT_SERVICE: LiteLLMCostBuilderService | None = None


def _get_service() -> LiteLLMCostBuilderService:
    global _DEFAULT_SERVICE
    if _DEFAULT_SERVICE is None:
        _DEFAULT_SERVICE = LiteLLMCostBuilderService(LangfuseCollectorClient())
    return _DEFAULT_SERVICE


def get_litellm_store_config() -> LiteLLMStoreConfig:
    return _get_service().get_store_config()


def ensure_litellm_store_schema() -> list[str]:
    return _get_service().store.ensure_schema()


def preview_litellm_cost_candidates(filters: LiteLLMCostFilters) -> LiteLLMCostDatasetPreview:
    return _get_service().preview_candidates(filters)


def create_litellm_cost_dataset(request: LiteLLMCostDatasetRequest) -> LiteLLMCostDatasetResult:
    return _get_service().create_dataset(request)
