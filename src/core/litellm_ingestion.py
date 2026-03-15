from __future__ import annotations

from src.core.litellm_store import LiteLLMStore, PostgresLiteLLMStore
from src.schemas.litellm_ingestion import (
    LiteLLMIngestionRequest,
    LiteLLMIngestionResult,
)


class LiteLLMIngestionWriterService:
    def __init__(self, store: LiteLLMStore | None = None) -> None:
        self.store = store or PostgresLiteLLMStore()

    def ingest_rows(self, request: LiteLLMIngestionRequest) -> LiteLLMIngestionResult:
        if not request.rows:
            return LiteLLMIngestionResult(
                requested_rows=0,
                upserted_rows=0,
                errors=["Yazmak icin en az bir LiteLLM ingestion row gereklidir."],
            )

        warnings = self.store.ensure_schema()
        upserted_rows = self.store.upsert_requests([row.model_dump(mode="python") for row in request.rows])
        return LiteLLMIngestionResult(
            requested_rows=len(request.rows),
            upserted_rows=upserted_rows,
            warnings=warnings,
            errors=[],
        )


_DEFAULT_SERVICE: LiteLLMIngestionWriterService | None = None


def _get_service() -> LiteLLMIngestionWriterService:
    global _DEFAULT_SERVICE
    if _DEFAULT_SERVICE is None:
        _DEFAULT_SERVICE = LiteLLMIngestionWriterService()
    return _DEFAULT_SERVICE


def ingest_litellm_rows(request: LiteLLMIngestionRequest) -> LiteLLMIngestionResult:
    return _get_service().ingest_rows(request)
