from __future__ import annotations

import os
import uuid
from datetime import datetime
from typing import Protocol

from src.schemas.experiment_runner import ExperimentMode, ExperimentRunHistoryResult, ExperimentRunRecord, ExperimentRunStatus


class RunHistoryStore(Protocol):
    def is_enabled(self) -> bool: ...

    def save_run(self, record: ExperimentRunRecord) -> str | None: ...

    def list_recent_runs(
        self,
        *,
        limit: int = 20,
        dataset_name: str | None = None,
        mode: ExperimentMode | None = None,
    ) -> ExperimentRunHistoryResult: ...


class PostgresRunHistoryStore:
    def __init__(self, dsn: str | None = None) -> None:
        self.dsn = dsn or os.getenv("DATABASE_URL") or os.getenv("POSTGRES_DSN")
        self._table_ensured = False

    def is_enabled(self) -> bool:
        return bool(self.dsn)

    def save_run(self, record: ExperimentRunRecord) -> str | None:
        if not self.is_enabled():
            return None
        self._ensure_table()
        import psycopg
        from psycopg.types.json import Jsonb

        with psycopg.connect(self.dsn) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO experiment_runs (
                        id, created_at, mode, dataset_name, run_name, description, status,
                        task_prompt_source, task_prompt_name, task_prompt_label, task_prompt_version, task_prompt_type, task_prompt_fingerprint,
                        judge_prompt_source, judge_prompt_name, judge_prompt_label, judge_prompt_version, judge_prompt_type, judge_prompt_fingerprint,
                        published_from_custom, published_at,
                        task_model, judge_model, metric_names, aggregate_metrics,
                        processed_items, failed_items, dataset_run_id, dataset_run_url, warnings, errors
                    ) VALUES (
                        %(id)s, %(created_at)s, %(mode)s, %(dataset_name)s, %(run_name)s, %(description)s, %(status)s,
                        %(task_prompt_source)s, %(task_prompt_name)s, %(task_prompt_label)s, %(task_prompt_version)s, %(task_prompt_type)s, %(task_prompt_fingerprint)s,
                        %(judge_prompt_source)s, %(judge_prompt_name)s, %(judge_prompt_label)s, %(judge_prompt_version)s, %(judge_prompt_type)s, %(judge_prompt_fingerprint)s,
                        %(published_from_custom)s, %(published_at)s,
                        %(task_model)s, %(judge_model)s, %(metric_names)s, %(aggregate_metrics)s,
                        %(processed_items)s, %(failed_items)s, %(dataset_run_id)s, %(dataset_run_url)s, %(warnings)s, %(errors)s
                    )
                    """,
                    {
                        "id": record.id,
                        "created_at": record.created_at,
                        "mode": record.mode.value,
                        "dataset_name": record.dataset_name,
                        "run_name": record.run_name,
                        "description": record.description,
                        "status": record.status.value,
                        "task_prompt_source": record.task_prompt_source.value,
                        "task_prompt_name": record.task_prompt_name,
                        "task_prompt_label": record.task_prompt_label,
                        "task_prompt_version": record.task_prompt_version,
                        "task_prompt_type": record.task_prompt_type.value if record.task_prompt_type else None,
                        "task_prompt_fingerprint": record.task_prompt_fingerprint,
                        "judge_prompt_source": record.judge_prompt_source.value,
                        "judge_prompt_name": record.judge_prompt_name,
                        "judge_prompt_label": record.judge_prompt_label,
                        "judge_prompt_version": record.judge_prompt_version,
                        "judge_prompt_type": record.judge_prompt_type.value if record.judge_prompt_type else None,
                        "judge_prompt_fingerprint": record.judge_prompt_fingerprint,
                        "published_from_custom": record.published_from_custom,
                        "published_at": record.published_at,
                        "task_model": record.task_model,
                        "judge_model": record.judge_model,
                        "metric_names": Jsonb(record.metric_names),
                        "aggregate_metrics": Jsonb([metric.model_dump() for metric in record.aggregate_metrics]),
                        "processed_items": record.processed_items,
                        "failed_items": record.failed_items,
                        "dataset_run_id": record.dataset_run_id,
                        "dataset_run_url": record.dataset_run_url,
                        "warnings": Jsonb(record.warnings),
                        "errors": Jsonb(record.errors),
                    },
                )
            conn.commit()
        return record.id

    def list_recent_runs(
        self,
        *,
        limit: int = 20,
        dataset_name: str | None = None,
        mode: ExperimentMode | None = None,
    ) -> ExperimentRunHistoryResult:
        if not self.is_enabled():
            return ExperimentRunHistoryResult(warnings=["DATABASE_URL veya POSTGRES_DSN ayarlanmamis."])
        self._ensure_table()
        import psycopg
        from psycopg.rows import dict_row

        filters: list[str] = []
        params: dict[str, object] = {"limit": limit}
        if dataset_name:
            filters.append("dataset_name = %(dataset_name)s")
            params["dataset_name"] = dataset_name
        if mode:
            filters.append("mode = %(mode)s")
            params["mode"] = mode.value
        where_clause = f"WHERE {' AND '.join(filters)}" if filters else ""
        with psycopg.connect(self.dsn, row_factory=dict_row) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    SELECT *
                    FROM experiment_runs
                    {where_clause}
                    ORDER BY created_at DESC
                    LIMIT %(limit)s
                    """,
                    params,
                )
                rows = cur.fetchall()
                cur.execute(
                    f"""
                    SELECT
                        COUNT(*) AS total_runs,
                        MAX(CASE WHEN status = 'succeeded' THEN created_at END) AS last_success_at,
                        MAX(CASE WHEN status = 'failed' THEN created_at END) AS last_error_at
                    FROM experiment_runs
                    {where_clause}
                    """,
                    {k: v for k, v in params.items() if k != "limit"},
                )
                summary = cur.fetchone() or {}
        records = [self._row_to_record(row) for row in rows]
        return ExperimentRunHistoryResult(
            records=records,
            total_runs=int(summary.get("total_runs") or 0),
            last_success_at=summary.get("last_success_at"),
            last_error_at=summary.get("last_error_at"),
        )

    def _ensure_table(self) -> None:
        if self._table_ensured or not self.is_enabled():
            return
        import psycopg

        with psycopg.connect(self.dsn) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS experiment_runs (
                        id TEXT PRIMARY KEY,
                        created_at TIMESTAMPTZ NOT NULL,
                        mode TEXT NOT NULL,
                        dataset_name TEXT NOT NULL,
                        run_name TEXT NOT NULL,
                        description TEXT NULL,
                        status TEXT NOT NULL,
                        task_prompt_source TEXT NOT NULL,
                        task_prompt_name TEXT NULL,
                        task_prompt_label TEXT NULL,
                        task_prompt_version INTEGER NULL,
                        task_prompt_type TEXT NULL,
                        task_prompt_fingerprint TEXT NULL,
                        judge_prompt_source TEXT NOT NULL,
                        judge_prompt_name TEXT NULL,
                        judge_prompt_label TEXT NULL,
                        judge_prompt_version INTEGER NULL,
                        judge_prompt_type TEXT NULL,
                        judge_prompt_fingerprint TEXT NULL,
                        published_from_custom BOOLEAN NOT NULL DEFAULT FALSE,
                        published_at TIMESTAMPTZ NULL,
                        task_model TEXT NULL,
                        judge_model TEXT NULL,
                        metric_names JSONB NOT NULL DEFAULT '[]'::jsonb,
                        aggregate_metrics JSONB NOT NULL DEFAULT '[]'::jsonb,
                        processed_items INTEGER NOT NULL DEFAULT 0,
                        failed_items INTEGER NOT NULL DEFAULT 0,
                        dataset_run_id TEXT NULL,
                        dataset_run_url TEXT NULL,
                        warnings JSONB NOT NULL DEFAULT '[]'::jsonb,
                        errors JSONB NOT NULL DEFAULT '[]'::jsonb
                    )
                    """
                )
                cur.execute(
                    "ALTER TABLE experiment_runs ADD COLUMN IF NOT EXISTS task_prompt_label TEXT NULL"
                )
                cur.execute(
                    "ALTER TABLE experiment_runs ADD COLUMN IF NOT EXISTS task_prompt_version INTEGER NULL"
                )
                cur.execute(
                    "ALTER TABLE experiment_runs ADD COLUMN IF NOT EXISTS judge_prompt_label TEXT NULL"
                )
                cur.execute(
                    "ALTER TABLE experiment_runs ADD COLUMN IF NOT EXISTS judge_prompt_version INTEGER NULL"
                )
                cur.execute(
                    "ALTER TABLE experiment_runs ADD COLUMN IF NOT EXISTS published_from_custom BOOLEAN NOT NULL DEFAULT FALSE"
                )
                cur.execute(
                    "ALTER TABLE experiment_runs ADD COLUMN IF NOT EXISTS published_at TIMESTAMPTZ NULL"
                )
                cur.execute(
                    "CREATE INDEX IF NOT EXISTS idx_experiment_runs_created_at ON experiment_runs (created_at DESC)"
                )
                cur.execute(
                    "CREATE INDEX IF NOT EXISTS idx_experiment_runs_dataset_name ON experiment_runs (dataset_name)"
                )
                cur.execute(
                    "CREATE INDEX IF NOT EXISTS idx_experiment_runs_mode ON experiment_runs (mode)"
                )
            conn.commit()
        self._table_ensured = True

    @staticmethod
    def _row_to_record(row: dict[str, object]) -> ExperimentRunRecord:
        return ExperimentRunRecord.model_validate(row)


class InMemoryRunHistoryStore:
    def __init__(self, records: list[ExperimentRunRecord] | None = None) -> None:
        self.records = list(records or [])

    def is_enabled(self) -> bool:
        return True

    def save_run(self, record: ExperimentRunRecord) -> str | None:
        self.records.insert(0, record)
        return record.id

    def list_recent_runs(
        self,
        *,
        limit: int = 20,
        dataset_name: str | None = None,
        mode: ExperimentMode | None = None,
    ) -> ExperimentRunHistoryResult:
        records = self.records
        if dataset_name:
            records = [record for record in records if record.dataset_name == dataset_name]
        if mode:
            records = [record for record in records if record.mode == mode]
        total_runs = len(records)
        last_success_at = next(
            (record.created_at for record in records if record.status == ExperimentRunStatus.SUCCEEDED),
            None,
        )
        last_error_at = next(
            (record.created_at for record in records if record.status == ExperimentRunStatus.FAILED),
            None,
        )
        records = records[:limit]
        return ExperimentRunHistoryResult(
            records=records,
            total_runs=total_runs,
            last_success_at=last_success_at,
            last_error_at=last_error_at,
        )



def build_run_record_id() -> str:
    return str(uuid.uuid4())
