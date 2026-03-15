from __future__ import annotations

import os
import re
from datetime import UTC, datetime
from typing import Any, Protocol

from src.schemas.litellm_cost_builder import (
    DEFAULT_LITELLM_LOG_TABLE,
    LiteLLMCostFilters,
    LiteLLMFieldMapping,
    LiteLLMStoreConfig,
)


_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class LiteLLMStore(Protocol):
    def get_config(self) -> LiteLLMStoreConfig: ...

    def list_requests(self, filters: LiteLLMCostFilters) -> tuple[list[dict[str, Any]], list[str]]: ...

    def ensure_schema(self) -> list[str]: ...

    def upsert_requests(self, rows: list[dict[str, Any]]) -> int: ...


class PostgresLiteLLMStore:
    def __init__(self, dsn: str | None = None, mapping: LiteLLMFieldMapping | None = None) -> None:
        self.dsn = dsn or os.getenv("LITELLM_DATABASE_URL") or os.getenv("DATABASE_URL")
        self.mapping = mapping or self._mapping_from_env()
        self.timeout_seconds = max(5, int(os.getenv("LITELLM_DB_TIMEOUT_SECONDS", "30")))
        self.auto_create_table = os.getenv("LITELLM_AUTO_CREATE_TABLE", "true").strip().lower() not in {
            "0",
            "false",
            "no",
        }
        self._table_bootstrapped = False

    def get_config(self) -> LiteLLMStoreConfig:
        warnings: list[str] = []
        if not self.dsn:
            warnings.append("LITELLM_DATABASE_URL veya DATABASE_URL ayarlanmamis.")
        missing_fields = self.mapping.missing_required_fields()
        if missing_fields:
            warnings.append("LiteLLM field mapping eksik.")
        return LiteLLMStoreConfig(
            enabled=bool(self.dsn and not missing_fields),
            dsn_present=bool(self.dsn),
            timeout_seconds=self.timeout_seconds,
            auto_create_table=self.auto_create_table,
            schema_mode="code_first",
            table_bootstrapped=self._table_bootstrapped,
            mapping=self.mapping,
            missing_fields=missing_fields,
            warnings=warnings,
        )

    def ensure_schema(self) -> list[str]:
        config = self.get_config()
        if not config.enabled or not self.dsn:
            return config.warnings
        if self._table_bootstrapped or not self.auto_create_table:
            return []

        import psycopg

        create_table_sql = self._create_table_sql()
        create_indexes_sql = self._create_index_statements()
        with psycopg.connect(self.dsn) as conn:
            with conn.cursor() as cur:
                cur.execute(create_table_sql)
                for statement in create_indexes_sql:
                    cur.execute(statement)
            conn.commit()
        self._table_bootstrapped = True
        return []

    def list_requests(self, filters: LiteLLMCostFilters) -> tuple[list[dict[str, Any]], list[str]]:
        config = self.get_config()
        if not config.enabled or not self.dsn:
            return [], config.warnings

        warnings = [*self.ensure_schema()]

        import psycopg
        from psycopg import sql
        from psycopg.rows import dict_row

        params: dict[str, Any] = {"limit": max(1, min(filters.limit, 1000))}
        select_clauses = [
            self._select_expr(self.mapping.id_column, "request_id", "text"),
            self._select_expr(self.mapping.created_at_column, "created_at", "timestamptz"),
            self._select_expr(self.mapping.model_column, "model_name", "text"),
            self._select_expr(self.mapping.provider_column, "provider", "text"),
            self._select_expr(self.mapping.total_cost_column, "total_cost", "double precision"),
            self._select_expr(self.mapping.input_tokens_column, "input_tokens", "integer"),
            self._select_expr(self.mapping.output_tokens_column, "output_tokens", "integer"),
            self._select_expr(self.mapping.total_tokens_column, "total_tokens", "integer"),
            self._select_expr(self.mapping.latency_ms_column, "latency_ms", "double precision"),
            self._select_expr(self.mapping.status_column, "status", "text"),
            self._select_expr(self.mapping.input_column, "request_input", "jsonb"),
            self._select_expr(self.mapping.output_column, "request_output", "jsonb"),
            self._select_expr(self.mapping.metadata_column, "metadata", "jsonb"),
            self._select_expr(self.mapping.langfuse_trace_id_column, "langfuse_trace_id", "text"),
            self._select_expr(self.mapping.langfuse_observation_id_column, "langfuse_observation_id", "text"),
        ]

        where_clauses = [sql.SQL("1=1")]
        if filters.from_date and self.mapping.created_at_column:
            where_clauses.append(
                sql.SQL("{} >= %(from_date)s").format(self._identifier(self.mapping.created_at_column))
            )
            params["from_date"] = filters.from_date
        if filters.to_date and self.mapping.created_at_column:
            where_clauses.append(
                sql.SQL("{} <= %(to_date)s").format(self._identifier(self.mapping.created_at_column))
            )
            params["to_date"] = filters.to_date
        if filters.model_names and self.mapping.model_column:
            where_clauses.append(
                sql.SQL("{} = ANY(%(model_names)s)").format(self._identifier(self.mapping.model_column))
            )
            params["model_names"] = filters.model_names
        if filters.providers:
            if self.mapping.provider_column:
                where_clauses.append(
                    sql.SQL("{} = ANY(%(providers)s)").format(self._identifier(self.mapping.provider_column))
                )
                params["providers"] = filters.providers
            else:
                warnings.append("Provider filtresi verildi ancak provider column map edilmemis.")
        if filters.statuses:
            if self.mapping.status_column:
                where_clauses.append(
                    sql.SQL("{} = ANY(%(statuses)s)").format(self._identifier(self.mapping.status_column))
                )
                params["statuses"] = filters.statuses
            else:
                warnings.append("Status filtresi verildi ancak status column map edilmemis.")

        total_tokens_expr = self._total_tokens_expression()
        if filters.min_cost is not None and self.mapping.total_cost_column:
            where_clauses.append(
                sql.SQL("{} >= %(min_cost)s").format(self._identifier(self.mapping.total_cost_column))
            )
            params["min_cost"] = filters.min_cost
        if filters.max_cost is not None and self.mapping.total_cost_column:
            where_clauses.append(
                sql.SQL("{} <= %(max_cost)s").format(self._identifier(self.mapping.total_cost_column))
            )
            params["max_cost"] = filters.max_cost
        if filters.min_total_tokens is not None and total_tokens_expr is not None:
            where_clauses.append(sql.SQL("{} >= %(min_total_tokens)s").format(total_tokens_expr))
            params["min_total_tokens"] = filters.min_total_tokens
        if filters.max_total_tokens is not None and total_tokens_expr is not None:
            where_clauses.append(sql.SQL("{} <= %(max_total_tokens)s").format(total_tokens_expr))
            params["max_total_tokens"] = filters.max_total_tokens
        if filters.min_latency_ms is not None and self.mapping.latency_ms_column:
            where_clauses.append(
                sql.SQL("{} >= %(min_latency_ms)s").format(self._identifier(self.mapping.latency_ms_column))
            )
            params["min_latency_ms"] = filters.min_latency_ms
        if filters.max_latency_ms is not None and self.mapping.latency_ms_column:
            where_clauses.append(
                sql.SQL("{} <= %(max_latency_ms)s").format(self._identifier(self.mapping.latency_ms_column))
            )
            params["max_latency_ms"] = filters.max_latency_ms
        if filters.require_langfuse_join:
            if self.mapping.langfuse_trace_id_column or self.mapping.langfuse_observation_id_column:
                join_checks: list[sql.Composed] = []
                if self.mapping.langfuse_trace_id_column:
                    join_checks.append(
                        sql.SQL("{} IS NOT NULL").format(self._identifier(self.mapping.langfuse_trace_id_column))
                    )
                if self.mapping.langfuse_observation_id_column:
                    join_checks.append(
                        sql.SQL("{} IS NOT NULL").format(
                            self._identifier(self.mapping.langfuse_observation_id_column)
                        )
                    )
                where_clauses.append(sql.SQL("(") + sql.SQL(" OR ").join(join_checks) + sql.SQL(")"))
            else:
                warnings.append("Langfuse join istendi ancak trace/observation id column'lari map edilmemis.")

        query = sql.SQL(
            "SELECT {fields} FROM {table} WHERE {where_clause} ORDER BY {created_at} DESC LIMIT %(limit)s"
        ).format(
            fields=sql.SQL(", ").join(select_clauses),
            table=self._identifier(self.mapping.table_name),
            where_clause=sql.SQL(" AND ").join(where_clauses),
            created_at=self._identifier(self.mapping.created_at_column),
        )

        with psycopg.connect(self.dsn, row_factory=dict_row) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    query,
                    params,
                    prepare=False,
                )
                rows = [dict(row) for row in cur.fetchall()]
        return rows, warnings

    def upsert_requests(self, rows: list[dict[str, Any]]) -> int:
        config = self.get_config()
        if not config.enabled or not self.dsn:
            raise ValueError("LiteLLM store kullanima hazir degil. DSN veya mapping eksik.")
        if not rows:
            return 0

        self.ensure_schema()

        import psycopg
        from psycopg import sql
        from psycopg.types.json import Jsonb

        columns = [
            self.mapping.id_column,
            self.mapping.created_at_column,
            self.mapping.model_column,
            self.mapping.provider_column,
            self.mapping.total_cost_column,
            self.mapping.input_tokens_column,
            self.mapping.output_tokens_column,
            self.mapping.total_tokens_column,
            self.mapping.latency_ms_column,
            self.mapping.status_column,
            self.mapping.input_column,
            self.mapping.output_column,
            self.mapping.metadata_column,
            self.mapping.langfuse_trace_id_column,
            self.mapping.langfuse_observation_id_column,
        ]
        placeholders = sql.SQL(", ").join(sql.Placeholder() for _ in columns)
        insert_query = sql.SQL(
            "INSERT INTO {table} ({columns}) VALUES ({placeholders}) "
            "ON CONFLICT ({pk}) DO UPDATE SET {updates}"
        ).format(
            table=self._identifier(self.mapping.table_name),
            columns=sql.SQL(", ").join(self._identifier(column) for column in columns),
            placeholders=placeholders,
            pk=self._identifier(self.mapping.id_column),
            updates=sql.SQL(", ").join(
                sql.SQL("{} = EXCLUDED.{}").format(self._identifier(column), self._identifier(column))
                for column in columns[1:]
            ),
        )

        values = []
        for row in rows:
            input_tokens = self._to_int(row.get("input_tokens"))
            output_tokens = self._to_int(row.get("output_tokens"))
            total_tokens = self._to_int(row.get("total_tokens"))
            if total_tokens == 0 and (input_tokens or output_tokens):
                total_tokens = input_tokens + output_tokens
            values.append(
                (
                    str(row.get("request_id")),
                    row.get("created_at") or datetime.now(UTC),
                    str(row.get("model_name")),
                    self._nullable_str(row.get("provider")),
                    self._to_float(row.get("total_cost")),
                    input_tokens,
                    output_tokens,
                    total_tokens,
                    self._nullable_float(row.get("latency_ms")),
                    self._nullable_str(row.get("status")),
                    Jsonb(row.get("request_input")) if row.get("request_input") is not None else None,
                    Jsonb(row.get("request_output")) if row.get("request_output") is not None else None,
                    Jsonb(row.get("metadata")) if row.get("metadata") is not None else None,
                    self._nullable_str(row.get("langfuse_trace_id")),
                    self._nullable_str(row.get("langfuse_observation_id")),
                )
            )

        with psycopg.connect(self.dsn) as conn:
            with conn.cursor() as cur:
                cur.executemany(insert_query, values)
            conn.commit()
        self._table_bootstrapped = True
        return len(values)

    @staticmethod
    def _identifier(name: str | None) -> Any:
        from psycopg import sql

        if not name:
            raise ValueError("Missing SQL identifier")
        parts = name.split(".")
        for part in parts:
            if not _IDENT_RE.match(part):
                raise ValueError(f"Invalid SQL identifier: {name}")
        return sql.Identifier(*parts)

    def _select_expr(self, column_name: str | None, alias: str, null_cast: str) -> Any:
        from psycopg import sql

        if column_name:
            return sql.SQL("{} AS {}").format(self._identifier(column_name), sql.Identifier(alias))
        return sql.SQL(f"NULL::{null_cast} AS {{}}").format(sql.Identifier(alias))

    def _total_tokens_expression(self) -> Any:
        from psycopg import sql

        if self.mapping.total_tokens_column:
            return self._identifier(self.mapping.total_tokens_column)
        if self.mapping.input_tokens_column and self.mapping.output_tokens_column:
            return sql.SQL("COALESCE({}, 0) + COALESCE({}, 0)").format(
                self._identifier(self.mapping.input_tokens_column),
                self._identifier(self.mapping.output_tokens_column),
            )
        return None

    @staticmethod
    def _mapping_from_env() -> LiteLLMFieldMapping:
        defaults = LiteLLMFieldMapping()
        return LiteLLMFieldMapping(
            table_name=os.getenv("LITELLM_LOG_TABLE") or defaults.table_name,
            id_column=os.getenv("LITELLM_ID_COLUMN") or defaults.id_column,
            created_at_column=os.getenv("LITELLM_CREATED_AT_COLUMN") or defaults.created_at_column,
            model_column=os.getenv("LITELLM_MODEL_COLUMN") or defaults.model_column,
            provider_column=os.getenv("LITELLM_PROVIDER_COLUMN") or defaults.provider_column,
            total_cost_column=os.getenv("LITELLM_COST_COLUMN") or defaults.total_cost_column,
            input_tokens_column=os.getenv("LITELLM_INPUT_TOKENS_COLUMN") or defaults.input_tokens_column,
            output_tokens_column=os.getenv("LITELLM_OUTPUT_TOKENS_COLUMN") or defaults.output_tokens_column,
            total_tokens_column=os.getenv("LITELLM_TOTAL_TOKENS_COLUMN") or defaults.total_tokens_column,
            latency_ms_column=os.getenv("LITELLM_LATENCY_MS_COLUMN") or defaults.latency_ms_column,
            status_column=os.getenv("LITELLM_STATUS_COLUMN") or defaults.status_column,
            input_column=os.getenv("LITELLM_INPUT_COLUMN") or defaults.input_column,
            output_column=os.getenv("LITELLM_OUTPUT_COLUMN") or defaults.output_column,
            metadata_column=os.getenv("LITELLM_METADATA_COLUMN") or defaults.metadata_column,
            langfuse_trace_id_column=os.getenv("LITELLM_LANGFUSE_TRACE_ID_COLUMN")
            or defaults.langfuse_trace_id_column,
            langfuse_observation_id_column=os.getenv("LITELLM_LANGFUSE_OBSERVATION_ID_COLUMN")
            or defaults.langfuse_observation_id_column,
        )

    def _create_table_sql(self) -> str:
        table_name = self.mapping.table_name or DEFAULT_LITELLM_LOG_TABLE
        table_sql = self._identifier_sql(table_name)
        return f"""
CREATE TABLE IF NOT EXISTS {table_sql} (
    {self._identifier_sql(self.mapping.id_column)} TEXT PRIMARY KEY,
    {self._identifier_sql(self.mapping.created_at_column)} TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    {self._identifier_sql(self.mapping.model_column)} TEXT NOT NULL,
    {self._identifier_sql(self.mapping.provider_column)} TEXT NULL,
    {self._identifier_sql(self.mapping.total_cost_column)} DOUBLE PRECISION NOT NULL DEFAULT 0,
    {self._identifier_sql(self.mapping.input_tokens_column)} INTEGER NOT NULL DEFAULT 0,
    {self._identifier_sql(self.mapping.output_tokens_column)} INTEGER NOT NULL DEFAULT 0,
    {self._identifier_sql(self.mapping.total_tokens_column)} INTEGER NOT NULL DEFAULT 0,
    {self._identifier_sql(self.mapping.latency_ms_column)} DOUBLE PRECISION NULL,
    {self._identifier_sql(self.mapping.status_column)} TEXT NULL,
    {self._identifier_sql(self.mapping.input_column)} JSONB NULL,
    {self._identifier_sql(self.mapping.output_column)} JSONB NULL,
    {self._identifier_sql(self.mapping.metadata_column)} JSONB NULL,
    {self._identifier_sql(self.mapping.langfuse_trace_id_column)} TEXT NULL,
    {self._identifier_sql(self.mapping.langfuse_observation_id_column)} TEXT NULL
)
""".strip()

    def _create_index_statements(self) -> list[str]:
        table_name = self.mapping.table_name or DEFAULT_LITELLM_LOG_TABLE
        table_sql = self._identifier_sql(table_name)
        safe_table_name = table_name.replace(".", "_")
        indexed_columns = [
            self.mapping.created_at_column,
            self.mapping.model_column,
            self.mapping.provider_column,
            self.mapping.status_column,
            self.mapping.langfuse_trace_id_column,
            self.mapping.langfuse_observation_id_column,
        ]
        statements: list[str] = []
        for column_name in indexed_columns:
            if not column_name:
                continue
            column_sql = self._identifier_sql(column_name)
            index_name = re.sub(r"[^a-z0-9_]", "_", f"idx_{safe_table_name}_{column_name}".lower())
            if len(index_name) > 63:
                index_name = index_name[:63]
            statements.append(f'CREATE INDEX IF NOT EXISTS "{index_name}" ON {table_sql} ({column_sql})')
        return statements

    @staticmethod
    def _identifier_sql(name: str | None) -> str:
        if not name:
            raise ValueError("Missing SQL identifier")
        parts = name.split(".")
        for part in parts:
            if not _IDENT_RE.match(part):
                raise ValueError(f"Invalid SQL identifier: {name}")
        return ".".join(f'"{part}"' for part in parts)

    @staticmethod
    def _to_int(value: Any) -> int:
        if value is None:
            return 0
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
        if value is None:
            return 0.0
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
        if value is None or value == "":
            return None
        return PostgresLiteLLMStore._to_float(value)

    @staticmethod
    def _nullable_str(value: Any) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        return text or None
