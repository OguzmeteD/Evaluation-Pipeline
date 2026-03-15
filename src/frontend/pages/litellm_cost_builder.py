from __future__ import annotations

import json
from datetime import date, datetime, time, timedelta
from typing import Any

try:
    import streamlit as st
except ModuleNotFoundError:  # pragma: no cover
    st = None

from src.core.experiment_runner import fetch_dataset_by_name
from src.core.litellm_ingestion import ingest_litellm_rows
from src.core.litellm_cost_builder import (
    create_litellm_cost_dataset,
    ensure_litellm_store_schema,
    get_litellm_store_config,
    preview_litellm_cost_candidates,
)
from src.schemas.litellm_ingestion import LiteLLMIngestionRequest, LiteLLMIngestionResult, LiteLLMIngestionRow
from src.schemas.litellm_cost_builder import (
    LiteLLMCostCandidateRow,
    LiteLLMCostDatasetPreview,
    LiteLLMCostDatasetRequest,
    LiteLLMCostDatasetResult,
    LiteLLMCostFilters,
)


PAGE_TITLE = "LiteLLM Cost Builder"


def render() -> None:
    if st is None:  # pragma: no cover
        raise RuntimeError("streamlit is required to render the LiteLLM Cost Builder page.")
    st.title(PAGE_TITLE)
    st.caption("Mevcut LiteLLM Postgres log/store verisinden request-level cost dataset olusturun.")

    config = get_litellm_store_config()
    config = _render_store_config(config)
    _render_ingestion_writer(config)
    filters = _build_filters()
    target = _render_dataset_target()

    if st.button(
        "Preview LiteLLM cost rows",
        type="primary",
        width="stretch",
        disabled=not config.enabled,
        key="litellm_cost_preview_button",
    ):
        st.session_state["litellm_cost_preview"] = preview_litellm_cost_candidates(filters)
        st.session_state.pop("litellm_cost_result", None)

    preview = st.session_state.get("litellm_cost_preview")
    if isinstance(preview, LiteLLMCostDatasetPreview):
        _render_preview(preview)

    selected_rows = _selected_rows_from_state(preview) if isinstance(preview, LiteLLMCostDatasetPreview) else []
    if st.button(
        "Create Langfuse dataset from LiteLLM costs",
        width="stretch",
        disabled=(
            not config.enabled
            or not target["dataset_name"]
            or bool(target["metadata_error"])
            or not selected_rows
        ),
        key="litellm_cost_create_button",
    ):
        st.session_state["litellm_cost_result"] = create_litellm_cost_dataset(
            LiteLLMCostDatasetRequest(
                dataset_name=target["dataset_name"],
                description=target["description"] or None,
                metadata=target["metadata"],
                filters=filters,
                rows=selected_rows,
            )
        )

    result = st.session_state.get("litellm_cost_result")
    if isinstance(result, LiteLLMCostDatasetResult):
        _render_result(result)


def _render_store_config(config):
    st.subheader("Store Config")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Configured", "yes" if config.enabled else "no")
    c2.metric("DSN present", "yes" if config.dsn_present else "no")
    c3.metric("Timeout", f"{config.timeout_seconds}s")
    c4.metric("Bootstrapped", "yes" if config.table_bootstrapped else "no")
    if config.auto_create_table:
        st.caption("Code-first tablo modu aktif. Preview sirasinda tablo yoksa otomatik olusturulur.")
    else:
        st.caption("Code-first tablo modu pasif. Tabloyu elle olusturman veya env ile mevcut tabloyu map etmen gerekir.")
    if st.button(
        "Ensure LiteLLM table exists",
        width="stretch",
        disabled=not config.dsn_present,
        key="litellm_ensure_schema_button",
    ):
        warnings = ensure_litellm_store_schema()
        if warnings:
            for warning in warnings:
                st.warning(warning)
        else:
            st.success("LiteLLM request log tablosu kontrol edildi ve hazir.")
        config = get_litellm_store_config()
    st.dataframe(
        [
            {
                "field": field,
                "value": value,
            }
            for field, value in config.mapping.model_dump().items()
        ],
        width="stretch",
        hide_index=True,
    )
    for warning in config.warnings:
        st.warning(warning)
    if config.missing_fields:
        st.caption("Eksik env mapping: " + ", ".join(config.missing_fields))
    return config


def _render_ingestion_writer(config) -> None:
    with st.expander("Ingestion Writer", expanded=False):
        st.caption("Canonical LiteLLM request row'larini JSON olarak tabloya upsert edin.")
        sample = [
            {
                "request_id": "req-123",
                "created_at": "2026-03-16T10:00:00Z",
                "model_name": "gpt-4.1-mini",
                "provider": "openai",
                "total_cost": 0.018,
                "input_tokens": 120,
                "output_tokens": 45,
                "total_tokens": 165,
                "latency_ms": 420.5,
                "status": "success",
                "request_input": {"messages": [{"role": "user", "content": "Hello"}]},
                "request_output": {"output_text": "Hi"},
                "metadata": {"source": "litellm_proxy"},
                "langfuse_trace_id": "trace-123",
            }
        ]
        st.code(json.dumps(sample, indent=2, ensure_ascii=False), language="json")
        payload = st.text_area(
            "Rows JSON",
            key="litellm_ingestion_payload",
            height=220,
            placeholder='[{"request_id":"req-123","model_name":"gpt-4.1-mini","total_cost":0.01}]',
        )
        rows, error = _parse_ingestion_rows(payload)
        if error:
            st.warning(error)
        if st.button(
            "Write rows to LiteLLM table",
            width="stretch",
            disabled=not config.enabled or bool(error) or not rows,
            key="litellm_ingestion_write_button",
        ):
            st.session_state["litellm_ingestion_result"] = ingest_litellm_rows(
                LiteLLMIngestionRequest(rows=rows)
            )
        result = st.session_state.get("litellm_ingestion_result")
        if isinstance(result, LiteLLMIngestionResult):
            c1, c2 = st.columns(2)
            c1.metric("Requested rows", result.requested_rows)
            c2.metric("Upserted rows", result.upserted_rows)
            for warning in result.warnings:
                st.warning(warning)
            for err in result.errors:
                st.error(err)


def _build_filters() -> LiteLLMCostFilters:
    st.subheader("Cost Filters")
    today = date.today()
    default_from = today - timedelta(days=14)
    date_range = st.date_input(
        "Date range",
        value=(default_from, today),
        key="litellm_cost_date_range",
    )
    if isinstance(date_range, tuple) and len(date_range) == 2:
        from_date, to_date = date_range
    else:
        from_date, to_date = default_from, today
    model_names = st.text_input(
        "Model names (comma-separated)",
        key="litellm_cost_models",
        placeholder="gpt-4.1, claude-3-7-sonnet",
    )
    providers = st.text_input(
        "Providers (comma-separated)",
        key="litellm_cost_providers",
        placeholder="openai, anthropic",
    )
    statuses = st.text_input(
        "Statuses (comma-separated)",
        key="litellm_cost_statuses",
        placeholder="success, error",
    )
    r1, r2, r3 = st.columns(3)
    with r1:
        min_cost = st.number_input("Min cost", min_value=0.0, value=0.0, step=0.01, key="litellm_min_cost")
        max_cost = st.number_input("Max cost", min_value=0.0, value=100.0, step=0.1, key="litellm_max_cost")
    with r2:
        min_tokens = st.number_input("Min total tokens", min_value=0, value=0, step=100, key="litellm_min_tokens")
        max_tokens = st.number_input("Max total tokens", min_value=0, value=500000, step=100, key="litellm_max_tokens")
    with r3:
        min_latency = st.number_input("Min latency ms", min_value=0.0, value=0.0, step=10.0, key="litellm_min_latency")
        max_latency = st.number_input("Max latency ms", min_value=0.0, value=600000.0, step=10.0, key="litellm_max_latency")
    require_join = st.checkbox(
        "Only rows with Langfuse join",
        key="litellm_require_join",
        value=False,
    )
    limit = st.slider("Preview limit", min_value=10, max_value=1000, value=100, key="litellm_cost_limit")
    return LiteLLMCostFilters(
        from_date=datetime.combine(from_date, time.min),
        to_date=datetime.combine(to_date, time.max),
        model_names=_split_csv(model_names),
        providers=_split_csv(providers),
        statuses=_split_csv(statuses),
        min_cost=None if min_cost <= 0 else float(min_cost),
        max_cost=None if max_cost <= 0 else float(max_cost),
        min_total_tokens=None if min_tokens <= 0 else int(min_tokens),
        max_total_tokens=None if max_tokens <= 0 else int(max_tokens),
        min_latency_ms=None if min_latency <= 0 else float(min_latency),
        max_latency_ms=None if max_latency <= 0 else float(max_latency),
        require_langfuse_join=require_join,
        limit=limit,
    )


def _render_dataset_target() -> dict[str, Any]:
    st.subheader("Create Langfuse Dataset")
    dataset_name = st.text_input(
        "Dataset name",
        key="litellm_cost_dataset_name",
        placeholder="litellm-cost-requests-v1",
    ).strip()
    description = st.text_area(
        "Description",
        key="litellm_cost_description",
        height=80,
        placeholder="LiteLLM request-level cost dataset",
    )
    metadata_text = st.text_area(
        "Metadata JSON (optional)",
        key="litellm_cost_metadata",
        height=100,
        placeholder='{"source":"litellm_cost_builder","grain":"request"}',
    )
    metadata, metadata_error = _parse_json_object(metadata_text)
    if metadata_error:
        st.warning(metadata_error)
    return {
        "dataset_name": dataset_name,
        "description": description,
        "metadata": metadata,
        "metadata_error": metadata_error,
    }


def _render_preview(preview: LiteLLMCostDatasetPreview) -> None:
    st.subheader("Preview")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Requests", preview.summary.total_requests)
    c2.metric("Total cost", _format_float(preview.summary.total_cost))
    c3.metric("Total tokens", preview.summary.total_tokens)
    c4.metric("Avg latency", _format_float(preview.summary.avg_latency_ms))
    for warning in preview.warnings:
        st.warning(warning)
    if not preview.rows:
        st.info("Secilen filtrelerle LiteLLM request kaydi bulunamadi.")
        return
    selected_request_ids = st.multiselect(
        "Selected requests for dataset",
        options=[row.request_id for row in preview.rows],
        default=[row.request_id for row in preview.rows],
        format_func=lambda value: _format_row_label(preview, value),
        key="litellm_cost_selected_request_ids",
    )
    st.metric("Selected for create", len(selected_request_ids))
    st.dataframe(
        [
            {
                "request_id": row.request_id,
                "created_at": row.created_at,
                "model_name": row.model_name,
                "provider": row.provider,
                "total_cost": row.total_cost,
                "total_tokens": row.total_tokens,
                "latency_ms": row.latency_ms,
                "status": row.status,
                "langfuse_trace_id": row.langfuse_trace_id,
            }
            for row in preview.rows
        ],
        width="stretch",
        hide_index=True,
    )
    selected_request_id = st.selectbox(
        "Request detail",
        options=[row.request_id for row in preview.rows],
        format_func=lambda value: _format_row_label(preview, value),
        key="litellm_cost_detail_request",
    )
    selected = next(row for row in preview.rows if row.request_id == selected_request_id)
    left, right = st.columns(2)
    with left:
        st.markdown("### Request input")
        _render_value_block("Request input", selected.request_input)
        st.markdown("### Request output")
        _render_value_block("Request output", selected.request_output)
    with right:
        st.markdown("### Metadata")
        _render_value_block("Metadata", selected.metadata)
        st.markdown("### Join summary")
        st.json(
            {
                "langfuse_trace_id": selected.langfuse_trace_id,
                "langfuse_observation_id": selected.langfuse_observation_id,
                "provider": selected.provider,
                "status": selected.status,
            }
        )


def _render_result(result: LiteLLMCostDatasetResult) -> None:
    st.subheader("Result")
    c1, c2, c3 = st.columns(3)
    c1.metric("Dataset name", result.dataset_name)
    c2.metric("Created items", result.created_items)
    c3.metric("Failed items", result.failed_items)
    if result.dataset_id:
        st.caption(f"Dataset ID: {result.dataset_id}")
    for warning in result.warnings:
        st.warning(warning)
    for error in result.errors:
        st.error(error)
    if result.item_ids:
        st.dataframe([{"item_id": item_id} for item_id in result.item_ids], width="stretch", hide_index=True)
    if result.dataset_id and not result.errors:
        if st.button("Open in Experiment Studio", width="stretch", key="litellm_cost_open_in_studio"):
            _send_dataset_to_experiment_studio(result.dataset_name)
            st.rerun()


def _selected_rows_from_state(preview: LiteLLMCostDatasetPreview) -> list[LiteLLMCostCandidateRow]:
    selected_ids = st.session_state.get("litellm_cost_selected_request_ids") or [
        row.request_id for row in preview.rows
    ]
    selected_set = set(selected_ids)
    return [row for row in preview.rows if row.request_id in selected_set]


def _parse_ingestion_rows(payload: str) -> tuple[list[LiteLLMIngestionRow], str | None]:
    if not payload.strip():
        return [], None
    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError as exc:
        return [], f"Gecersiz ingestion JSON: {exc}"
    if isinstance(parsed, dict):
        parsed = [parsed]
    if not isinstance(parsed, list):
        return [], "Ingestion payload JSON object veya JSON array olmalidir."
    rows: list[LiteLLMIngestionRow] = []
    try:
        for item in parsed:
            if not isinstance(item, dict):
                return [], "Her ingestion satiri JSON object olmali."
            rows.append(LiteLLMIngestionRow.model_validate(item))
    except Exception as exc:
        return [], f"Ingestion row dogrulanamadi: {exc}"
    return rows, None


def _format_row_label(preview: LiteLLMCostDatasetPreview, request_id: str) -> str:
    row = next(item for item in preview.rows if item.request_id == request_id)
    return f"{row.model_name or 'unknown-model'} | cost={_format_float(row.total_cost)} | {request_id}"


def _send_dataset_to_experiment_studio(dataset_name: str) -> None:
    st.session_state["studio_dataset_name"] = dataset_name
    try:
        st.session_state["studio_dataset"] = fetch_dataset_by_name(dataset_name)
        st.session_state["studio_prompt_apply_message"] = (
            f"Dataset '{dataset_name}' Experiment Studio icine yuklendi."
        )
    except Exception as exc:  # pragma: no cover
        st.session_state["studio_prompt_apply_message"] = (
            f"Dataset adi tasindi ancak Experiment Studio fetch basarisiz oldu: {exc}"
        )
    st.session_state["pending_active_page"] = "Experiment Studio"


def _parse_json_object(value: str) -> tuple[dict[str, Any] | None, str | None]:
    if not value.strip():
        return None, None
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        return None, f"Metadata JSON parse edilemedi: {exc}"
    if not isinstance(parsed, dict):
        return None, "Metadata JSON object olmalidir."
    return parsed, None


def _split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _format_float(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def _render_value_block(label: str, value: Any) -> None:
    if value is None:
        st.caption("Veri yok.")
        return
    if isinstance(value, (dict, list)):
        st.json(value)
        return
    if isinstance(value, (int, float, bool)):
        st.code(str(value), language="text")
        return
    if isinstance(value, str):
        stripped = value.strip()
        if stripped:
            try:
                parsed = json.loads(stripped)
            except json.JSONDecodeError:
                parsed = None
            if isinstance(parsed, (dict, list)):
                st.json(parsed)
                return
        st.text_area(label, value=value, height=180, disabled=True)
        return
    st.code(str(value), language="text")
