from __future__ import annotations

import json
from datetime import date, datetime, time, timedelta
from typing import Any

try:
    import streamlit as st
except ModuleNotFoundError:  # pragma: no cover
    st = None

from src.core.dataset_builder import create_langfuse_dataset, preview_dataset_candidates
from src.core.experiment_runner import fetch_dataset_by_name
from src.schemas.dataset_builder import (
    DatasetBuilderFilters,
    DatasetCandidateResult,
    DatasetCandidateTrace,
    DatasetCreationRequest,
    DatasetCreationResult,
    DatasetMetricThreshold,
)
from src.schemas.experiment_runner import PRESET_METRIC_RUBRICS


PAGE_TITLE = "Dataset Builder"


def render() -> None:
    if st is None:  # pragma: no cover
        raise RuntimeError("streamlit is required to render the Dataset Builder page.")
    st.title(PAGE_TITLE)
    st.caption("Score threshold ile basarili trace'leri secip Langfuse dataset olusturun.")

    target = _render_dataset_target()
    filters = _render_score_filters()
    preview_disabled = not filters.metric_thresholds

    if st.button(
        "Preview candidate traces",
        type="primary",
        width="stretch",
        disabled=preview_disabled,
        key="dataset_builder_preview_button",
    ):
        st.session_state["dataset_builder_preview"] = preview_dataset_candidates(filters)
        st.session_state.pop("dataset_builder_result", None)

    preview = st.session_state.get("dataset_builder_preview")
    if isinstance(preview, DatasetCandidateResult):
        _render_candidate_preview(preview)

    selected_candidates = (
        _selected_candidates_from_state(preview) if isinstance(preview, DatasetCandidateResult) else []
    )
    create_disabled = preview_disabled or not selected_candidates
    if st.button(
        "Create dataset in Langfuse",
        width="stretch",
        disabled=create_disabled or bool(target["metadata_error"]) or not target["dataset_name"],
        key="dataset_builder_create_button",
    ):
        st.session_state["dataset_builder_result"] = create_langfuse_dataset(
            DatasetCreationRequest(
                dataset_name=target["dataset_name"],
                description=target["description"] or None,
                metadata=target["metadata"],
                candidates=selected_candidates,
            )
        )

    result = st.session_state.get("dataset_builder_result")
    if isinstance(result, DatasetCreationResult):
        _render_creation_result(result)


def _render_dataset_target() -> dict[str, Any]:
    st.subheader("Dataset Target")
    dataset_name = st.text_input(
        "Dataset name",
        key="dataset_builder_name",
        placeholder="high-score-traces-v1",
    ).strip()
    description = st.text_area(
        "Description",
        key="dataset_builder_description",
        height=80,
        placeholder="Score threshold ile secilen trace'lerden uretilen dataset.",
    )
    metadata_text = st.text_area(
        "Metadata JSON (optional)",
        key="dataset_builder_metadata",
        height=100,
        placeholder='{"created_by": "dataset_builder", "selection": "score_threshold_match"}',
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


def _render_score_filters() -> DatasetBuilderFilters:
    st.subheader("Score Filters")
    today = date.today()
    default_from = today - timedelta(days=14)
    date_range = st.date_input(
        "Date range",
        value=(default_from, today),
        key="dataset_builder_date_range",
    )
    if isinstance(date_range, tuple) and len(date_range) == 2:
        from_date, to_date = date_range
    else:
        from_date, to_date = default_from, today

    preset_metrics = st.multiselect(
        "Preset metrics",
        options=list(PRESET_METRIC_RUBRICS.keys()),
        default=["helpfulness", "correctness"],
        key="dataset_builder_preset_metrics",
    )
    custom_metric = st.text_input(
        "Custom metric name",
        key="dataset_builder_custom_metric",
        placeholder="brand_safety",
    ).strip()
    selected_metrics = _build_metric_names(preset_metrics, custom_metric)
    thresholds = _render_metric_threshold_inputs(selected_metrics)

    session_ids = st.text_input(
        "Session IDs (comma-separated, optional)",
        key="dataset_builder_session_ids",
        placeholder="session-1, session-2",
    )
    experiment_id = st.text_input(
        "Experiment ID (optional)",
        key="dataset_builder_experiment_id",
    ).strip()
    limit = st.slider(
        "Candidate limit",
        min_value=10,
        max_value=500,
        value=100,
        key="dataset_builder_limit",
    )
    return DatasetBuilderFilters(
        from_date=datetime.combine(from_date, time.min),
        to_date=datetime.combine(to_date, time.max),
        experiment_id=experiment_id or None,
        session_ids=_split_csv(session_ids),
        metric_thresholds=thresholds,
        limit=limit,
    )


def _render_metric_threshold_inputs(metric_names: list[str]) -> list[DatasetMetricThreshold]:
    thresholds: list[DatasetMetricThreshold] = []
    if not metric_names:
        st.info("Preview almak icin en az bir metric secin.")
        return thresholds
    st.markdown("### Metric thresholds")
    for metric_name in metric_names:
        c1, c2 = st.columns([1, 2])
        with c1:
            min_score = st.number_input(
                f"{metric_name} min score",
                min_value=0.0,
                max_value=1.0,
                value=0.8,
                step=0.05,
                key=f"dataset_builder_threshold_{metric_name}",
            )
        with c2:
            judge_name = st.text_input(
                f"{metric_name} judge name (optional)",
                key=f"dataset_builder_judge_{metric_name}",
                placeholder="judge_name",
            ).strip()
        thresholds.append(
            DatasetMetricThreshold(
                metric_name=metric_name,
                min_score=float(min_score),
                judge_name=judge_name or None,
            )
        )
    return thresholds


def _render_candidate_preview(preview: DatasetCandidateResult) -> None:
    st.subheader("Candidate Preview")
    c1, c2, c3 = st.columns(3)
    c1.metric("Matched traces", preview.total_candidates)
    c2.metric(
        "Average score",
        _format_float(_average_candidate_score(preview)),
    )
    selected_trace_ids = st.multiselect(
        "Selected traces for dataset",
        options=[candidate.trace_id for candidate in preview.candidates],
        default=[candidate.trace_id for candidate in preview.candidates],
        format_func=lambda value: _format_candidate_label(preview, value),
        key="dataset_builder_selected_trace_ids",
    )
    c3.metric("Selected for create", len(selected_trace_ids))
    for warning in preview.warnings:
        st.warning(warning)
    if not preview.candidates:
        st.info("Filtrelerle eslesen trace bulunamadi.")
        return
    rows = [
        {
            "trace_id": candidate.trace_id,
            "trace_name": candidate.trace_name,
            "matched_metrics": ", ".join(candidate.matched_metrics),
            "avg_score": candidate.avg_score,
            "session_id": candidate.session_id,
            "experiment_id": candidate.experiment_id,
        }
        for candidate in preview.candidates
    ]
    st.dataframe(rows, width="stretch", hide_index=True)
    selected_trace_id = st.selectbox(
        "Trace detail",
        options=[candidate.trace_id for candidate in preview.candidates],
        format_func=lambda value: _format_candidate_label(preview, value),
        key="dataset_builder_selected_trace",
    )
    selected = next(candidate for candidate in preview.candidates if candidate.trace_id == selected_trace_id)
    left, right = st.columns(2)
    with left:
        st.markdown("### Input")
        _render_value_block("Trace input", selected.input_payload)
        st.markdown("### Expected output")
        _render_value_block("Trace output", selected.output_payload)
    with right:
        st.markdown("### Score summary")
        st.dataframe(
            [score.model_dump() for score in selected.score_summary],
            width="stretch",
            hide_index=True,
        )
        st.markdown("### Metadata")
        _render_value_block("Trace metadata", selected.metadata)


def _render_creation_result(result: DatasetCreationResult) -> None:
    st.subheader("Create Dataset")
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
        st.markdown("### Created item IDs")
        st.dataframe([{"item_id": item_id} for item_id in result.item_ids], width="stretch", hide_index=True)
    if result.dataset_id and not result.errors:
        if st.button(
            "Open in Experiment Studio",
            width="stretch",
            key="dataset_builder_open_in_studio",
        ):
            _send_dataset_to_experiment_studio(result.dataset_name)
            st.rerun()


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


def _build_metric_names(preset_metrics: list[str], custom_metric: str) -> list[str]:
    names = list(preset_metrics)
    if custom_metric and custom_metric not in names:
        names.append(custom_metric)
    return names


def _split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _average_candidate_score(preview: DatasetCandidateResult) -> float | None:
    values = [candidate.avg_score for candidate in preview.candidates if candidate.avg_score is not None]
    if not values:
        return None
    return sum(values) / len(values)


def _format_float(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def _format_candidate_label(preview: DatasetCandidateResult, trace_id: str) -> str:
    candidate = next(row for row in preview.candidates if row.trace_id == trace_id)
    return f"{candidate.trace_name or trace_id} | score={_format_float(candidate.avg_score)}"


def _selected_candidates_from_state(preview: DatasetCandidateResult) -> list[DatasetCandidateTrace]:
    selected_ids = st.session_state.get("dataset_builder_selected_trace_ids") or [
        candidate.trace_id for candidate in preview.candidates
    ]
    selected_set = set(selected_ids)
    return [candidate for candidate in preview.candidates if candidate.trace_id in selected_set]


def _send_dataset_to_experiment_studio(dataset_name: str) -> None:
    st.session_state["studio_dataset_name"] = dataset_name
    try:
        st.session_state["studio_dataset"] = fetch_dataset_by_name(dataset_name)
        st.session_state["studio_prompt_apply_message"] = (
            f"Dataset '{dataset_name}' Experiment Studio icine yuklendi."
        )
    except Exception as exc:  # pragma: no cover - runtime/network dependent
        st.session_state["studio_prompt_apply_message"] = (
            f"Dataset adi tasindi ancak Experiment Studio fetch basarisiz oldu: {exc}"
        )
    st.session_state["pending_active_page"] = "Experiment Studio"


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
