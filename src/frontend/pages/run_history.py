from __future__ import annotations

from collections import defaultdict

try:
    import pandas as pd
    import streamlit as st
except ModuleNotFoundError:  # pragma: no cover - optional during backend tests
    pd = None
    st = None

from src.core.experiment_runner import list_recent_experiment_runs
from src.schemas.experiment_runner import ExperimentMode, ExperimentRunHistoryResult, ExperimentRunRecord


PAGE_TITLE = "Run History"
STATUS_COLORS = {
    "succeeded": "#BED754",
    "failed": "#E3651D",
}


def render() -> None:
    if st is None:  # pragma: no cover - defensive runtime guard
        raise RuntimeError("streamlit is required to render the Run History page.")
    st.title(PAGE_TITLE)
    st.caption("PostgreSQL'de saklanan son experiment calistirmalarini ozet olarak inceleyin.")

    limit = st.slider("Limit", min_value=5, max_value=100, value=20, key="history_limit")
    dataset_name = st.text_input("Dataset filter", key="history_dataset_name")
    mode_value = st.selectbox(
        "Mode filter",
        options=["all", ExperimentMode.PROMPT_RUNNER.value, ExperimentMode.REEVALUATE_EXISTING.value],
        key="history_mode",
    )
    selected_statuses = st.multiselect(
        "Status filter",
        options=["succeeded", "failed"],
        default=["succeeded", "failed"],
        key="history_statuses",
    )
    history = list_recent_experiment_runs(
        limit=limit,
        dataset_name=dataset_name or None,
        mode=None if mode_value == "all" else ExperimentMode(mode_value),
    )
    _render_history(history, selected_statuses=selected_statuses)


def _render_history(
    history: ExperimentRunHistoryResult,
    *,
    selected_statuses: list[str],
) -> None:
    filtered_records = filter_history_records(history.records, selected_statuses)
    c1, c2, c3 = st.columns(3)
    c1.metric("Total runs", len(filtered_records))
    c2.metric("Last success", str(history.last_success_at) if history.last_success_at else "n/a")
    c3.metric("Last error", str(history.last_error_at) if history.last_error_at else "n/a")
    _render_status_badges(selected_statuses)
    for warning in history.warnings:
        st.warning(warning)
    if not filtered_records:
        st.info("Kayitli run bulunamadi.")
        return
    _render_aggregate_trend(filtered_records)
    table_rows = build_history_table_rows(filtered_records)
    if pd is not None:
        table_frame = pd.DataFrame(table_rows)
        styled = table_frame.style.map(_style_status_badge, subset=["status_badge"])
        st.dataframe(styled, width="stretch", hide_index=True)
    else:  # pragma: no cover
        st.dataframe(table_rows, width="stretch", hide_index=True)
    selected = st.selectbox(
        "Run detail",
        options=[record.id for record in filtered_records],
        format_func=lambda value: next(record.run_name for record in filtered_records if record.id == value),
        key="history_selected_record",
    )
    record = next(record for record in filtered_records if record.id == selected)
    left, right = st.columns([1, 1])
    with left:
        st.markdown("### Aggregate metrics")
        st.dataframe([metric.model_dump() for metric in record.aggregate_metrics], width="stretch", hide_index=True)
        if record.dataset_run_url:
            st.markdown(f"[Open Langfuse Run]({record.dataset_run_url})")
    with right:
        st.markdown("### Prompt summary")
        st.json(
            {
                "task_prompt_source": record.task_prompt_source.value,
                "task_prompt_name": record.task_prompt_name,
                "task_prompt_label": record.task_prompt_label,
                "task_prompt_type": record.task_prompt_type.value if record.task_prompt_type else None,
                "judge_prompt_source": record.judge_prompt_source.value,
                "judge_prompt_name": record.judge_prompt_name,
                "judge_prompt_label": record.judge_prompt_label,
                "judge_prompt_type": record.judge_prompt_type.value if record.judge_prompt_type else None,
                "published_from_custom": record.published_from_custom,
                "published_at": str(record.published_at) if record.published_at else None,
            }
        )
        if record.warnings:
            st.markdown("### Warnings")
            st.json(record.warnings)
        if record.errors:
            st.markdown("### Errors")
            st.json(record.errors)


def filter_history_records(
    records: list[ExperimentRunRecord],
    selected_statuses: list[str],
) -> list[ExperimentRunRecord]:
    if not selected_statuses:
        return records
    allowed = set(selected_statuses)
    return [record for record in records if record.status.value in allowed]


def build_aggregate_trend_rows(records: list[ExperimentRunRecord]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for record in records:
        for metric in record.aggregate_metrics:
            rows.append(
                {
                    "created_at": record.created_at,
                    "metric_name": metric.name,
                    "measure": "average_score",
                    "series_name": f"{metric.name}: average_score",
                    "value": metric.average_score,
                    "run_name": record.run_name,
                    "status": record.status.value,
                }
            )
            rows.append(
                {
                    "created_at": record.created_at,
                    "metric_name": metric.name,
                    "measure": "count",
                    "series_name": f"{metric.name}: count",
                    "value": metric.count,
                    "run_name": record.run_name,
                    "status": record.status.value,
                }
            )
        rows.append(
            {
                "created_at": record.created_at,
                "metric_name": "run",
                "measure": "failed_items",
                "series_name": "run: failed_items",
                "value": record.failed_items,
                "run_name": record.run_name,
                "status": record.status.value,
            }
        )
        rows.append(
            {
                "created_at": record.created_at,
                "metric_name": "run",
                "measure": "processed_items",
                "series_name": "run: processed_items",
                "value": record.processed_items,
                "run_name": record.run_name,
                "status": record.status.value,
            }
        )
    rows.sort(key=lambda row: (row["created_at"], str(row["metric_name"])))
    return rows


def _render_aggregate_trend(records: list[ExperimentRunRecord]) -> None:
    st.markdown("### Aggregate Trend")
    trend_rows = build_aggregate_trend_rows(records)
    if not trend_rows:
        st.info("Trend icin aggregate metric verisi bulunamadi.")
        return
    measures = ["average_score", "count", "failed_items", "processed_items"]
    selected_measures = st.multiselect(
        "Trend measures",
        options=measures,
        default=["average_score", "count", "failed_items"],
        key="history_trend_measures",
    )
    if not selected_measures:
        st.info("Trend gostermek icin en az bir olcu secin.")
        return
    chart_rows = [row for row in trend_rows if row["measure"] in selected_measures]
    if pd is None:  # pragma: no cover - streamlit runtime always includes pandas
        st.dataframe(chart_rows, width="stretch", hide_index=True)
        return
    chart_frame = pd.DataFrame(chart_rows)
    pivot = chart_frame.pivot_table(
        index="created_at",
        columns="series_name",
        values="value",
        aggfunc="mean",
    ).sort_index()
    st.line_chart(pivot, height=280, width="stretch")
    st.dataframe(chart_frame, width="stretch", hide_index=True)


def build_history_table_rows(records: list[ExperimentRunRecord]) -> list[dict[str, object]]:
    return [
        {
            "created_at": record.created_at,
            "run_name": record.run_name,
            "dataset_name": record.dataset_name,
            "mode": record.mode.value,
            "status": record.status.value,
            "status_badge": record.status.value.upper(),
            "prompt_source": f"task={record.task_prompt_source.value}, judge={record.judge_prompt_source.value}",
            "prompt_label": f"task={record.task_prompt_label or 'n/a'}, judge={record.judge_prompt_label or 'n/a'}",
            "metric_count": len(record.metric_names),
            "processed_items": record.processed_items,
            "failed_items": record.failed_items,
        }
        for record in records
    ]


def _render_status_badges(selected_statuses: list[str]) -> None:
    if not selected_statuses:
        st.caption("Status filter: tum statusler")
        return
    badges = []
    for status in selected_statuses:
        color = STATUS_COLORS.get(status, "#750E21")
        badges.append(
            f'<span style="display:inline-block;padding:0.22rem 0.55rem;margin-right:0.35rem;'
            f'border-radius:999px;background:{color};color:#191919;font-weight:700;">{status}</span>'
        )
    st.markdown("".join(badges), unsafe_allow_html=True)


def _style_status_badge(value: object) -> str:
    text = str(value).lower()
    status = text.replace("_badge", "").replace(" ", "").lower()
    color = STATUS_COLORS.get(status, "#750E21")
    return (
        f"background-color: {color};"
        "color: #191919;"
        "font-weight: 700;"
        "border-radius: 999px;"
        "text-align: center;"
    )
