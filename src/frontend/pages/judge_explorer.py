from __future__ import annotations

from datetime import date, datetime, time, timedelta
from typing import Any

try:
    import streamlit as st
except ModuleNotFoundError:  # pragma: no cover - optional during backend tests
    st = None

from src.core.judger import get_evaluation_dataset
from src.schemas.evaluation_dataset import EvaluationDataset, GenerationRow, JudgeDatasetFilters


def render() -> None:
    if st is None:  # pragma: no cover - defensive runtime guard
        raise RuntimeError("streamlit is required to render the Judge Explorer page.")
    st.title("Langfuse Judge Explorer")
    st.caption(
        "LLM-as-a-Judge skorlarini, system promptlari ve generation kayitlarini hibrit dataset olarak inceleyin."
    )

    filters = _build_filters()
    if st.sidebar.button("Load dataset", type="primary", width="stretch", key="load_dataset_button"):
        st.session_state["dataset"] = get_evaluation_dataset(filters)

    dataset = st.session_state.get("dataset")
    if not isinstance(dataset, EvaluationDataset):
        st.info("Filtreleri secip dataset'i yukleyin.")
        return

    _render_metrics(dataset)
    _render_trace_table(dataset)
    _render_generation_details(dataset)


def _build_filters() -> JudgeDatasetFilters:
    st.sidebar.header("Judge Filters")
    today = date.today()
    default_from = today - timedelta(days=7)
    date_range = st.sidebar.date_input("Date range", value=(default_from, today), key="judge_date_range")
    if isinstance(date_range, tuple) and len(date_range) == 2:
        from_date, to_date = date_range
    else:
        from_date = default_from
        to_date = today

    experiment_id = st.sidebar.text_input("Experiment ID", key="judge_experiment_id")
    trace_id = st.sidebar.text_input("Trace ID", key="judge_trace_id")
    judge_names = st.sidebar.text_input("Judge names (comma-separated)", key="judge_names")
    score_names = st.sidebar.text_input("Score names (comma-separated)", key="score_names")
    min_score = st.sidebar.number_input(
        "Minimum score",
        min_value=0.0,
        max_value=1.0,
        value=0.0,
        step=0.05,
        key="judge_min_score",
    )
    limit = st.sidebar.slider("Row limit", min_value=10, max_value=500, value=100, key="judge_limit")

    return JudgeDatasetFilters(
        from_date=datetime.combine(from_date, time.min),
        to_date=datetime.combine(to_date, time.max),
        experiment_id=experiment_id or None,
        trace_ids=_split_csv(trace_id),
        judge_names=_split_csv(judge_names),
        score_names=_split_csv(score_names),
        min_score=min_score,
        limit=limit,
    )


def _render_metrics(dataset: EvaluationDataset) -> None:
    metric1, metric2, metric3, metric4 = st.columns(4)
    metric1.metric("Traces", dataset.meta.counts.traces)
    metric2.metric("Generations", dataset.meta.counts.rows)
    metric3.metric("Average score", _format_optional_number(dataset.meta.average_score))
    metric4.metric("Prompt coverage", f"{dataset.meta.prompt_coverage * 100:.1f}%")
    if dataset.meta.warnings:
        for warning in dataset.meta.warnings:
            st.warning(warning)


def _render_trace_table(dataset: EvaluationDataset) -> None:
    st.subheader("Trace overview")
    rows = [
        {
            "trace_id": trace.trace_id,
            "trace_name": trace.trace_name,
            "session_id": trace.session_id,
            "experiment_id": trace.experiment_id,
            "observation_count": trace.observation_count,
            "avg_score": trace.avg_score,
            "has_prompt": trace.has_prompt,
            "has_generation": trace.has_generation,
            "latency_ms": trace.latency_ms,
            "total_cost": trace.total_cost,
        }
        for trace in dataset.traces
    ]
    st.dataframe(rows, width="stretch", hide_index=True)


def _render_generation_details(dataset: EvaluationDataset) -> None:
    st.subheader("Generation drill-down")
    trace_options = {_trace_label(trace.trace_id, trace.trace_name): trace.trace_id for trace in dataset.traces}
    if not trace_options:
        st.info("No traces available for drill-down.")
        return

    selected_label = st.selectbox("Selected trace", list(trace_options.keys()), key="selected_trace")
    selected_trace_id = trace_options[selected_label]
    rows = [row for row in dataset.rows if row.trace_id == selected_trace_id]
    if not rows:
        st.info("No generation rows for the selected trace.")
        return

    st.dataframe(
        [
            {
                "observation_id": row.observation_id,
                "observation_name": row.observation_name,
                "model": row.model,
                "score_count": len(row.judge_scores),
                "latency_ms": row.latency_ms,
                "total_cost": row.total_cost,
                "prompt_source": row.prompt_source.value,
                "has_prompt": row.has_prompt,
                "has_generation": row.has_generation,
            }
            for row in rows
        ],
        width="stretch",
        hide_index=True,
    )

    row_labels = {_row_label(row): row for row in rows}
    chosen_row_label = st.selectbox("Selected generation", list(row_labels.keys()), key="selected_generation")
    selected_row = row_labels[chosen_row_label]

    left, right = st.columns([1, 1])
    with left:
        st.markdown("### Prompt and output")
        st.text_area("System prompt", value=selected_row.system_prompt or "", height=200)
        st.text_area("Generation", value=selected_row.generation_text or "", height=240)
        st.json(selected_row.prompt_messages)
    with right:
        st.markdown("### Judge and raw payloads")
        st.dataframe(
            [
                {
                    "judge_name": score.judge_name,
                    "score_name": score.score_name,
                    "score_value": score.score_value,
                    "score_label": score.score_label,
                    "comment": score.score_comment,
                    "source": score.score_source,
                }
                for score in selected_row.judge_scores
            ],
            width="stretch",
            hide_index=True,
        )
        st.markdown("#### Input payload")
        st.json(_json_safe(selected_row.input_payload))
        st.markdown("#### Output payload")
        st.json(_json_safe(selected_row.output_payload))


def _trace_label(trace_id: str, trace_name: str | None) -> str:
    return f"{trace_name} ({trace_id})" if trace_name else trace_id


def _row_label(row: GenerationRow) -> str:
    if row.observation_name and row.observation_id:
        return f"{row.observation_name} ({row.observation_id})"
    return row.observation_id or row.trace_id


def _split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _format_optional_number(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def _json_safe(value: Any) -> Any:
    return {} if value is None else value
