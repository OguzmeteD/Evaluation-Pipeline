from __future__ import annotations

from datetime import date, datetime, time, timedelta

try:
    import streamlit as st
except ModuleNotFoundError:  # pragma: no cover
    st = None

from src.core.metrics_analytics import get_tool_judge_dataset
from src.schemas.metrics_analytics import TOOL_EVALUATOR_PRESETS, ToolJudgeDataset, ToolJudgeFilters


PAGE_TITLE = "Tool Judge"


def render() -> None:
    if st is None:  # pragma: no cover
        raise RuntimeError("streamlit is required to render the Tool Judge page.")
    st.title(PAGE_TITLE)
    st.caption("Tool observation metrics ve evaluator score ozetlerini birlikte inceleyin.")

    filters = _build_filters()
    if st.sidebar.button("Load tool judge", type="primary", width="stretch", key="load_tool_judge"):
        st.session_state["tool_judge"] = get_tool_judge_dataset(filters)

    dataset = st.session_state.get("tool_judge")
    if not isinstance(dataset, ToolJudgeDataset):
        st.info("Tool isimlerini ve evaluator filtrelerini girip dataset'i yukleyin.")
        return

    _render_summary(dataset)
    _render_tool_rows(dataset)
    _render_evaluator_rows(dataset)


def build_evaluator_selection(preset_names: list[str], custom_names: str) -> list[str]:
    result = list(preset_names)
    for name in custom_names.split(","):
        cleaned = name.strip()
        if cleaned and cleaned not in result:
            result.append(cleaned)
    return result


def _build_filters() -> ToolJudgeFilters:
    st.sidebar.header("Tool Judge Filters")
    today = date.today()
    default_from = today - timedelta(days=14)
    date_range = st.sidebar.date_input(
        "Date range",
        value=(default_from, today),
        key="tool_judge_date_range",
    )
    if isinstance(date_range, tuple) and len(date_range) == 2:
        from_date, to_date = date_range
    else:
        from_date, to_date = default_from, today
    tool_names = st.sidebar.text_area(
        "Tool names (comma-separated)",
        key="tool_judge_tool_names",
        height=120,
        placeholder="rag_search, embedding_lookup",
    )
    environment = st.sidebar.text_input("Environment", key="tool_judge_environment")
    observation_types = st.sidebar.multiselect(
        "Observation types",
        options=["GENERATION", "SPAN", "EVENT"],
        default=[],
        key="tool_judge_observation_types",
    )
    preset_evaluators = st.sidebar.multiselect(
        "Preset evaluators",
        options=TOOL_EVALUATOR_PRESETS,
        default=["rag", "embedding"],
        key="tool_judge_presets",
    )
    custom_evaluators = st.sidebar.text_input(
        "Custom evaluators (comma-separated)",
        key="tool_judge_custom_evaluators",
        placeholder="faithfulness, answer_relevance",
    )
    limit = st.sidebar.slider("Row limit", min_value=10, max_value=500, value=100, key="tool_judge_limit")
    return ToolJudgeFilters(
        from_date=datetime.combine(from_date, time.min),
        to_date=datetime.combine(to_date, time.max),
        tool_names=_split_csv(tool_names),
        environment=environment or None,
        observation_types=observation_types,
        evaluator_names=build_evaluator_selection(preset_evaluators, custom_evaluators),
        limit=limit,
    )


def _render_summary(dataset: ToolJudgeDataset) -> None:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Tools", dataset.summary.tool_count)
    c2.metric("Observations", dataset.summary.observation_count)
    c3.metric("Total cost", _format_float(dataset.summary.total_cost))
    c4.metric("Total tokens", dataset.summary.total_tokens)
    if dataset.summary.avg_latency_ms is not None:
        st.caption(f"Average latency: {dataset.summary.avg_latency_ms:.3f} ms")
    for warning in dataset.warnings:
        st.warning(warning)


def _render_tool_rows(dataset: ToolJudgeDataset) -> None:
    st.subheader("Tool metrics")
    st.dataframe(
        [row.model_dump() for row in dataset.tool_rows],
        width="stretch",
        hide_index=True,
    )


def _render_evaluator_rows(dataset: ToolJudgeDataset) -> None:
    st.subheader("Evaluator scores")
    if not dataset.evaluator_rows:
        st.info("Evaluator score verisi bulunamadi.")
        return
    st.dataframe(
        [
            {
                **row.model_dump(),
                "categorical_breakdown": dict(row.categorical_breakdown),
            }
            for row in dataset.evaluator_rows
        ],
        width="stretch",
        hide_index=True,
    )


def _split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _format_float(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"
