from __future__ import annotations

from datetime import date, datetime, time, timedelta

try:
    import streamlit as st
except ModuleNotFoundError:  # pragma: no cover
    st = None

from src.core.metrics_analytics import get_prompt_analytics_dataset
from src.schemas.metrics_analytics import MetricsTimeGranularity, PromptAnalyticsDataset, PromptMetricsFilters


PAGE_TITLE = "Prompt Analytics"


def render() -> None:
    if st is None:  # pragma: no cover
        raise RuntimeError("streamlit is required to render the Prompt Analytics page.")
    st.title(PAGE_TITLE)
    st.caption("Prompt version ve run history etkisini cost, latency ve token bazinda inceleyin.")

    filters = _build_filters()
    if st.sidebar.button("Load prompt analytics", type="primary", width="stretch", key="load_prompt_analytics"):
        st.session_state["prompt_analytics"] = get_prompt_analytics_dataset(filters)

    dataset = st.session_state.get("prompt_analytics")
    if not isinstance(dataset, PromptAnalyticsDataset):
        st.info("Filtreleri secip prompt analytics dataset'ini yukleyin.")
        return

    _render_summary(dataset)
    _render_versions(dataset)
    _render_trend(dataset)
    _render_runs(dataset)


def _build_filters() -> PromptMetricsFilters:
    st.sidebar.header("Prompt Analytics Filters")
    today = date.today()
    default_from = today - timedelta(days=14)
    date_range = st.sidebar.date_input(
        "Date range",
        value=(default_from, today),
        key="prompt_analytics_date_range",
    )
    if isinstance(date_range, tuple) and len(date_range) == 2:
        from_date, to_date = date_range
    else:
        from_date, to_date = default_from, today

    prompt_name = st.sidebar.text_input("Prompt name", key="prompt_analytics_prompt_name")
    versions_csv = st.sidebar.text_input("Prompt versions (comma-separated)", key="prompt_analytics_versions")
    dataset_name = st.sidebar.text_input("Dataset name", key="prompt_analytics_dataset_name")
    run_name = st.sidebar.text_input("Run name contains", key="prompt_analytics_run_name")
    environment = st.sidebar.text_input("Environment", key="prompt_analytics_environment")
    model_name = st.sidebar.text_input("Model name", key="prompt_analytics_model")
    granularity = st.sidebar.selectbox(
        "Time granularity",
        options=["none"] + [item.value for item in MetricsTimeGranularity],
        index=4,
        key="prompt_analytics_granularity",
    )
    limit = st.sidebar.slider("Row limit", min_value=10, max_value=500, value=100, key="prompt_analytics_limit")
    return PromptMetricsFilters(
        from_date=datetime.combine(from_date, time.min),
        to_date=datetime.combine(to_date, time.max),
        prompt_name=prompt_name or None,
        prompt_versions=_split_int_csv(versions_csv),
        run_name=run_name or None,
        dataset_name=dataset_name or None,
        environment=environment or None,
        model_name=model_name or None,
        time_granularity=None if granularity == "none" else MetricsTimeGranularity(granularity),
        limit=limit,
    )


def _render_summary(dataset: PromptAnalyticsDataset) -> None:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Observations", dataset.summary.total_observations)
    c2.metric("Total cost", _format_float(dataset.summary.total_cost))
    c3.metric("Avg latency", _format_float(dataset.summary.avg_latency_ms, suffix=" ms"))
    c4.metric("Total tokens", dataset.summary.total_tokens)
    for warning in dataset.warnings:
        st.warning(warning)


def _render_versions(dataset: PromptAnalyticsDataset) -> None:
    st.subheader("Prompt version comparison")
    st.dataframe(
        [row.model_dump() for row in dataset.version_rows],
        width="stretch",
        hide_index=True,
    )


def _render_trend(dataset: PromptAnalyticsDataset) -> None:
    st.subheader("Time trend")
    if not dataset.trend_rows:
        st.info("Trend verisi bulunamadi.")
        return
    st.dataframe(
        [row.model_dump() for row in dataset.trend_rows],
        width="stretch",
        hide_index=True,
    )


def _render_runs(dataset: PromptAnalyticsDataset) -> None:
    st.subheader("Run history enrichment")
    if not dataset.run_rows:
        st.info("Run history ile eslesen kayit bulunamadi.")
        return
    st.dataframe(
        [row.model_dump() for row in dataset.run_rows],
        width="stretch",
        hide_index=True,
    )


def _split_int_csv(value: str) -> list[int]:
    result: list[int] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            result.append(int(item))
        except ValueError:
            continue
    return result


def _format_float(value: float | None, *, suffix: str = "") -> str:
    return "n/a" if value is None else f"{value:.3f}{suffix}"
