from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import streamlit as st
except ModuleNotFoundError:  # pragma: no cover - optional during backend tests
    st = None

from src.frontend.pages import (
    dataset_builder,
    experiment_studio,
    judge_explorer,
    prompt_analytics,
    run_history,
    tool_judge,
)
from src.frontend.prompt_coach_widget import render_prompt_coach_widget


PAGES = {
    "Judge Explorer": judge_explorer.render,
    "Experiment Studio": experiment_studio.render,
    "Dataset Builder": dataset_builder.render,
    "Prompt Analytics": prompt_analytics.render,
    "Tool Judge": tool_judge.render,
    "Run History": run_history.render,
}


def main() -> None:
    if st is None:  # pragma: no cover - defensive runtime guard
        raise RuntimeError("streamlit is required to run the application.")
    st.set_page_config(page_title="Langfuse Workbench", layout="wide")
    _inject_app_theme()
    _consume_pending_page_switch()
    st.sidebar.title("Workbench")
    page_name = st.sidebar.radio("Pages", options=list(PAGES.keys()), key="active_page")
    PAGES[page_name]()
    render_prompt_coach_widget(active_page=page_name)


def _consume_pending_page_switch() -> None:
    pending_page = st.session_state.pop("pending_active_page", None)
    if pending_page in PAGES:
        st.session_state["active_page"] = pending_page


def _inject_app_theme() -> None:
    st.markdown(
        """
        <style>
        :root {
            --theme-bg: #191919;
            --theme-panel: #241b1d;
            --theme-panel-2: #2e1d21;
            --theme-accent: #750E21;
            --theme-accent-2: #E3651D;
            --theme-highlight: #BED754;
            --theme-text: #f7f3e9;
            --theme-text-muted: #d8cfbf;
            --theme-border: rgba(190, 215, 84, 0.28);
        }
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(227, 101, 29, 0.22), transparent 28%),
                radial-gradient(circle at top right, rgba(190, 215, 84, 0.10), transparent 24%),
                linear-gradient(180deg, #191919 0%, #241b1d 55%, #191919 100%);
            color: var(--theme-text);
        }
        [data-testid="stSidebar"] {
            background:
                linear-gradient(180deg, rgba(117, 14, 33, 0.96) 0%, rgba(25, 25, 25, 0.98) 100%);
            border-right: 1px solid rgba(190, 215, 84, 0.18);
        }
        [data-testid="stSidebar"] * {
            color: var(--theme-text);
        }
        .stApp, .stApp p, .stApp li, .stApp label, .stApp span, .stApp div, .stApp h1, .stApp h2, .stApp h3 {
            color: var(--theme-text);
        }
        .stCaption, .stMarkdown small {
            color: var(--theme-text-muted) !important;
        }
        [data-testid="stMetric"] {
            background: linear-gradient(180deg, rgba(117, 14, 33, 0.24), rgba(36, 27, 29, 0.92));
            border: 1px solid var(--theme-border);
            border-radius: 16px;
            padding: 0.75rem 0.9rem;
        }
        [data-testid="stMetricLabel"], [data-testid="stMetricValue"] {
            color: var(--theme-text) !important;
        }
        [data-baseweb="input"] > div,
        [data-baseweb="select"] > div,
        [data-testid="stTextInputRootElement"] > div,
        [data-testid="stNumberInputRootElement"] > div,
        .stTextArea textarea,
        .stTextInput input {
            background: rgba(36, 27, 29, 0.96) !important;
            color: var(--theme-text) !important;
            border: 1px solid rgba(227, 101, 29, 0.45) !important;
        }
        .stTextArea textarea::placeholder,
        .stTextInput input::placeholder {
            color: rgba(247, 243, 233, 0.55) !important;
        }
        [data-baseweb="select"] svg,
        .stTextInput button svg {
            fill: var(--theme-highlight) !important;
        }
        .stButton > button,
        .stDownloadButton > button {
            background: linear-gradient(135deg, #750E21 0%, #E3651D 100%);
            color: #fff7eb !important;
            border: 1px solid rgba(190, 215, 84, 0.22);
            border-radius: 12px;
        }
        .stButton > button:hover,
        .stDownloadButton > button:hover {
            border-color: rgba(190, 215, 84, 0.6);
            color: #ffffff !important;
        }
        [data-testid="stDataFrame"],
        [data-testid="stTable"] {
            background: rgba(36, 27, 29, 0.88);
            border: 1px solid rgba(190, 215, 84, 0.15);
            border-radius: 14px;
        }
        [data-testid="stAlertContainer"] {
            border-radius: 14px;
        }
        [data-testid="stToolbar"] {
            right: 1rem;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 0.5rem;
        }
        .stTabs [data-baseweb="tab"] {
            background: rgba(117, 14, 33, 0.32);
            border-radius: 12px 12px 0 0;
            color: var(--theme-text);
        }
        .stTabs [aria-selected="true"] {
            background: rgba(227, 101, 29, 0.28) !important;
            border-bottom-color: var(--theme-highlight) !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
