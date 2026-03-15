from __future__ import annotations

from typing import Any

try:
    import streamlit as st
except ModuleNotFoundError:  # pragma: no cover
    st = None

from src.core.experiment_runner import list_prompts
from src.core.prompt_coach_agent import fallback_prompt_coach_response, get_prompt_coach_response
from src.schemas.prompt_coach import PromptApplyTarget, PromptCoachRequest, PromptCoachResponse


def render_prompt_coach_widget(*, active_page: str) -> None:
    if st is None:  # pragma: no cover
        raise RuntimeError("streamlit is required to render Prompt Coach.")
    _inject_prompt_coach_css()
    if hasattr(st, "popover"):
        with st.popover("PC", help="Prompt Coach", width="content"):
            _render_prompt_coach_body(active_page=active_page)
    else:  # pragma: no cover
        with st.expander("Prompt Coach"):
            _render_prompt_coach_body(active_page=active_page)


def apply_recommended_prompt(session_state: dict[str, Any], *, target: PromptApplyTarget, prompt: str) -> None:
    targets: list[str] = []
    if target in {PromptApplyTarget.TASK, PromptApplyTarget.BOTH}:
        targets.append("task")
    if target in {PromptApplyTarget.JUDGE, PromptApplyTarget.BOTH}:
        targets.append("judge")
    session_state["studio_pending_prompt_apply"] = {
        "targets": targets,
        "prompt": prompt,
    }


def _render_prompt_coach_body(*, active_page: str) -> None:
    st.markdown("### Prompt Coach")
    st.caption("Prompt iyilestirme, onay/revizyon karari ve web destekli judge onerileri.")
    _render_visible_prompt_versions()
    request_text = st.text_area(
        "What should change?",
        key="prompt_coach_request",
        height=120,
        placeholder="Promptu daha net yap, retrieval durumunda hallucination azalt, judge rubric ekle...",
    )
    if st.button("Ask coach", key="prompt_coach_submit", width="stretch"):
        if request_text.strip():
            st.session_state["prompt_coach_response"] = _ask_prompt_coach(
                PromptCoachRequest(
                    user_request=request_text.strip(),
                    active_page=active_page,
                    current_task_prompt=st.session_state.get("studio_task_custom_prompt"),
                    current_judge_prompt=st.session_state.get("studio_judge_custom_prompt"),
                    current_task_prompt_name=st.session_state.get("studio_task_prompt_name"),
                    current_judge_prompt_name=st.session_state.get("studio_judge_prompt_name"),
                    current_task_prompt_label=st.session_state.get("studio_task_prompt_label"),
                    current_judge_prompt_label=st.session_state.get("studio_judge_prompt_label"),
                    current_system_context=_build_context_summary(active_page=active_page),
                )
            )
        else:
            st.warning("Prompt Coach icin bir istek girin.")

    response = st.session_state.get("prompt_coach_response")
    if isinstance(response, PromptCoachResponse):
        _render_prompt_coach_response(response)


def _ask_prompt_coach(request: PromptCoachRequest) -> PromptCoachResponse:
    try:
        return get_prompt_coach_response(request)
    except Exception as exc:  # pragma: no cover - runtime/network dependent
        return fallback_prompt_coach_response(str(exc))


def _render_prompt_coach_response(response: PromptCoachResponse) -> None:
    st.success(f"Decision: {response.decision.value}")
    st.write(response.summary)
    if response.reasons:
        st.markdown("**Reasons**")
        st.write("\n".join(f"- {reason}" for reason in response.reasons))
    if response.recommended_prompt:
        st.markdown("**Recommended prompt**")
        st.text_area(
            "Suggested prompt",
            value=response.recommended_prompt,
            key="prompt_coach_recommended_prompt",
            height=180,
        )
        left, right = st.columns(2)
        with left:
            if st.button("Apply to task", key="prompt_coach_apply_task", width="stretch"):
                apply_recommended_prompt(
                    st.session_state,
                    target=PromptApplyTarget.TASK,
                    prompt=response.recommended_prompt,
                )
                st.session_state["studio_prompt_apply_message"] = "Task prompt guncellendi."
                st.rerun()
        with right:
            if st.button("Apply to judge", key="prompt_coach_apply_judge", width="stretch"):
                apply_recommended_prompt(
                    st.session_state,
                    target=PromptApplyTarget.JUDGE,
                    prompt=response.recommended_prompt,
                )
                st.session_state["studio_prompt_apply_message"] = "Judge prompt guncellendi."
                st.rerun()
    if response.judge_guidance:
        st.markdown("**Judge guidance**")
        st.write(response.judge_guidance)
    if response.suggested_evaluators:
        st.markdown("**Suggested evaluators**")
        st.write(", ".join(response.suggested_evaluators))
    if response.web_sources:
        st.markdown("**Web sources**")
        for source in response.web_sources:
            st.markdown(f"- [{source.title}]({source.url})")
            if source.snippet:
                st.caption(source.snippet)
    for warning in response.warnings:
        st.warning(warning)


def _render_visible_prompt_versions() -> None:
    current_refs = _collect_current_prompt_refs(st.session_state)
    _clear_stale_visible_prompt_versions(st.session_state, current_refs)
    with st.expander("MCP-visible prompt versions", expanded=False):
        st.caption("Langfuse MCP tarafinda gorunen mevcut task/judge prompt versiyon ozeti.")
        if current_refs:
            st.caption(
                "Active refs: "
                + ", ".join(f"{ref['target']}={ref['name']}" for ref in current_refs)
            )
        else:
            st.caption("Mevcut task/judge prompt name bilgisi yok.")
            return
        _render_visible_prompt_versions_table(current_refs)


def _render_visible_prompt_versions_table(current_refs: list[dict[str, str]]) -> None:
    if not current_refs:
        return
    if st.button("Load visible prompt versions", key="prompt_coach_load_visible_prompts", width="stretch"):
        try:
            rows: list[dict[str, Any]] = []
            for ref in current_refs:
                prompt_rows = list_prompts(name=ref["name"], limit=10)
                rows.extend(_build_visible_prompt_rows(ref["target"], prompt_rows))
            st.session_state["prompt_coach_visible_prompts"] = rows
            st.session_state["prompt_coach_visible_prompts_refs"] = current_refs
        except Exception as exc:  # pragma: no cover
            st.session_state["prompt_coach_visible_prompts_error"] = str(exc)
    rows = st.session_state.get("prompt_coach_visible_prompts", [])
    error = st.session_state.pop("prompt_coach_visible_prompts_error", None)
    if error:
        st.warning(f"Prompt version listesi alinamadi: {error}")
    if rows:
        target_counts = _summarize_visible_prompt_rows(rows)
        metrics = st.columns(max(len(target_counts), 1))
        for index, (target, count) in enumerate(target_counts.items()):
            metrics[index].metric(f"{target.title()} prompt rows", count)
        st.dataframe(rows, width="stretch", hide_index=True)
    else:
        st.caption("Listeyi yuklemek icin butonu kullanin.")


def _build_context_summary(*, active_page: str) -> str:
    context_lines = [f"Active page: {active_page}"]
    dataset_name = st.session_state.get("studio_dataset_name")
    if dataset_name:
        context_lines.append(f"Dataset name: {dataset_name}")
    if st.session_state.get("studio_mode"):
        context_lines.append(f"Experiment mode: {st.session_state['studio_mode']}")
    if st.session_state.get("studio_task_prompt_name"):
        context_lines.append(f"Task prompt name: {st.session_state['studio_task_prompt_name']}")
    if st.session_state.get("studio_judge_prompt_name"):
        context_lines.append(f"Judge prompt name: {st.session_state['studio_judge_prompt_name']}")
    if st.session_state.get("tool_judge_tool_names"):
        context_lines.append(f"Tool names: {st.session_state['tool_judge_tool_names']}")
    return "\n".join(context_lines)


def _collect_current_prompt_refs(session_state: dict[str, Any]) -> list[dict[str, str]]:
    refs: list[dict[str, str]] = []
    for target in ["task", "judge"]:
        prompt_name = session_state.get(f"studio_{target}_prompt_name")
        if prompt_name:
            refs.append({"target": target, "name": prompt_name})
    return refs


def _build_visible_prompt_rows(target: str, prompt_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in prompt_rows:
        rows.append(
            {
                "target": target,
                "name": row.get("name"),
                "type": row.get("type"),
                "versions": ", ".join(str(version) for version in row.get("versions", [])),
                "labels": ", ".join(row.get("labels", [])),
                "tags": ", ".join(row.get("tags", [])),
                "last_updated_at": row.get("lastUpdatedAt") or row.get("last_updated_at"),
            }
        )
    return rows


def _summarize_visible_prompt_rows(rows: list[dict[str, Any]]) -> dict[str, int]:
    summary: dict[str, int] = {}
    for row in rows:
        target = str(row.get("target") or "unknown")
        summary[target] = summary.get(target, 0) + 1
    return summary


def _clear_stale_visible_prompt_versions(
    session_state: dict[str, Any],
    current_refs: list[dict[str, str]],
) -> None:
    stored_refs = session_state.get("prompt_coach_visible_prompts_refs")
    if stored_refs == current_refs:
        return
    session_state.pop("prompt_coach_visible_prompts", None)
    session_state["prompt_coach_visible_prompts_refs"] = current_refs


def _inject_prompt_coach_css() -> None:
    st.markdown(
        """
        <style>
        div[data-testid="stPopover"] {
            position: fixed;
            right: 1.25rem;
            bottom: 1.25rem;
            z-index: 9999;
        }
        div[data-testid="stPopover"] > button {
            width: 3.5rem;
            height: 3.5rem;
            border-radius: 999px;
            background: linear-gradient(135deg, #E3651D 0%, #750E21 100%);
            color: #fff7eb;
            border: 1px solid rgba(190, 215, 84, 0.32);
            box-shadow: 0 12px 30px rgba(0, 0, 0, 0.24);
            font-weight: 700;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
