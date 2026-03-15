from __future__ import annotations

import json
from typing import Any

try:
    import streamlit as st
except ModuleNotFoundError:  # pragma: no cover - optional during backend tests
    st = None

from src.core.experiment_runner import (
    fetch_dataset_by_name,
    list_prompts,
    publish_prompt,
    resolve_prompt,
    run_llm_judge_on_existing_results,
    run_dataset_reevaluation,
    run_prompt_experiment,
)
from src.schemas.experiment_runner import (
    EvaluationScope,
    EvaluatorMetricSpec,
    ExperimentExecutionRequest,
    ExperimentExecutionResult,
    ExperimentMode,
    PRESET_METRIC_RUBRICS,
    PromptPublishTarget,
    PromptResolutionRequest,
    PromptSource,
    PromptTarget,
    PromptType,
    PublishedPromptRequest,
    PublishedPromptResult,
    ResolvedPrompt,
)


PAGE_TITLE = "Experiment Studio"


def render() -> None:
    if st is None:  # pragma: no cover - defensive runtime guard
        raise RuntimeError("streamlit is required to render the Experiment Studio page.")
    _consume_pending_prompt_apply()
    _consume_pending_prompt_browser_selection()
    st.title(PAGE_TITLE)
    st.caption(
        "Dataset bazli custom evaluator tanimlayin, Langfuse prompt adi veya custom prompt ile experiment calistirin ve sonucu history'e kaydedin."
    )
    _inject_css()
    message = st.session_state.pop("studio_prompt_apply_message", None)
    if message:
        st.success(message)

    dataset_name = st.text_input(
        "Dataset name",
        key="studio_dataset_name",
        placeholder="customer-support-eval-set",
    )
    fetch_col, info_col = st.columns([1, 3])
    with fetch_col:
        if st.button(
            "Fetch dataset",
            type="primary",
            width="stretch",
            key="fetch_dataset_button",
        ):
            _load_dataset(dataset_name)
    with info_col:
        st.markdown(
            '<div class="studio-hint">Dataset name ile Langfuse dataset cekilir. Prompt secimi, metric tanimi ve run execution ayni sayfada yapilir.</div>',
            unsafe_allow_html=True,
        )

    dataset = st.session_state.get("studio_dataset")
    if dataset is not None:
        _render_dataset_loader(dataset)

    task_prompt = _render_prompt_block(title="Task Prompt Source", prefix="task", target=PromptTarget.TASK)
    judge_prompt = _render_prompt_block(title="Judge Prompt Source", prefix="judge", target=PromptTarget.JUDGE)
    _render_prompt_publish_section(prefix="task", title="Publish task prompt", target=PromptPublishTarget.TASK, prompt_state=task_prompt)
    _render_prompt_publish_section(prefix="judge", title="Publish judge prompt", target=PromptPublishTarget.JUDGE, prompt_state=judge_prompt)
    _render_run_with_published_prompt()

    preset_metrics = st.multiselect(
        "Preset score metrics",
        options=list(PRESET_METRIC_RUBRICS.keys()),
        default=["helpfulness", "correctness"],
        key="studio_preset_metrics",
    )
    custom_name = st.text_input(
        "Custom metric name",
        key="studio_custom_metric_name",
        placeholder="brand_safety",
    )
    custom_rubric = st.text_area(
        "Custom metric rubric",
        key="studio_custom_metric_rubric",
        height=100,
        placeholder="Bu metrik neyi olcuyor?",
    )
    metrics = build_metric_specs_from_form(preset_metrics, custom_name, custom_rubric)
    _render_metric_preview(metrics)

    mode = st.radio(
        "Execution mode",
        options=[ExperimentMode.PROMPT_RUNNER.value, ExperimentMode.REEVALUATE_EXISTING.value],
        format_func=lambda value: "Prompt Runner" if value == ExperimentMode.PROMPT_RUNNER.value else "Re-evaluate Existing",
        horizontal=True,
        key="studio_mode",
    )
    judge_model = st.text_input(
        "Judge model",
        value=st.session_state.get("studio_judge_model", ""),
        key="studio_judge_model",
        placeholder="ENV varsayilani veya provider:model formati",
    )
    run_name = st.text_input("Run name (optional)", key="studio_run_name")
    description = st.text_area(
        "Description (optional)",
        key="studio_description",
        height=80,
    )
    concurrency = st.slider(
        "Concurrency",
        min_value=1,
        max_value=20,
        value=5,
        key="studio_concurrency",
    )

    scope = EvaluationScope.OBSERVATIONS
    task_model = None
    if mode == ExperimentMode.PROMPT_RUNNER.value:
        task_model = st.text_input(
            "Task model",
            value=st.session_state.get("studio_task_model", ""),
            key="studio_task_model",
            placeholder="ENV varsayilani veya provider:model formati",
        )
    else:
        allowed_scopes = available_scopes(dataset.items if dataset else [])
        default_scope = allowed_scopes[0] if allowed_scopes else EvaluationScope.OBSERVATIONS.value
        selected_scope = st.selectbox(
            "Scope",
            options=allowed_scopes or [EvaluationScope.OBSERVATIONS.value],
            format_func=lambda value: value.title(),
            index=(allowed_scopes.index(default_scope) if allowed_scopes else 0),
            key="studio_scope",
        )
        scope = EvaluationScope(selected_scope)
        if dataset is not None:
            missing_count = count_missing_scope_ids(dataset.items, scope)
            if missing_count:
                st.warning(
                    f"{missing_count} item secilen scope icin source id icermiyor; reevaluation sirasinda atlanacak."
                )

    validation_errors = validate_run_form(
        dataset=dataset,
        mode=ExperimentMode(mode),
        metrics=metrics,
        task_prompt=task_prompt,
        judge_prompt=judge_prompt,
        task_model=task_model,
        judge_model=judge_model,
    )
    for error in validation_errors:
        st.warning(error)

    execute_disabled = bool(validation_errors)
    if st.button(
        "Run experiment",
        width="stretch",
        disabled=execute_disabled,
        key="studio_execute_button",
    ):
        _execute(
            request=ExperimentExecutionRequest(
                dataset_name=dataset.dataset_name,
                mode=ExperimentMode(mode),
                judge_prompt=judge_prompt["custom_prompt"] or None,
                judge_model=judge_model or None,
                metrics=metrics,
                run_name=run_name or None,
                description=description or None,
                task_system_prompt=task_prompt["custom_prompt"] or None,
                task_model=task_model or None,
                max_concurrency=concurrency,
                scope=scope,
                task_prompt_source=task_prompt["source"],
                task_prompt_name=_effective_prompt_name("task", task_prompt),
                task_prompt_label=_effective_prompt_label("task", task_prompt),
                task_prompt_version=_effective_prompt_version("task", task_prompt),
                task_prompt_type=task_prompt["prompt_type"],
                judge_prompt_source=judge_prompt["source"],
                judge_prompt_name=_effective_prompt_name("judge", judge_prompt),
                judge_prompt_label=_effective_prompt_label("judge", judge_prompt),
                judge_prompt_version=_effective_prompt_version("judge", judge_prompt),
                judge_prompt_type=judge_prompt["prompt_type"],
                use_published_task_prompt=bool(st.session_state.get("studio_task_use_published_prompt")),
                use_published_judge_prompt=bool(st.session_state.get("studio_judge_use_published_prompt")),
            )
        )

    result = st.session_state.get("studio_result")
    if isinstance(result, ExperimentExecutionResult):
        if result.history_record_id:
            st.success(f"Run history kaydi olusturuldu: {result.history_record_id}")
        _render_result(result)


def build_metric_specs_from_form(
    preset_metrics: list[str],
    custom_metric_name: str,
    custom_metric_rubric: str,
) -> list[EvaluatorMetricSpec]:
    metrics = [
        EvaluatorMetricSpec(
            name=name,
            rubric=PRESET_METRIC_RUBRICS.get(name),
            is_custom=False,
        )
        for name in preset_metrics
    ]
    if custom_metric_name.strip():
        metrics.append(
            EvaluatorMetricSpec(
                name=custom_metric_name.strip(),
                rubric=custom_metric_rubric.strip() or None,
                is_custom=True,
            )
        )
    return metrics


def available_scopes(items: list[Any]) -> list[str]:
    scopes: list[str] = []
    if any(getattr(item, "source_observation_id", None) for item in items):
        scopes.append(EvaluationScope.OBSERVATIONS.value)
    if any(getattr(item, "source_trace_id", None) for item in items):
        scopes.append(EvaluationScope.TRACES.value)
    return scopes


def count_missing_scope_ids(items: list[Any], scope: EvaluationScope) -> int:
    count = 0
    for item in items:
        entity_id = (
            getattr(item, "source_observation_id", None)
            if scope == EvaluationScope.OBSERVATIONS
            else getattr(item, "source_trace_id", None)
        )
        if not entity_id:
            count += 1
    return count


def validate_run_form(
    *,
    dataset: Any,
    mode: ExperimentMode,
    metrics: list[EvaluatorMetricSpec],
    task_prompt: dict[str, Any],
    judge_prompt: dict[str, Any],
    task_model: str | None,
    judge_model: str | None,
) -> list[str]:
    errors: list[str] = []
    if dataset is None:
        errors.append("Once dataset fetch edin.")
    if not metrics:
        errors.append("En az bir metric secin veya custom metric ekleyin.")
    if not _prompt_is_ready(prefix="judge", prompt_state=judge_prompt):
        errors.append("Judge prompt kaynagini resolve edin veya custom prompt girin.")
    if mode == ExperimentMode.PROMPT_RUNNER:
        if not _prompt_is_ready(prefix="task", prompt_state=task_prompt):
            errors.append("Task prompt kaynagini resolve edin veya custom prompt girin.")
        if not task_model:
            errors.append("Prompt Runner icin task model gerekli.")
    if not judge_model:
        errors.append("Judge model gerekli.")
    return errors


def _prompt_is_ready(*, prefix: str, prompt_state: dict[str, Any]) -> bool:
    if prompt_state["is_ready"]:
        return True
    if st is None:  # pragma: no cover
        return False
    if st.session_state.get(f"studio_{prefix}_use_published_prompt"):
        published = st.session_state.get(f"studio_{prefix}_published_prompt")
        return isinstance(published, PublishedPromptResult) and bool(published.prompt_name)
    return False


def _effective_prompt_name(prefix: str, prompt_state: dict[str, Any]) -> str | None:
    if st.session_state.get(f"studio_{prefix}_use_published_prompt"):
        published = st.session_state.get(f"studio_{prefix}_published_prompt")
        if isinstance(published, PublishedPromptResult):
            return published.prompt_name
    return prompt_state["prompt_name"]


def _effective_prompt_label(prefix: str, prompt_state: dict[str, Any]) -> str | None:
    if st.session_state.get(f"studio_{prefix}_use_published_prompt"):
        published = st.session_state.get(f"studio_{prefix}_published_prompt")
        if isinstance(published, PublishedPromptResult):
            return published.prompt_label
    return prompt_state["prompt_label"]


def _effective_prompt_version(prefix: str, prompt_state: dict[str, Any]) -> int | None:
    if st.session_state.get(f"studio_{prefix}_use_published_prompt"):
        published = st.session_state.get(f"studio_{prefix}_published_prompt")
        if isinstance(published, PublishedPromptResult):
            return published.prompt_version
    return prompt_state["prompt_version"]


def _consume_pending_prompt_apply() -> None:
    pending = st.session_state.pop("studio_pending_prompt_apply", None)
    if not pending:
        return
    prompt = str(pending.get("prompt", ""))
    for prefix in pending.get("targets", []):
        st.session_state[f"studio_{prefix}_use_langfuse"] = False
        st.session_state[f"studio_{prefix}_custom_prompt"] = prompt
        st.session_state.pop(f"studio_{prefix}_prompt_resolution", None)


def _consume_pending_prompt_browser_selection() -> None:
    pending = st.session_state.pop("studio_pending_prompt_browser_selection", None)
    if not pending:
        return
    prefix = pending["prefix"]
    st.session_state[f"studio_{prefix}_use_langfuse"] = True
    st.session_state[f"studio_{prefix}_prompt_name"] = pending["prompt_name"]
    st.session_state[f"studio_{prefix}_prompt_label"] = pending["prompt_label"]
    st.session_state[f"studio_{prefix}_prompt_version"] = pending["prompt_version"]
    st.session_state.pop(f"studio_{prefix}_prompt_resolution", None)


def _render_prompt_block(*, title: str, prefix: str, target: PromptTarget) -> dict[str, Any]:
    st.subheader(title)
    use_langfuse = st.toggle(
        "Use Langfuse prompt name",
        key=f"studio_{prefix}_use_langfuse",
        value=False,
    )
    source = PromptSource.LANGFUSE_PROMPT if use_langfuse else PromptSource.CUSTOM_PROMPT
    prompt_name = st.text_input(
        "Prompt name",
        key=f"studio_{prefix}_prompt_name",
        disabled=not use_langfuse,
    )
    label_col, version_col, type_col = st.columns([1, 1, 1])
    with label_col:
        prompt_label = st.text_input(
            "Prompt label",
            key=f"studio_{prefix}_prompt_label",
            disabled=not use_langfuse,
        )
    with version_col:
        raw_version = st.text_input(
            "Prompt version",
            key=f"studio_{prefix}_prompt_version",
            disabled=not use_langfuse,
        )
    with type_col:
        prompt_type_value = st.selectbox(
            "Prompt type",
            options=[PromptType.TEXT.value, PromptType.CHAT.value],
            key=f"studio_{prefix}_prompt_type",
            index=0,
        )
    prompt_type = PromptType(prompt_type_value)
    _render_prompt_browser(prefix=prefix, prompt_type=prompt_type)
    if prompt_type == PromptType.CHAT:
        custom_prompt, custom_messages = _render_chat_prompt_editor(prefix=prefix)
    else:
        custom_prompt = st.text_area(
            "Custom prompt",
            key=f"studio_{prefix}_custom_prompt",
            height=140,
            placeholder="Langfuse prompt bulunamazsa veya custom prompt kullanmak isterseniz bu alani doldurun.",
        )
        custom_messages = []
    state_key = f"studio_{prefix}_prompt_resolution"
    if st.button("Fetch prompt", key=f"studio_{prefix}_fetch_prompt", disabled=not use_langfuse):
        try:
            version = int(raw_version) if raw_version.strip() else None
            resolution = resolve_prompt(
                PromptResolutionRequest(
                    source=source,
                    target=target,
                    prompt_name=prompt_name or None,
                    prompt_label=prompt_label or None,
                    prompt_version=version,
                    prompt_type=prompt_type,
                    custom_prompt=custom_prompt or None,
                )
            )
            st.session_state[state_key] = resolution
        except Exception as exc:  # pragma: no cover - streamlit path
            st.session_state[state_key] = None
            st.error(f"Prompt fetch hatasi: {exc}")

    resolution = st.session_state.get(state_key)
    if source == PromptSource.CUSTOM_PROMPT and custom_prompt.strip():
        resolution = resolve_prompt(
            PromptResolutionRequest(
                source=PromptSource.CUSTOM_PROMPT,
                target=target,
                prompt_type=prompt_type,
                custom_prompt=custom_prompt,
            )
        )
    if resolution is not None:
        _render_prompt_preview(resolution)

    version = int(raw_version) if raw_version.strip().isdigit() else None
    is_ready = bool(resolution) or bool(custom_prompt.strip() and source == PromptSource.CUSTOM_PROMPT)
    return {
        "prefix": prefix,
        "source": source,
        "prompt_name": prompt_name or None,
        "prompt_label": prompt_label or None,
        "prompt_version": version,
        "prompt_type": prompt_type,
        "custom_prompt": custom_prompt,
        "custom_messages": custom_messages,
        "resolution": resolution,
        "is_ready": is_ready,
    }


def _render_prompt_publish_section(
    *,
    prefix: str,
    title: str,
    target: PromptPublishTarget,
    prompt_state: dict[str, Any],
) -> None:
    st.markdown(f"### {title}")
    publish_name = st.text_input(
        "Publish prompt name",
        value=prompt_state["prompt_name"] or st.session_state.get(f"studio_{prefix}_publish_name", ""),
        key=f"studio_{prefix}_publish_name",
        placeholder=f"{prefix}-prompt-name",
    )
    left, right = st.columns([2, 1])
    with left:
        publish_label = st.text_input(
            "Publish label (optional)",
            key=f"studio_{prefix}_publish_label",
            placeholder="production",
        )
    with right:
        use_next_run = st.toggle(
            "Use for next run",
            value=True,
            key=f"studio_{prefix}_publish_use_next_run",
        )
    commit_message = st.text_input(
        "Commit message (optional)",
        key=f"studio_{prefix}_publish_commit_message",
        placeholder="Prompt Coach revise -> accepted",
    )
    published = st.session_state.get(f"studio_{prefix}_published_prompt")
    if st.button(f"Publish {prefix} prompt", key=f"studio_{prefix}_publish_button", width="stretch"):
        try:
            if not prompt_state["is_ready"]:
                raise ValueError("Publish icin once prompt source resolve edilmeli veya custom prompt girilmeli.")
            resolved = prompt_state.get("resolution").resolved_prompt if prompt_state.get("resolution") else None
            messages = resolved.messages if resolved else prompt_state.get("custom_messages", [])
            prompt_text = (resolved.compiled_text if resolved else prompt_state["custom_prompt"]) or ""
            publish_result = publish_prompt(
                PublishedPromptRequest(
                    target=target,
                    prompt_name=publish_name.strip(),
                    prompt_type=prompt_state["prompt_type"],
                    prompt_text=prompt_text,
                    messages=messages,
                    label=publish_label.strip() or None,
                    commit_message=commit_message.strip() or None,
                    use_for_next_run=use_next_run,
                    source=(resolved.source if resolved else prompt_state["source"]),
                    source_fingerprint=(resolved.fingerprint if resolved else None),
                )
            )
            st.session_state[f"studio_{prefix}_published_prompt"] = publish_result
            st.session_state[f"studio_{prefix}_use_published_prompt"] = publish_result.use_for_next_run
            st.success(
                f"Published: {publish_result.prompt_name} v{publish_result.prompt_version or 'n/a'}"
            )
        except Exception as exc:  # pragma: no cover
            st.error(f"Prompt publish hatasi: {exc}")
    if isinstance(published, PublishedPromptResult):
        st.caption(
            f"Published prompt: {published.prompt_name} | version={published.prompt_version or 'n/a'} | label={published.prompt_label or 'n/a'}"
        )
        st.toggle(
            "Use published version in next run",
            key=f"studio_{prefix}_use_published_prompt",
            value=published.use_for_next_run,
        )


def _render_run_with_published_prompt() -> None:
    st.markdown("### Run With Published Prompt")
    task_published = st.session_state.get("studio_task_published_prompt")
    judge_published = st.session_state.get("studio_judge_published_prompt")
    rows = []
    for name, value in [("task", task_published), ("judge", judge_published)]:
        if isinstance(value, PublishedPromptResult):
            rows.append(
                {
                    "target": name,
                    "prompt_name": value.prompt_name,
                    "version": value.prompt_version,
                    "label": value.prompt_label,
                    "use_in_next_run": bool(st.session_state.get(f"studio_{name}_use_published_prompt")),
                }
            )
    if rows:
        st.dataframe(rows, width="stretch", hide_index=True)
    else:
        st.caption("Henuz publish edilmis prompt yok.")


def _render_prompt_browser(*, prefix: str, prompt_type: PromptType) -> None:
    st.markdown("#### Published Prompt Browser")
    search_col, label_col, action_col = st.columns([2, 2, 1])
    with search_col:
        search_name = st.text_input("Search name", key=f"studio_{prefix}_browser_name", placeholder="support")
    with label_col:
        search_label = st.text_input("Search label", key=f"studio_{prefix}_browser_label", placeholder="production")
    with action_col:
        load = st.button("Load", key=f"studio_{prefix}_browser_load", width="stretch")
    if load:
        try:
            rows = list_prompts(
                name=search_name.strip() or None,
                label=search_label.strip() or None,
                tag="published",
                limit=25,
            )
            filtered = [row for row in rows if str(row.get("type", "")).lower() == prompt_type.value]
            st.session_state[f"studio_{prefix}_browser_results"] = filtered
        except Exception as exc:  # pragma: no cover
            st.error(f"Prompt browser hatasi: {exc}")
    rows = st.session_state.get(f"studio_{prefix}_browser_results", [])
    if rows:
        st.dataframe(_build_prompt_browser_rows(rows), width="stretch", hide_index=True)
        options = _build_prompt_browser_options(rows)
        selected = st.selectbox(
            "Select published prompt",
            options=list(options.keys()),
            key=f"studio_{prefix}_browser_selected",
        )
        if st.button("Use selected prompt", key=f"studio_{prefix}_browser_apply", width="stretch"):
            selected_row = options[selected]
            st.session_state["studio_pending_prompt_browser_selection"] = {
                "prefix": prefix,
                "prompt_name": selected_row["name"],
                "prompt_label": selected_row["labels"][0] if selected_row.get("labels") else "",
                "prompt_version": str(max(selected_row.get("versions") or [1])),
            }
            st.rerun()
    else:
        st.caption("Published prompt listesi henuz yuklenmedi.")


def _render_chat_prompt_editor(*, prefix: str) -> tuple[str, list[dict[str, str]]]:
    st.markdown("#### Chat Prompt Editor")
    rows = st.session_state.get(f"studio_{prefix}_chat_messages")
    if not rows:
        rows = [{"role": "system", "content": ""}, {"role": "user", "content": ""}]
        st.session_state[f"studio_{prefix}_chat_messages"] = rows
    edited = st.data_editor(
        rows,
        key=f"studio_{prefix}_chat_editor",
        width="stretch",
        num_rows="dynamic",
        column_config={
            "role": st.column_config.SelectboxColumn(
                "Role",
                options=["system", "user", "assistant", "placeholder"],
                required=True,
            ),
            "content": st.column_config.TextColumn("Content", required=True),
        },
        hide_index=True,
    )
    normalized = _normalize_chat_editor_rows(edited)
    st.session_state[f"studio_{prefix}_chat_messages"] = normalized
    serialized = _serialize_chat_messages(normalized)
    st.text_area(
        "Compiled chat preview",
        value=serialized,
        key=f"studio_{prefix}_chat_preview",
        height=140,
        disabled=True,
    )
    return serialized, normalized


def _render_prompt_preview(resolution: Any) -> None:
    resolved: ResolvedPrompt = resolution.resolved_prompt
    badge = resolved.source.value
    st.markdown(f"**Source:** `{badge}`  |  **Type:** `{resolved.prompt_type.value if resolved.prompt_type else 'n/a'}`")
    if resolved.prompt_name:
        st.caption(
            f"Prompt: {resolved.prompt_name} | version={resolved.prompt_version or 'n/a'} | label={resolved.prompt_label or 'n/a'}"
        )
    for warning in resolution.warnings:
        st.warning(warning)
    preview = resolved.compiled_text or ""
    st.text_area("Prompt preview", value=preview, height=120, disabled=True)
    if resolved.prompt_type == PromptType.CHAT and resolved.messages:
        st.dataframe(resolved.messages, width="stretch", hide_index=True)


def _load_dataset(dataset_name: str) -> None:
    if not dataset_name.strip():
        st.error("Dataset name zorunlu.")
        return
    try:
        st.session_state["studio_dataset"] = fetch_dataset_by_name(dataset_name.strip())
        st.session_state.pop("studio_result", None)
    except Exception as exc:  # pragma: no cover - streamlit path
        st.error(f"Dataset fetch sirasinda hata: {exc}")


def _execute(request: ExperimentExecutionRequest) -> None:
    result = (
        run_prompt_experiment(request)
        if request.mode == ExperimentMode.PROMPT_RUNNER
        else run_llm_judge_on_existing_results(request)
    )
    st.session_state["studio_result"] = result


def _render_dataset_loader(dataset: Any) -> None:
    st.subheader("Dataset Loader")
    meta1, meta2, meta3 = st.columns(3)
    meta1.metric("Dataset ID", dataset.dataset_id)
    meta2.metric("Items", dataset.total_items)
    meta3.metric("Warnings", len(dataset.warnings))
    if dataset.description:
        st.caption(dataset.description)
    for warning in dataset.warnings:
        st.warning(warning)
    st.dataframe(
        [
            {
                "id": item.id,
                "status": item.status,
                "source_trace_id": item.source_trace_id,
                "source_observation_id": item.source_observation_id,
                "input": _preview(item.input),
                "expected_output": _preview(item.expected_output),
            }
            for item in dataset.items
        ],
        width="stretch",
        hide_index=True,
    )


def _render_metric_preview(metrics: list[EvaluatorMetricSpec]) -> None:
    st.subheader("Evaluator Builder")
    if not metrics:
        st.info("En az bir metric secin veya custom metric ekleyin.")
        return
    st.dataframe(
        [
            {
                "metric": metric.name,
                "rubric": metric.rubric or "Global judge prompt kullanilir.",
                "custom": metric.is_custom,
            }
            for metric in metrics
        ],
        width="stretch",
        hide_index=True,
    )


def _render_result(result: ExperimentExecutionResult) -> None:
    st.subheader("Results")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Processed", result.processed_items)
    c2.metric("Failed", result.failed_items)
    c3.metric("Aggregate metrics", len(result.aggregate_metrics))
    c4.metric("Status", result.status.value)
    if result.dataset_run_url:
        st.markdown(f"[Open Langfuse Run]({result.dataset_run_url})")
    for warning in result.warnings:
        st.warning(warning)
    for error in result.errors:
        st.error(error)
    if result.task_prompt_summary or result.judge_prompt_summary:
        st.markdown("### Prompt summary")
        left, right = st.columns(2)
        with left:
            if result.task_prompt_summary:
                st.json(result.task_prompt_summary.model_dump())
        with right:
            if result.judge_prompt_summary:
                st.json(result.judge_prompt_summary.model_dump())
    if result.published_task_prompt or result.published_judge_prompt:
        st.markdown("### Published prompt summary")
        rows = []
        for value in [result.published_task_prompt, result.published_judge_prompt]:
            if value:
                rows.append(value.model_dump())
        st.dataframe(rows, width="stretch", hide_index=True)
    if result.aggregate_metrics:
        st.markdown("### Aggregate metric summary")
        st.dataframe(
            [metric.model_dump() for metric in result.aggregate_metrics],
            width="stretch",
            hide_index=True,
        )
    st.markdown("### Item results")
    st.dataframe(
        [
            {
                "dataset_item_id": row.dataset_item_id,
                "entity_id": row.entity_id,
                "entity_type": row.entity_type,
                "trace_id": row.trace_id,
                "evaluation_count": len(row.evaluations),
                "output": _preview(row.output),
            }
            for row in result.item_results
        ],
        width="stretch",
        hide_index=True,
    )
    if result.item_results:
        detail_options = {_result_label(row): row for row in result.item_results}
        selected_key = st.selectbox(
            "Result detail",
            list(detail_options.keys()),
            key="studio_result_detail",
        )
        selected = detail_options[selected_key]
        left, right = st.columns([1, 1])
        with left:
            st.markdown("#### Input")
            _render_value_block("Input payload", selected.input)
            st.markdown("#### Expected output")
            _render_value_block("Expected output payload", selected.expected_output)
            st.markdown("#### Output")
            _render_value_block("Output payload", selected.output)
        with right:
            st.markdown("#### Evaluations")
            if selected.evaluations:
                st.dataframe(
                    [evaluation.model_dump() for evaluation in selected.evaluations],
                    width="stretch",
                    hide_index=True,
                )
            else:
                st.info("Bu item icin evaluation kaydi olusmadi.")


def _result_label(row: Any) -> str:
    label = row.dataset_item_id or row.entity_id or "result"
    if row.entity_type:
        return f"{label} [{row.entity_type}]"
    return label


def _preview(value: Any) -> str:
    if value is None:
        return ""
    text = str(value)
    return text if len(text) <= 120 else f"{text[:117]}..."


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


def _normalize_chat_editor_rows(rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    normalized: list[dict[str, str]] = []
    for row in rows:
        role = str(row.get("role", "")).strip()
        content = str(row.get("content", "")).strip()
        if not role and not content:
            continue
        normalized.append({"role": role or "user", "content": content})
    return normalized


def _serialize_chat_messages(messages: list[dict[str, str]]) -> str:
    return "\n".join(f"[{message['role']}] {message['content']}" for message in messages).strip()


def _build_prompt_browser_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "name": row.get("name"),
            "type": row.get("type"),
            "versions": ", ".join(str(version) for version in row.get("versions", [])),
            "labels": ", ".join(row.get("labels", [])),
            "tags": ", ".join(row.get("tags", [])),
            "last_updated_at": row.get("lastUpdatedAt") or row.get("last_updated_at"),
        }
        for row in rows
    ]


def _build_prompt_browser_options(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    options: dict[str, dict[str, Any]] = {}
    for row in rows:
        latest_version = max(row.get("versions") or [1])
        label = row.get("labels", [None])[0]
        key = f"{row.get('name')} | v{latest_version} | label={label or 'n/a'}"
        options[key] = row
    return options


def _inject_css() -> None:
    st.markdown(
        """
        <style>
        .studio-hint {
            background: linear-gradient(135deg, rgba(117, 14, 33, 0.76), rgba(36, 27, 29, 0.96));
            border: 1px solid rgba(190, 215, 84, 0.28);
            color: #f7f3e9;
            border-radius: 14px;
            padding: 0.85rem 1rem;
            min-height: 3rem;
            display: flex;
            align-items: center;
            box-shadow: 0 12px 30px rgba(0, 0, 0, 0.18);
        }
        .studio-hint strong {
            color: #BED754;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
