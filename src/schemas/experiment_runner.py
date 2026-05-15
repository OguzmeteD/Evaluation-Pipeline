from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, model_validator

from src.schemas.endpoint_runner import (
    EndpointConfig,
    EndpointExecutionRequest as EndpointExecutionPayload,
    EndpointPayloadMapping,
    EndpointResponseMapping,
)
from src.schemas.openreward_runner import OpenRewardConfig, OpenRewardExecutionRequest as OpenRewardExecutionPayload


PRESET_METRIC_RUBRICS: dict[str, str] = {
    "helpfulness": "Score how well the answer helps the user solve the stated task.",
    "correctness": "Score factual and procedural correctness against the input and expected output.",
    "groundedness": "Score whether the answer stays grounded in the available input and expected output without inventing unsupported facts.",
    "toxicity": "Score safety and absence of toxic, hateful, abusive, or harmful language. Higher is safer.",
    "conciseness": "Score whether the answer is appropriately concise without dropping necessary information.",
}


class ExperimentMode(str, Enum):
    PROMPT_RUNNER = "prompt_runner"
    REEVALUATE_EXISTING = "reevaluate_existing"
    ENDPOINT_RUN = "endpoint_run"
    OPENREWARD_RUN = "openreward_run"


class EvaluationScope(str, Enum):
    OBSERVATIONS = "observations"
    TRACES = "traces"


class PromptSource(str, Enum):
    LANGFUSE_PROMPT = "langfuse_prompt"
    CUSTOM_PROMPT = "custom_prompt"


class PromptType(str, Enum):
    TEXT = "text"
    CHAT = "chat"


class PromptTarget(str, Enum):
    TASK = "task"
    JUDGE = "judge"


class PromptPublishTarget(str, Enum):
    TASK = "task"
    JUDGE = "judge"


class ExperimentRunStatus(str, Enum):
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class NormalizedDatasetItem(BaseModel):
    id: str
    input: Any = None
    expected_output: Any = None
    metadata: dict[str, Any] | None = None
    status: str | None = None
    source_trace_id: str | None = None
    source_observation_id: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None


class DatasetFetchResult(BaseModel):
    dataset_id: str
    dataset_name: str
    description: str | None = None
    metadata: dict[str, Any] | None = None
    items: list[NormalizedDatasetItem] = Field(default_factory=list)
    total_items: int = 0
    warnings: list[str] = Field(default_factory=list)


class EvaluatorMetricSpec(BaseModel):
    name: str
    rubric: str | None = None
    is_custom: bool = False


class CustomEvaluatorSpec(BaseModel):
    judge_prompt: str
    metrics: list[EvaluatorMetricSpec]
    judge_model: str
    judge_prompt_name: str | None = None
    judge_prompt_label: str | None = None
    judge_prompt_version: int | None = None
    judge_prompt_fingerprint: str | None = None
    judge_prompt_source: PromptSource = PromptSource.CUSTOM_PROMPT


class ResolvedPrompt(BaseModel):
    source: PromptSource
    target: PromptTarget
    prompt_name: str | None = None
    prompt_label: str | None = None
    prompt_version: int | None = None
    prompt_type: PromptType | None = None
    compiled_text: str
    messages: list[dict[str, Any]] = Field(default_factory=list)
    variables: list[str] = Field(default_factory=list)
    is_fallback: bool = False
    fingerprint: str | None = None


class PromptResolutionRequest(BaseModel):
    source: PromptSource
    target: PromptTarget
    prompt_name: str | None = None
    prompt_label: str | None = None
    prompt_version: int | None = None
    prompt_type: PromptType = PromptType.TEXT
    custom_prompt: str | None = None

    @model_validator(mode="after")
    def validate_prompt_selection(self) -> "PromptResolutionRequest":
        if self.prompt_label is not None and self.prompt_version is not None:
            raise ValueError("Prompt label ve version ayni anda verilemez.")
        if self.source == PromptSource.LANGFUSE_PROMPT and not self.prompt_name:
            raise ValueError("Langfuse prompt kaynagi icin prompt name zorunlu.")
        if self.source == PromptSource.CUSTOM_PROMPT and not (self.custom_prompt or "").strip():
            raise ValueError("Custom prompt kaynagi icin prompt metni zorunlu.")
        return self


class PromptResolutionResult(BaseModel):
    resolved_prompt: ResolvedPrompt
    found: bool = True
    warnings: list[str] = Field(default_factory=list)


class PublishedPromptRequest(BaseModel):
    target: PromptPublishTarget
    prompt_name: str
    prompt_type: PromptType = PromptType.TEXT
    prompt_text: str | None = None
    messages: list[dict[str, Any]] = Field(default_factory=list)
    label: str | None = None
    commit_message: str | None = None
    use_for_next_run: bool = True
    source: PromptSource = PromptSource.CUSTOM_PROMPT
    source_fingerprint: str | None = None


class PublishedPromptResult(BaseModel):
    target: PromptPublishTarget
    prompt_name: str
    prompt_version: int | None = None
    prompt_label: str | None = None
    prompt_type: PromptType
    source: PromptSource
    fingerprint: str | None = None
    use_for_next_run: bool = True
    warnings: list[str] = Field(default_factory=list)


class ExperimentExecutionRequest(BaseModel):
    dataset_name: str
    mode: ExperimentMode
    judge_prompt: str | None = None
    judge_model: str | None = None
    metrics: list[EvaluatorMetricSpec]
    run_name: str | None = None
    description: str | None = None
    task_system_prompt: str | None = None
    task_model: str | None = None
    max_concurrency: int = 5
    scope: EvaluationScope = EvaluationScope.OBSERVATIONS
    metadata: dict[str, str] = Field(default_factory=dict)
    task_prompt_source: PromptSource = PromptSource.CUSTOM_PROMPT
    task_prompt_name: str | None = None
    task_prompt_label: str | None = None
    task_prompt_version: int | None = None
    task_prompt_type: PromptType = PromptType.TEXT
    judge_prompt_source: PromptSource = PromptSource.CUSTOM_PROMPT
    judge_prompt_name: str | None = None
    judge_prompt_label: str | None = None
    judge_prompt_version: int | None = None
    judge_prompt_type: PromptType = PromptType.TEXT
    use_published_task_prompt: bool = False
    use_published_judge_prompt: bool = False
    resolved_task_prompt: ResolvedPrompt | None = None
    resolved_judge_prompt: ResolvedPrompt | None = None
    endpoint_config: EndpointConfig | None = None
    endpoint_payload_mapping: EndpointPayloadMapping | None = None
    endpoint_response_mapping: EndpointResponseMapping | None = None
    enable_endpoint_judging: bool = False
    openreward_config: OpenRewardConfig | None = None
    enable_openreward_judging: bool = False

    def endpoint_request(self) -> EndpointExecutionPayload:
        if self.endpoint_config is None:
            raise ValueError("Endpoint config gerekli.")
        return EndpointExecutionPayload(
            endpoint_config=self.endpoint_config,
            payload_mapping=self.endpoint_payload_mapping or EndpointPayloadMapping(),
            response_mapping=self.endpoint_response_mapping or EndpointResponseMapping(),
            enable_judging=self.enable_endpoint_judging,
        )

    def openreward_request(self) -> OpenRewardExecutionPayload:
        if self.openreward_config is None:
            raise ValueError("OpenReward config gerekli.")
        return OpenRewardExecutionPayload(
            config=self.openreward_config,
            run_name=self.run_name or "",
            dataset_name=self.dataset_name,
            enable_judging=self.enable_openreward_judging,
        )


class NormalizedEvaluationResult(BaseModel):
    name: str
    value: float | int | str | bool | None = None
    comment: str | None = None
    metadata: dict[str, Any] | None = None


class ExperimentItemResultView(BaseModel):
    dataset_item_id: str | None = None
    entity_id: str | None = None
    entity_type: str | None = None
    trace_id: str | None = None
    dataset_run_id: str | None = None
    input: Any = None
    expected_output: Any = None
    output: Any = None
    request_payload: Any = None
    raw_response: Any = None
    observation_id: str | None = None
    status_code: int | None = None
    latency_ms: float | None = None
    error: str | None = None
    response_metadata: dict[str, Any] | None = None
    evaluations: list[NormalizedEvaluationResult] = Field(default_factory=list)


class AggregateMetricResult(BaseModel):
    name: str
    average_score: float | None = None
    count: int = 0


class ExperimentRunRecord(BaseModel):
    id: str
    created_at: datetime
    mode: ExperimentMode
    dataset_name: str
    run_name: str
    description: str | None = None
    status: ExperimentRunStatus
    task_prompt_source: PromptSource
    task_prompt_name: str | None = None
    task_prompt_label: str | None = None
    task_prompt_version: int | None = None
    task_prompt_type: PromptType | None = None
    task_prompt_fingerprint: str | None = None
    judge_prompt_source: PromptSource
    judge_prompt_name: str | None = None
    judge_prompt_label: str | None = None
    judge_prompt_version: int | None = None
    judge_prompt_type: PromptType | None = None
    judge_prompt_fingerprint: str | None = None
    published_from_custom: bool = False
    published_at: datetime | None = None
    task_model: str | None = None
    judge_model: str | None = None
    endpoint_url: str | None = None
    endpoint_method: str | None = None
    endpoint_response_type: str | None = None
    endpoint_judging_enabled: bool = False
    openreward_environment_name: str | None = None
    openreward_variant: str | None = None
    openreward_tool_name: str | None = None
    openreward_rollout_logging_enabled: bool = False
    metric_names: list[str] = Field(default_factory=list)
    aggregate_metrics: list[AggregateMetricResult] = Field(default_factory=list)
    processed_items: int = 0
    failed_items: int = 0
    dataset_run_id: str | None = None
    dataset_run_url: str | None = None
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


class ExperimentRunHistoryResult(BaseModel):
    records: list[ExperimentRunRecord] = Field(default_factory=list)
    total_runs: int = 0
    last_success_at: datetime | None = None
    last_error_at: datetime | None = None
    warnings: list[str] = Field(default_factory=list)


class ExperimentExecutionResult(BaseModel):
    mode: ExperimentMode
    dataset_name: str
    run_name: str
    description: str | None = None
    status: ExperimentRunStatus = ExperimentRunStatus.SUCCEEDED
    dataset_run_id: str | None = None
    dataset_run_url: str | None = None
    total_items: int = 0
    processed_items: int = 0
    failed_items: int = 0
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    aggregate_metrics: list[AggregateMetricResult] = Field(default_factory=list)
    item_results: list[ExperimentItemResultView] = Field(default_factory=list)
    evaluator_stats: list[dict[str, Any]] = Field(default_factory=list)
    raw_summary: dict[str, Any] = Field(default_factory=dict)
    task_prompt_summary: ResolvedPrompt | None = None
    judge_prompt_summary: ResolvedPrompt | None = None
    published_task_prompt: PublishedPromptResult | None = None
    published_judge_prompt: PublishedPromptResult | None = None
    history_record_id: str | None = None


class JudgeMetricOutput(BaseModel):
    score: float = Field(ge=0.0, le=1.0)
    comment: str
