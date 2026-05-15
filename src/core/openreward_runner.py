from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:  # pragma: no cover
    from openreward import Rollout
    from openreward.client import OpenReward


@dataclass
class _FallbackTask:
    server_name: str
    environment_name: str
    task_spec: dict[str, Any]
    namespace: str | None


@dataclass
class _FallbackSystemMessage:
    content: str
    type: str = "system_message"


@dataclass
class _FallbackUserMessage:
    content: str
    type: str = "user_message"


@dataclass
class _FallbackToolCall:
    name: str
    content: str
    call_id: str
    type: str = "tool_call"


@dataclass
class _FallbackToolResult:
    content: str
    call_id: str
    type: str = "tool_result"

from src.schemas.experiment_runner import NormalizedDatasetItem
from src.schemas.openreward_runner import (
    OpenRewardConfig,
    OpenRewardExecutionRequest,
    OpenRewardExecutionResult,
    OpenRewardItemResult,
)


class OpenRewardClientLike(Protocol):
    @property
    def environments(self) -> Any: ...

    @property
    def rollout(self) -> Any: ...

    def close(self) -> None: ...


class OpenRewardRunnerService:
    def __init__(self, client_factory: Any | None = None) -> None:
        self.client_factory = client_factory or self._default_client_factory

    def run_openreward_execution(
        self,
        *,
        items: list[NormalizedDatasetItem],
        request: OpenRewardExecutionRequest,
    ) -> OpenRewardExecutionResult:
        client = self.client_factory(base_url=request.config.base_url)
        warnings: list[str] = []
        item_results: list[OpenRewardItemResult] = []
        try:
            environment = client.environments.get(request.config.environment_name, variant=request.config.variant)
            for item in items:
                item_results.append(
                    self._run_single_item(
                        client=client,
                        environment=environment,
                        item=item,
                        request=request,
                    )
                )
        finally:
            client.close()

        failed_items = sum(1 for row in item_results if row.error)
        rewards = [row.reward for row in item_results if row.error is None and row.reward is not None]
        finished_count = sum(1 for row in item_results if row.error is None and row.finished is True)
        latencies = [row.latency_ms for row in item_results if row.error is None and row.latency_ms is not None]
        processed_items = len(item_results)
        successful_items = processed_items - failed_items
        summary = {
            "success_rate": (successful_items / processed_items) if processed_items else 0.0,
            "average_reward": (sum(rewards) / len(rewards)) if rewards else None,
            "finished_rate": (finished_count / successful_items) if successful_items else 0.0,
            "avg_latency_ms": (sum(latencies) / len(latencies)) if latencies else None,
        }
        if failed_items:
            warnings.append(f"{failed_items} OpenReward item'i basarisiz oldu.")
        return OpenRewardExecutionResult(
            processed_items=processed_items,
            failed_items=failed_items,
            item_results=item_results,
            warnings=warnings,
            errors=[],
            summary=summary,
        )

    def _run_single_item(
        self,
        *,
        client: OpenRewardClientLike,
        environment: Any,
        item: NormalizedDatasetItem,
        request: OpenRewardExecutionRequest,
    ) -> OpenRewardItemResult:
        started = time.perf_counter()
        config = request.config
        context = self._build_context(item)
        task_spec = self._build_task_spec(item=item, config=config, context=context)
        task = self._task_type()(
            server_name=environment.server,
            environment_name=environment.name,
            task_spec=task_spec,
            namespace=environment.namespace,
        )
        rollout = self._create_rollout(
            client=client,
            request=request,
            item=item,
            task_spec=task_spec,
        )
        try:
            with environment.session(task=task) as session:
                prompt_blocks = session.get_prompt()
                available_tools = [tool.name for tool in session.list_tools()]
                if config.tool_name not in available_tools:
                    raise ValueError(
                        f"OpenReward tool bulunamadi: {config.tool_name}. Mevcut tool'lar: {', '.join(available_tools) or 'yok'}"
                    )

                tool_input = self._build_tool_input(item=item, config=config, context=context)
                self._log_rollout_prompt(
                    rollout=rollout,
                    prompt_blocks=prompt_blocks,
                    item=item,
                )
                call_id = item.id
                if rollout is not None:
                    rollout.log(
                        self._tool_call_type()(
                            name=config.tool_name,
                            content=json.dumps(tool_input, ensure_ascii=True, default=str),
                            call_id=call_id,
                        )
                    )
                tool_output = session.call_tool(config.tool_name, tool_input)
                latency_ms = (time.perf_counter() - started) * 1000.0
                tool_output_dict = self._tool_output_to_dict(tool_output)
                if rollout is not None:
                    rollout.log(
                        self._tool_result_type()(
                            content=json.dumps(tool_output_dict, ensure_ascii=True, default=str),
                            call_id=call_id,
                        ),
                        reward=tool_output.reward,
                        is_finished=tool_output.finished,
                        metadata=tool_output.metadata or {},
                    )
                return OpenRewardItemResult(
                    dataset_item_id=item.id,
                    session_id=session.sid,
                    prompt_blocks=[self._block_to_dict(block) for block in prompt_blocks],
                    available_tools=available_tools,
                    tool_input=tool_input,
                    tool_output=tool_output_dict,
                    output=self._serialize_blocks(tool_output.blocks),
                    reward=tool_output.reward,
                    finished=tool_output.finished,
                    rollout_id=getattr(rollout, "event_id", None),
                    rollout_url=self._build_rollout_url(client, rollout),
                    latency_ms=latency_ms,
                    metadata=tool_output.metadata or {},
                )
        except Exception as exc:
            return OpenRewardItemResult(
                dataset_item_id=item.id,
                tool_input=self._build_tool_input(item=item, config=config, context=context),
                rollout_id=getattr(rollout, "event_id", None),
                rollout_url=self._build_rollout_url(client, rollout),
                latency_ms=(time.perf_counter() - started) * 1000.0,
                error=str(exc),
            )

    @staticmethod
    def _build_context(item: NormalizedDatasetItem) -> dict[str, Any]:
        return {
            "input": item.input,
            "expected_output": item.expected_output,
            "metadata": item.metadata or {},
            "dataset_item_id": item.id,
            "source_trace_id": item.source_trace_id,
            "source_observation_id": item.source_observation_id,
        }

    def _build_task_spec(
        self,
        *,
        item: NormalizedDatasetItem,
        config: OpenRewardConfig,
        context: dict[str, Any],
    ) -> dict[str, Any]:
        if config.task_spec_template:
            payload = self._inject_placeholders(config.task_spec_template, context)
            if not isinstance(payload, dict):
                raise ValueError("OpenReward task spec template bir JSON object olmali.")
            return payload
        if isinstance(item.input, dict):
            return dict(item.input)
        return {
            "input": item.input,
            "expected_output": item.expected_output,
            "metadata": item.metadata or {},
            "dataset_item_id": item.id,
        }

    def _build_tool_input(
        self,
        *,
        item: NormalizedDatasetItem,
        config: OpenRewardConfig,
        context: dict[str, Any],
    ) -> dict[str, Any]:
        if config.tool_input_template:
            payload = self._inject_placeholders(config.tool_input_template, context)
            if not isinstance(payload, dict):
                raise ValueError("OpenReward tool input template bir JSON object olmali.")
            return payload
        value = item.expected_output if item.expected_output is not None else item.input
        return {config.tool_input_field_name: value}

    def _inject_placeholders(self, value: Any, context: dict[str, Any]) -> Any:
        if isinstance(value, dict):
            return {key: self._inject_placeholders(child, context) for key, child in value.items()}
        if isinstance(value, list):
            return [self._inject_placeholders(child, context) for child in value]
        if isinstance(value, str):
            for key, replacement in context.items():
                token = f"{{{{{key}}}}}"
                if value == token:
                    return replacement
                if token in value:
                    value = value.replace(token, self._replacement_to_string(replacement))
            return value
        return value

    @staticmethod
    def _replacement_to_string(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        try:
            return json.dumps(value, ensure_ascii=True, default=str)
        except TypeError:
            return str(value)

    @staticmethod
    def _block_to_dict(block: Any) -> dict[str, Any]:
        data = {"type": getattr(block, "type", None)}
        if getattr(block, "type", None) == "text":
            data["text"] = getattr(block, "text", "")
        if getattr(block, "type", None) == "image":
            data["mimeType"] = getattr(block, "mimeType", None)
            data["data"] = getattr(block, "data", None)
        detail = getattr(block, "detail", None)
        if detail is not None:
            data["detail"] = detail
        return data

    def _tool_output_to_dict(self, tool_output: Any) -> dict[str, Any]:
        return {
            "blocks": [self._block_to_dict(block) for block in getattr(tool_output, "blocks", [])],
            "metadata": getattr(tool_output, "metadata", None),
            "reward": getattr(tool_output, "reward", None),
            "finished": getattr(tool_output, "finished", False),
        }

    @staticmethod
    def _serialize_blocks(blocks: list[Any]) -> Any:
        if len(blocks) == 1 and getattr(blocks[0], "type", None) == "text":
            return getattr(blocks[0], "text", "")
        return [OpenRewardRunnerService._block_to_dict(block) for block in blocks]

    def _create_rollout(
        self,
        *,
        client: OpenRewardClientLike,
        request: OpenRewardExecutionRequest,
        item: NormalizedDatasetItem,
        task_spec: dict[str, Any],
    ) -> Any | None:
        if not request.config.log_rollout:
            return None
        return client.rollout.create(
            run_name=request.config.rollout_run_name or request.run_name,
            rollout_name=item.id,
            environment=request.config.environment_name,
            variant=request.config.variant,
            metadata={
                "dataset_name": request.dataset_name,
                "dataset_item_id": item.id,
                "source_trace_id": item.source_trace_id,
                "source_observation_id": item.source_observation_id,
            },
            task_spec=task_spec,
            print_messages=request.config.print_rollout_messages,
        )

    def _log_rollout_prompt(
        self,
        *,
        rollout: Any | None,
        prompt_blocks: list[Any],
        item: NormalizedDatasetItem,
    ) -> None:
        if rollout is None:
            return
        prompt_text = self._serialize_blocks(prompt_blocks)
        rollout.log(self._system_message_type()(content=_stringify_payload(prompt_text)))
        rollout.log(self._user_message_type()(content=_stringify_payload(item.input)))

    @staticmethod
    def _build_rollout_url(client: OpenRewardClientLike, rollout: Any | None) -> str | None:
        if rollout is None:
            return None
        web_base_url = getattr(client, "_web_base_url", None)
        event_id = getattr(rollout, "event_id", None)
        if not web_base_url or not event_id:
            return None
        return f"{str(web_base_url).rstrip('/')}/rollout/{event_id}"

    @staticmethod
    def _default_client_factory(**kwargs: Any) -> Any:
        from openreward.client import OpenReward

        return OpenReward(**kwargs)

    @staticmethod
    def _task_type() -> Any:
        try:
            from openreward.api.environments.types import Task

            return Task
        except ModuleNotFoundError:  # pragma: no cover - test/runtime fallback
            return _FallbackTask

    @staticmethod
    def _system_message_type() -> Any:
        try:
            from openreward.api.rollouts.serializers.base import SystemMessage

            return SystemMessage
        except ModuleNotFoundError:  # pragma: no cover
            return _FallbackSystemMessage

    @staticmethod
    def _user_message_type() -> Any:
        try:
            from openreward.api.rollouts.serializers.base import UserMessage

            return UserMessage
        except ModuleNotFoundError:  # pragma: no cover
            return _FallbackUserMessage

    @staticmethod
    def _tool_call_type() -> Any:
        try:
            from openreward.api.rollouts.serializers.base import ToolCall

            return ToolCall
        except ModuleNotFoundError:  # pragma: no cover
            return _FallbackToolCall

    @staticmethod
    def _tool_result_type() -> Any:
        try:
            from openreward.api.rollouts.serializers.base import ToolResult

            return ToolResult
        except ModuleNotFoundError:  # pragma: no cover
            return _FallbackToolResult


def _stringify_payload(payload: Any) -> str:
    if payload is None:
        return ""
    if isinstance(payload, str):
        return payload
    return json.dumps(payload, ensure_ascii=True, default=str)
