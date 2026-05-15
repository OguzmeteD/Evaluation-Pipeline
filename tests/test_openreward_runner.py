from __future__ import annotations

import unittest
from dataclasses import dataclass
from typing import Any

from src.core.openreward_runner import OpenRewardRunnerService
from src.schemas.experiment_runner import NormalizedDatasetItem
from src.schemas.openreward_runner import OpenRewardConfig, OpenRewardExecutionRequest


@dataclass
class FakeTextBlock:
    text: str
    detail: dict[str, Any] | None = None
    type: str = "text"


@dataclass
class FakeToolSpec:
    name: str


@dataclass
class FakeToolOutput:
    blocks: list[Any]
    metadata: dict[str, Any] | None = None
    reward: float | None = None
    finished: bool = False


class FakeSession:
    def __init__(self, *, output: FakeToolOutput) -> None:
        self.sid = "sid-123"
        self._output = output
        self.tool_calls: list[tuple[str, dict[str, Any]]] = []

    def __enter__(self) -> "FakeSession":
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def get_prompt(self) -> list[Any]:
        return [FakeTextBlock(text="Solve it.")]

    def list_tools(self) -> list[Any]:
        return [FakeToolSpec(name="submit")]

    def call_tool(self, tool_name: str, input: dict[str, Any]) -> FakeToolOutput:
        self.tool_calls.append((tool_name, input))
        return self._output


class FakeEnvironment:
    def __init__(self, output: FakeToolOutput) -> None:
        self.server = "env"
        self.name = "env"
        self.namespace = "owner"
        self.variant = None
        self.output = output
        self.sessions: list[FakeSession] = []

    def session(self, *, task: Any) -> FakeSession:
        session = FakeSession(output=self.output)
        self.sessions.append(session)
        return session


class FakeEnvironmentsAPI:
    def __init__(self, environment: FakeEnvironment) -> None:
        self.environment = environment
        self.calls: list[dict[str, Any]] = []

    def get(self, name: str, variant: str | None = None) -> FakeEnvironment:
        self.calls.append({"name": name, "variant": variant})
        return self.environment


class FakeRollout:
    def __init__(self, event_id: str = "rollout-1") -> None:
        self.event_id = event_id
        self.logged: list[dict[str, Any]] = []

    def log(self, message: Any, reward: float | None = None, is_finished: bool | None = False, metadata: dict[str, Any] | None = None) -> None:
        self.logged.append(
            {
                "message_type": getattr(message, "type", None),
                "reward": reward,
                "is_finished": is_finished,
                "metadata": metadata,
            }
        )


class FakeRolloutAPI:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.rollouts: list[FakeRollout] = []

    def create(self, **kwargs: Any) -> FakeRollout:
        self.calls.append(kwargs)
        rollout = FakeRollout()
        self.rollouts.append(rollout)
        return rollout


class FakeOpenRewardClient:
    def __init__(self, output: FakeToolOutput) -> None:
        self.environments = FakeEnvironmentsAPI(FakeEnvironment(output))
        self.rollout = FakeRolloutAPI()
        self._web_base_url = "https://openreward.ai"
        self.closed = False

    def close(self) -> None:
        self.closed = True


class OpenRewardRunnerServiceTest(unittest.TestCase):
    def test_tool_input_template_injects_expected_output(self) -> None:
        output = FakeToolOutput(blocks=[FakeTextBlock(text="Correct")], reward=1.0, finished=True)
        client = FakeOpenRewardClient(output)
        service = OpenRewardRunnerService(client_factory=lambda **_: client)
        item = NormalizedDatasetItem(id="item-1", input="What is 2+2?", expected_output="4", metadata={"difficulty": "easy"})
        result = service.run_openreward_execution(
            items=[item],
            request=OpenRewardExecutionRequest(
                config=OpenRewardConfig(
                    environment_name="owner/env",
                    tool_name="submit",
                    tool_input_template={"answer": "{{expected_output}}"},
                    task_spec_template={"question": "{{input}}"},
                ),
                run_name="or-run",
                dataset_name="dataset",
            ),
        )
        self.assertEqual(result.failed_items, 0)
        self.assertEqual(result.item_results[0].tool_input, {"answer": "4"})
        self.assertEqual(result.item_results[0].reward, 1.0)
        self.assertEqual(result.summary["average_reward"], 1.0)

    def test_rollout_logging_creates_rollout_and_url(self) -> None:
        output = FakeToolOutput(blocks=[FakeTextBlock(text="Done")], reward=0.5, finished=True)
        client = FakeOpenRewardClient(output)
        service = OpenRewardRunnerService(client_factory=lambda **_: client)
        item = NormalizedDatasetItem(id="item-2", input="Prompt", expected_output="Answer")
        result = service.run_openreward_execution(
            items=[item],
            request=OpenRewardExecutionRequest(
                config=OpenRewardConfig(
                    environment_name="owner/env",
                    tool_name="submit",
                    log_rollout=True,
                ),
                run_name="or-run",
                dataset_name="dataset",
            ),
        )
        self.assertEqual(len(client.rollout.calls), 1)
        self.assertEqual(result.item_results[0].rollout_url, "https://openreward.ai/rollout/rollout-1")
        self.assertTrue(client.closed)


if __name__ == "__main__":
    unittest.main()
