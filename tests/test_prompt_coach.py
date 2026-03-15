from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from src.core.langfuse_mcp import _build_basic_auth_token, _build_mcp_endpoint
from src.core.prompt_coach_agent import PydanticAIPromptCoachGateway
from src.frontend.pages.experiment_studio import _consume_pending_prompt_apply
from src.core.web_search import DuckDuckGoHTMLParser, _normalize_duckduckgo_url
from src.frontend.prompt_coach_widget import (
    _build_visible_prompt_rows,
    _clear_stale_visible_prompt_versions,
    _summarize_visible_prompt_rows,
    apply_recommended_prompt,
)
from src.schemas.prompt_coach import PromptApplyTarget, PromptCoachDecision, PromptCoachRequest, PromptCoachResponse


class PromptCoachTest(unittest.TestCase):
    def test_normalize_duckduckgo_redirect_url(self) -> None:
        url = "https://duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fdocs"
        self.assertEqual(_normalize_duckduckgo_url(url), "https://example.com/docs")

    def test_duckduckgo_parser_extracts_results(self) -> None:
        parser = DuckDuckGoHTMLParser()
        parser.feed(
            """
            <a class="result__a" href="https://duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fguide">Prompt Guide</a>
            <a class="result__snippet">Helpful prompt engineering reference.</a>
            """
        )
        self.assertEqual(parser.results[0].title, "Prompt Guide")
        self.assertEqual(parser.results[0].url, "https://example.com/guide")

    def test_apply_recommended_prompt_queues_task_and_judge_update(self) -> None:
        state: dict[str, str | bool] = {}
        apply_recommended_prompt(state, target=PromptApplyTarget.BOTH, prompt="Improved prompt")
        self.assertEqual(
            state["studio_pending_prompt_apply"],
            {"targets": ["task", "judge"], "prompt": "Improved prompt"},
        )

    def test_consume_pending_prompt_apply_updates_studio_state(self) -> None:
        state: dict[str, str | bool | dict[str, object]] = {
            "studio_pending_prompt_apply": {"targets": ["task", "judge"], "prompt": "Improved prompt"},
            "studio_task_use_langfuse": True,
            "studio_judge_use_langfuse": True,
            "studio_task_prompt_resolution": {"dummy": True},
            "studio_judge_prompt_resolution": {"dummy": True},
        }
        with patch("src.frontend.pages.experiment_studio.st", SimpleNamespace(session_state=state)):
            _consume_pending_prompt_apply()
        self.assertFalse(state["studio_task_use_langfuse"])
        self.assertFalse(state["studio_judge_use_langfuse"])
        self.assertEqual(state["studio_task_custom_prompt"], "Improved prompt")
        self.assertEqual(state["studio_judge_custom_prompt"], "Improved prompt")
        self.assertNotIn("studio_task_prompt_resolution", state)
        self.assertNotIn("studio_judge_prompt_resolution", state)
        self.assertNotIn("studio_pending_prompt_apply", state)

    def test_gateway_run_registers_tool_without_runcontext_name_error(self) -> None:
        response = PromptCoachResponse(
            decision=PromptCoachDecision.REVISE,
            summary="ok",
            judge_guidance="guidance",
        )

        class FakeRunResult:
            def __init__(self, output: PromptCoachResponse) -> None:
                self.output = output

        class FakeAgent:
            def __init__(self, *args, **kwargs) -> None:
                self.tool_functions = []
                self.kwargs = kwargs

            def tool(self, func=None, **kwargs):
                def decorator(tool_func):
                    self.tool_functions.append(tool_func)
                    return tool_func

                if func is not None:
                    return decorator(func)
                return decorator

            def run_sync(self, prompt, deps=None):
                return FakeRunResult(response)

        gateway = PydanticAIPromptCoachGateway(model_name="openai:gpt-4.1")
        with patch("src.core.prompt_coach_agent.Agent", FakeAgent):
            result = gateway.run(PromptCoachRequest(user_request="Make the judge prompt stricter"))
        self.assertEqual(result.summary, "ok")

    def test_gateway_passes_langfuse_mcp_server_as_toolset(self) -> None:
        response = PromptCoachResponse(
            decision=PromptCoachDecision.APPROVE,
            summary="ok",
        )
        created_agents = []

        class FakeRunResult:
            def __init__(self, output: PromptCoachResponse) -> None:
                self.output = output

        class FakeAgent:
            def __init__(self, *args, **kwargs) -> None:
                self.tool_functions = []
                self.kwargs = kwargs
                created_agents.append(self)

            def tool(self, func=None, **kwargs):
                def decorator(tool_func):
                    self.tool_functions.append(tool_func)
                    return tool_func
                if func is not None:
                    return decorator(func)
                return decorator

            def run_sync(self, prompt, deps=None):
                return FakeRunResult(response)

        fake_mcp_server = object()
        with patch("src.core.prompt_coach_agent.Agent", FakeAgent), patch(
            "src.core.prompt_coach_agent.build_langfuse_mcp_server", return_value=fake_mcp_server
        ):
            gateway = PydanticAIPromptCoachGateway(model_name="openai:gpt-4.1")
            gateway.run(PromptCoachRequest(user_request="Review the judge prompt"))
        self.assertEqual(created_agents[0].kwargs["toolsets"], [fake_mcp_server])

    def test_langfuse_mcp_helpers_build_endpoint_and_auth(self) -> None:
        self.assertEqual(_build_mcp_endpoint("https://cloud.langfuse.com"), "https://cloud.langfuse.com/api/public/mcp")
        self.assertEqual(_build_mcp_endpoint("https://cloud.langfuse.com/api/public/mcp"), "https://cloud.langfuse.com/api/public/mcp")
        self.assertEqual(_build_basic_auth_token("pk", "sk"), "cGs6c2s=")

    def test_build_visible_prompt_rows_and_summary(self) -> None:
        rows = _build_visible_prompt_rows(
            "judge",
            [
                {
                    "name": "judge-prompt",
                    "type": "text",
                    "versions": [3, 2, 1],
                    "labels": ["prod"],
                    "tags": ["published"],
                    "lastUpdatedAt": "2026-03-09T00:00:00Z",
                }
            ],
        )
        self.assertEqual(rows[0]["name"], "judge-prompt")
        self.assertEqual(rows[0]["versions"], "3, 2, 1")
        self.assertEqual(_summarize_visible_prompt_rows(rows), {"judge": 1})

    def test_clear_stale_visible_prompt_versions_resets_previous_rows(self) -> None:
        state = {
            "prompt_coach_visible_prompts": [{"target": "task", "name": "old"}],
            "prompt_coach_visible_prompts_refs": [{"target": "task", "name": "old"}],
        }
        _clear_stale_visible_prompt_versions(state, [{"target": "task", "name": "new"}])
        self.assertNotIn("prompt_coach_visible_prompts", state)
        self.assertEqual(state["prompt_coach_visible_prompts_refs"], [{"target": "task", "name": "new"}])


if __name__ == "__main__":
    unittest.main()
