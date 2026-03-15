from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from src.frontend.pages.experiment_studio import (
    _build_prompt_browser_options,
    _build_prompt_browser_rows,
    _consume_pending_prompt_browser_selection,
    _normalize_chat_editor_rows,
    _render_value_block,
    _serialize_chat_messages,
)


class _FakeStreamlit:
    def __init__(self) -> None:
        self.calls: list[tuple[str, object]] = []

    def caption(self, value):
        self.calls.append(("caption", value))

    def json(self, value):
        self.calls.append(("json", value))

    def code(self, value, language="text"):
        self.calls.append(("code", value))

    def text_area(self, label, value="", height=0, disabled=False):
        self.calls.append(("text_area", {"label": label, "value": value, "height": height, "disabled": disabled}))


class ExperimentStudioPageTest(unittest.TestCase):
    def test_render_value_block_uses_text_area_for_plain_string(self) -> None:
        fake_st = _FakeStreamlit()
        with patch("src.frontend.pages.experiment_studio.st", fake_st):
            _render_value_block("Input payload", "Arcelik buzdolabi oner")
        self.assertEqual(fake_st.calls[0][0], "text_area")

    def test_render_value_block_uses_json_for_json_string(self) -> None:
        fake_st = _FakeStreamlit()
        with patch("src.frontend.pages.experiment_studio.st", fake_st):
            _render_value_block("Input payload", '{"query":"Arcelik"}')
        self.assertEqual(fake_st.calls[0], ("json", {"query": "Arcelik"}))

    def test_normalize_chat_editor_rows_filters_empty_rows(self) -> None:
        rows = _normalize_chat_editor_rows(
            [
                {"role": "system", "content": "You are helpful."},
                {"role": "", "content": ""},
                {"role": "user", "content": "Merhaba"},
            ]
        )
        self.assertEqual(
            rows,
            [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Merhaba"},
            ],
        )

    def test_serialize_chat_messages_formats_preview(self) -> None:
        preview = _serialize_chat_messages(
            [{"role": "system", "content": "Be strict"}, {"role": "user", "content": "Score this"}]
        )
        self.assertEqual(preview, "[system] Be strict\n[user] Score this")

    def test_build_prompt_browser_rows_and_options(self) -> None:
        raw_rows = [
            {
                "name": "judge-support",
                "type": "text",
                "versions": [1, 2],
                "labels": ["production"],
                "tags": ["published"],
                "last_updated_at": "2026-03-09T10:00:00Z",
            }
        ]
        rows = _build_prompt_browser_rows(raw_rows)
        self.assertEqual(rows[0]["versions"], "1, 2")
        options = _build_prompt_browser_options(raw_rows)
        self.assertIn("judge-support | v2 | label=production", options)

    def test_consume_pending_prompt_browser_selection_updates_state(self) -> None:
        state = {
            "studio_pending_prompt_browser_selection": {
                "prefix": "judge",
                "prompt_name": "judge-support",
                "prompt_label": "production",
                "prompt_version": "2",
            },
            "studio_judge_prompt_resolution": {"dummy": True},
        }
        with patch("src.frontend.pages.experiment_studio.st", SimpleNamespace(session_state=state)):
            _consume_pending_prompt_browser_selection()
        self.assertTrue(state["studio_judge_use_langfuse"])
        self.assertEqual(state["studio_judge_prompt_name"], "judge-support")
        self.assertEqual(state["studio_judge_prompt_label"], "production")
        self.assertEqual(state["studio_judge_prompt_version"], "2")
        self.assertNotIn("studio_judge_prompt_resolution", state)
        self.assertNotIn("studio_pending_prompt_browser_selection", state)


if __name__ == "__main__":
    unittest.main()
