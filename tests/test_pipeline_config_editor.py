from __future__ import annotations

import unittest

from ui.services.pipeline_config_editor import (
    parse_config_text,
    should_refresh_editor_widget,
    sync_editor_text,
)


class PipelineConfigEditorTests(unittest.TestCase):
    def test_sync_editor_text_refreshes_untouched_buffer(self) -> None:
        current_text, baseline_text = sync_editor_text(
            current_text='{"pipeline": {"analyzers": [{"name": "a"}]}}',
            baseline_text='{"pipeline": {"analyzers": [{"name": "a"}]}}',
            generated_text='{"pipeline": {"analyzers": [{"name": "b"}]}}',
        )

        self.assertEqual(current_text, '{"pipeline": {"analyzers": [{"name": "b"}]}}')
        self.assertEqual(baseline_text, '{"pipeline": {"analyzers": [{"name": "b"}]}}')

    def test_sync_editor_text_preserves_manual_edits(self) -> None:
        current_text, baseline_text = sync_editor_text(
            current_text='{"pipeline": {"analyzers": [{"name": "custom"}]}}',
            baseline_text='{"pipeline": {"analyzers": [{"name": "a"}]}}',
            generated_text='{"pipeline": {"analyzers": [{"name": "b"}]}}',
        )

        self.assertEqual(current_text, '{"pipeline": {"analyzers": [{"name": "custom"}]}}')
        self.assertEqual(baseline_text, '{"pipeline": {"analyzers": [{"name": "b"}]}}')

    def test_parse_config_text_requires_object(self) -> None:
        with self.assertRaisesRegex(ValueError, "oggetto JSON"):
            parse_config_text("[]")

    def test_visible_widget_refresh_uses_previous_baseline(self) -> None:
        self.assertTrue(
            should_refresh_editor_widget(
                widget_text='{"pipeline": {"analyzers": [{"name": "a"}]}}',
                previous_baseline_text='{"pipeline": {"analyzers": [{"name": "a"}]}}',
            )
        )

    def test_visible_widget_keeps_manual_override(self) -> None:
        self.assertFalse(
            should_refresh_editor_widget(
                widget_text='{"pipeline": {"analyzers": [{"name": "custom"}]}}',
                previous_baseline_text='{"pipeline": {"analyzers": [{"name": "a"}]}}',
            )
        )


if __name__ == "__main__":
    unittest.main()
