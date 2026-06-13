from __future__ import annotations

import io

from sef.cli.output import CliMessageRenderer


def test_cli_renderer_formats_plain_status_when_color_is_disabled() -> None:
    renderer = CliMessageRenderer(stream=io.StringIO(), enable_color=False)

    assert renderer.status("warning", "Optional extra is missing") == "SEF  warning        Optional extra is missing"
    assert renderer.detail("suggestion", "Install sef[opencv]") == "  suggestion: Install sef[opencv]"


def test_cli_renderer_styles_brand_and_severity_when_color_is_enabled() -> None:
    renderer = CliMessageRenderer(stream=io.StringIO(), enable_color=True)

    message = renderer.status("error", "Missing dependency")

    assert "\033[1m\033[36mSEF\033[0m" in message
    assert "\033[31merror\033[0m" in message
    assert "Missing dependency" in message
